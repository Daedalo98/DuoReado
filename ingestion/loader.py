"""ingestion/loader.py
Folder-based ingestion. Any PDF dropped into data/pdfs/ becomes a book.

Language resolution order (first hit wins):
    1. <name>.pdf.meta.json   → {"language": "es", "title": "...", "author": "..."}
    2. filename suffix        → don_quijote_es.pdf → "es"
    3. config.DEFAULT_LANGUAGE

Chapter resolution order (first hit wins):
    1. <name>.pdf.chapters.json → manual specs (edited from the web UI)
    2. PDF TOC / heading regex   → auto-detected
    3. A single "Full Text" chapter

Every chapter carries an explicit {title, start_page, end_page}. A sidecar
value of 0 for end_page means "up to the next start, or end of book".
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable
from dataclasses import asdict

from config import (
    DEFAULT_LANGUAGE,
    MAX_SENTENCES_PER_WORD,
    PDFS_DIR,
    log,
    ensure_directories,

)
from ingestion.json_builder import (
    book_path,
    build_book_index,
    load_book_index,
    load_granted_lemmas,
    save_book_index,
    save_book_text,
    save_sentences,
    sentences_path,
    text_path,
)

from ingestion.pdf_extractor import PdfExtractionResult, extract_pdf_pages
from ingestion.text_normalizer import extract_tokens, split_sentences
from models.schemas import BookIndex, ChapterInfo, Occurrence

ProgressFn = Callable[[str, int, int], None] | None


# ── Discovery & naming ──────────────────────────────────────────────────
def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "book"


def book_id_for_pdf(pdf_path: str | Path) -> str:
    return _slugify(Path(pdf_path).stem)


def discover_pdfs(pdfs_dir: str | Path | None = None) -> list[Path]:
    folder = Path(pdfs_dir) if pdfs_dir else PDFS_DIR
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.pdf") if p.is_file())


# ── Sidecars ────────────────────────────────────────────────────────────
def meta_sidecar_path(pdf_path: str | Path) -> Path:
    pdf = Path(pdf_path)
    return pdf.with_suffix(pdf.suffix + ".meta.json")


def chapter_sidecar_path(pdf_path: str | Path) -> Path:
    pdf = Path(pdf_path)
    return pdf.with_suffix(pdf.suffix + ".chapters.json")


def read_sidecar_metadata(pdf_path: Path) -> dict:
    sidecar = meta_sidecar_path(pdf_path)
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_chapter_sidecar(pdf_path: str | Path) -> list[dict] | None:
    """Return ``[{title, start_page, end_page}, ...]`` or None if absent."""
    path = chapter_sidecar_path(pdf_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    chapters = data.get("chapters") if isinstance(data, dict) else None
    if not isinstance(chapters, list) or not chapters:
        return None
    out = []
    for i, c in enumerate(chapters):
        if not isinstance(c, dict):
            continue
        out.append({
            "title": str(c.get("title") or f"Chapter {i+1}").strip(),
            "start_page": int(c.get("start_page") or 1),
            "end_page": int(c.get("end_page") or 0),
        })
    return out or None


def write_chapter_sidecar(pdf_path: str | Path, chapters: list[dict]) -> Path:
    """Validate, normalize, and save ``<pdf>.chapters.json``."""
    path = chapter_sidecar_path(pdf_path)
    normalized = []
    for i, c in enumerate(chapters):
        normalized.append({
            "title": str(c.get("title") or f"Chapter {i+1}").strip(),
            "start_page": int(c.get("start_page") or 1),
            "end_page": int(c.get("end_page") or 0),
        })
    path.write_text(
        json.dumps({"chapters": normalized}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ── Language ────────────────────────────────────────────────────────────
def infer_language(pdf_path: Path, default: str = DEFAULT_LANGUAGE) -> str:
    sidecar_lang = (read_sidecar_metadata(pdf_path).get("language") or "").strip().lower()
    if len(sidecar_lang) >= 2:
        return sidecar_lang[:2]
    stem = pdf_path.stem
    if "_" in stem:
        suffix = stem.rsplit("_", 1)[1].lower()
        if len(suffix) == 2 and suffix.isalpha():
            return suffix
    return default


def is_ingested(pdf_path: Path) -> bool:
    return book_path(book_id_for_pdf(pdf_path)).exists()


# ── Chapter slicing ─────────────────────────────────────────────────────
def _normalize_chapter_specs(
    specs: list | None,
    total_pages: int,
) -> list[dict] | None:
    """Accept tuples or dicts; clamp to [1, total]; fill missing ends;
    resolve overlaps by clamping the *previous* chapter's end_page."""
    if not specs:
        return None

    out: list[dict] = []
    for i, spec in enumerate(specs):
        if isinstance(spec, (tuple, list)) and len(spec) >= 2:
            title, start = spec[0], spec[1]
            end = spec[2] if len(spec) >= 3 else 0
        elif isinstance(spec, dict):
            title = spec.get("title", f"Chapter {i+1}")
            start = spec.get("start_page", 1)
            end = spec.get("end_page", 0) or 0
        else:
            continue
        start = max(1, min(total_pages, int(start)))
        end = int(end) if end else 0
        if end:
            end = max(start, min(total_pages, end))
        out.append({
            "title": str(title).strip() or f"Chapter {i+1}",
            "start_page": start,
            "end_page": end,
        })

    if not out:
        return None
    out.sort(key=lambda c: c["start_page"])

    # Fill missing ends from the next start (or total_pages).
    for i, ch in enumerate(out):
        if not ch["end_page"]:
            ch["end_page"] = (
                max(ch["start_page"], out[i + 1]["start_page"] - 1)
                if i + 1 < len(out)
                else total_pages
            )

    # Clamp overlaps so ranges are strictly monotonic.
    for i in range(len(out) - 1):
        if out[i]["end_page"] >= out[i + 1]["start_page"]:
            out[i]["end_page"] = out[i + 1]["start_page"] - 1

    return out


def _build_chapters(
    extraction: PdfExtractionResult,
    specs: list | None,
) -> list[ChapterInfo]:
    # Priority: explicit specs → auto-detected → single full-text chapter.
    normalized = (
        _normalize_chapter_specs(specs, extraction.total_pages)
        or _normalize_chapter_specs(extraction.suggested_chapters, extraction.total_pages)
        or _normalize_chapter_specs([("Full Text", 1)], extraction.total_pages)
    )
    chapters: list[ChapterInfo] = []
    for idx, ch in enumerate(normalized):
        # NOTE: we intentionally do NOT populate `text`.
        # The full prose lives in data/texts/<book_id>.txt (see save_book_text).
        # Keeping ChapterInfo light is what makes the main JSON browsable.
        chapters.append(ChapterInfo(
            chapter_id=f"ch_{idx + 1}",
            number=idx + 1,
            title=ch["title"],
            start_page=ch["start_page"],
            end_page=ch["end_page"],
        ))
    return chapters



# ── Tokenisation ────────────────────────────────────────────────────────
def _tokenize(
    chapters: list[ChapterInfo],
    extraction: PdfExtractionResult,
    language: str,
    granted: set[str],
) -> dict[str, list[Occurrence]]:
    """Fill chapter.word_counts / word_lemmas; return sentences-per-lemma.

    Granted lemmas are dropped here, so they never reach the vocabulary JSON.
    Also logs how many were filtered and a sample, for full transparency.
    """
    sentences_by_lemma: dict[str, list[Occurrence]] = {}
    filtered_unique: set[str] = set()

    for ch in chapters:
        counts: Counter[str] = Counter()
        order: list[str] = []
        pages = [p for p in extraction.pages if ch.start_page <= p.page_number <= ch.end_page]

        for page in pages:
            for sentence in split_sentences(page.text):
                if not sentence:
                    continue
                for tok in extract_tokens(sentence, language=language):
                    lemma = (tok.lemma or "").strip().lower()
                    if len(lemma) < 2:
                        continue
                    if lemma in granted:
                        filtered_unique.add(lemma)
                        continue

                    counts[lemma] += 1
                    if lemma not in order:
                        order.append(lemma)

                    bucket = sentences_by_lemma.setdefault(lemma, [])
                    if len(bucket) < MAX_SENTENCES_PER_WORD:
                        if not any(o.sentence == sentence for o in bucket):
                            bucket.append(Occurrence(
                                chapter_id=ch.chapter_id,
                                page=page.page_number,
                                snippet=lemma,
                                sentence=sentence,
                            ))

        ch.word_counts = dict(counts)
        ch.word_lemmas = order

    # Transparency: show what the granted filter removed.
    if granted:
        sample = sorted(filtered_unique)[:15]
        more = "…" if len(filtered_unique) > 15 else ""
        log.info(
            "Granted filter (%s): %d unique lemma(s) excluded — %s%s",
            language, len(filtered_unique), ", ".join(sample), more,
        )
    return sentences_by_lemma

# ── Public API ──────────────────────────────────────────────────────────
def ingest_pdf(
    pdf_path: str | Path,
    *,
    language: str | None = None,
    title: str | None = None,
    author: str | None = None,
    chapter_specs: list | None = None,
    force: bool = False,
    on_progress: ProgressFn = None,
) -> BookIndex:
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    book_id = book_id_for_pdf(pdf)
    if not force and book_path(book_id).exists():
        if on_progress:
            on_progress("already-ingested", 1, 1)
        return load_book_index(book_id)

    meta = read_sidecar_metadata(pdf)
    lang = (language or meta.get("language") or infer_language(pdf))[:2]

    # ── 1. Extract text ────────────────────────────────────────────────
    if on_progress: on_progress("extracting", 1, 6)
    extraction = extract_pdf_pages(pdf)

    # ── 2. Save full text to data/texts/<book_id>.txt ─────────────────
    if on_progress: on_progress("saving text file", 2, 6)
    save_book_text(book_id, extraction.pages)

    # ── 3. Slice chapters (no text embedded) ──────────────────────────
    if on_progress: on_progress("slicing chapters", 3, 6)
    specs = chapter_specs if chapter_specs is not None else read_chapter_sidecar(pdf)
    chapters = _build_chapters(extraction, specs)

    # ── 4. Tokenise + save sentence sidecar ───────────────────────────
    if on_progress: on_progress("tokenizing", 4, 6)
    granted = load_granted_lemmas(lang)
    sentences_by_lemma = _tokenize(chapters, extraction, lang, granted)
    save_sentences(book_id, sentences_by_lemma)

    # ── 5. Build the small main JSON ──────────────────────────────────
    if on_progress: on_progress("aggregating vocabulary", 5, 6)
    book = build_book_index(
        book_id=book_id,
        title=title or meta.get("title") or extraction.title,
        language=lang,
        source_pdf=str(pdf),
        chapters=chapters,
        granted=granted,
        metadata={
            "author": author or meta.get("author") or extraction.author,
            "total_pages": extraction.total_pages,
            "granted_count": len(granted),
        },
    )

    # ── 6. Save + report sizes ────────────────────────────────────────
    if on_progress: on_progress("saving book index", 6, 6)
    save_book_index(book)

    def _size(p: Path) -> str:
        if not p.exists(): return "0 KB"
        n = p.stat().st_size
        return f"{n // 1024} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"

    log.info(
        "Ingest sizes for %s — main=%s, sentences=%s, text=%s",
        book_id,
        _size(book_path(book_id)),
        _size(sentences_path(book_id)),
        _size(text_path(book_id)),
    )
    return book


def ingest_folder(*, force: bool = False, on_progress: ProgressFn = None) -> list[dict]:
    results: list[dict] = []
    for pdf in discover_pdfs():
        try:
            book = ingest_pdf(pdf, force=force, on_progress=on_progress)
            results.append({
                "pdf": pdf.name, "book_id": book.book_id,
                "words": len(book.words), "chapters": len(book.chapters),
                "ok": True,
            })
        except Exception as exc:
            results.append({"pdf": pdf.name, "ok": False, "error": str(exc)})
    return results


# ── CLI ────────────────────────────────────────────────────────────────
def _print_progress(stage: str, current: int, total: int) -> None:
    print(f"  [{current}/{total}] {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DuoReado folder ingestion")
    parser.add_argument("--pdf", help="ingest one PDF")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    if args.scan:
        for r in ingest_folder(force=args.force, on_progress=_print_progress):
            print(r)
    elif args.pdf:
        book = ingest_pdf(args.pdf, language=args.lang, force=args.force,
                          on_progress=_print_progress)
        print(f"{book.book_id}: {len(book.words)} words, {len(book.chapters)} chapters")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
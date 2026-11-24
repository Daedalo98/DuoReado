""" ingestion/loader.py

Folder-based ingestion. Any PDF dropped into data/pdfs/ becomes a book.

Language resolution order (first hit wins):
    1. <name>.pdf.meta.json  →  {"language": "es", "title": "...", "author": "..."}
    2. filename suffix       →  don_quijote_es.pdf  →  "es"
    3. config.DEFAULT_LANGUAGE

Re-ingestion is safe: user progress lives in UserProgress, never in the book JSON.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from config import (
    DEFAULT_LANGUAGE,
    MAX_SENTENCES_PER_WORD,
    PDFS_DIR,
    ensure_directories,
)
from ingestion.json_builder import (
    book_path,
    build_book_index,
    load_book_index,
    load_granted_lemmas,
    save_book_index,
)
from ingestion.pdf_extractor import PdfExtractionResult, extract_pdf_pages
from ingestion.text_normalizer import extract_tokens, split_sentences
from models.schemas import BookIndex, ChapterInfo, Occurrence

ProgressFn = Callable[[str, int, int], None] | None


# ---------------------------------------------------------------------------
# Discovery & naming
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "book"


def book_id_for_pdf(pdf_path: str | Path) -> str:
    """Stable id derived from the filename — no page count, so renames are free."""
    return _slugify(Path(pdf_path).stem)


def discover_pdfs(pdfs_dir: str | Path | None = None) -> list[Path]:
    folder = Path(pdfs_dir) if pdfs_dir else PDFS_DIR
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.pdf") if p.is_file())


# ---------------------------------------------------------------------------
# Metadata & language inference
# ---------------------------------------------------------------------------

def read_sidecar_metadata(pdf_path: Path) -> dict:
    """Optional <name>.pdf.meta.json next to the PDF."""
    sidecar = pdf_path.with_suffix(pdf_path.suffix + ".meta.json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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


# ---------------------------------------------------------------------------
# Chapter slicing
# ---------------------------------------------------------------------------

def _build_chapters(
    extraction: PdfExtractionResult,
    specs: list[tuple[str, int]] | None,
) -> list[ChapterInfo]:
    raw = specs or extraction.suggested_chapters or [("Full Text", 1)]
    sorted_specs = sorted(
        (
            (str(t).strip() or f"Chapter {i + 1}",
             max(1, min(extraction.total_pages, int(sp))))
            for i, (t, sp) in enumerate(raw)
        ),
        key=lambda x: x[1],
    )
    chapters: list[ChapterInfo] = []
    for idx, (title, start_p) in enumerate(sorted_specs):
        end_p = (
            sorted_specs[idx + 1][1] - 1
            if idx + 1 < len(sorted_specs)
            else extraction.total_pages
        )
        end_p = max(start_p, end_p)
        pages = [p for p in extraction.pages if start_p <= p.page_number <= end_p]
        text = "\n\n".join(p.text for p in pages if p.text.strip())
        chapters.append(ChapterInfo(
            chapter_id=f"ch_{idx + 1}",
            number=idx + 1,
            title=title,
            start_page=start_p,
            end_page=end_p,
            text=text,
        ))
    return chapters


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenize(
    chapters: list[ChapterInfo],
    extraction: PdfExtractionResult,
    language: str,
    granted: set[str],
) -> dict[str, list[Occurrence]]:
    """Fill chapter.word_counts / word_lemmas; return sentences-per-lemma.

    Granted lemmas are dropped here, so they never reach the vocabulary JSON.
    """
    sentences_by_lemma: dict[str, list[Occurrence]] = {}

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
                    if len(lemma) < 2 or lemma in granted:
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

    return sentences_by_lemma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: str | Path,
    *,
    language: str | None = None,
    title: str | None = None,
    author: str | None = None,
    chapter_specs: list[tuple[str, int]] | None = None,
    force: bool = False,
    on_progress: ProgressFn = None,
) -> BookIndex:
    """Ingest one PDF into data/books/<slug>.json. Returns the BookIndex."""
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

    if on_progress: on_progress("extracting", 1, 5)
    extraction = extract_pdf_pages(pdf)

    if on_progress: on_progress("slicing chapters", 2, 5)
    chapters = _build_chapters(extraction, chapter_specs)

    if on_progress: on_progress("tokenizing", 3, 5)
    granted = load_granted_lemmas(lang)
    sentences_by_lemma = _tokenize(chapters, extraction, lang, granted)

    if on_progress: on_progress("aggregating vocabulary", 4, 5)
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
        sentences_by_lemma=sentences_by_lemma,   # ← one-line patch, see §3
    )

    if on_progress: on_progress("saving", 5, 5)
    save_book_index(book)
    return book


def ingest_folder(
    *,
    force: bool = False,
    on_progress: ProgressFn = None,
) -> list[dict]:
    """Ingest every PDF in data/pdfs/ that has no JSON yet (or all, if force)."""
    results: list[dict] = []
    for pdf in discover_pdfs():
        try:
            book = ingest_pdf(pdf, force=force, on_progress=on_progress)
            results.append({
                "pdf": pdf.name,
                "book_id": book.book_id,
                "words": len(book.words),
                "chapters": len(book.chapters),
                "ok": True,
            })
        except Exception as exc:
            results.append({"pdf": pdf.name, "ok": False, "error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_progress(stage: str, current: int, total: int) -> None:
    print(f"  [{current}/{total}] {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DuoReado folder ingestion")
    parser.add_argument("--pdf", help="ingest one PDF")
    parser.add_argument("--scan", action="store_true", help="ingest every new PDF in data/pdfs/")
    parser.add_argument("--lang", default=None, help="override language code")
    parser.add_argument("--force", action="store_true", help="rebuild even if JSON exists")
    args = parser.parse_args()

    ensure_directories()

    if args.scan:
        for result in ingest_folder(force=args.force, on_progress=_print_progress):
            print(result)
    elif args.pdf:
        book = ingest_pdf(args.pdf, language=args.lang, force=args.force,
                          on_progress=_print_progress)
        print(f"{book.book_id}: {len(book.words)} words, {len(book.chapters)} chapters")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
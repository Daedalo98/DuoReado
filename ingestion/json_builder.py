"""
ingestion/json_builder.py
Builds the clean .txt export and the structured .json book/word index.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

from config import BOOKS_DIR, TEXTS_DIR, ensure_directories
from ingestion.pdf_extractor import PdfExtractionResult, extract_pdf_pages
from ingestion.text_normalizer import extract_tokens, split_sentences
from models.schemas import BookIndex, ChapterInfo, Occurrence, WordEntry, utc_now_iso


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "book"


def get_book_json_path(book_id: str) -> Path:
    ensure_directories()
    return BOOKS_DIR / f"{book_id}.json"


def list_saved_books() -> list[dict[str, str]]:
    """Return summary metadata for all previously built book JSON files."""
    ensure_directories()
    books: list[dict[str, str]] = []
    for json_file in sorted(BOOKS_DIR.glob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            meta = raw.get("metadata", {})
            books.append(
                {
                    "book_id": meta.get("book_id", json_file.stem),
                    "title": meta.get("title", json_file.stem),
                    "author": meta.get("author", "Unknown"),
                    "language": meta.get("language", "en"),
                    "json_path": str(json_file),
                }
            )
        except Exception:
            continue
    return books


def load_book_index(json_path: str | Path) -> BookIndex:
    """Load a BookIndex from a JSON file."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return BookIndex.from_dict(data)


def save_book_index(book: BookIndex, json_path: str | Path | None = None) -> Path:
    """
    Atomically write BookIndex to disk (write to temp file, then os.replace)
    so user accuracy/progress is never corrupted by an unexpected exit.
    """
    ensure_directories()
    book.updated_at = utc_now_iso()
    target = Path(json_path) if json_path else get_book_json_path(book.book_id)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = target.with_suffix(".json.tmp")
    payload = json.dumps(book.to_dict(), ensure_ascii=False, indent=2)
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, target)
    return target


def build_book_index(
    pdf_path: str | Path,
    title: str,
    author: str,
    language: str,
    chapter_specs: list[tuple[str, int]] | None = None,
    existing_book: BookIndex | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    extraction: PdfExtractionResult | None = None,
) -> BookIndex:
    """
    Full ingestion pipeline:
    1. Extract pages from PDF (if not already extracted).
    2. Write cleaned full text to a .txt file.
    3. Slice pages into chapters based on chapter_specs ([(title, start_page), ...]).
    4. Extract sentences and normalized word lemmas per chapter/page.
    5. Build/merge BookIndex and save JSON.
    """
    ensure_directories()
    pdf_file = Path(pdf_path)

    if extraction is None:
        if on_progress:
            on_progress("Extracting text from PDF pages...", 1, 4)
        extraction = extract_pdf_pages(pdf_file)

    book_id = existing_book.book_id if existing_book else f"{_slugify(title)}_{extraction.total_pages}p"
    txt_path = TEXTS_DIR / f"{book_id}.txt"

    # 1. Export full clean text to .txt
    if on_progress:
        on_progress("Writing cleaned text file...", 2, 4)
    full_txt_parts = [
        f"--- PAGE {p.page_number} ---\n{p.text}" for p in extraction.pages if p.text.strip()
    ]
    txt_path.write_text("\n\n".join(full_txt_parts), encoding="utf-8")

    # 2. Build chapter ranges from [(title, start_page), ...]
    raw_specs = chapter_specs if chapter_specs else extraction.suggested_chapters
    if not raw_specs:
        raw_specs = [("Chapter 1", 1)]

    # Sort and clamp start pages to [1, total_pages]
    sorted_specs = sorted(
        [(t.strip() or f"Chapter {i+1}", max(1, min(extraction.total_pages, int(sp))))
         for i, (t, sp) in enumerate(raw_specs)],
        key=lambda x: x[1],
    )

    chapters: list[ChapterInfo] = []
    for idx, (ch_title, start_p) in enumerate(sorted_specs):
        if idx + 1 < len(sorted_specs):
            next_start = sorted_specs[idx + 1][1]
            end_p = max(start_p, next_start - 1)
        else:
            end_p = extraction.total_pages

        ch_pages = [
            p for p in extraction.pages if start_p <= p.page_number <= end_p
        ]
        ch_text = "\n\n".join(p.text for p in ch_pages if p.text.strip())
        chapters.append(
            ChapterInfo(
                chapter_id=f"ch_{idx + 1}",
                title=ch_title,
                start_page=start_p,
                end_page=end_p,
                text=ch_text,
                word_lemmas=[],
            )
        )

    # 3. Process words per chapter & page
    if on_progress:
        on_progress("Tokenizing chapters and indexing words...", 3, 4)

    words_map: dict[str, WordEntry] = {}
    previous_words = existing_book.words if existing_book else {}

    for ch in chapters:
        ch_lemma_set: set[str] = set()

        if on_progress:
            # Map chapter progress smoothly across steps 3 -> 4 (out of 100 total units)
            ch_step = 25 + int(((chapters.index(ch) + 1) / max(1, len(chapters))) * 65)
            on_progress(
                f"Indexing '{ch.title}' (p.{ch.start_page}-{ch.end_page}) — {len(words_map)} lemmas so far",
                ch_step,
                100,
            )
            
        ch_pages = [
            p for p in extraction.pages if ch.start_page <= p.page_number <= ch.end_page
        ]

        for page_obj in ch_pages:
            sentences = split_sentences(page_obj.text)
            for sent in sentences:
                tokens = extract_tokens(sent, language=language)
                for tok in tokens:
                    lemma = tok.lemma
                    ch_lemma_set.add(lemma)

                    if lemma not in words_map:
                        prev = previous_words.get(lemma)
                        words_map[lemma] = WordEntry(
                            lemma=lemma,
                            pos=tok.pos,
                            known_status=prev.known_status if prev else "new",
                            attempts=prev.attempts if prev else 0,
                            correct=prev.correct if prev else 0,
                            accuracy=prev.accuracy if prev else 0.0,
                            streak=prev.streak if prev else 0,
                            last_reviewed=prev.last_reviewed if prev else None,
                            cached_translation=prev.cached_translation if prev else "",
                        )

                    entry = words_map[lemma]
                    if not entry.pos and tok.pos:
                        entry.pos = tok.pos

                    entry.total_count += 1
                    entry.forms[tok.surface] = entry.forms.get(tok.surface, 0) + 1
                    entry.by_chapter[ch.chapter_id] = entry.by_chapter.get(ch.chapter_id, 0) + 1

                    if page_obj.page_number not in entry.pages:
                        entry.pages.append(page_obj.page_number)

                    # Store up to 6 example sentences per word to keep JSON fast & compact
                    if len(entry.occurrences) < 6:
                        entry.occurrences.append(
                            Occurrence(
                                chapter_id=ch.chapter_id,
                                page=page_obj.page_number,
                                sentence=sent,
                            )
                        )

        ch.word_lemmas = sorted(ch_lemma_set)

    # Sort page lists cleanly
    for entry in words_map.values():
        entry.pages.sort()

    if on_progress:
        on_progress("Saving book JSON index...", 4, 4)

    now = utc_now_iso()
    book = BookIndex(
        book_id=book_id,
        title=title.strip() or extraction.title,
        author=author.strip() or extraction.author,
        language=(language or "en").strip().lower(),
        total_pages=extraction.total_pages,
        source_pdf=str(pdf_file.resolve()),
        clean_txt_path=str(txt_path.resolve()),
        created_at=existing_book.created_at if existing_book else now,
        updated_at=now,
        chapters=chapters,
        words=words_map,
    )

    save_book_index(book)
    return book
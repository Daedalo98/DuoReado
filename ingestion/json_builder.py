"""ingestion/json_builder.py

Phase 1 of the pipeline. Produces the *skeleton* vocabulary JSON:

    {"words": {"abuelo": {"occurrences": 14, "first_chapter": 2,
                          "entries": [], "linguistics": {...}, "semantic_keywords": []}}}

The empty fields are filled later by ai/enrichment.py.

Design notes
------------
- Granted words (`data/granted/granted_<lang>.json`) are removed BEFORE
  aggregation so they never appear in any output structure.
- The loader is language-agnostic: pass "es", "fr", "pt", ...
- `build_book_index` accepts an already-parsed list of ChapterInfo. PDF
  parsing / tokenization belong to the caller; this module only aggregates.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from config import DATA_DIR
from models.schemas import BookIndex, ChapterInfo, VocabularyEntry

BOOKS_DIR = DATA_DIR / "books"
GRANTED_DIR = DATA_DIR / "granted"

# Keys inside the granted JSON that hold prose, not words.
_NOTE_KEYS = {"note", "notes", "comment", "comments", "explanation", "gloss"}


# ---------------------------------------------------------------------------
# Granted-words loader (multi-language)
# ---------------------------------------------------------------------------

def _granted_paths(language: str) -> list[Path]:
    """Ordered search paths for the granted file of `language`."""
    return [
        GRANTED_DIR / f"granted_{language}.json",     # canonical
        DATA_DIR / f"granted_{language}.json",        # legacy location
        DATA_DIR / "granted.json",                    # single-language fallback
    ]


def _flatten_granted(node, out: set[str], parent_key: str | None = None) -> None:
    """Recursively walk the granted JSON and collect every word string.

    - Strings whose *parent key* is a prose key (`note`, ...) are skipped.
    - Both sides of `accented_homonym_pairs` are collected, because both
      are considered granted by the user.
    - Lowercasing makes matching case-insensitive (most lemmatizers already
      return lowercase, but this is safe either way).
    """
    if isinstance(node, str):
        if parent_key in _NOTE_KEYS:
            return
        s = node.strip().lower()
        if s:
            out.add(s)
    elif isinstance(node, dict):
        for key, value in node.items():
            _flatten_granted(value, out, parent_key=key)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _flatten_granted(item, out, parent_key=parent_key)


def load_granted_lemmas(language: str) -> set[str]:
    """Return the set of lemmas the user already knows for `language`.

    Returns an empty set if no granted file exists — the pipeline still
    works, it just does not filter anything.
    """
    for path in _granted_paths(language):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out: set[str] = set()
        _flatten_granted(raw, out)
        return out
    return set()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _chapter_counter(chapter: ChapterInfo) -> Counter[str]:
    """Prefer the explicit word_counts; fall back to counting word_lemmas."""
    if chapter.word_counts:
        return Counter(chapter.word_counts)
    return Counter(chapter.word_lemmas)


def _filter_chapter(chapter: ChapterInfo, granted: set[str]) -> ChapterInfo:
    """Return a copy of `chapter` with granted lemmas removed from its lists."""
    kept_lemmas = [l for l in chapter.word_lemmas if l.lower() not in granted]
    kept_counts = {
        k: v for k, v in chapter.word_counts.items() if k.lower() not in granted
    }
    return replace(chapter, word_lemmas=kept_lemmas, word_counts=kept_counts)


def build_book_index(
    *,
    book_id: str,
    title: str,
    language: str,
    source_pdf: str,
    chapters: Iterable[ChapterInfo],
    granted: set[str] | None = None,
    metadata: dict | None = None,
) -> BookIndex:
    """Assemble a BookIndex from already-parsed chapters.

    `granted` is optional; if omitted it is loaded from disk for `language`.
    Passing it explicitly is the recommended way to unit-test the builder.
    """
    granted = granted if granted is not None else load_granted_lemmas(language)

    book = BookIndex(
        book_id=book_id,
        title=title,
        language=language,
        source_pdf=source_pdf,
        metadata=dict(metadata or {}),
    )

    for chapter in chapters:
        # 1) Strip granted lemmas so they never reach the output.
        filtered = _filter_chapter(chapter, granted)
        book.chapters.append(filtered)

        # 2) Aggregate ONLY the cheap fields. Enrichment stays empty.
        counts = _chapter_counter(filtered)
        for lemma, count in counts.items():
            if not lemma:
                continue
            entry = book.words.setdefault(lemma, VocabularyEntry())
            entry.occurrences += count
            if entry.first_chapter == 0:
                entry.first_chapter = filtered.number

    return book


# ---------------------------------------------------------------------------
# Persistence (atomic write)
# ---------------------------------------------------------------------------

def book_path(book_id: str, books_dir: Path | None = None) -> Path:
    return (books_dir or BOOKS_DIR) / f"{book_id}.json"


def save_book_index(book: BookIndex, books_dir: Path | None = None) -> Path:
    """Write the book JSON atomically (tmp + os.replace)."""
    path = book_path(book.book_id, books_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(book.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def load_book_index(book_id: str, books_dir: Path | None = None) -> BookIndex:
    path = book_path(book_id, books_dir)
    if not path.exists():
        raise FileNotFoundError(f"Book JSON not found: {path}")
    return BookIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_book_ids(books_dir: Path | None = None) -> list[str]:
    d = books_dir or BOOKS_DIR
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
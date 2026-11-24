"""ingestion/json_builder.py
BookIndex serialisation + the missing helpers that other modules rely on:
    * book_path(book_id)                — path to <book_id>.json
    * load_granted_lemmas(language)     — set of lemmas to skip at build time
    * build_book_index(...)             — aggregate loader output into BookIndex
    * load_book_index / save_book_index — encoding-safe, atomic I/O
    * list_book_ids / list_saved_books  — discovery helpers

Re-ingestion is always safe because user progress lives in UserProgress,
never in the BookIndex JSON.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    BOOKS_DIR,
    GRANTED_DIR,
    TEXTS_DIR,
    ensure_directories,
    log,
    safe_read_text,
    safe_write_text,
)
from models.schemas import (
    BookIndex,
    ChapterInfo,
    Occurrence,
    VocabularyEntry,
)


# ── Small helpers ────────────────────────────────────────────────────────
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "book"


# ── Paths ────────────────────────────────────────────────────────────────
def book_path(book_id: str) -> Path:
    """Path to the on-disk JSON for a given book id."""
    ensure_directories()
    return BOOKS_DIR / f"{book_id}.json"


# Backwards-compatible alias (older code imported `get_book_json_path`).
get_book_json_path = book_path

def sentences_path(book_id: str) -> Path:
    ensure_directories()
    return BOOKS_DIR / f"{book_id}.sentences.json"


def text_path(book_id: str) -> Path:
    ensure_directories()
    return TEXTS_DIR / f"{book_id}.txt"

# ── Text sidecar (paginated prose) ──────────────────────────────────────
_PAGE_MARKER_RE = re.compile(r"^---\s*PAGE\s+(\d+)\s*---\s*$", re.MULTILINE)

def save_book_text(book_id: str, pages: list) -> Path:
    """Write a paginated text file: ``--- PAGE N ---`` followed by page text.

    Accepts any object with ``.page_number`` and ``.text`` (ExtractedPage).
    """
    path = text_path(book_id)
    parts = [
        f"--- PAGE {p.page_number} ---\n{p.text}"
        for p in pages
        if p.text.strip()
    ]
    safe_write_text(path, "\n\n".join(parts))
    log.info("Saved full text → %s (%d KB)", path, path.stat().st_size // 1024)
    return path

def read_chapter_text(book_id: str, chapter: ChapterInfo) -> str:
    """Return the chapter's prose.

    Prefers the paginated ``.txt`` file. Falls back to ``chapter.text``
    (embedded in old book JSONs before this split) so previously-ingested
    books keep working.
    """
    if chapter.text:
        return chapter.text

    path = text_path(book_id)
    if not path.exists():
        return ""
    raw = safe_read_text(path)

    # Split into alternating (page_number, page_text) chunks.
    pages_by_num: dict[int, str] = {}
    parts = _PAGE_MARKER_RE.split(raw)
    for i in range(1, len(parts) - 1, 2):
        try:
            num = int(parts[i])
        except ValueError:
            continue
        pages_by_num[num] = parts[i + 1].strip()

    selected = [
        pages_by_num[n]
        for n in range(chapter.start_page, chapter.end_page + 1)
        if n in pages_by_num
    ]
    return "\n\n".join(selected)


# ── Sentences sidecar ───────────────────────────────────────────────────
def save_sentences(book_id: str, sentences_by_lemma: dict[str, list[Occurrence]]) -> Path:
    path = sentences_path(book_id)
    payload: dict[str, list[dict]] = {}
    for lemma, occs in sentences_by_lemma.items():
        if not occs:
            continue
        payload[lemma] = [asdict(o) for o in occs]
    safe_write_text(path, json.dumps(payload, ensure_ascii=False))
    log.info("Saved sentences → %s (%d KB, %d lemmas)",
             path, path.stat().st_size // 1024, len(payload))
    return path


def load_sentences(book_id: str) -> dict[str, list[Occurrence]]:
    path = sentences_path(book_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(safe_read_text(path))
    except Exception as exc:
        log.warning("Could not load sentences for %s: %s", book_id, exc)
        return {}
    out: dict[str, list[Occurrence]] = {}
    for lemma, occs in data.items():
        try:
            out[lemma] = [Occurrence(**o) for o in occs]
        except Exception:
            continue
    return out


# ── Granted lemmas ──────────────────────────────────────────────────────
# The granted file may be a flat list ["a", "el", …], the old
# {"lemmas": [...]} shape, OR the rich structured layout you have for
# Spanish (nested categories + accented_homonym_pairs). The walker below
# handles all three.
#
# Rules:
#   * every *string value* becomes a lemma (lowercased, punctuation stripped)
#   * multi-word expressions are split into their individual words, so
#     "sin embargo" grants both "sin" and "embargo" — matches what the
#     tokenizer actually produces
#   * keys named note / description / comment / hint are skipped, so prose
#     like "article vs. pronoun (he/him)" never becomes a "lemma"
#   * the file can live in several conventional places; we search them in order
#     and fall back to a bounded rglob over the project root

_PROSE_KEYS = {"note", "description", "comment", "hint", "explanation"}

_STRIP_CHARS = ".,;:!?\"'()[]{}«»¿¡“”‘’—–"


def _collect_granted(value: Any, out: set[str]) -> None:
    """Recursively flatten a granted-words JSON value into `out`."""
    if isinstance(value, str):
        # Multi-word phrase → individual words. Strip surrounding punctuation.
        for token in value.lower().replace(",", " ").split():
            token = token.strip(_STRIP_CHARS)
            if token:
                out.add(token)
    elif isinstance(value, list):
        for item in value:
            _collect_granted(item, out)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in _PROSE_KEYS:
                continue
            _collect_granted(item, out)


def _read_granted_file(path: Path) -> set[str]:
    try:
        data = json.loads(safe_read_text(path))
    except Exception as exc:
        log.error("Could not parse granted-lemma file %s: %s", path, exc)
        return set()
    out: set[str] = set()
    _collect_granted(data, out)
    log.info("Loaded %d granted lemma(s) from %s", len(out), path)
    return out


def load_granted_lemmas(language: str) -> set[str]:
    """Return the set of lemmas the user is *not* studying for `language`.

    Missing file → empty set (nothing filtered), with a warning.
    """
    lang = (language or "en").lower()[:2]

    path = GRANTED_DIR / f"granted_{lang}.json"
    if path.exists():
        return _read_granted_file(path)
    else:
        log.warning("No granted-lemma file found at %s", path)
        return set()

# ── Load / save BookIndex ────────────────────────────────────────────────
def load_book_index(book_id_or_path: str | Path) -> BookIndex:
    """Load a BookIndex from an id (`don_quijote_es`) or a full path."""
    path = Path(book_id_or_path)
    if not path.suffix:            # caller passed an id, not a path
        path = book_path(str(path))
    raw = safe_read_text(path)
    return BookIndex.from_dict(json.loads(raw))


def save_book_index(book: BookIndex) -> Path:
    """Atomically write a BookIndex to disk (temp + os.replace)."""
    ensure_directories()
    target = book_path(book.book_id)
    tmp = target.with_suffix(".json.tmp")
    payload = json.dumps(book.to_dict(), ensure_ascii=False, indent=2)
    safe_write_text(tmp, payload)
    os.replace(tmp, target)
    log.debug("Saved book index → %s", target)
    return target


# ── Discovery ────────────────────────────────────────────────────────────
def list_book_ids() -> list[str]:
    ensure_directories()
    return sorted(
        p.stem for p in BOOKS_DIR.glob("*.json")
        if not p.stem.endswith(".sentences")
    )


def list_saved_books() -> list[dict[str, str]]:
    ensure_directories()
    out: list[dict[str, str]] = []
    for json_file in sorted(BOOKS_DIR.glob("*.json")):
        if json_file.stem.endswith(".sentences"):
            continue
        try:
            data = json.loads(safe_read_text(json_file))
            out.append({
                "book_id": data.get("book_id", json_file.stem),
                "title": data.get("title", json_file.stem),
                "author": data.get("metadata", {}).get("author", "Unknown"),
                "language": data.get("language", "en"),
                "json_path": str(json_file),
            })
        except Exception as exc:
            log.warning("Skipping corrupt book file %s: %s", json_file, exc)
    return out


# ── Aggregation (loader) ────────────────────────────────────────────────
def build_book_index(
    *,
    book_id: str,
    title: str,
    language: str,
    source_pdf: str,
    chapters: list[ChapterInfo],
    granted: set[str],
    metadata: dict[str, Any] | None = None,
) -> BookIndex:
    """Aggregate tokenised chapters into a BookIndex.

    Sentences are NOT embedded here — the loader saves them to a sidecar
    via ``save_sentences``. This keeps the main JSON small.
    """
    words: dict[str, VocabularyEntry] = {}
    for ch in chapters:
        for lemma, count in ch.word_counts.items():
            if lemma in granted:
                continue
            entry = words.get(lemma)
            if entry is None:
                entry = VocabularyEntry()
                words[lemma] = entry
            entry.occurrences += int(count)
            if entry.first_chapter == 0 or ch.number < entry.first_chapter:
                entry.first_chapter = ch.number

    return BookIndex(
        book_id=book_id,
        title=title,
        language=language,
        source_pdf=source_pdf,
        metadata=dict(metadata or {}),
        chapters=chapters,
        words=words,
    )


# ── Called by the CLI in loader.py ──────────────────────────────────────
def _print_progress(stage: str, current: int, total: int) -> None:
    print(f"  [{current}/{total}] {stage}")
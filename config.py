"""config.py
Central configuration for file paths, Ollama settings, learning thresholds,
and shared utilities (encoding-safe file I/O, progress-callback type).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Tuple

import chardet

# ── Logging setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Project directories ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"          # NEW — imported by loader / phone_bridge
BOOKS_DIR = DATA_DIR / "books"
TEXTS_DIR = DATA_DIR / "texts"
CACHE_DIR = DATA_DIR / "cache"
USERS_DIR = DATA_DIR / "users"
GRANTED_DIR = DATA_DIR / "granted"    # granted_<lang>.json per-language lists

# ── Ollama settings ─────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "granite4.2:3b"
DECISION_MODEL = "tev1:4b"
OLLAMA_TIMEOUT_SECONDS = 180
DEFAULT_BATCH_SIZE = 10
DEFAULT_SENTENCES_PER_WORD = 4
MAX_SENTENCES_PER_WORD = 6

# ── Languages ────────────────────────────────────────────────────────────
DEFAULT_LANGUAGE = "en"               # NEW — used by loader.infer_language fallback

# ── spaCy models ────────────────────────────────────────────────────────
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "it": "it_core_news_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "pt": "pt_core_news_sm",
}

# ── Mastery & readiness ────────────────────────────────────────────────
MASTERED_MIN_ATTEMPTS = 3
MASTERED_MIN_ACCURACY = 0.80
CHAPTER_READINESS_TARGET = 0.90

# ── Shared types ────────────────────────────────────────────────────────
# The single progress-callback signature used everywhere:
#   (message: str, current_step: int, total_steps: int) -> None
ProgressCallback = Callable[[str, int, int], None]


# ── Encoding-safe helpers ───────────────────────────────────────────────
def safe_read_text(
    path: Path | str,
    encodings: Tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1"),
) -> str:
    """Read a text file, trying each encoding in turn, then chardet.

    Never raises UnicodeDecodeError: if all explicit encodings fail we
    let `chardet` guess (which handles PDF/HTML/legacy bytes) and finally
    fall back to latin-1 with replacement so the caller always gets a str.
    """
    path = Path(path)
    raw = path.read_bytes()

    # 1. Try the caller's preferred encodings.
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as exc:
            log.warning("Could not decode %s as %s: %s", path, enc, exc)

    # 2. Try chardet (handles PDFs / HTML / older charsets).
    detected = (chardet.detect(raw) or {}).get("encoding") or "latin-1"
    log.warning("Falling back to chardet encoding %r for %s", detected, path)
    try:
        return raw.decode(detected, errors="replace")
    except (LookupError, UnicodeDecodeError):
        # 3. Last-resort: latin-1 always decodes (with possible '?').
        return raw.decode("latin-1", errors="replace")


def safe_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """Write text to disk, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def ensure_directories() -> None:
    """Create all data directories if they do not exist yet."""
    for folder in (DATA_DIR, PDFS_DIR, BOOKS_DIR, TEXTS_DIR, CACHE_DIR, USERS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
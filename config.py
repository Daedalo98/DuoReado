""" config.py — central configuration for file paths, Ollama, thresholds. """
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# --- Ingestion roots -------------------------------------------------------
PDFS_DIR    = DATA_DIR / "pdfs"       # ← NEW: drop PDFs here
BOOKS_DIR   = DATA_DIR / "books"      # generated vocabulary JSON
GRANTED_DIR = DATA_DIR / "granted"    # ← NEW: granted_es.json, granted_fr.json, ...
USERS_DIR   = DATA_DIR / "users"      # ← NEW: per-user progress
CACHE_DIR   = DATA_DIR / "cache"      # Ollama response cache

# Legacy folder kept only so old .txt exports do not break; nothing reads it.
TEXTS_DIR   = DATA_DIR / "texts"

# Default language when the filename gives no hint (e.g. "don_quijote.pdf").
DEFAULT_LANGUAGE = "es"

# How many book sentences to keep per word (used by the quiz + reader).
MAX_SENTENCES_PER_WORD = 5

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "granite4.2:3b"
DECISION_MODEL = "tev1:4b"
OLLAMA_TIMEOUT_SECONDS = 180

# spaCy models by 2-letter language code
SPACY_MODELS = {
    "en": "en_core_web_sm", "it": "it_core_news_sm", "es": "es_core_news_sm",
    "fr": "fr_core_news_sm", "de": "de_core_news_sm", "pt": "pt_core_news_sm",
}

# Mastery / readiness thresholds
MASTERED_MIN_ATTEMPTS = 3
MASTERED_MIN_ACCURACY = 0.80
CHAPTER_READINESS_TARGET = 0.90


def ensure_directories() -> None:
    for folder in (DATA_DIR, PDFS_DIR, BOOKS_DIR, GRANTED_DIR, USERS_DIR, CACHE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
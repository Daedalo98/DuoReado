"""
config.py
Central configuration for file paths, Ollama settings, and learning thresholds.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Where imported books (.json), exported clean texts (.txt), and AI caches live
DATA_DIR = BASE_DIR / "data"
BOOKS_DIR = DATA_DIR / "books"
TEXTS_DIR = DATA_DIR / "texts"
CACHE_DIR = DATA_DIR / "cache"
GRANTED_ES_PATH = Path(__file__).resolve().parent / "spanish" / "granted_es.json"


# Default Ollama server settings
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "granite4.2:3b"
DECISION_MODEL = "tev1:4b"
OLLAMA_TIMEOUT_SECONDS = 180

DEFAULT_BATCH_SIZE = 10
DEFAULT_SENTENCES_PER_WORD = 4

# Optional spaCy model names by 2-letter language code.
# If spaCy or the language model isn't installed, text_normalizer gracefully
# falls back to a clean regex/lowercase normalizer.
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "it": "it_core_news_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "pt": "pt_core_news_sm",
}

# Mastery & Chapter Readiness thresholds
MASTERED_MIN_ATTEMPTS = 3
MASTERED_MIN_ACCURACY = 0.80       # 80%+ accuracy across >= 3 attempts
CHAPTER_READINESS_TARGET = 0.90    # 90% of chapter words known/mastered unlocks "smooth reading"


def ensure_directories() -> None:
    """Create data directories if they do not exist yet."""
    for folder in (DATA_DIR, BOOKS_DIR, TEXTS_DIR, CACHE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
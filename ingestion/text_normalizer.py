"""
ingestion/text_normalizer.py
Sentence segmentation and word normalization/lemmatization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import SPACY_MODELS

# Matches words including accented Latin letters and internal apostrophes/hyphens
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…»”\"])\s+")

_SPACY_CACHE: dict[str, object | None] = {}


@dataclass
class TokenInfo:
    surface: str  # Exact form in text, e.g., "andavano"
    lemma: str    # Normalized base form, e.g., "andare"
    pos: str      # Part of speech tag if available, e.g., "VERB"


def _get_nlp(language: str):
    """Lazily load and cache a spaCy pipeline for the target language."""
    lang = (language or "en").lower()[:2]
    if lang in _SPACY_CACHE:
        return _SPACY_CACHE[lang]

    model_name = SPACY_MODELS.get(lang)
    if not model_name:
        _SPACY_CACHE[lang] = None
        return None

    try:
        import spacy  # type: ignore
        nlp = spacy.load(model_name, disable=["ner", "parser"])
        # Add sentencizer so sentence boundaries work cleanly
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _SPACY_CACHE[lang] = nlp
        return nlp
    except Exception:
        _SPACY_CACHE[lang] = None
        return None


def split_sentences(text: str) -> list[str]:
    """Split text into readable sentences."""
    if not text.strip():
        return []
    # Normalize internal newlines to spaces for clean sentence display
    flat = re.sub(r"\s+", " ", text.strip())
    parts = SENTENCE_SPLIT_RE.split(flat)
    return [s.strip() for s in parts if s.strip()]


def extract_tokens(text: str, language: str = "en") -> list[TokenInfo]:
    """
    Extract normalized word tokens (surface form, lemma, POS) from text.
    Uses spaCy when installed for the language; falls back to lowercase regex tokens.
    """
    nlp = _get_nlp(language)
    if nlp is not None:
        doc = nlp(text)
        tokens: list[TokenInfo] = []
        for tok in doc:
            if tok.is_alpha and len(tok.text) > 1:
                lemma = (tok.lemma_ or tok.text).strip().lower()
                # Guard against spaCy returning "-PRON-" or weird punctuation
                if not lemma or lemma.startswith("-") or not WORD_RE.fullmatch(lemma):
                    lemma = tok.text.lower()
                tokens.append(
                    TokenInfo(
                        surface=tok.text,
                        lemma=lemma,
                        pos=tok.pos_ or "",
                    )
                )
        return tokens

    # Fallback tokenizer when spaCy model is not installed
    tokens = []
    for match in WORD_RE.finditer(text):
        surface = match.group(0)
        if len(surface) > 1:
            tokens.append(
                TokenInfo(
                    surface=surface,
                    lemma=surface.lower(),
                    pos="",
                )
            )
    return tokens
"""ai/enrichment.py
Shared enrichment service: turns a bare lemma into a fully populated
`VocabularyEntry` (forms, translations, explanation, edge cases, keywords).

Design rules
------------
* The prompt is JSON-only, so parsing is deterministic (see `_extract_json_object`).
* We enrich *the lemma*, never the chapter context — this module is book-agnostic.
* Every helper here is idempotent: calling it twice on a non-empty entry is a no-op.
"""
from __future__ import annotations

from ai.ollama_client import OllamaClient, OllamaError, _extract_json_object
from models.schemas import WordFormEntry, WordLinguistics, VocabularyEntry

# Prompt intentionally tight: only a JSON object, no prose.
ENRICH_SYSTEM = """You are a precise bilingual lexicographer.
Return ONLY a valid JSON object with this shape:
{
  "entries": [
    {"form": "<inflected form>", "relation": "base|plural|past|...",
     "translations": ["...", "..."]}
  ],
  "explanation": "<1-3 sentence usage note>",
  "edge_cases": ["<common mistake or subtlety>", "..."],
  "semantic_keywords": ["<single word>", "..."]
}
Rules:
- At least one entry MUST have relation "base".
- translations are in the user's native language.
- No markdown fences. No prose outside the JSON object."""


def _parse(raw: str) -> dict | None:
    """Wrapper so we can swap parsers later without touching call sites."""
    return _extract_json_object(raw)


def enrich_if_empty(
    entry: VocabularyEntry,
    lemma: str,
    language: str,
    client: OllamaClient,
    native_language: str = "en",
) -> bool:
    """Populate `entry` in-place if it has no translations yet.

    Returns True iff new data was written. When False, the caller should
    treat the entry as already enriched (and skip re-saving the book).
    """
    # Idempotency guard — already done.
    if entry.has_translations:
        return False

    prompt = (
        f"Target language: {language}. Native language: {native_language}.\n"
        f"Enrich the {language} lemma '{lemma}'."
    )

    try:
        raw = client.generate(prompt, system=ENRICH_SYSTEM, use_cache=True)
    except OllamaError:
        return False

    data = _parse(raw)
    if not isinstance(data, dict):
        return False

    # ── entries ──────────────────────────────────────────────────────
    raw_entries = data.get("entries") or []
    new_entries: list[WordFormEntry] = []
    for e in raw_entries:
        if not isinstance(e, dict):
            continue
        form = str(e.get("form") or lemma).strip()
        relation = str(e.get("relation") or "base").strip() or "base"
        translations = [str(t).strip() for t in (e.get("translations") or []) if str(t).strip()]
        if not translations:
            continue
        new_entries.append(
            WordFormEntry(form=form, relation=relation, translations=translations)
        )
    if not new_entries:
        # No usable translations → don't mark enriched.
        return False
    entry.entries = new_entries

    # ── linguistics ──────────────────────────────────────────────────
    ling = data.get("linguistics") if isinstance(data.get("linguistics"), dict) else {}
    explanation = str(data.get("explanation") or ling.get("explanation") or "").strip()
    edge_cases = [
        str(x).strip()
        for x in (data.get("edge_cases") or ling.get("edge_cases") or [])
        if str(x).strip()
    ]
    entry.linguistics = WordLinguistics(
        explanation=explanation,
        edge_cases=edge_cases,
    )

    # ── semantic keywords ────────────────────────────────────────────
    kws = [
        str(k).strip().lstrip("#")
        for k in (data.get("semantic_keywords") or [])
        if str(k).strip()
    ]
    entry.semantic_keywords = kws

    return True
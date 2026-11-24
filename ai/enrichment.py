"""ai/enrichment.py

Standalone vocabulary enrichment. Takes **only a lemma and a language**;
it never sees a BookIndex, a chapter, or a UI session. This is what makes
the AI section a self-contained service.

It calls the transport layer (`OllamaClient.generate`) and then parses the
response with the same tolerant parser the client already ships
(`_extract_json_object`), aliased here as `_parse_json` so the rest of the
module reads cleanly.
"""
from __future__ import annotations

from typing import Any

from ai.ollama_client import OllamaClient, OllamaError, _extract_json_object
from models.schemas import VocabularyEntry, WordFormEntry, WordLinguistics

# Alias so the body of the module can call a stable name.
_parse_json = _extract_json_object


ENRICH_SYSTEM = """You are a lexicographer.
Return ONLY a valid JSON object with this exact schema:
{
  "entries": [{"form": str, "relation": str, "translations": [str]}],
  "linguistics": {"explanation": str, "edge_cases": [str]},
  "semantic_keywords": [str]
}
Rules:
- "relation" is one of: base, feminine variation, masculine variation,
  plural variation, feminine plural variation, diminutive, augmentative,
  related form.
- Include the base form first.
- Keep the explanation under 60 words.
- Provide 3 to 6 semantic keywords.
No markdown fences. No prose outside the JSON object."""


def enrich_word(
    lemma: str,
    language: str,
    client: OllamaClient,
    *,
    native_language: str = "en",
) -> dict[str, Any] | None:
    """Ask the LLM to describe `lemma`. Returns the parsed payload or None."""
    prompt = (
        f"Target language: {language}. Native language: {native_language}.\n"
        f"Word: '{lemma}'.\n"
        "List the base form, gender/number variations, diminutives and "
        "common related forms. Add a concise linguistic explanation, "
        "edge cases (colloquial / idiomatic uses) and semantic keywords."
    )
    try:
        raw = client.generate(prompt, system=ENRICH_SYSTEM, use_cache=True)
    except OllamaError:
        return None
    return _parse_json(raw)


def apply_enrichment(entry: VocabularyEntry, payload: dict[str, Any]) -> None:
    """Merge the LLM payload into a (previously empty) VocabularyEntry."""
    entry.entries = [
        WordFormEntry(
            form=e.get("form", ""),
            relation=e.get("relation", "base"),
            translations=list(e.get("translations", [])),
        )
        for e in payload.get("entries", [])
    ]
    ling = payload.get("linguistics", {}) or {}
    entry.linguistics = WordLinguistics(
        explanation=ling.get("explanation", ""),
        edge_cases=list(ling.get("edge_cases", [])),
    )
    entry.semantic_keywords = list(payload.get("semantic_keywords", []))


def enrich_if_empty(
    entry: VocabularyEntry,
    lemma: str,
    language: str,
    client: OllamaClient,
    *,
    native_language: str = "en",
) -> bool:
    """Convenience: enrich only if the entry has not been enriched yet.

    Returns True if enrichment happened, False otherwise (already rich or
    LLM failure). Does **not** persist — callers save the book JSON.
    """
    if entry.entries:
        return False
    payload = enrich_word(lemma, language, client, native_language=native_language)
    if not payload:
        return False
    apply_enrichment(entry, payload)
    return True
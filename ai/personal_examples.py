"""ai/personal_examples.py

Standalone personalized sentence generator. Inputs: a lemma, the target
language, and the user's native language. Output: a PersonalExample.

The book is never consulted — this is the canonical "disconnect from the
book" module for Requirement 3.
"""
from __future__ import annotations

from ai.ollama_client import OllamaClient, OllamaError, _extract_json_object
from models.schemas import PersonalExample

_parse_json = _extract_json_object


EXAMPLE_SYSTEM = """You write example sentences for language learners.
Return ONLY a valid JSON object: {"sentence": str, "translation": str}.
- "sentence" is natural, 8–16 words, in the target language.
- "translation" is a faithful translation into the native language.
- Use one inflected form of the requested word that fits naturally.
No markdown fences. No prose outside the JSON object."""


def generate_example(
    lemma: str,
    *,
    target_language: str,
    native_language: str,
    client: OllamaClient,
    context_note: str = "",
) -> PersonalExample | None:
    """Generate one personalized example sentence for `lemma`."""
    extra = f"\nExtra context: {context_note}" if context_note else ""
    prompt = (
        f"Target language: {target_language}. Native language: {native_language}.\n"
        f"Write ONE natural example sentence using '{lemma}'.{extra}"
    )
    try:
        raw = client.generate(prompt, system=EXAMPLE_SYSTEM, use_cache=True)
    except OllamaError:
        return None
    data = _parse_json(raw)
    if not isinstance(data, dict):
        return None
    sentence = (data.get("sentence") or "").strip()
    translation = (data.get("translation") or "").strip()
    if not sentence or not translation:
        return None
    return PersonalExample(
        lemma=lemma,
        sentence=sentence,
        translation=translation,
        source="llm",
    )
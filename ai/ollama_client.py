"""
ai/ollama_client.py
Client for local Ollama LLM inference with response caching and structured prompts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from config import (
    CACHE_DIR,
    DECISION_MODEL,
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT_SECONDS,
    ensure_directories,
)

def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Safely parse a JSON object from LLM output, stripping markdown fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening ```json or ``` and closing ```
        lines = [ln for ln in lines[1:] if ln.strip() != "```"]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        # Attempt to slice from first '{' to last '}' if prose surrounded the JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an invalid response."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        ensure_directories()

    def list_models(self) -> list[str]:
        """Return names of locally installed Ollama models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return models or [self.model]
        except Exception:
            return [self.model]

    def _cache_key(self, prompt: str, system: str) -> Path:
        digest = hashlib.sha256(f"{self.model}||{system}||{prompt}".encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{digest}.json"

    def generate(self, prompt: str, system: str = "", use_cache: bool = True) -> str:
        """
        Send a prompt to Ollama's /api/generate endpoint (non-streaming for reliable parsing).
        Caches responses on disk so repeated lookups are instant.
        """
        cache_path = self._cache_key(prompt, system)
        if use_cache and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("response"):
                    return cached["response"]
            except Exception:
                pass

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise OllamaError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Ensure `ollama serve` is running and model '{self.model}' is pulled."
            ) from exc

        text = (data.get("response") or "").strip()
        if not text:
            raise OllamaError("Ollama returned an empty response.")

        if use_cache:
            try:
                cache_path.write_text(json.dumps({"response": text}, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        return text

    def translate_word(self, word: str, source_lang: str, context_sentence: str = "") -> str:
        """Return a concise translation and core meaning of a word in context."""
        system = "You are a concise, accurate language tutor. Give brief, clear answers."
        prompt = (
            f"Target language: {source_lang}\n"
            f"Word: '{word}'\n"
            f"Context sentence: '{context_sentence}'\n\n"
            "Provide:\n"
            "1) Primary English meaning(s) (1 short line)\n"
            "2) How it functions in this context (1 short line)"
        )
        return self.generate(prompt, system=system)

    def explain_sentence(self, sentence: str, source_lang: str) -> str:
        """Translate a sentence into English and explain key grammar, vocabulary, and nuance."""
        system = (
            "You are an encouraging, clear language coach helping a learner read a book "
            "in a foreign language. Keep explanations structured and easy to scan."
        )
        prompt = (
            f"Source language: {source_lang}\n"
            f"Sentence: \"{sentence}\"\n\n"
            "Respond with:\n"
            "**Natural Translation:** ...\n"
            "**Literal / Structure Note:** ...\n"
            "**Key Words & Grammar:** bullet points of tricky words, tenses, or idioms."
        )
        return self.generate(prompt, system=system)

    def generate_cloze_question(self, lemma: str, source_lang: str, fallback_sentence: str = "") -> dict[str, str]:
        """
        Generate a fill-in-the-blank sentence using `lemma` in `source_lang`.
        Returns a dict with 'sentence_with_blank', 'answer', and 'english_hint'.
        """
        system = (
            "You create language learning fill-in-the-blank exercises. "
            "Always reply with valid JSON only, no markdown fences."
        )
        prompt = (
            f"Create one natural sentence in {source_lang} that uses the word '{lemma}' "
            f"(inflected naturally if needed). Replace the target word in the sentence with '_____'.\n"
            "Return JSON with exact keys:\n"
            '{"sentence_with_blank": "...", "answer": "...", "english_hint": "..."}'
        )
        raw = self.generate(prompt, system=system, use_cache=True)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "", 1).strip()

        try:
            parsed = json.loads(cleaned)
            if "sentence_with_blank" in parsed and "answer" in parsed:
                return {
                    "sentence_with_blank": str(parsed["sentence_with_blank"]),
                    "answer": str(parsed["answer"]),
                    "english_hint": str(parsed.get("english_hint", "")),
                }
        except Exception:
            pass

        # Fallback if model didn't return strict JSON
        blanked = fallback_sentence or f"_____ ({lemma})"
        if fallback_sentence and lemma.lower() in fallback_sentence.lower():
            blanked = fallback_sentence.replace(lemma, "_____")
        return {
            "sentence_with_blank": blanked,
            "answer": lemma,
            "english_hint": f"Base word: {lemma}",
        }

    def grade_translation(
        self,
        source_sentence: str,
        user_translation: str,
        source_lang: str,
    ) -> str:
        """Evaluate a user's translation of a book sentence and give constructive feedback."""
        system = (
            "You are a supportive language tutor grading a learner's translation into English. "
            "Be encouraging, accurate, and concise."
        )
        prompt = (
            f"Source language: {source_lang}\n"
            f"Original sentence: \"{source_sentence}\"\n"
            f"Learner's English translation: \"{user_translation}\"\n\n"
            "Provide:\n"
            "**Score / Verdict:** (e.g., Spot-on / Good with minor tweaks / Needs another look)\n"
            "**Reference Translation:** ...\n"
            "**Feedback:** brief explanation of what worked or what nuance was missed."
        )
        return self.generate(prompt, system=system, use_cache=False)

    def enrich_word_entry(
        self,
        lemma: str,
        source_lang: str,
        num_sentences: int,
        existing_count: int = 0,
        context_sentence: str = "",
    ) -> dict[str, Any]:
        """
        Generate linguistic metadata and example sentences for a word when JSON data
        is missing or fewer than `num_sentences` sentences exist.

        Returns a normalized dict with:
        - translations: list[str]
        - false_friends: list[str]
        - edge_meanings: list[str]
        - sentences: list[dict[str, str]]  (each has sentence, translation, related_word)
        """
        needed_sentences = max(0, int(num_sentences) - int(existing_count))
        system = (
            "You are an expert lexicographer and language tutor. "
            "Respond strictly with valid JSON only—no extra prose outside the JSON object."
        )
        prompt = (
            f"Analyze the target-language word/lemma '{lemma}' (language code: {source_lang}).\n"
            f"Book context snippet (if any): '{context_sentence}'\n"
            f"Generate up to {max(1, needed_sentences)} new example sentences.\n"
            "Include related or inflected word forms where natural (e.g., gender/number/tense "
            "variants or closely related words like 'abuelo', 'abuela', 'abuelos').\n\n"
            "Return JSON with this exact schema:\n"
            "{\n"
            '  "translations": ["common english meaning 1", "common english meaning 2"],\n'
            '  "false_friends": ["false friend or common mistranslation note"],\n'
            '  "edge_meanings": ["idiomatic, figurative, or peculiar usage note"],\n'
            '  "sentences": [\n'
            "    {\n"
            '      "sentence": "Example sentence in target language.",\n'
            '      "translation": "Natural English translation.",\n'
            '      "related_word": "surface or related word used in the sentence"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        raw = self.generate(prompt, system=system, use_cache=True)
        data = _extract_json_object(raw) or {}

        def _clean_str_list(key: str) -> list[str]:
            val = data.get(key, [])
            if not isinstance(val, list):
                return []
            return [str(x).strip() for x in val if str(x).strip()]

        raw_sentences = data.get("sentences", [])
        cleaned_sentences: list[dict[str, str]] = []
        if isinstance(raw_sentences, list):
            for item in raw_sentences[: max(0, needed_sentences)]:
                if isinstance(item, dict):
                    sent = str(item.get("sentence", "")).strip()
                    trans = str(item.get("translation", "")).strip()
                    rel = str(item.get("related_word", lemma)).strip() or lemma
                    if sent:
                        cleaned_sentences.append(
                            {
                                "sentence": sent,
                                "translation": trans or "(Translation pending)",
                                "related_word": rel,
                                "source": "llm",
                            }
                        )

        return {
            "translations": _clean_str_list("translations"),
            "false_friends": _clean_str_list("false_friends"),
            "edge_meanings": _clean_str_list("edge_meanings"),
            "sentences": cleaned_sentences,
        }


    def judge_translation_with_decision_model(
        self,
        source_sentence: str,
        user_translation: str,
        source_lang: str,
        decision_model: str = DECISION_MODEL,
    ) -> dict[str, Any]:
        """
        Use a fast decision model (default: tev1:4b) to decide whether `user_translation`
        is an acceptable translation of `source_sentence`.
        Returns {"valid": bool, "reason": str}.
        """
        system = (
            "You are a fast translation validator. Decide if the learner's translation "
            "accurately captures the meaning of the source sentence. "
            "Reply strictly with JSON: {\"valid\": true or false, \"reason\": \"short reason\"}."
        )
        prompt = (
            f"Source language: {source_lang}\n"
            f"Source sentence: {source_sentence}\n"
            f"Learner translation: {user_translation}\n"
            "Is the learner translation valid?"
        )

        # Temporarily run against decision_model while preserving self.model
        original_model = self.model
        try:
            self.model = decision_model
            raw = self.generate(prompt, system=system, use_cache=False)
        finally:
            self.model = original_model

        parsed = _extract_json_object(raw)
        if parsed is not None and "valid" in parsed:
            is_valid = bool(parsed["valid"])
            reason = str(parsed.get("reason", "")).strip() or (
                "Accepted by decision model." if is_valid else "Meaning mismatch detected."
            )
            return {"valid": is_valid, "reason": reason}

        # Fallback heuristic if model returned plain text
        lower = raw.lower()
        is_valid = ("true" in lower or "valid" in lower) and "invalid" not in lower and "not valid" not in lower
        return {"valid": is_valid, "reason": raw.strip()[:200]}
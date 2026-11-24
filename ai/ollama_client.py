"""ai/ollama_client.py
Client for local Ollama LLM inference with response caching.
Now uses config.safe_read_text / safe_write_text to avoid UTF‑8 crashes.
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
    log,
    safe_read_text,
    safe_write_text,
)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Safely parse a JSON object from LLM output, stripping markdown fences."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines[1:] if ln.strip() != "```"]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        # Try to slice from first '{' to last '}'
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
        log.debug("OllamaClient initialised with model=%s", model)

    def list_models(self) -> list[str]:
        """Return names of locally installed Ollama models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return models or [self.model]
        except Exception as exc:
            log.warning("Could not list Ollama models: %s", exc)
            return [self.model]

    def _cache_key(self, prompt: str, system: str) -> Path:
        digest = hashlib.sha256(
            f"{self.model}||{system}||{prompt}".encode("utf-8")
        ).hexdigest()
        return CACHE_DIR / f"{digest}.json"

    def generate(self, prompt: str, system: str = "", use_cache: bool = True) -> str:
        """
        Send a prompt to Ollama's /api/generate endpoint.
        Caches responses on disk so repeated lookups are instant.
        All file reads/writes go through config.safe_* helpers.
        """
        cache_path = self._cache_key(prompt, system)

        # ── 1. Try cache first ────────────────────────────────────────
        if use_cache and cache_path.exists():
            try:
                raw = safe_read_text(cache_path)
                cached = json.loads(raw)
                if cached.get("response"):
                    log.debug("Cache hit for prompt %s", cache_path.name)
                    return cached["response"]
            except Exception as exc:
                log.warning("Could not read cache %s: %s", cache_path, exc)

        # ── 2. Call Ollama ──────────────────────────────────────────
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        log.info("Calling Ollama model %s", self.model)
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log.error("Ollama request failed: %s", exc)
            raise OllamaError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Ensure `ollama serve` is running and model '{self.model}' is pulled."
            ) from exc

        text = (data.get("response") or "").strip()
        if not text:
            raise OllamaError("Ollama returned an empty response.")

        # ── 3. Persist to cache ──────────────────────────────────────
        if use_cache:
            try:
                safe_write_text(
                    cache_path,
                    json.dumps({"response": text}, ensure_ascii=False),
                )
                log.debug("Cached response to %s", cache_path.name)
            except Exception as exc:
                log.warning("Could not write cache %s: %s", cache_path, exc)

        return text
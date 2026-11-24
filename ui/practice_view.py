"""ui/practice_view.py

Framework-agnostic practice controller.

The rendering layer (PyQt6 / Tkinter / Textual / ...) only needs to:
    1. call  session.build_queue(chapter_number=...)          once per session
    2. call  session.next_question()                           on every prompt
    3. call  session.submit_answer(correct: bool)              on every reply
    4. call  session.refresh_chapter_scores()                  when the user stops

All AI calls are lemma-driven:
    - session.enrich_current()   → ai/enrichment.enrich_if_empty
    - session.generate_example() → ai/personal_examples.generate_example

There is no book-context injection anywhere in this file.
"""
from __future__ import annotations

from typing import Any

from ai.enrichment import enrich_if_empty
from ai.ollama_client import OllamaClient
from ai.personal_examples import generate_example
from ingestion.json_builder import save_book_index
from learning import mastery
from learning.user_store import save_user_progress
from models.schemas import BookIndex, UserProgress, WordKnowledge


class PracticeSession:
    def __init__(
        self,
        book: BookIndex,
        progress: UserProgress,
        client: OllamaClient,
    ) -> None:
        self.book = book
        self.progress = progress
        self.client = client
        self._queue: list[str] = []
        self._current_lemma: str | None = None
        self._current_chapter_number: int | None = None

    # ------------------------------------------------------------------ queue

    def build_queue(
        self,
        *,
        chapter_number: int | None = None,
        limit: int = 20,
    ) -> list[str]:
        """Pick words to practice. Priority: learning > new; mastered/skipped out."""
        if chapter_number is not None:
            chapter = self.book.chapter_by_number(chapter_number)
            if chapter is None:
                self._queue = []
                return []
            self._current_chapter_number = chapter.number
            lemmas = [l for l in chapter.word_lemmas if l in self.book.words]
        else:
            self._current_chapter_number = None
            lemmas = list(self.book.words.keys())

        ranked: list[tuple[int, str]] = []
        for lemma in lemmas:
            wk = self.progress.word_knowledge.get(lemma)
            status = wk.known_status if wk else "new"
            if status in ("mastered", "skipped"):
                continue
            ranked.append((0 if status == "learning" else 1, lemma))
        ranked.sort()
        self._queue = [lemma for _, lemma in ranked[:limit]]
        self._current_lemma = None
        return list(self._queue)

    # -------------------------------------------------------------- questions

    def next_question(self) -> dict[str, Any] | None:
        """Return the next prompt payload, or None if the queue is empty."""
        if not self._queue:
            return None
        lemma = self._queue.pop(0)
        self._current_lemma = lemma
        entry = self.book.words.get(lemma)
        wk = self.progress.word_knowledge.get(lemma) or WordKnowledge(lemma=lemma)
        examples = self.progress.personal_examples.get(lemma, [])
        return {
            "lemma": lemma,
            "entry": entry.to_dict() if entry else None,
            "knowledge": wk.to_dict(),
            "examples": [e.to_dict() for e in examples],
            "is_enriched": bool(entry and entry.entries),
        }

    def submit_answer(self, correct: bool) -> dict[str, Any]:
        """Record the user's result for the current question."""
        if not self._current_lemma:
            return {}
        wk = mastery.record_attempt(self.progress, self._current_lemma, correct)
        save_user_progress(self.progress)
        return {"lemma": self._current_lemma, "knowledge": wk.to_dict()}

    # ------------------------------------------------------------- enrichment

    def enrich_current(self) -> bool:
        """Enrich the current word on the shared book JSON. Returns True on success."""
        if not self._current_lemma:
            return False
        entry = self.book.words.get(self._current_lemma)
        if entry is None:
            return False
        changed = enrich_if_empty(
            entry,
            self._current_lemma,
            self.book.language,
            self.client,
            native_language=self.progress.native_language,
        )
        if changed:
            save_book_index(self.book)
        return changed

    def generate_example(self) -> dict[str, Any] | None:
        """Generate and persist a personalized example for the current word."""
        if not self._current_lemma:
            return None
        example = generate_example(
            self._current_lemma,
            target_language=self.book.language,
            native_language=self.progress.native_language,
            client=self.client,
        )
        if example is None:
            return None
        self.progress.personal_examples.setdefault(self._current_lemma, []).append(example)
        save_user_progress(self.progress)
        return example.to_dict()

    # -------------------------------------------------------------- lifecycle

    def refresh_chapter_scores(self) -> None:
        """Recompute every chapter score and persist. Call when the user stops."""
        mastery.refresh_chapter_scores(self.book, self.progress)
        save_user_progress(self.progress)
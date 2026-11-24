"""ui/reader_view.py

Framework-agnostic reader controller.

The rendering layer only needs to:
    - call  session.open_chapter(n)     to switch chapters
    - call  session.chapter_tokens()    to get colored tokens
    - call  session.word_detail(lemma)  to populate a popup
    - call  session.mark_skipped(lemma) when the user flags "I know this"

AI calls are lemma-driven and go through the *shared* ai modules — no book
context is injected into the prompts.
"""
from __future__ import annotations

from typing import Any

from ai.enrichment import enrich_if_empty
from ai.ollama_client import OllamaClient
from ai.personal_examples import generate_example
from ingestion.json_builder import save_book_index
from learning import mastery
from learning.user_store import save_user_progress
from models.schemas import BookIndex, ChapterInfo, UserProgress, WordKnowledge

_STATUS_COLORS = {
    "new": "#c0392b",        # red   — unknown
    "learning": "#d68910",   # amber — learning
    "mastered": "#1e8449",   # green — known
    "skipped": "#5d6d7e",    # grey  — skipped by user
    "granted": "#95a5a6",    # dim   — granted (should not appear post-filter)
}


class ReaderSession:
    def __init__(
        self,
        book: BookIndex,
        progress: UserProgress,
        client: OllamaClient,
    ) -> None:
        self.book = book
        self.progress = progress
        self.client = client
        self.current_chapter: ChapterInfo | None = None

    # ---------------------------------------------------------------- reading

    def open_chapter(self, number: int) -> ChapterInfo | None:
        self.current_chapter = self.book.chapter_by_number(number)
        if self.current_chapter is None:
            return None
        # Remember this book as active for the user.
        if self.book.book_id not in self.progress.active_book_ids:
            self.progress.active_book_ids.append(self.book.book_id)
            save_user_progress(self.progress)
        return self.current_chapter

    def chapter_tokens(self) -> list[dict[str, str]]:
        """Return the chapter's lemmas with a color code for rendering."""
        if self.current_chapter is None:
            return []
        out: list[dict[str, str]] = []
        for lemma in self.current_chapter.word_lemmas:
            if lemma not in self.book.words:
                out.append({"lemma": lemma, "status": "granted",
                            "color": _STATUS_COLORS["granted"]})
                continue
            wk = self.progress.word_knowledge.get(lemma)
            status = wk.known_status if wk else "new"
            out.append({"lemma": lemma, "status": status,
                        "color": _STATUS_COLORS.get(status, _STATUS_COLORS["new"])})
        return out

    # ---------------------------------------------------------------- popups

    def word_detail(self, lemma: str) -> dict[str, Any]:
        """Everything the popup needs to render."""
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

    # --------------------------------------------------------------- actions

    def mark_skipped(self, lemma: str) -> dict[str, Any]:
        wk = mastery.mark_word_skipped(self.progress, lemma)
        save_user_progress(self.progress)
        return wk.to_dict()

    def enrich(self, lemma: str) -> bool:
        entry = self.book.words.get(lemma)
        if entry is None:
            return False
        changed = enrich_if_empty(
            entry, lemma, self.book.language, self.client,
            native_language=self.progress.native_language,
        )
        if changed:
            save_book_index(self.book)
        return changed

    def generate_example(self, lemma: str) -> dict[str, Any] | None:
        example = generate_example(
            lemma,
            target_language=self.book.language,
            native_language=self.progress.native_language,
            client=self.client,
        )
        if example is None:
            return None
        self.progress.personal_examples.setdefault(lemma, []).append(example)
        save_user_progress(self.progress)
        return example.to_dict()

    # -------------------------------------------------------------- lifecycle

    def refresh_chapter_scores(self) -> None:
        mastery.refresh_chapter_scores(self.book, self.progress)
        save_user_progress(self.progress)
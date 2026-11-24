"""
models/schemas.py
Data structures representing a parsed book, its chapters, word occurrences,
and the user's learning accuracy/mastery state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Occurrence:
    """A single appearance of a word in the book."""
    chapter_id: str
    page: int
    sentence: str


@dataclass
class ChapterInfo:
    """Metadata and text boundaries for a single chapter."""
    chapter_id: str
    title: str
    start_page: int  # 1-based page number
    end_page: int    # 1-based page number (inclusive)
    text: str = ""
    word_lemmas: list[str] = field(default_factory=list)


@dataclass
class WordEntry:
    """
    Represents a normalized dictionary/base word (lemma) across the book
    along with all surface forms, occurrences, and user mastery metrics.
    """
    lemma: str
    pos: str = ""
    forms: dict[str, int] = field(default_factory=dict)          # surface_form -> count
    total_count: int = 0
    by_chapter: dict[str, int] = field(default_factory=dict)     # chapter_id -> count
    pages: list[int] = field(default_factory=list)
    occurrences: list[Occurrence] = field(default_factory=list)
    translations: list[str] = field(default_factory=list)          # (b) most common direct translations
    false_friends: list[str] = field(default_factory=list)         # (c) false friends & common mistranslations
    edge_meanings: list[str] = field(default_factory=list)         # (d) edge meanings & peculiar uses
    enriched_sentences: list[dict[str, str]] = field(default_factory=list)
    # Each item in enriched_sentences:
    # {"sentence": "...", "translation": "...", "related_word": "...", "source": "book|llm"}

    # User learning & accuracy tracking
    known_status: str = "new"  # "new" | "learning" | "mastered" | "skipped"
    attempts: int = 0
    correct: int = 0
    accuracy: float = 0.0
    streak: int = 0
    last_reviewed: str | None = None
    cached_translation: str = ""

    def record_attempt(self, is_correct: bool) -> None:
        """Update accuracy, streak, timestamp, and mastery status after an attempt."""
        if self.known_status == "skipped":
            self.known_status = "learning"

        self.attempts += 1
        if is_correct:
            self.correct += 1
            self.streak += 1
        else:
            self.streak = 0

        self.accuracy = round(self.correct / self.attempts, 4) if self.attempts > 0 else 0.0
        self.last_reviewed = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "lemma": self.lemma,
            "pos": self.pos,
            "forms": self.forms,
            "total_count": self.total_count,
            "by_chapter": self.by_chapter,
            "pages": self.pages,
            "occurrences": [asdict(occ) for occ in self.occurrences],
            "known_status": self.known_status,
            "attempts": self.attempts,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "streak": self.streak,
            "last_reviewed": self.last_reviewed,
            "cached_translation": self.cached_translation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordEntry":
        """Reconstruct a WordEntry from a dictionary loaded from JSON."""
        occurrences = [
            Occurrence(
                chapter_id=occ.get("chapter_id", "ch_1"),
                page=int(occ.get("page", 1)),
                sentence=occ.get("sentence", ""),
            )
            for occ in data.get("occurrences", [])
        ]
        return cls(
            lemma=data["lemma"],
            pos=data.get("pos", ""),
            forms=dict(data.get("forms", {})),
            total_count=int(data.get("total_count", 0)),
            by_chapter=dict(data.get("by_chapter", {})),
            pages=list(data.get("pages", [])),
            occurrences=occurrences,
            known_status=data.get("known_status", "new"),
            attempts=int(data.get("attempts", 0)),
            correct=int(data.get("correct", 0)),
            accuracy=float(data.get("accuracy", 0.0)),
            streak=int(data.get("streak", 0)),
            last_reviewed=data.get("last_reviewed"),
            cached_translation=data.get("cached_translation", ""),
        )


@dataclass
class BookIndex:
    """Top-level JSON structure stored per book."""
    book_id: str
    title: str
    author: str
    language: str
    total_pages: int
    source_pdf: str
    clean_txt_path: str
    created_at: str
    updated_at: str
    chapters: list[ChapterInfo] = field(default_factory=list)
    words: dict[str, WordEntry] = field(default_factory=dict)  # keyed by lemma

    def get_chapter(self, chapter_id: str) -> ChapterInfo | None:
        """Find a chapter by its ID."""
        for ch in self.chapters:
            if ch.chapter_id == chapter_id:
                return ch
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire book index for JSON export."""
        return {
            "metadata": {
                "book_id": self.book_id,
                "title": self.title,
                "author": self.author,
                "language": self.language,
                "total_pages": self.total_pages,
                "source_pdf": self.source_pdf,
                "clean_txt_path": self.clean_txt_path,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "total_unique_words": len(self.words),
                "total_chapters": len(self.chapters),
            },
            "chapters": [
                {
                    "chapter_id": ch.chapter_id,
                    "title": ch.title,
                    "start_page": ch.start_page,
                    "end_page": ch.end_page,
                    "text": ch.text,
                    "word_lemmas": ch.word_lemmas,
                }
                for ch in self.chapters
            ],
            "words": {lemma: entry.to_dict() for lemma, entry in self.words.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BookIndex":
        """Deserialize a BookIndex from JSON data."""
        meta = data.get("metadata", {})
        chapters = [
            ChapterInfo(
                chapter_id=ch["chapter_id"],
                title=ch["title"],
                start_page=int(ch["start_page"]),
                end_page=int(ch["end_page"]),
                text=ch.get("text", ""),
                word_lemmas=list(ch.get("word_lemmas", [])),
            )
            for ch in data.get("chapters", [])
        ]
        words = {
            lemma: WordEntry.from_dict(w_data)
            for lemma, w_data in data.get("words", {}).items()
        }
        return cls(
            book_id=meta.get("book_id", "book"),
            title=meta.get("title", "Untitled Book"),
            author=meta.get("author", "Unknown Author"),
            language=meta.get("language", "en"),
            total_pages=int(meta.get("total_pages", len(chapters))),
            source_pdf=meta.get("source_pdf", ""),
            clean_txt_path=meta.get("clean_txt_path", ""),
            created_at=meta.get("created_at", utc_now_iso()),
            updated_at=meta.get("updated_at", utc_now_iso()),
            chapters=chapters,
            words=words,
        )
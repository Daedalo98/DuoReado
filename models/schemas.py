"""models/schemas.py

Design overview
---------------
BookIndex          : the shared, regenerable vocabulary extracted from a PDF.
VocabularyEntry    : what a word *is* (static, language-level knowledge).
UserProgress       : what a *user* knows about those words (per-user, persistent).

The split is deliberate: the book JSON can be rebuilt from the PDF at any
time without destroying a user's history, and the AI layer only ever needs
a lemma string.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Book side (shared, regenerable)
# ---------------------------------------------------------------------------

@dataclass
class Occurrence:
    """One occurrence of a word inside a chapter. Build-time only; not saved
    in the final book JSON — we only persist the aggregate counts."""
    chapter_id: str
    page: int = 0
    snippet: str = ""
    sentence: str = ""
    translation: str = ""


@dataclass
class ChapterInfo:
    chapter_id: str
    number: int                       # 1-based chapter index (used for first_chapter)
    title: str = ""
    start_page: int = 0
    end_page: int = 0
    word_lemmas: list[str] = field(default_factory=list)
    word_counts: dict[str, int] = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChapterInfo":
        return cls(
            chapter_id=data.get("chapter_id", ""),
            number=data.get("number", 0),
            title=data.get("title", ""),
            start_page=data.get("start_page", 0),
            end_page=data.get("end_page", 0),
            word_lemmas=list(data.get("word_lemmas", [])),
            word_counts=dict(data.get("word_counts", {})),
            text=data.get("text", ""),
        )


@dataclass
class WordFormEntry:
    """A morphological / relational variant of the base word."""
    form: str
    relation: str = "base"
    translations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordFormEntry":
        return cls(
            form=data.get("form", ""),
            relation=data.get("relation", "base"),
            translations=list(data.get("translations", [])),
        )


@dataclass
class WordLinguistics:
    explanation: str = ""
    edge_cases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordLinguistics":
        return cls(
            explanation=data.get("explanation", ""),
            edge_cases=list(data.get("edge_cases", [])),
        )


@dataclass
class VocabularyEntry:
    occurrences: int = 0
    first_chapter: int = 0
    sentences: list[Occurrence] = field(default_factory=list)   # ← NEW
    entries: list[WordFormEntry] = field(default_factory=list)
    linguistics: WordLinguistics = field(default_factory=WordLinguistics)
    semantic_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrences": self.occurrences,
            "first_chapter": self.first_chapter,
            "sentences": [asdict(s) for s in self.sentences],   # ← NEW
            "entries": [e.to_dict() for e in self.entries],
            "linguistics": self.linguistics.to_dict(),
            "semantic_keywords": list(self.semantic_keywords),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VocabularyEntry":
        return cls(
            occurrences=data.get("occurrences", 0),
            first_chapter=data.get("first_chapter", 0),
            sentences=[Occurrence(**s) for s in data.get("sentences", [])],   # ← NEW
            entries=[WordFormEntry.from_dict(e) for e in data.get("entries", [])],
            linguistics=WordLinguistics.from_dict(data.get("linguistics", {})),
            semantic_keywords=list(data.get("semantic_keywords", [])),
        )

@dataclass
class BookIndex:
    book_id: str
    title: str
    language: str                    # BCP-47-ish code: "es", "fr", "pt", ...
    source_pdf: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    chapters: list[ChapterInfo] = field(default_factory=list)
    words: dict[str, VocabularyEntry] = field(default_factory=dict)

    def chapter_by_number(self, number: int) -> ChapterInfo | None:
        for c in self.chapters:
            if c.number == number:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_id": self.book_id,
            "title": self.title,
            "language": self.language,
            "source_pdf": self.source_pdf,
            "metadata": self.metadata,
            "chapters": [c.to_dict() for c in self.chapters],
            "words": {k: v.to_dict() for k, v in self.words.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BookIndex":
        return cls(
            book_id=data.get("book_id", ""),
            title=data.get("title", ""),
            language=data.get("language", ""),
            source_pdf=data.get("source_pdf", ""),
            metadata=dict(data.get("metadata", {})),
            chapters=[ChapterInfo.from_dict(c) for c in data.get("chapters", [])],
            words={k: VocabularyEntry.from_dict(v) for k, v in data.get("words", {}).items()},
        )


# ---------------------------------------------------------------------------
# User side (personal, persistent)
# ---------------------------------------------------------------------------

@dataclass
class ChapterScore:
    chapter_id: str
    chapter_number: int
    average_correctness: float = 0.0
    known_words: int = 0
    total_words: int = 0
    score: float = 0.0               # average_correctness * known_words / total_words

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChapterScore":
        return cls(
            chapter_id=data.get("chapter_id", ""),
            chapter_number=data.get("chapter_number", 0),
            average_correctness=data.get("average_correctness", 0.0),
            known_words=data.get("known_words", 0),
            total_words=data.get("total_words", 0),
            score=data.get("score", 0.0),
        )


@dataclass
class WordKnowledge:
    lemma: str
    attempts: int = 0
    correct: int = 0
    accuracy: float = 0.0
    known_status: str = "new"        # new | learning | mastered | skipped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordKnowledge":
        return cls(
            lemma=data.get("lemma", ""),
            attempts=data.get("attempts", 0),
            correct=data.get("correct", 0),
            accuracy=data.get("accuracy", 0.0),
            known_status=data.get("known_status", "new"),
        )


@dataclass
class PersonalExample:
    lemma: str
    sentence: str
    translation: str
    source: str = "llm"              # "llm" | "book"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonalExample":
        return cls(
            lemma=data.get("lemma", ""),
            sentence=data.get("sentence", ""),
            translation=data.get("translation", ""),
            source=data.get("source", "llm"),
        )


@dataclass
class UserProgress:
    """Everything that is personal. Saved to data/users/<user_id>/progress.json.

    Note: `chapter_scores`, `word_knowledge`, `personal_examples` are all keyed
    by lemma / chapter_id and are updated *by the mastery layer*, never by
    the book builder.
    """
    user_id: str
    native_language: str = "en"
    active_book_ids: list[str] = field(default_factory=list)
    chapter_scores: dict[str, ChapterScore] = field(default_factory=dict)
    word_knowledge: dict[str, WordKnowledge] = field(default_factory=dict)
    personal_examples: dict[str, list[PersonalExample]] = field(default_factory=dict)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "native_language": self.native_language,
            "active_book_ids": list(self.active_book_ids),
            "chapter_scores": {k: v.to_dict() for k, v in self.chapter_scores.items()},
            "word_knowledge": {k: v.to_dict() for k, v in self.word_knowledge.items()},
            "personal_examples": {
                k: [e.to_dict() for e in v]
                for k, v in self.personal_examples.items()
            },
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProgress":
        return cls(
            user_id=data.get("user_id", ""),
            native_language=data.get("native_language", "en"),
            active_book_ids=list(data.get("active_book_ids", [])),
            chapter_scores={
                k: ChapterScore.from_dict(v)
                for k, v in data.get("chapter_scores", {}).items()
            },
            word_knowledge={
                k: WordKnowledge.from_dict(v)
                for k, v in data.get("word_knowledge", {}).items()
            },
            personal_examples={
                k: [PersonalExample.from_dict(e) for e in v]
                for k, v in data.get("personal_examples", {}).items()
            },
            updated_at=data.get("updated_at", ""),
        )
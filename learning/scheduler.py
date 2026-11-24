"""learning/scheduler.py
Builds quiz questions from the shared BookIndex + per-user UserProgress.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

from models.schemas import (
    BookIndex,
    ChapterInfo,
    UserProgress,
    VocabularyEntry,
    WordKnowledge,
)

# ── tunables ────────────────────────────────────────────────────────────
_CHOICES = 4
_MAX_DISTRACTORS = 12
_EXCLUDED_STATUSES = {"mastered", "skipped"}


# ── result type ─────────────────────────────────────────────────────────
@dataclass
class QuizQuestion:
    lemma: str
    prompt: str
    choices: list[str]
    correct_index: int
    sentence: str = ""
    sentence_translation: str = ""
    chapter_id: str = ""

    def to_dict(self) -> dict:
        return {
            "lemma": self.lemma,
            "prompt": self.prompt,
            "choices": list(self.choices),
            "correct_index": self.correct_index,
            "sentence": self.sentence,
            "sentence_translation": self.sentence_translation,
            "chapter_id": self.chapter_id,
        }


# ── helpers ─────────────────────────────────────────────────────────────
def _all_translations(entry: VocabularyEntry) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for e in entry.entries:
        for t in e.translations:
            k = t.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(t.strip())
    return out


def _primary_translation(entry: VocabularyEntry) -> str | None:
    for e in entry.entries:
        if e.relation == "base" and e.translations:
            return e.translations[0]
    for e in entry.entries:
        if e.translations:
            return e.translations[0]
    return None


def pick_sentence(entry: VocabularyEntry, chapter_id: str) -> tuple[str, str]:
    for occ in entry.sentences:
        if occ.chapter_id == chapter_id and occ.sentence:
            return occ.sentence, occ.translation
    for occ in entry.sentences:
        if occ.sentence:
            return occ.sentence, occ.translation
    return "", ""


def _chapter_lemmas(chapter: ChapterInfo, book: BookIndex) -> list[str]:
    return [l for l in chapter.word_lemmas if l in book.words]


def _status_of(progress: UserProgress, lemma: str) -> str:
    wk: WordKnowledge | None = progress.word_knowledge.get(lemma)
    return wk.known_status if wk else "new"


def _priority_key(
    lemma: str,
    entry: VocabularyEntry,
    progress: UserProgress,
) -> tuple[int, int, str]:
    status = _status_of(progress, lemma)
    prio = {"learning": 0, "new": 1}.get(status, 2)
    return (prio, -entry.occurrences, lemma)


def _sample_distractors(
    correct: str,
    pool: Iterable[tuple[str, VocabularyEntry]],
    n: int,
    rng: random.Random,
) -> list[str]:
    correct_lower = correct.strip().lower()
    pool_list = list(pool)
    rng.shuffle(pool_list)
    picked: list[str] = []
    seen: set[str] = {correct_lower}
    for _lemma, entry in pool_list:
        for t in _all_translations(entry):
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            picked.append(t)
            if len(picked) >= n:
                return picked
    return picked


# ── public API ──────────────────────────────────────────────────────────
def build_quiz(
    book: BookIndex,
    progress: UserProgress,
    *,
    chapter_number: int | None = None,
    n: int = 10,
    include_mastered: bool = False,
    rng: random.Random | None = None,
) -> list[QuizQuestion]:
    rng = rng or random.Random()
    chapter: ChapterInfo | None = (
        book.chapter_by_number(chapter_number) if chapter_number is not None else None
    )
    if chapter_number is not None and chapter is None:
        return []

    candidates = (
        _chapter_lemmas(chapter, book) if chapter else list(book.words.keys())
    )

    eligible: list[tuple[str, VocabularyEntry]] = []
    for lemma in candidates:
        entry = book.words.get(lemma)
        if entry is None or not entry.entries:
            continue
        if not include_mastered and _status_of(progress, lemma) in _EXCLUDED_STATUSES:
            continue
        if _primary_translation(entry) is None:
            continue
        eligible.append((lemma, entry))

    eligible.sort(key=lambda le: _priority_key(le[0], le[1], progress))
    picked = eligible[:n]

    picked_lemmas = {l for l, _ in picked}
    distractor_pool = [
        (l, e) for l, e in book.words.items()
        if l not in picked_lemmas and e.entries
    ][:_MAX_DISTRACTORS]

    questions: list[QuizQuestion] = []
    chapter_id = chapter.chapter_id if chapter else ""
    for lemma, entry in picked:
        correct = _primary_translation(entry)
        if correct is None:
            continue
        wrongs = _sample_distractors(correct, distractor_pool, _CHOICES - 1, rng)
        if len(wrongs) < _CHOICES - 1:
            wrongs.extend([t for t in _all_translations(entry) if t != correct][
                : _CHOICES - 1 - len(wrongs)
            ])
        if not wrongs:
            continue
        choices = [correct] + wrongs
        rng.shuffle(choices)
        sentence, translation = pick_sentence(entry, chapter_id)
        questions.append(QuizQuestion(
            lemma=lemma,
            prompt=lemma,
            choices=choices,
            correct_index=choices.index(correct),
            sentence=sentence,
            sentence_translation=translation,
            chapter_id=chapter_id,
        ))
    return questions


def build_queue(
    book: BookIndex,
    progress: UserProgress,
    *,
    chapter_number: int | None = None,
    limit: int = 20,
) -> list[str]:
    chapter = book.chapter_by_number(chapter_number) if chapter_number else None
    candidates = (
        _chapter_lemmas(chapter, book) if chapter else list(book.words.keys())
    )
    ranked: list[tuple[int, int, str]] = []
    for lemma in candidates:
        entry = book.words[lemma]
        if _status_of(progress, lemma) in _EXCLUDED_STATUSES:
            continue
        ranked.append(_priority_key(lemma, entry, progress))
    ranked.sort()
    return [lemma for _, _, lemma in ranked[:limit]]


# ── back-compat shims (single copy) ────────────────────────────────────
@dataclass
class PracticeItem(QuizQuestion):
    @property
    def word(self) -> str:
        return self.lemma

    @property
    def options(self) -> list[str]:
        return self.choices


def build_chapter_practice_queue(
    chapter_number: int,
    book: BookIndex,
    progress: UserProgress,
    *,
    limit: int = 10,
    include_mastered: bool = False,
) -> list[PracticeItem]:
    questions = build_quiz(
        book, progress,
        chapter_number=chapter_number,
        n=limit,
        include_mastered=include_mastered,
    )
    return [PracticeItem(**asdict(q)) for q in questions]


def build_meaning_options(
    correct: str,
    pool: Iterable[str],
    n: int = 3,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    rng = rng or random.Random()
    correct_lower = correct.strip().lower()
    pool_list = [p for p in pool if p.strip().lower() != correct_lower]
    rng.shuffle(pool_list)
    seen: set[str] = {correct_lower}
    out: list[str] = []
    for candidate in pool_list:
        key = candidate.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= n:
            break
    return out
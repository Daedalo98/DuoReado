"""learning/scheduler.py

Builds quiz questions from the shared BookIndex + per-user UserProgress.

There is no WordEntry anywhere:
    - What the word is (translations, sentences, edge cases)  → VocabularyEntry
    - What the user knows (status, attempts, accuracy)        → WordKnowledge

Public API
----------
    build_quiz(book, progress, *, chapter_number=None, n=10) -> list[QuizQuestion]
    pick_sentence(entry, chapter_id)                         -> (sentence, translation)

The QuizQuestion is a *pure value*: no dataclasses reference the schemas,
so the phone bridge / UI can serialize it directly to JSON.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Iterable

from models.schemas import (
    BookIndex,
    ChapterInfo,
    UserProgress,
    VocabularyEntry,
    WordKnowledge,
)

# --- tunables ---------------------------------------------------------------
_CHOICES = 4                # number of options per question
_MAX_DISTRACTORS = 12       # cap when sampling wrong translations
_EXCLUDED_STATUSES = {"mastered", "skipped"}


# --- result type ------------------------------------------------------------

@dataclass
class QuizQuestion:
    lemma: str
    prompt: str                              # usually the lemma itself
    choices: list[str]                       # translations shown to the user
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


# --- helpers: shape adapters ------------------------------------------------

def _all_translations(entry: VocabularyEntry) -> list[str]:
    """Flatten and dedupe every translation across all form entries.

    Why flatten? On the old schema `word.translations` was a single list.
    The new schema nests translations under each morphological form. For a
    quiz prompt we just need "any correct answer for the base lemma".
    """
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
    """Return the base form's first translation, or the first we can find."""
    for e in entry.entries:
        if e.relation == "base" and e.translations:
            return e.translations[0]
    for e in entry.entries:
        if e.translations:
            return e.translations[0]
    return None


def pick_sentence(
    entry: VocabularyEntry,
    chapter_id: str,
) -> tuple[str, str]:
    """Return (sentence, translation) for this word inside `chapter_id`.

    Priority:
        1. A sentence whose `chapter_id` matches and has a filled `sentence`.
        2. Any sentence with a filled `sentence`.
        3. The first empty-occurrence snippet (page-only fallback).

    If nothing usable exists, return ("", "") — the caller renders the
    question without a context sentence.
    """
    for occ in entry.sentences:
        if occ.chapter_id == chapter_id and occ.sentence:
            return occ.sentence, occ.translation
    for occ in entry.sentences:
        if occ.sentence:
            return occ.sentence, occ.translation
    return "", ""


def _chapter_lemmas(chapter: ChapterInfo, book: BookIndex) -> list[str]:
    return [l for l in chapter.word_lemmas if l in book.words]


# --- candidate selection ----------------------------------------------------

def _status_of(progress: UserProgress, lemma: str) -> str:
    wk: WordKnowledge | None = progress.word_knowledge.get(lemma)
    return wk.known_status if wk else "new"


def _priority_key(
    lemma: str,
    entry: VocabularyEntry,
    progress: UserProgress,
) -> tuple[int, int, str]:
    """Lower tuple = higher priority in the quiz.

    priority, -occurrences, lemma
      priority 0 = learning       (user is actively studying)
      priority 1 = new            (never seen, decent to introduce)
      priority 2 = mastered       (only if include_mastered=True)
    """
    status = _status_of(progress, lemma)
    if status == "learning":
        prio = 0
    elif status == "new":
        prio = 1
    else:
        prio = 2
    return (prio, -entry.occurrences, lemma)


# --- distractors ------------------------------------------------------------

def _sample_distractors(
    correct: str,
    pool: Iterable[tuple[str, VocabularyEntry]],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Pick `n` wrong translations from `pool`, excluding `correct`."""
    correct_lower = correct.strip().lower()
    pool_list: list[tuple[str, VocabularyEntry]] = list(pool)
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


# --- public API -------------------------------------------------------------

def build_quiz(
    book: BookIndex,
    progress: UserProgress,
    *,
    chapter_number: int | None = None,
    n: int = 10,
    include_mastered: bool = False,
    rng: random.Random | None = None,
) -> list[QuizQuestion]:
    """Return up to `n` QuizQuestions.

    chapter_number=None → draw from the entire book.
    Only enriched words (with at least one translation) become questions;
    un-enriched words must be enriched first (ai/enrichment.py).
    """
    rng = rng or random.Random()
    chapter: ChapterInfo | None = (
        book.chapter_by_number(chapter_number) if chapter_number is not None else None
    )
    if chapter_number is not None and chapter is None:
        return []

    candidates: list[str] = (
        _chapter_lemmas(chapter, book) if chapter else list(book.words.keys())
    )

    # 1. Filter to words we can actually quiz.
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

    # 2. Rank by priority and take the first `n`.
    eligible.sort(key=lambda le: _priority_key(le[0], le[1], progress))
    picked = eligible[:n]

    # 3. Distractor pool: enriched words not already among the picked ones.
    picked_lemmas = {l for l, _ in picked}
    distractor_pool = [
        (l, e) for l, e in book.words.items()
        if l not in picked_lemmas and e.entries
    ]
    distractor_pool = distractor_pool[:_MAX_DISTRACTORS]

    # 4. Build the questions.
    questions: list[QuizQuestion] = []
    chapter_id = chapter.chapter_id if chapter else ""
    for lemma, entry in picked:
        correct = _primary_translation(entry)
        if correct is None:              # defensive: filtered above
            continue

        wrongs = _sample_distractors(correct, distractor_pool, _CHOICES - 1, rng)
        if len(wrongs) < _CHOICES - 1:
            # Not enough distractors (tiny book). Fall back to reusing any
            # translations from the same word's other forms — still wrong for
            # the base lemma.
            extra = [t for t in _all_translations(entry) if t != correct]
            wrongs.extend(extra[: _CHOICES - 1 - len(wrongs)])
        if not wrongs:
            continue

        choices = [correct] + wrongs
        rng.shuffle(choices)
        correct_index = choices.index(correct)

        sentence, translation = pick_sentence(entry, chapter_id)
        questions.append(
            QuizQuestion(
                lemma=lemma,
                prompt=lemma,
                choices=choices,
                correct_index=correct_index,
                sentence=sentence,
                sentence_translation=translation,
                chapter_id=chapter_id,
            )
        )
    return questions


# --- back-compat: queue builder used by phone_bridge / practice_view -------

def build_queue(
    book: BookIndex,
    progress: UserProgress,
    *,
    chapter_number: int | None = None,
    limit: int = 20,
) -> list[str]:
    """Plain list of lemmas to practise (no questions, no distractors).

    Kept because PracticeSession and phone_bridge call it. It is *not* the
    same thing as build_quiz — it doesn't filter by "has translations",
    because the caller may want to enrich them on the fly.
    """
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


# ---------------------------------------------------------------------------
# Back-compat aliases and wrappers for the pre-refactor API
#
# These names lived on learning/scheduler.py before the BookIndex/UserProgress
# split. They are kept so unchanged call sites compile. Everything routes to
# the canonical new functions above — there is no duplicated logic here.
# ---------------------------------------------------------------------------

@dataclass
class PracticeItem(QuizQuestion):
    """Back-compat subclass of QuizQuestion with the old field names.

    Old callers read `.word` and `.options`; new callers read `.lemma` and
    `.choices`. Both work. No new data, just read-only properties.
    """
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
    """Back-compat wrapper for build_quiz, returning PracticeItem instances.

    The exact old signature is unknown; this is the clean, documented form:
        build_chapter_practice_queue(chapter_number, book, progress, limit=N)

    If a call site uses (book, progress, chapter_number) order instead,
    swap the arguments at that site — this wrapper stays unchanged.
    """
    questions = build_quiz(
        book,
        progress,
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
    """Back-compat: return up to `n` wrong translations drawn from `pool`.

    Old signature took a flat pool of translation strings; that is still the
    contract here. If a call site passes VocabularyEntry objects, flatten
    them with `_all_translations(e)` at the call site — this wrapper stays
    focused on the common case.
    """
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


# === BACK-COMPAT SHIMS — append at the END of learning/scheduler.py ===
# These names lived on scheduler before the BookIndex/UserProgress split.
# Everything routes to the canonical new functions above; no duplicated logic.
import random as _random_backcompat      # local alias so this block is self-contained
from dataclasses import asdict as _asdict_backcompat

@dataclass
class PracticeItem(QuizQuestion):
    """Back-compat subclass of QuizQuestion with the old field names."""
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
    """Back-compat wrapper for build_quiz, returning PracticeItem instances."""
    questions = build_quiz(
        book, progress,
        chapter_number=chapter_number,
        n=limit,
        include_mastered=include_mastered,
    )
    return [PracticeItem(**_asdict_backcompat(q)) for q in questions]


def build_meaning_options(
    correct: str,
    pool: Iterable[str],
    n: int = 3,
    *,
    rng: _random_backcompat.Random | None = None,
) -> list[str]:
    """Back-compat: return up to `n` wrong translations drawn from `pool`."""
    rng = rng or _random_backcompat.Random()
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
# === end of back-compat shims ===
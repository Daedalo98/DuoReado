"""learning/mastery.py

The bridge between the shared BookIndex and the per-user UserProgress.

Everything here is *derived* from:
    - VocabularyEntry   (what the word is, from the book JSON)
    - WordKnowledge     (what the user knows, from the user JSON)
"""
from __future__ import annotations

from models.schemas import (
    BookIndex,
    ChapterInfo,
    ChapterScore,
    UserProgress,
    WordKnowledge,
)

# ---------------------------------------------------------------------------
# Thresholds (same as before, now applied to WordKnowledge)
# ---------------------------------------------------------------------------

LEARNING_MIN_ATTEMPTS = 1
MASTERED_MIN_ATTEMPTS = 3
MASTERED_MIN_ACCURACY = 0.80
CHAPTER_READINESS_TARGET = 0.90

_READY_STATUSES = {"mastered", "skipped"}


# ---------------------------------------------------------------------------
# Per-word updates
# ---------------------------------------------------------------------------

def refresh_word_mastery(wk: WordKnowledge) -> str:
    """Recompute `known_status` from attempts/correct. Mutates `wk`."""
    if wk.known_status == "skipped":
        return "skipped"
    if wk.attempts == 0:
        wk.known_status = "new"
    elif wk.attempts >= MASTERED_MIN_ATTEMPTS and wk.accuracy >= MASTERED_MIN_ACCURACY:
        wk.known_status = "mastered"
    else:
        wk.known_status = "learning"
    return wk.known_status


def record_attempt(progress: UserProgress, lemma: str, correct: bool) -> WordKnowledge:
    """Record one practice attempt and persist it on `progress` in memory."""
    wk = progress.word_knowledge.get(lemma)
    if wk is None:
        wk = WordKnowledge(lemma=lemma)
        progress.word_knowledge[lemma] = wk
    wk.attempts += 1
    if correct:
        wk.correct += 1
    wk.accuracy = wk.correct / wk.attempts
    refresh_word_mastery(wk)
    return wk


def mark_word_skipped(progress: UserProgress, lemma: str) -> WordKnowledge:
    wk = progress.word_knowledge.get(lemma) or WordKnowledge(lemma=lemma)
    wk.known_status = "skipped"
    progress.word_knowledge[lemma] = wk
    return wk


# ---------------------------------------------------------------------------
# Chapter-level views
# ---------------------------------------------------------------------------

def _chapter_lemmas(chapter: ChapterInfo, book: BookIndex) -> list[str]:
    """Lemmas in this chapter that exist in the vocabulary (granted ones
    were already removed at build time)."""
    return [l for l in chapter.word_lemmas if l in book.words]


def chapter_readiness(
    chapter: ChapterInfo,
    book: BookIndex,
    progress: UserProgress,
) -> dict[str, float | int]:
    """How close is the user to being able to read this chapter comfortably."""
    lemmas = _chapter_lemmas(chapter, book)
    total = len(lemmas)
    if total == 0:
        return {
            "total_words": 0, "ready_words": 0, "learning_words": 0,
            "new_words": 0, "readiness_ratio": 1.0,
            "target_ratio": CHAPTER_READINESS_TARGET,
            "words_needed_for_target": 0,
        }

    ready = learning = new = 0
    for lemma in lemmas:
        wk = progress.word_knowledge.get(lemma)
        status = wk.known_status if wk else "new"
        if status in _READY_STATUSES:
            ready += 1
        elif status == "learning":
            learning += 1
        else:
            new += 1

    needed_total = int(total * CHAPTER_READINESS_TARGET + 0.9999)
    return {
        "total_words": total,
        "ready_words": ready,
        "learning_words": learning,
        "new_words": new,
        "readiness_ratio": ready / total,
        "target_ratio": CHAPTER_READINESS_TARGET,
        "words_needed_for_target": max(0, needed_total - ready),
    }


def compute_chapter_score(
    chapter: ChapterInfo,
    book: BookIndex,
    progress: UserProgress,
) -> ChapterScore:
    """average_correctness × (known_words / total_words).

    - `average_correctness` is the mean of `accuracy` over words the user has
      attempted in this chapter (0.0 if none yet).
    - `known_words` counts mastered + skipped.
    """
    lemmas = _chapter_lemmas(chapter, book)
    total = len(lemmas)
    if total == 0:
        return ChapterScore(
            chapter_id=chapter.chapter_id,
            chapter_number=chapter.number,
            average_correctness=1.0,
            known_words=0,
            total_words=0,
            score=1.0,
        )

    known = 0
    accuracies: list[float] = []
    for lemma in lemmas:
        wk = progress.word_knowledge.get(lemma)
        if wk is None:
            continue
        accuracies.append(wk.accuracy)
        if wk.known_status in _READY_STATUSES:
            known += 1

    avg = sum(accuracies) / len(accuracies) if accuracies else 0.0
    ratio_known = known / total
    return ChapterScore(
        chapter_id=chapter.chapter_id,
        chapter_number=chapter.number,
        average_correctness=round(avg, 4),
        known_words=known,
        total_words=total,
        score=round(avg * ratio_known, 4),
    )


def refresh_chapter_scores(book: BookIndex, progress: UserProgress) -> None:
    """Recompute every chapter score for this book and store it on `progress`."""
    for chapter in book.chapters:
        progress.chapter_scores[chapter.chapter_id] = compute_chapter_score(
            chapter, book, progress
        )
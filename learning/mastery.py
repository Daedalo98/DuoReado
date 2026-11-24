"""
learning/mastery.py
Computes per-word mastery state and chapter reading readiness metrics.
"""

from __future__ import annotations

from config import CHAPTER_READINESS_TARGET, MASTERED_MIN_ACCURACY, MASTERED_MIN_ATTEMPTS
from models.schemas import BookIndex, ChapterInfo, WordEntry


def refresh_word_mastery(entry: WordEntry) -> str:
    """
    Recompute `known_status` based on attempts, accuracy, and streak.
    """
    if entry.known_status == "skipped":
        return "skipped"

    if entry.attempts == 0:
        entry.known_status = "new"
    elif (
        entry.attempts >= MASTERED_MIN_ATTEMPTS
        and entry.accuracy >= MASTERED_MIN_ACCURACY
        and entry.streak >= 1
    ):
        entry.known_status = "mastered"
    else:
        entry.known_status = "learning"

    return entry.known_status


def mark_word_known(entry: WordEntry) -> None:
    """Allow user to mark a word as already known / skipped."""
    entry.known_status = "skipped"
    if entry.attempts == 0:
        entry.accuracy = 1.0


def chapter_readiness(book: BookIndex, chapter: ChapterInfo) -> dict[str, float | int]:
    """
    Compute readiness stats for a chapter:
    - total_words: unique lemmas in chapter
    - ready_words: words marked 'mastered' or 'skipped'
    - learning_words: words currently in progress
    - new_words: untouched words
    - readiness_ratio: ready_words / total_words (0.0 to 1.0)
    - target_ratio: CHAPTER_READINESS_TARGET
    - words_needed_for_target: how many more words to reach target_ratio
    """
    lemmas = [l for l in chapter.word_lemmas if l in book.words]
    total = len(lemmas)
    if total == 0:
        return {
            "total_words": 0,
            "ready_words": 0,
            "learning_words": 0,
            "new_words": 0,
            "readiness_ratio": 1.0,
            "target_ratio": CHAPTER_READINESS_TARGET,
            "words_needed_for_target": 0,
        }

    ready = 0
    learning = 0
    new = 0

    for lemma in lemmas:
        status = book.words[lemma].known_status
        if status in ("mastered", "skipped"):
            ready += 1
        elif status == "learning":
            learning += 1
        else:
            new += 1

    ratio = ready / total
    needed_total = int(total * CHAPTER_READINESS_TARGET + 0.9999)
    needed_more = max(0, needed_total - ready)

    return {
        "total_words": total,
        "ready_words": ready,
        "learning_words": learning,
        "new_words": new,
        "readiness_ratio": ratio,
        "target_ratio": CHAPTER_READINESS_TARGET,
        "words_needed_for_target": needed_more,
    }
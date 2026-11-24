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


# ---------------------------------------------------------------------------
# Manual status overrides (used by UI buttons and the phone bridge)
#
# All helpers are pure mutations on `progress` — the CALLER is responsible
# for calling save_user_progress(progress) afterwards. This matches the
# pattern of record_attempt() and keeps persistence a single concern.
# ---------------------------------------------------------------------------

def _get_or_create(progress: UserProgress, lemma: str) -> WordKnowledge:
    """Fetch the WordKnowledge for `lemma`, creating a fresh one if absent."""
    wk = progress.word_knowledge.get(lemma)
    if wk is None:
        wk = WordKnowledge(lemma=lemma)
        progress.word_knowledge[lemma] = wk
    return wk


def mark_word_known(progress: UserProgress, lemma: str) -> WordKnowledge:
    """Force-mark a word as mastered — the user says 'I know this'.

    Historical attempts/accuracy are preserved so that unlater `mark_word_new`
    can restore a sensible state. Status becomes 'mastered', which excludes
    the word from future queues (see scheduler._EXCLUDED_STATUSES).
    """
    wk = _get_or_create(progress, lemma)
    wk.known_status = "mastered"
    return wk


def mark_word_skipped(progress: UserProgress, lemma: str) -> WordKnowledge:
    """Force-mark a word as skipped — like 'known', but semantically "ignored"."""
    wk = _get_or_create(progress, lemma)
    wk.known_status = "skipped"
    return wk


def mark_word_new(progress: UserProgress, lemma: str) -> WordKnowledge:
    """Reset a word to 'new' — the 'forget this word' action.

    Also wipes the accuracy counter so the scheduler treats it as unseen
    again. If you prefer to preserve history, drop the two following lines.
    """
    wk = _get_or_create(progress, lemma)
    wk.known_status = "new"
    wk.attempts = 0
    wk.correct = 0
    wk.accuracy = 0.0
    return wk


def mark_word_learning(progress: UserProgress, lemma: str) -> WordKnowledge:
    """Force-mark a word as 'learning' (in the queue, not yet mastered)."""
    wk = _get_or_create(progress, lemma)
    wk.known_status = "learning"
    return wk


def set_word_status(
    progress: UserProgress,
    lemma: str,
    status: str,
) -> WordKnowledge:
    """Generic override — useful for admin / bulk tooling / CLI scripts."""
    valid = {"new", "learning", "mastered", "skipped"}
    if status not in valid:
        raise ValueError(
            f"Invalid status {status!r}; expected one of {sorted(valid)}"
        )
    wk = _get_or_create(progress, lemma)
    wk.known_status = status
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

# ---------------------------------------------------------------------------
# Back-compat wrappers around the old mastery API
# ---------------------------------------------------------------------------

def chapter_is_ready(
    chapter: ChapterInfo,
    book: BookIndex,
    progress: UserProgress,
    target: float = CHAPTER_READINESS_TARGET,
) -> bool:
    """Boolean shortcut for chapter_readiness(...)['readiness_ratio'] >= target."""
    return chapter_readiness(chapter, book, progress)["readiness_ratio"] >= target


def build_session_queue(
    book: BookIndex,
    progress: UserProgress,
    *,
    chapter_number: int | None = None,
    limit: int = 20,
) -> list[str]:
    """Forward to learning.scheduler.build_queue (the canonical implementation)."""
    from learning.scheduler import build_queue        # local import: avoid cycle
    return build_queue(book, progress, chapter_number=chapter_number, limit=limit)


def build_session_plan(
    book: BookIndex,
    progress: UserProgress,
    *,
    chapter_number: int | None = None,
    limit: int = 20,
) -> dict:
    """Older callers may expect a dict with a queue and a count.

    If a caller expects a *bare list*, change `.get('queue')` to a plain return —
    the wrapper below returns both so either shape works:
        build_session_plan(...)['queue']   → list[str]
        build_session_plan(...)['count']   → int
    """
    queue = build_session_queue(
        book, progress, chapter_number=chapter_number, limit=limit
    )
    return {"queue": queue, "count": len(queue)}
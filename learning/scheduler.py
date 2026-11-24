"""
learning/scheduler.py
Selects words and context sentences for a chapter practice session.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from models.schemas import BookIndex, ChapterInfo, WordEntry


@dataclass
class PracticeItem:
    word: WordEntry
    chapter_id: str
    sample_sentence: str
    page: int


def _pick_sentence_for_chapter(word: WordEntry, chapter_id: str) -> tuple[str, int]:
    """Choose a sentence from the target chapter if available, else any occurrence."""
    for occ in word.occurrences:
        if occ.chapter_id == chapter_id and occ.sentence.strip():
            return occ.sentence.strip(), occ.page
    for occ in word.occurrences:
        if occ.sentence.strip():
            return occ.sentence.strip(), occ.page
    return f"Example context for '{word.lemma}'.", 1


def build_chapter_practice_queue(
    book: BookIndex,
    chapter: ChapterInfo,
    batch_size: int = 15,
    include_skipped: bool = False,
) -> list[PracticeItem]:
    """
    Build an ordered list of PracticeItems for `chapter`.
    Priority order:
    1) Low-accuracy / struggling words ('learning' with lowest accuracy)
    2) Untouched ('new') words sorted by frequency in this chapter (most frequent first)
    3) Mastered words (for occasional reinforcement)
    """
    candidates: list[WordEntry] = []
    for lemma in chapter.word_lemmas:
        entry = book.words.get(lemma)
        if not entry:
            continue
        if entry.known_status == "skipped" and not include_skipped:
            continue
        candidates.append(entry)

    def priority_key(w: WordEntry) -> tuple[float, int, str]:
        ch_count = w.by_chapter.get(chapter.chapter_id, 0)
        knowledge_score = -1.0 if w.attempts == 0 else w.accuracy
        return (knowledge_score, -ch_count, w.lemma)

    candidates.sort(key=priority_key)
    selected = candidates[:batch_size]

    items: list[PracticeItem] = []
    for w in selected:
        sent, page = _pick_sentence_for_chapter(w, chapter.chapter_id)
        items.append(
            PracticeItem(
                word=w,
                chapter_id=chapter.chapter_id,
                sample_sentence=sent,
                page=page,
            )
        )
    return items


def build_meaning_options(
    target_word: WordEntry,
    book: BookIndex,
    chapter: ChapterInfo,
    num_options: int = 4,
) -> list[str]:
    """
    Build multiple-choice options (each labeled with its lemma + known translation/hint).
    """
    pool = [
        book.words[l]
        for l in chapter.word_lemmas
        if l != target_word.lemma and l in book.words
    ]
    random.shuffle(pool)
    distractors = pool[: max(0, num_options - 1)]

    def format_choice(w: WordEntry) -> str:
        if w.cached_translation:
            first_line = w.cached_translation.splitlines()[0]
            return f"{w.lemma}  —  {first_line[:60]}"
        forms_str = ", ".join(list(w.forms.keys())[:2])
        return f"{w.lemma}  (forms: {forms_str or w.lemma})"

    choices = [format_choice(target_word)] + [format_choice(d) for d in distractors]
    random.shuffle(choices)
    return choices
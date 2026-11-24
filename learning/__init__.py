"""learning/__init__.py

Public API of the learning package. Each submodule is imported defensively
so a missing legacy name in one module does not brick the entire package.
"""

# --- mastery ----------------------------------------------------------------
try:
    from .mastery import (                                     # noqa: F401
        CHAPTER_READINESS_TARGET,
        MASTERED_MIN_ACCURACY,
        MASTERED_MIN_ATTEMPTS,
        chapter_is_ready,
        chapter_readiness,
        compute_chapter_score,
        mark_word_known,
        mark_word_learning,
        mark_word_new,
        mark_word_skipped,
        record_attempt,
        refresh_chapter_scores,
        refresh_word_mastery,
        set_word_status,
    )
except ImportError as _exc:                                    # pragma: no cover
    import warnings
    warnings.warn(f"learning.mastery partial import: {_exc}", stacklevel=2)

# --- scheduler --------------------------------------------------------------
try:
    from .scheduler import (                                   # noqa: F401
        PracticeItem,
        QuizQuestion,
        build_chapter_practice_queue,
        build_meaning_options,
        build_queue,
        build_quiz,
        pick_sentence,
    )
except ImportError as _exc:                                    # pragma: no cover
    import warnings
    warnings.warn(f"learning.scheduler partial import: {_exc}", stacklevel=2)
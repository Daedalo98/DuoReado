"""learning/user_store.py
Per-user persistence. One file per user: data/users/<id>/progress.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from config import USERS_DIR, log, safe_read_text
from models.schemas import UserProgress, utc_now_iso


def user_dir(user_id: str) -> Path:
    safe = user_id.replace("/", "_").replace("\\", "_") or "default"
    path = USERS_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_progress_path(user_id: str) -> Path:
    """Public — the bridge uses this to read/write raw JSON."""
    return user_dir(user_id) / "progress.json"


def load_user_progress(user_id: str) -> UserProgress:
    path = user_progress_path(user_id)
    if not path.exists():
        return UserProgress(user_id=user_id)
    try:
        return UserProgress.from_dict(json.loads(safe_read_text(path)))
    except Exception as exc:
        log.warning("Corrupt progress for %s (%s) — starting fresh", user_id, exc)
        return UserProgress(user_id=user_id)


def save_user_progress(progress: UserProgress) -> Path:
    progress.updated_at = utc_now_iso()
    path = user_progress_path(progress.user_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(progress.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    log.debug("Saved user progress → %s", path)
    return path
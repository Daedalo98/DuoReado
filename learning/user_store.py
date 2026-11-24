# learning/user_store.py  — NEW FILE

import json, os
from pathlib import Path
from config import DATA_DIR          # e.g. DATA_DIR / "users"
from models.schemas import UserProgress, utc_now_iso

USERS_DIR = DATA_DIR / "users"

def _user_path(user_id: str) -> Path:
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    return USERS_DIR / user_id / "progress.json"

def load_user_progress(user_id: str) -> UserProgress:
    path = _user_path(user_id)
    if not path.exists():
        return UserProgress(user_id=user_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Reconstruct nested objects here (omitted for brevity — mirror from_dict pattern)
    return UserProgress.from_dict(raw)

def save_user_progress(progress: UserProgress) -> Path:
    progress.updated_at = utc_now_iso()
    path = _user_path(progress.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)            # atomic — same pattern as books[reference:13]
    return path
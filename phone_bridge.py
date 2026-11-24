"""phone_bridge.py

PC-side HTTP/JSON bridge.

The PC keeps the authoritative BookIndex (shared vocabulary) and UserProgress
(personal learning state) on disk. The phone only renders and sends actions.

Endpoints (all return JSON unless noted):
    GET  /                      -> phone_ui.html
    GET  /api/books             -> list available book files
    POST /api/select            -> {book_id, chapter_number, user_id}
    POST /api/session/start     -> {limit}
    GET  /api/session/next      -> next practice item
    POST /api/session/answer    -> {correct}
    GET  /api/progress          -> chapter scores + readiness
    GET  /api/word/<lemma>      -> word entry + user knowledge + examples
    POST /api/word/<lemma>/skip -> mark skipped
    POST /api/word/<lemma>/enrich -> run Ollama enrichment on the shared book

Run:
    python phone_bridge.py --host 0.0.0.0 --port 8765 --user default
"""
from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# --- Project imports (post-refactor) ---------------------------------------
from ai.enrichment import enrich_if_empty
from ai.ollama_client import OllamaClient
from config import DEFAULT_MODEL, ensure_directories
from ingestion.json_builder import (
    list_book_ids,
    load_book_index,
    save_book_index,
)
from learning import mastery
from learning.user_store import load_user_progress, save_user_progress
from models.schemas import BookIndex, ChapterInfo, UserProgress, VocabularyEntry

from ingestion.loader import (
    book_id_for_pdf, discover_pdfs, infer_language, ingest_folder, ingest_pdf, is_ingested,
)
from config import PDFS_DIR

BASE_DIR = Path(__file__).resolve().parent
UI_HTML_PATH = BASE_DIR / "phone_ui.html"


# ---------------------------------------------------------------------------
# Global state (guarded by a lock because ThreadingHTTPServer is concurrent)
# ---------------------------------------------------------------------------

class BridgeState:
    def __init__(self, user_id: str, model_name: str = DEFAULT_MODEL) -> None:
        self.lock = threading.Lock()
        self.user_id = user_id
        self.ollama = OllamaClient(model=model_name)

        self.book: BookIndex | None = None
        self.chapter: ChapterInfo | None = None
        self.progress: UserProgress = load_user_progress(user_id)

        # Practice queue lives in memory for the duration of one session.
        self.queue: list[str] = []
        self.current_lemma: str | None = None

        self.feedback: str = "Ready."
        self.trace: list[str] = []

    # -- small helpers ------------------------------------------------------

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.trace.append(f"[{ts}] {message}")
            # keep the last 200 lines only
            self.trace = self.trace[-200:]

    def _chapter(self, number: int) -> ChapterInfo | None:
        if self.book is None:
            return None
        return self.book.chapter_by_number(number)

    # -- session control ----------------------------------------------------

    def select(self, book_id: str, chapter_number: int) -> dict[str, Any]:
        with self.lock:
            self.book = load_book_index(book_id)
            self.progress = load_user_progress(self.user_id)
            self.chapter = self._chapter(chapter_number)
            if self.chapter is None:
                raise ValueError(f"Chapter {chapter_number} not found in {book_id}")
            self.queue.clear()
            self.current_lemma = None
            self.feedback = f"Loaded {self.book.title} — chapter {chapter_number}."
            self.log(self.feedback)
            return self._snapshot()

    def build_queue(self, limit: int = 20) -> list[str]:
        with self.lock:
            if self.book is None or self.chapter is None:
                raise ValueError("No book/chapter selected")
            lemmas = [l for l in self.chapter.word_lemmas if l in self.book.words]

            # Priority: learning > new. Mastered/skipped words are excluded.
            ranked: list[tuple[int, str]] = []
            for lemma in lemmas:
                wk = self.progress.word_knowledge.get(lemma)
                status = wk.known_status if wk else "new"
                if status in ("mastered", "skipped"):
                    continue
                ranked.append((0 if status == "learning" else 1, lemma))

            ranked.sort()
            self.queue = [lemma for _, lemma in ranked[:limit]]
            self.current_lemma = None
            self.feedback = f"Queue built with {len(self.queue)} words."
            self.log(self.feedback)
            return list(self.queue)

    def next_item(self) -> dict[str, Any] | None:
        with self.lock:
            if not self.queue:
                self.current_lemma = None
                return None
            lemma = self.queue.pop(0)
            self.current_lemma = lemma
            return self._word_payload(lemma)

    def submit_answer(self, correct: bool) -> dict[str, Any]:
        with self.lock:
            if not self.current_lemma:
                raise ValueError("No active question")
            wk = mastery.record_attempt(self.progress, self.current_lemma, correct)
            save_user_progress(self.progress)
            self.feedback = (
                f"{self.current_lemma}: {'✓' if correct else '✗'} "
                f"(accuracy {wk.accuracy:.0%}, status {wk.known_status})"
            )
            self.log(self.feedback)
            return {"lemma": self.current_lemma, "knowledge": wk.to_dict()}

    # -- read-only snapshots -----------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        if self.book is None:
            return {"book": None, "chapter": None, "user": self.progress.to_dict()}
        return {
            "book": {
                "book_id": self.book.book_id,
                "title": self.book.title,
                "language": self.book.language,
            },
            "chapter": {
                "number": self.chapter.number,
                "title": self.chapter.title,
                "total_words": len(
                    [l for l in self.chapter.word_lemmas if l in self.book.words]
                ),
            }
            if self.chapter
            else None,
            "user": self.progress.to_dict(),
        }

    def _word_payload(self, lemma: str) -> dict[str, Any]:
        entry = self.book.words.get(lemma) if self.book else None
        wk = self.progress.word_knowledge.get(lemma)
        examples = self.progress.personal_examples.get(lemma, [])
        return {
            "lemma": lemma,
            "entry": entry.to_dict() if entry else None,
            "knowledge": wk.to_dict()
            if wk
            else {"lemma": lemma, "known_status": "new", "accuracy": 0.0},
            "examples": [e.to_dict() for e in examples],
            "is_enriched": bool(entry and entry.entries),
        }

    def word_detail(self, lemma: str) -> dict[str, Any]:
        with self.lock:
            return self._word_payload(lemma)

    def progress_snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.book is None:
                return {"chapters": [], "user": self.progress.to_dict()}
            # Recompute scores for the currently loaded book only.
            mastery.refresh_chapter_scores(self.book, self.progress)
            save_user_progress(self.progress)
            chapters = []
            for ch in self.book.chapters:
                r = mastery.chapter_readiness(ch, self.book, self.progress)
                score = self.progress.chapter_scores.get(ch.chapter_id)
                chapters.append(
                    {
                        "chapter_id": ch.chapter_id,
                        "number": ch.number,
                        "title": ch.title,
                        "readiness": r,
                        "score": score.to_dict() if score else None,
                    }
                )
            return {"chapters": chapters, "user": self.progress.to_dict()}

    # -- mutations ----------------------------------------------------------

    def mark_skipped(self, lemma: str) -> dict[str, Any]:
        with self.lock:
            wk = mastery.mark_word_skipped(self.progress, lemma)
            save_user_progress(self.progress)
            self.log(f"Skipped {lemma}")
            return wk.to_dict()

    def mark_word_known(self, lemma: str) -> dict[str, Any]:
        with self.lock:
            wk = mastery.mark_word_known(self.progress, lemma)
            save_user_progress(self.progress)
            self.log(f"Marked {lemma} as known")
            return wk.to_dict()

    def enrich(self, lemma: str) -> bool:
        with self.lock:
            if self.book is None:
                raise ValueError("No book loaded")
            entry = self.book.words.get(lemma)
            if entry is None:
                return False
            changed = enrich_if_empty(
                entry,
                lemma,
                self.book.language,
                self.ollama,
                native_language=self.progress.native_language,
            )
            if changed:
                save_book_index(self.book)
                self.log(f"Enriched {lemma}")
            return changed

    def list_pdfs(self) -> list[dict]:
        out = []
        for pdf in discover_pdfs():
            out.append({
                "name": pdf.name,
                "book_id": book_id_for_pdf(pdf),
                "language": infer_language(pdf),
                "ingested": is_ingested(pdf),
                "size_kb": round(pdf.stat().st_size / 1024, 1),
            })
        return out

    def ingest_one(self, filename: str, language: str | None = None, force: bool = False) -> dict:
        pdf = PDFS_DIR / Path(filename).name
        if not pdf.exists():
            raise ValueError(f"PDF not found: {filename}")
        book = ingest_pdf(
            pdf, language=language, force=force,
            on_progress=lambda s, c, t: self.log(f"ingest {s} ({c}/{t})"),
        )
        self.log(f"Ingested {book.book_id}: {len(book.words)} words")
        return {"book_id": book.book_id, "words": len(book.words), "chapters": len(book.chapters)}

    def ingest_all(self, force: bool = False) -> list[dict]:
        return ingest_folder(force=force,
                            on_progress=lambda s, c, t: self.log(f"scan {s} ({c}/{t})"))

    def save_upload(self, filename: str, data: bytes) -> dict:
        safe = Path(filename).name
        if not safe.lower().endswith(".pdf"):
            raise ValueError("Only .pdf files are accepted")
        if not data:
            raise ValueError("Empty upload")
        PDFS_DIR.mkdir(parents=True, exist_ok=True)
        target = PDFS_DIR / safe
        target.write_bytes(data)
        self.log(f"Uploaded {safe} ({len(data)} bytes)")
        return {"name": safe, "size": len(data)}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

STATE: BridgeState


class BridgeHandler(BaseHTTPRequestHandler):
    # -- plumbing ----------------------------------------------------------

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _send(self, status: int, body: Any, content_type: str = "application/json") -> None:
        if isinstance(body, (dict, list)):
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            payload = body.encode("utf-8")
        else:
            payload = bytes(body)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        # Allow a phone browser on the same LAN to call the bridge.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def _err(self, status: int, message: str) -> None:
        self._send(status, {"ok": False, "error": message})

    # -- routing -----------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, b"")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                html = UI_HTML_PATH.read_text(encoding="utf-8")
                self._send(200, html, "text/html; charset=utf-8")
                return

            if path == "/api/books":
                self._send(200, {"ok": True, "books": list_book_ids()})
                return

            if path == "/api/state":
                self._send(200, {"ok": True, "state": STATE._snapshot()})
                return

            if path == "/api/session/next":
                item = STATE.next_item()
                self._send(200, {"ok": True, "item": item, "feedback": STATE.feedback})
                return

            if path == "/api/progress":
                self._send(200, {"ok": True, **STATE.progress_snapshot()})
                return

            if path.startswith("/api/word/"):
                lemma = path[len("/api/word/"):]
                self._send(200, {"ok": True, "word": STATE.word_detail(lemma)})
                return
            
            if path == "/api/pdfs":
                self._send(200, {"ok": True, "pdfs": STATE.list_pdfs()})
                return

            self._err(404, f"Unknown path: {path}")
        except Exception as exc:
            self._err(500, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json()

            if path == "/api/select":
                book_id = body.get("book_id")
                chapter_number = int(body.get("chapter_number", 1))
                snapshot = STATE.select(book_id, chapter_number)
                self._send(200, {"ok": True, "state": snapshot})
                return

            if path == "/api/session/start":
                limit = int(body.get("limit", 20))
                queue = STATE.build_queue(limit=limit)
                self._send(200, {"ok": True, "queue": queue})
                return

            if path == "/api/session/answer":
                correct = bool(body.get("correct"))
                result = STATE.submit_answer(correct)
                self._send(200, {"ok": True, **result})
                return

            if path.startswith("/api/word/") and path.endswith("/skip"):
                lemma = path[len("/api/word/"):-len("/skip")]
                wk = STATE.mark_skipped(lemma)
                self._send(200, {"ok": True, "knowledge": wk})
                return

            if path.startswith("/api/word/") and path.endswith("/enrich"):
                lemma = path[len("/api/word/"):-len("/enrich")]
                changed = STATE.enrich(lemma)
                self._send(200, {"ok": True, "enriched": changed})
                return

            if path == "/api/ingest":
                body = self._read_json()
                filename = body.get("filename")
                if filename:
                    result = STATE.ingest_one(filename, body.get("language"), bool(body.get("force")))
                else:
                    result = STATE.ingest_all(bool(body.get("force")))
                self._send(200, {"ok": True, "result": result})
                return

            if path == "/api/upload":
                # raw-body upload: Content-Type: application/pdf, ?name=<filename>
                from urllib.parse import parse_qs
                name = (parse_qs(urlparse(self.path).query).get("name") or [""])[0]
                length = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(length) if length else b""
                self._send(200, {"ok": True, "upload": STATE.save_upload(name, data)})
                return

            self._err(404, f"Unknown path: {path}")
        except Exception as exc:
            self._err(400, str(exc))

    # quieter logs
    def log_message(self, fmt: str, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DuoReado phone bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--user", default="default", help="UserProgress id")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    ensure_directories()

    global STATE
    STATE = BridgeState(user_id=args.user, model_name=args.model)

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    print(f"DuoReado bridge listening on http://{args.host}:{args.port}")
    print(f"Serving {UI_HTML_PATH.name} at /")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
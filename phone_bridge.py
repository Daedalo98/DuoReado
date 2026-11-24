"""phone_bridge.py
PC-side HTTP/JSON bridge.

New in this revision
--------------------
*   GET  /api/book/<id>                     → summary + chapter list
*   GET  /api/words?book_id=X&chapter=N&q=&limit=&offset=
*   GET  /api/pdf/pages?name=foo.pdf        → {total_pages, suggested_chapters}
*   GET  /api/pdf/chapters?name=foo.pdf     → {chapters, source: sidecar|auto|none}
*   POST /api/pdf/chapters?name=foo.pdf     → saves <pdf>.chapters.json
*   GET  /api/raw/<kind>/<id>               → {text}
*   POST /api/raw/<kind>/<id>               → {text} → validates + writes
*   GET  /api/raw/<kind>/<id>/download      → attachment
*   POST /api/word/<lemma>/known            → marks mastered

Upload bug (byte 0xe2) is fixed: only the JSON branches call
``self._read_json()``; the upload branch reads raw bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


from ai.enrichment import enrich_if_empty
from ai.ollama_client import OllamaClient
from config import DEFAULT_MODEL, PDFS_DIR, ensure_directories, safe_read_text

from dataclasses import asdict
from ingestion.json_builder import (
    book_path,
    list_book_ids,
    load_book_index,
    load_sentences,
    read_chapter_text,
    save_book_index,
    sentences_path,
    text_path,
)

from ingestion.loader import (
    book_id_for_pdf,
    chapter_sidecar_path,
    discover_pdfs,
    infer_language,
    is_ingested,
    ingest_pdf,
    read_chapter_sidecar,
    write_chapter_sidecar,
)
from ingestion.pdf_extractor import pdf_page_info
from learning import mastery

from learning.user_store import (
    load_user_progress,
    save_user_progress,
    user_progress_path,
)
from models.schemas import (
    BookIndex, 
    ChapterInfo, 
    UserProgress,
    Occurrence,
)

BASE_DIR = Path(__file__).resolve().parent
UI_HTML_PATH = BASE_DIR / "phone_ui.html"


# ── Live task tracker ───────────────────────────────────────────────────
class LiveTask:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = False
        self.label = ""
        self.stage = ""
        self.current = 0
        self.total = 0
        self.started_at = 0.0
        self.eta_seconds: float | None = None

    def begin(self, label: str, total: int) -> None:
        with self.lock:
            self.active = True
            self.label = label
            self.stage = "starting"
            self.current = 0
            self.total = max(0, int(total))
            self.started_at = time.time()
            self.eta_seconds = None

    def update(self, stage: str, current: int, total: int) -> None:
        with self.lock:
            self.stage = stage
            self.current = int(current)
            self.total = max(1, int(total))
            elapsed = time.time() - self.started_at
            ratio = self.current / self.total
            self.eta_seconds = (elapsed / ratio) * (1 - ratio) if ratio > 0 else None

    def finish(self, label: str = "") -> None:
        with self.lock:
            self.active = False
            if label:
                self.label = label
            self.stage = "done"
            self.eta_seconds = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "active": self.active, "label": self.label, "stage": self.stage,
                "current": self.current, "total": self.total,
                "eta_seconds": round(self.eta_seconds, 1)
                    if self.eta_seconds is not None else None,
            }


# ── Bridge state ────────────────────────────────────────────────────────
class BridgeState:
    def __init__(self, user_id: str, model_name: str = DEFAULT_MODEL) -> None:
        self.lock = threading.Lock()
        self.user_id = user_id
        self.ollama = OllamaClient(model=model_name)
        self.book: BookIndex | None = None
        self.chapter: ChapterInfo | None = None
        self.progress: UserProgress = load_user_progress(user_id)
        self.queue: list[str] = []
        self.current_lemma: str | None = None
        self.feedback: str = "Ready."
        self.trace: list[str] = []
        self.task = LiveTask()

        # Small single-slot cache for /api/words searches on large books.
        self._cached_book_id: str | None = None
        self._cached_book: BookIndex | None = None

        self._cached_sentences: dict[str, list[Occurrence]] | None = None
        self._cached_sentences_book: str | None = None

    # -- logging -------------------------------------------------------
    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.trace.append(f"[{ts}] {message}")
            self.trace = self.trace[-300:]
        print(f"[{ts}] {message}")

    # -- book / sentence caches ---------------------------------------
    def _get_book(self, book_id: str) -> BookIndex:
        if self._cached_book_id == book_id and self._cached_book is not None:
            return self._cached_book
        self._cached_book = load_book_index(book_id)
        self._cached_book_id = book_id
        return self._cached_book

    def _get_sentences(self, book_id: str) -> dict[str, list[Occurrence]]:
        if self._cached_sentences_book == book_id and self._cached_sentences is not None:
            return self._cached_sentences
        self._cached_sentences = load_sentences(book_id)
        self._cached_sentences_book = book_id
        return self._cached_sentences

    def _invalidate_book_cache(self, book_id: str | None = None) -> None:
        if book_id is None or self._cached_book_id == book_id:
            self._cached_book_id = None
            self._cached_book = None
            self._cached_sentences_book = None
            self._cached_sentences = None

    # -- session ------------------------------------------------------
    def select(self, book_id: str, chapter_number: int) -> dict[str, Any]:
        with self.lock:
            self.book = load_book_index(book_id)
            self._cached_book_id = book_id
            self._cached_book = self.book
            self.progress = load_user_progress(self.user_id)
            self.chapter = self.book.chapter_by_number(chapter_number)
            if self.chapter is None:
                raise ValueError(f"Chapter {chapter_number} not found in {book_id}")
            self.queue.clear()
            self.current_lemma = None
            self.feedback = f"Loaded {self.book.title} — chapter {chapter_number}."
            self.log(self.feedback)
            return self._snapshot()

    def build_queue(self, limit: int = 20) -> list[str]:
        with self.lock:
            if not (self.book and self.chapter):
                raise ValueError("No book/chapter selected")
            lemmas = [l for l in self.chapter.word_lemmas if l in self.book.words]
            ranked: list[tuple[int, str]] = []
            for lemma in lemmas:
                wk = self.progress.word_knowledge.get(lemma)
                status = wk.known_status if wk else "new"
                if status in ("mastered", "skipped"):
                    continue
                ranked.append((0 if status == "learning" else 1, lemma))
            ranked.sort()
            self.queue = [l for _, l in ranked[:limit]]
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

    # -- snapshots ----------------------------------------------------
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
            } if self.chapter else None,
            "user": self.progress.to_dict(),
        }

    # -- word payload, merged with sidecar sentences ------------------
    def _word_payload(self, lemma: str) -> dict[str, Any]:
        entry = self.book.words.get(lemma) if self.book else None
        payload = entry.to_dict() if entry else None
        # New books: sentences live in the sidecar, not in the main JSON.
        if payload is not None and not payload.get("sentences"):
            occs = self._get_sentences(self.book.book_id).get(lemma, [])
            if occs:
                payload["sentences"] = [asdict(o) for o in occs]
        wk = self.progress.word_knowledge.get(lemma)
        examples = self.progress.personal_examples.get(lemma, [])
        return {
            "lemma": lemma,
            "entry": payload,
            "knowledge": wk.to_dict()
                if wk else {"lemma": lemma, "known_status": "new", "accuracy": 0.0},
            "examples": [e.to_dict() for e in examples],
            "is_enriched": bool(entry and entry.entries),
        }

    # -- word search: sort + filter ----------------------------------
    def list_words(
        self,
        book_id: str,
        *,
        chapter_number: int | None = None,
        q: str = "",
        sort: str = "alpha",
        keyword: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        with self.lock:
            book = self._get_book(book_id)
            if chapter_number:
                ch = book.chapter_by_number(chapter_number)
                if ch is None:
                    raise ValueError(f"Chapter {chapter_number} not in {book_id}")
                lemmas = [l for l in ch.word_lemmas if l in book.words]
            else:
                lemmas = list(book.words.keys())

            # Filter: search substring
            if q:
                ql = q.lower()
                lemmas = [l for l in lemmas if ql in l.lower()]

            # Filter: keyword (exact, case-insensitive)
            if keyword:
                kw = keyword.lower()
                lemmas = [
                    l for l in lemmas
                    if any(k.lower() == kw for k in book.words[l].semantic_keywords)
                ]

            # Sort
            if sort == "freq":
                lemmas.sort(key=lambda l: (-book.words[l].occurrences, l))
            elif sort == "keyword":
                def _key(l: str):
                    kws = book.words[l].semantic_keywords
                    if kws:
                        return (0, kws[0].lower(), -book.words[l].occurrences, l)
                    return (1, "", -book.words[l].occurrences, l)
                lemmas.sort(key=_key)
            else:  # alpha
                lemmas.sort()

            total = len(lemmas)
            page = lemmas[offset: offset + limit]

            items = []
            for lemma in page:
                entry = book.words[lemma]
                wk = self.progress.word_knowledge.get(lemma)
                translation = ""
                for e in entry.entries:
                    if e.relation == "base" and e.translations:
                        translation = e.translations[0]; break
                if not translation:
                    for e in entry.entries:
                        if e.translations:
                            translation = e.translations[0]; break
                items.append({
                    "lemma": lemma,
                    "status": wk.known_status if wk else "new",
                    "occurrences": entry.occurrences,
                    "first_chapter": entry.first_chapter,
                    "is_enriched": bool(entry.entries),
                    "translation": translation,
                    "keywords": entry.semantic_keywords,
                })
            return {
                "total": total, "offset": offset, "limit": limit,
                "chapter_number": chapter_number,
                "sort": sort, "keyword": keyword,
                "words": items,
            }

    # -- unique keywords across the book -----------------------------
    def list_keywords(self, book_id: str) -> dict[str, Any]:
        with self.lock:
            book = self._get_book(book_id)
            counter: dict[str, int] = {}
            for entry in book.words.values():
                for kw in entry.semantic_keywords:
                    k = kw.lower()
                    counter[k] = counter.get(k, 0) + 1
            ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            return {
                "book_id": book_id,
                "keywords": [{"keyword": k, "count": c} for k, c in ordered],
            }
          
    def word_detail(self, lemma: str) -> dict[str, Any]:
        with self.lock:
            return self._word_payload(lemma)

    # -- chapter text (from .txt sidecar) ----------------------------
    def chapter_text(self, book_id: str, chapter_number: int) -> dict[str, Any]:
        with self.lock:
            book = self._get_book(book_id)
            chapter = book.chapter_by_number(chapter_number)
            if chapter is None:
                raise ValueError(f"Chapter {chapter_number} not in {book_id}")
            text = read_chapter_text(book_id, chapter)
            return {
                "book_id": book_id,
                "chapter_number": chapter.number,
                "title": chapter.title,
                "start_page": chapter.start_page,
                "end_page": chapter.end_page,
                "text": text,
            }

    # -- size reporting ---------------------------------------------
    def file_sizes(self, book_id: str) -> dict[str, Any]:
        def _size(p: Path) -> int:
            return p.stat().st_size if p.exists() else 0
        return {
            "main_bytes": _size(book_path(book_id)),
            "sentences_bytes": _size(sentences_path(book_id)),
            "text_bytes": _size(text_path(book_id)),
        }        

    def progress_snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.book is None:
                return {"chapters": [], "user": self.progress.to_dict()}
            mastery.refresh_chapter_scores(self.book, self.progress)
            save_user_progress(self.progress)
            chapters = []
            for ch in self.book.chapters:
                r = mastery.chapter_readiness(ch, self.book, self.progress)
                score = self.progress.chapter_scores.get(ch.chapter_id)
                chapters.append({
                    "chapter_id": ch.chapter_id, "number": ch.number,
                    "title": ch.title, "readiness": r,
                    "score": score.to_dict() if score else None,
                })
            return {"chapters": chapters, "user": self.progress.to_dict()}

    # -- book summary + word search -----------------------------------
    def book_summary(self, book_id: str) -> dict[str, Any]:
        with self.lock:
            book = self._get_book(book_id)
            return {
                "book_id": book.book_id,
                "title": book.title,
                "language": book.language,
                "total_words": len(book.words),
                "chapters": [
                    {
                        "number": c.number, "title": c.title,
                        "start_page": c.start_page, "end_page": c.end_page,
                        "word_count": len([l for l in c.word_lemmas if l in book.words]),
                    }
                    for c in book.chapters
                ],
            }

    def list_words(
        self,
        book_id: str,
        *,
        chapter_number: int | None = None,
        q: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        with self.lock:
            book = self._get_book(book_id)
            if chapter_number:
                ch = book.chapter_by_number(chapter_number)
                if ch is None:
                    raise ValueError(f"Chapter {chapter_number} not in {book_id}")
                lemmas = [l for l in ch.word_lemmas if l in book.words]
            else:
                lemmas = list(book.words.keys())

            lemmas.sort()
            if q:
                ql = q.lower()
                lemmas = [l for l in lemmas if ql in l.lower()]

            total = len(lemmas)
            page = lemmas[offset: offset + limit]

            items = []
            for lemma in page:
                entry = book.words[lemma]
                wk = self.progress.word_knowledge.get(lemma)
                translation = ""
                for e in entry.entries:
                    if e.relation == "base" and e.translations:
                        translation = e.translations[0]
                        break
                if not translation:
                    for e in entry.entries:
                        if e.translations:
                            translation = e.translations[0]
                            break
                items.append({
                    "lemma": lemma,
                    "status": wk.known_status if wk else "new",
                    "occurrences": entry.occurrences,
                    "first_chapter": entry.first_chapter,
                    "is_enriched": bool(entry.entries),
                    "translation": translation,
                })
            return {
                "total": total, "offset": offset, "limit": limit,
                "chapter_number": chapter_number, "words": items,
            }

    # -- mutations ----------------------------------------------------
    def mark_skipped(self, lemma: str) -> dict[str, Any]:
        with self.lock:
            wk = mastery.mark_word_skipped(self.progress, lemma)
            save_user_progress(self.progress)
            self.log(f"Skipped {lemma}")
            return wk.to_dict()

    def mark_known(self, lemma: str) -> dict[str, Any]:
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
            self.task.begin(f"Enriching '{lemma}'", total=1)
            self.log(f"Enriching '{lemma}' with Ollama…")
            try:
                changed = enrich_if_empty(
                    entry, lemma, self.book.language, self.ollama,
                    native_language=self.progress.native_language,
                )
            finally:
                self.task.finish(f"Enrichment for '{lemma}' done")
            if changed:
                save_book_index(self.book)
                self._invalidate_book_cache(self.book.book_id)
                self.log(f"Enriched {lemma}")
            return changed

    # -- library ------------------------------------------------------
    def list_pdfs(self) -> list[dict]:
        out = []
        for pdf in discover_pdfs():
            out.append({
                "name": pdf.name,
                "book_id": book_id_for_pdf(pdf),
                "language": infer_language(pdf),
                "ingested": is_ingested(pdf),
                "has_chapters": chapter_sidecar_path(pdf).exists(),
                "size_kb": round(pdf.stat().st_size / 1024, 1),
            })
        return out

    # -- ingestion now includes size info ----------------------------
    def ingest_one(self, filename: str, language: str | None = None, force: bool = False) -> dict:
        pdf = PDFS_DIR / Path(filename).name
        if not pdf.exists():
            raise ValueError(f"PDF not found: {filename}")
        self.task.begin(f"Ingesting {pdf.name}", total=6)
        self.log(f"Ingest start: {pdf.name}")
        try:
            book = ingest_pdf(
                pdf, language=language, force=force,
                on_progress=lambda s, c, t: (
                    self.task.update(s, c, t),
                    self.log(f"{pdf.name} → {s} ({c}/{t})"),
                ),
            )
        finally:
            self.task.finish(f"Ingested {pdf.name}")
        self._invalidate_book_cache(book.book_id)
        sizes = self.file_sizes(book.book_id)
        self.log(
            f"Ingested {book.book_id}: {len(book.words)} words, "
            f"{len(book.chapters)} chapters — "
            f"main={sizes['main_bytes']//1024}KB, "
            f"sentences={sizes['sentences_bytes']//1024}KB, "
            f"text={sizes['text_bytes']//1024}KB"
        )
        return {
            "book_id": book.book_id, "words": len(book.words),
            "chapters": len(book.chapters), **sizes,
        }

    def ingest_all(self, force: bool = False) -> list[dict]:
        pdfs = discover_pdfs()
        self.task.begin(f"Ingesting {len(pdfs)} PDF(s)", total=max(1, len(pdfs)))
        results: list[dict] = []
        try:
            for idx, pdf in enumerate(pdfs, start=1):
                self.task.update(f"ingest {pdf.name}", idx - 1, len(pdfs))
                self.log(f"[{idx}/{len(pdfs)}] {pdf.name}")
                try:
                    book = ingest_pdf(
                        pdf, force=force,
                        on_progress=lambda s, c, t: self.log(f"  {pdf.name} → {s} ({c}/{t})"),
                    )
                    self._invalidate_book_cache(book.book_id)
                    results.append({
                        "pdf": pdf.name, "book_id": book.book_id,
                        "words": len(book.words), "chapters": len(book.chapters),
                        "ok": True,
                    })
                except Exception as exc:
                    results.append({"pdf": pdf.name, "ok": False, "error": str(exc)})
                self.task.update(f"done {pdf.name}", idx, len(pdfs))
        finally:
            self.task.finish(f"Scanned {len(pdfs)} PDF(s)")
        return results

    def save_upload(self, filename: str, data: bytes) -> dict:
        safe = Path(filename).name
        if not safe.lower().endswith(".pdf"):
            raise ValueError("Only .pdf files are accepted")
        if not data:
            raise ValueError("Empty upload")
        PDFS_DIR.mkdir(parents=True, exist_ok=True)
        target = PDFS_DIR / safe
        target.write_bytes(data)
        self.log(f"Uploaded {safe} ({len(data):,} bytes)")
        return {"name": safe, "size": len(data)}

    # -- chapter specs editor -----------------------------------------
    def pdf_pages(self, filename: str) -> dict[str, Any]:
        pdf = PDFS_DIR / Path(filename).name
        if not pdf.exists():
            raise ValueError(f"PDF not found: {filename}")
        info = pdf_page_info(pdf)
        sidecar = read_chapter_sidecar(pdf)
        return {
            "filename": pdf.name,
            "total_pages": info["total_pages"],
            "suggested_chapters": info["suggested_chapters"],
            "current_chapters": sidecar or info["suggested_chapters"],
            "source": "sidecar" if sidecar else "auto",
        }

    def save_chapter_specs(self, filename: str, chapters: list[dict]) -> dict[str, Any]:
        pdf = PDFS_DIR / Path(filename).name
        if not pdf.exists():
            raise ValueError(f"PDF not found: {filename}")

        # ── Validate against total_pages before writing ────────────────
        total = pdf_page_info(pdf)["total_pages"]
        prev_end = 0
        for i, c in enumerate(chapters, start=1):
            start = int(c.get("start_page", 0))
            end = int(c.get("end_page", 0)) or total
            if not (1 <= start <= total):
                raise ValueError(
                    f"Chapter {i}: start_page {start} out of [1, {total}]"
                )
            if end < start:
                raise ValueError(
                    f"Chapter {i}: end_page {end} < start_page {start}"
                )
            if end > total:
                raise ValueError(
                    f"Chapter {i}: end_page {end} > {total}"
                )
            if start <= prev_end:
                raise ValueError(
                    f"Chapter {i}: overlaps previous (starts at {start}, "
                    f"previous ended at {prev_end})"
                )
            prev_end = end

        write_chapter_sidecar(pdf, chapters)
        self.log(f"Saved chapter specs for {pdf.name} ({len(chapters)} chapters)")
        return {
            "filename": pdf.name,
            "chapters": read_chapter_sidecar(pdf) or [],
        }

    # -- raw JSON editing ---------------------------------------------
    def _raw_path(self, kind: str, ident: str) -> Path:
        if kind == "book":
            return book_path(ident)
        if kind == "user":
            return user_progress_path(ident)
        raise ValueError(f"Unknown kind {kind!r}")

    def get_raw_json(self, kind: str, ident: str) -> str:
        path = self._raw_path(kind, ident)
        if not path.exists():
            raise FileNotFoundError(f"{kind} {ident} not found")
        return safe_read_text(path)

    def save_raw_json(self, kind: str, ident: str, text: str) -> dict[str, Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        # Validate against the expected schema before we overwrite.
        if kind == "book":
            try:
                BookIndex.from_dict(data)
            except Exception as exc:
                raise ValueError(f"Not a valid BookIndex: {exc}") from exc
        elif kind == "user":
            try:
                UserProgress.from_dict(data)
            except Exception as exc:
                raise ValueError(f"Not a valid UserProgress: {exc}") from exc
        else:
            raise ValueError(f"Unknown kind {kind!r}")

        path = self._raw_path(kind, ident)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

        # Invalidate in-memory caches so subsequent calls see the change.
        with self.lock:
            if kind == "book":
                self._invalidate_book_cache(ident)
                if self.book and self.book.book_id == ident:
                    self.book = load_book_index(ident)
                    self.log(f"Reloaded book {ident} after raw JSON edit")
            elif kind == "user" and ident == self.user_id:
                self.progress = load_user_progress(self.user_id)
                self.log(f"Reloaded user progress for {ident} after raw JSON edit")

        self.log(f"Saved raw JSON: {kind}/{ident}")
        return {"path": str(path), "bytes": len(text)}


# ── HTTP handler ────────────────────────────────────────────────────────
STATE: BridgeState


class BridgeHandler(BaseHTTPRequestHandler):
    # -- plumbing -----------------------------------------------------
    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            return {}

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
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def _send_attachment(self, filename: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, message: str) -> None:
        self._send(status, {"ok": False, "error": message})

    # -- GET ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send(200, UI_HTML_PATH.read_text(encoding="utf-8"),
                           "text/html; charset=utf-8")
                return

            if path == "/api/books":
                self._send(200, {"ok": True, "books": list_book_ids()}); return

            if path == "/api/state":
                self._send(200, {"ok": True, "state": STATE._snapshot()}); return

            if path == "/api/session/next":
                item = STATE.next_item()
                self._send(200, {"ok": True, "item": item, "feedback": STATE.feedback}); return

            if path == "/api/progress":
                self._send(200, {"ok": True, **STATE.progress_snapshot()}); return

            if path == "/api/progress/live":
                self._send(200, {"ok": True, "task": STATE.task.snapshot()}); return

            if path == "/api/log":
                tail = int((qs.get("n") or ["40"])[0])
                with STATE.lock:
                    lines = STATE.trace[-tail:]
                self._send(200, {"ok": True, "lines": lines}); return

            if path == "/api/pdfs":
                self._send(200, {"ok": True, "pdfs": STATE.list_pdfs()}); return

            # /api/pdf/pages?name=foo.pdf
            if path == "/api/pdf/pages":
                name = (qs.get("name") or [""])[0]
                self._send(200, {"ok": True, **STATE.pdf_pages(name)}); return

            # /api/pdf/chapters?name=foo.pdf
            if path == "/api/pdf/chapters":
                name = (qs.get("name") or [""])[0]
                self._send(200, {"ok": True, **STATE.pdf_pages(name)}); return

            # /api/book/<book_id>
            if path.startswith("/api/book/"):
                book_id = path[len("/api/book/"):]
                self._send(200, {"ok": True, **STATE.book_summary(book_id)}); return

            # ── GET routing additions ──────────────────────────────────────────────
            if path == "/api/words":
                book_id = (qs.get("book_id") or [""])[0]
                if not book_id:
                    self._err(400, "book_id is required"); return
                ch_raw = (qs.get("chapter") or [""])[0]
                chapter_number = int(ch_raw) if ch_raw else None
                q = (qs.get("q") or [""])[0]
                sort = (qs.get("sort") or ["alpha"])[0]
                keyword = (qs.get("keyword") or [""])[0]
                limit = min(500, max(1, int((qs.get("limit") or ["100"])[0])))
                offset = max(0, int((qs.get("offset") or ["0"])[0]))
                self._send(200, {"ok": True, **STATE.list_words(
                    book_id, chapter_number=chapter_number, q=q,
                    sort=sort, keyword=keyword,
                    limit=limit, offset=offset,
                )}); return

            if path == "/api/keywords":
                book_id = (qs.get("book_id") or [""])[0]
                if not book_id:
                    self._err(400, "book_id is required"); return
                self._send(200, {"ok": True, **STATE.list_keywords(book_id)}); return

            if path == "/api/chapter/text":
                book_id = (qs.get("book_id") or [""])[0]
                ch_raw = (qs.get("chapter") or ["1"])[0]
                if not book_id:
                    self._err(400, "book_id is required"); return
                self._send(200, {"ok": True, **STATE.chapter_text(book_id, int(ch_raw))}); return

            # /api/raw/<kind>/<id> and /api/raw/<kind>/<id>/download
            if path.startswith("/api/raw/"):
                rest = path[len("/api/raw/"):]
                if rest.endswith("/download"):
                    ident_path = rest[: -len("/download")]
                    kind, _, ident = ident_path.partition("/")
                    text = STATE.get_raw_json(kind, ident)
                    self._send_attachment(f"{kind}_{ident}.json",
                                          text.encode("utf-8"))
                    return
                kind, _, ident = rest.partition("/")
                text = STATE.get_raw_json(kind, ident)
                self._send(200, {"ok": True, "kind": kind, "id": ident, "text": text})
                return

            if path.startswith("/api/word/"):
                lemma = path[len("/api/word/"):]
                self._send(200, {"ok": True, "word": STATE.word_detail(lemma)}); return

            self._err(404, f"Unknown path: {path}")
        except FileNotFoundError as exc:
            self._err(404, str(exc))
        except Exception as exc:
            self._err(500, str(exc))

    # -- POST ---------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            # ── Binary upload: NO JSON parsing on the body. ────────────
            if path == "/api/upload":
                name = (qs.get("name") or [""])[0]
                length = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(length) if length else b""
                self._send(200, {"ok": True, "upload": STATE.save_upload(name, data)})
                return

            # ── All other endpoints are JSON. ──────────────────────────
            body = self._read_json()

            if path == "/api/select":
                snapshot = STATE.select(
                    body.get("book_id"), int(body.get("chapter_number", 1))
                )
                self._send(200, {"ok": True, "state": snapshot}); return

            if path == "/api/session/start":
                queue = STATE.build_queue(limit=int(body.get("limit", 20)))
                self._send(200, {"ok": True, "queue": queue}); return

            if path == "/api/session/answer":
                result = STATE.submit_answer(bool(body.get("correct")))
                self._send(200, {"ok": True, **result}); return

            if path.startswith("/api/word/") and path.endswith("/skip"):
                lemma = path[len("/api/word/"):-len("/skip")]
                self._send(200, {"ok": True, "knowledge": STATE.mark_skipped(lemma)}); return

            if path.startswith("/api/word/") and path.endswith("/known"):
                lemma = path[len("/api/word/"):-len("/known")]
                self._send(200, {"ok": True, "knowledge": STATE.mark_known(lemma)}); return

            if path.startswith("/api/word/") and path.endswith("/enrich"):
                lemma = path[len("/api/word/"):-len("/enrich")]
                changed = STATE.enrich(lemma)
                self._send(200, {"ok": True, "enriched": changed}); return

            if path == "/api/ingest":
                filename = body.get("filename")
                if filename:
                    result = STATE.ingest_one(
                        filename, body.get("language"), bool(body.get("force"))
                    )
                else:
                    result = STATE.ingest_all(bool(body.get("force")))
                self._send(200, {"ok": True, "result": result}); return

            # /api/pdf/chapters?name=foo.pdf
            if path == "/api/pdf/chapters":
                name = (qs.get("name") or [""])[0]
                chapters = body.get("chapters") or []
                if not isinstance(chapters, list) or not chapters:
                    self._err(400, "chapters must be a non-empty list"); return
                result = STATE.save_chapter_specs(name, chapters)
                self._send(200, {"ok": True, **result}); return

            # /api/raw/<kind>/<id>
            if path.startswith("/api/raw/"):
                kind, _, ident = path[len("/api/raw/"):].partition("/")
                text = body.get("text")
                if not isinstance(text, str):
                    self._err(400, "text field is required"); return
                result = STATE.save_raw_json(kind, ident, text)
                self._send(200, {"ok": True, **result}); return

            self._err(404, f"Unknown path: {path}")
        except Exception as exc:
            self._err(400, str(exc))

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


# ── Entry point ────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="DuoReado phone bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--user", default="default")
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
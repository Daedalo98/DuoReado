"""ui/main_window.py
Self-contained Tkinter frontend:
  * Book/chapter/model selectors (real widgets — no placeholders).
  * Per-tab frames built here, driven by PracticeSession / ReaderSession.
  * Ingest tab wires directly into ingestion.loader with a live progress bar
    + ETA (mirrors phone_bridge's LiveTask).
  * Trace panel shows every internal step.
"""
from __future__ import annotations

import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ai.ollama_client import OllamaClient
from config import DEFAULT_MODEL, PDFS_DIR, log
from ingestion.json_builder import (
    list_saved_books, 
    load_book_index, 
    save_book_index,
    read_chapter_text,
)

from ingestion.loader import book_id_for_pdf, discover_pdfs, ingest_pdf
from learning.mastery import chapter_readiness
from learning.user_store import load_user_progress, save_user_progress
from models.schemas import BookIndex, ChapterInfo
from ui.practice_view import PracticeSession
from ui.reader_view import ReaderSession

USER_ID = "default"


class MainWindow(ttk.Frame):
    # ── init ───────────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root, padding=10)
        self.root = root
        self.pack(fill="both", expand=True)

        self.ollama = OllamaClient(model=DEFAULT_MODEL)
        self.progress = load_user_progress(USER_ID)

        self.book: BookIndex | None = None
        self.chapter: ChapterInfo | None = None
        self.saved_books: list[dict[str, str]] = []

        self.book_var = tk.StringVar()
        self.chapter_var = tk.StringVar(value="1")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.readiness_var = tk.StringVar(value="Readiness: —")

        # Live task tracking (mirrors phone_bridge.LiveTask).
        self._task_started_at = 0.0
        self._task_total = 0
        self._task_current = 0

        self._build_top_bar()
        self._build_notebook()
        self._build_trace_panel()
        self._refresh_book_list(select_latest=True)

    # ── Trace / progress ──────────────────────────────────────────────
    def trace(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"

        def _append() -> None:
            self.trace_text.configure(state="normal")
            self.trace_text.insert("end", line)
            self.trace_text.see("end")
            self.trace_text.configure(state="disabled")

        self.after(0, _append)
        log.info(message)

    def clear_trace(self) -> None:
        self.trace_text.configure(state="normal")
        self.trace_text.delete("1.0", "end")
        self.trace_text.configure(state="disabled")

    def _progress_begin(self, label: str, total: int) -> None:
        self._task_started_at = time.time()
        self._task_total = max(1, total)
        self._task_current = 0
        self.progress_label_var.set(label)
        self.progress_bar["value"] = 0
        self.progress_eta_var.set("")
        self.trace(label)

    def _progress_update(self, stage: str, current: int, total: int) -> None:
        self._task_current = current
        self._task_total = max(1, total)
        pct = 100.0 * current / self._task_total
        elapsed = time.time() - self._task_started_at
        ratio = current / self._task_total if self._task_total else 0
        eta = (elapsed / ratio) * (1 - ratio) if ratio > 0 else 0
        self.progress_bar["value"] = pct
        self.progress_label_var.set(stage)
        self.progress_eta_var.set(f"{int(eta)}s left" if eta > 0 else "")
        self.trace(f"{stage} ({current}/{total})")
        self.update_idletasks()

    def _progress_end(self, label: str) -> None:
        self.progress_bar["value"] = 100
        self.progress_eta_var.set("")
        self.progress_label_var.set(label)
        self.trace(label)

    # ── Top bar ───────────────────────────────────────────────────────
    def _build_top_bar(self) -> None:
        bar = ttk.LabelFrame(self, text="Active Book, Chapter & Ollama Model", padding=10)
        bar.pack(fill="x", pady=(0, 8))

        ttk.Label(bar, text="Book:").grid(row=0, column=0, sticky="w")
        self.book_combo = ttk.Combobox(bar, textvariable=self.book_var, width=40, state="readonly")
        self.book_combo.grid(row=0, column=1, sticky="w", padx=4)
        self.book_combo.bind("<<ComboboxSelected>>", self._on_book_selected)

        ttk.Label(bar, text="Chapter:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.chapter_combo = ttk.Combobox(
            bar, textvariable=self.chapter_var, width=6, state="readonly"
        )
        self.chapter_combo.grid(row=0, column=3, sticky="w")
        self.chapter_combo.bind("<<ComboboxSelected>>", self._on_chapter_selected)

        ttk.Label(bar, text="Model:").grid(row=0, column=4, sticky="w", padx=(12, 0))
        self.model_combo = ttk.Combobox(
            bar, textvariable=self.model_var, width=22,
            values=self.ollama.list_models(),
        )
        self.model_combo.grid(row=0, column=5, sticky="w")

        ttk.Button(bar, text="Refresh books", command=self._refresh_book_list)\
            .grid(row=0, column=6, padx=(12, 0))
        ttk.Label(bar, textvariable=self.readiness_var)\
            .grid(row=1, column=0, columnspan=7, sticky="w", pady=(6, 0))

        # Progress bar + ETA
        prog = ttk.Frame(self)
        prog.pack(fill="x", pady=(0, 6))
        self.progress_bar = ttk.Progressbar(prog, orient="horizontal", mode="determinate")
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_label_var = tk.StringVar(value="Ready.")
        self.progress_eta_var = tk.StringVar(value="")
        ttk.Label(prog, textvariable=self.progress_label_var, width=42).pack(side="left", padx=6)
        ttk.Label(prog, textvariable=self.progress_eta_var, width=10).pack(side="left")

    # ── Notebook ──────────────────────────────────────────────────────
    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.ingest_tab = self._build_ingest_tab()
        self.practice_tab = self._build_practice_tab()
        self.reader_tab = self._build_reader_tab()

        self.notebook.add(self.ingest_tab, text="1. Import PDF & Index")
        self.notebook.add(self.practice_tab, text="2. Practice Words")
        self.notebook.add(self.reader_tab, text="3. Read Chapter")

    # ── Ingest tab ────────────────────────────────────────────────────
    def _build_ingest_tab(self) -> ttk.Frame:
        f = ttk.Frame(self.notebook, padding=10)
        ttk.Label(
            f,
            text=f"PDFs live in {PDFS_DIR}. Drop them there, then click the button.",
            foreground="#475569",
        ).pack(anchor="w")

        row = ttk.Frame(f); row.pack(fill="x", pady=8)
        ttk.Button(row, text="Rescan folder & ingest all",
                   command=self._ingest_all).pack(side="left")
        ttk.Button(row, text="Add single PDF…",
                   command=self._ingest_one).pack(side="left", padx=6)

        self.ingest_list = ttk.Treeview(
            f, columns=("name", "lang", "ingested", "size"),
            show="headings", height=10,
        )
        for c, t, w in [("name", "File", 320), ("lang", "Lang", 60),
                        ("ingested", "Ingested", 90), ("size", "Size KB", 90)]:
            self.ingest_list.heading(c, text=t)
            self.ingest_list.column(c, width=w, anchor="w")
        self.ingest_list.pack(fill="both", expand=True)

        ttk.Button(f, text="Refresh list", command=self._refresh_pdf_list).pack(pady=6)
        self._refresh_pdf_list()
        return f

    # ── Practice tab ──────────────────────────────────────────────────
    def _build_practice_tab(self) -> ttk.Frame:
        f = ttk.Frame(self.notebook, padding=10)
        self.practice_session: PracticeSession | None = None
        self.practice_current: dict | None = None

        self.p_lemma_var = tk.StringVar(value="—")
        ttk.Label(f, textvariable=self.p_lemma_var,
                  font=("Helvetica", 22, "bold")).pack(pady=8)
        self.p_meta_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.p_meta_var, foreground="#64748b").pack()
        self.p_details = tk.Text(f, height=14, wrap="word", state="disabled")
        self.p_details.pack(fill="both", expand=True, pady=10)

        btn = ttk.Frame(f); btn.pack(fill="x")
        ttk.Button(btn, text="Start session", command=self._practice_start).pack(side="left")
        ttk.Button(btn, text="Reveal", command=self._practice_reveal).pack(side="left", padx=4)
        ttk.Button(btn, text="I knew it", command=lambda: self._practice_answer(True)).pack(side="left")
        ttk.Button(btn, text="I didn't", command=lambda: self._practice_answer(False)).pack(side="left", padx=4)
        ttk.Button(btn, text="Skip", command=self._practice_skip).pack(side="left")
        ttk.Button(btn, text="Enrich", command=self._practice_enrich).pack(side="left", padx=4)
        return f

    # ── Reader tab ────────────────────────────────────────────────────
    def _build_reader_tab(self) -> ttk.Frame:
        f = ttk.Frame(self.notebook, padding=10)
        self.reader_session: ReaderSession | None = None
        self.reader_text = tk.Text(f, wrap="word")
        self.reader_text.pack(fill="both", expand=True)
        ttk.Button(f, text="Open chapter", command=self._reader_open).pack(pady=6)
        return f

    # ── Trace panel (bottom) ─────────────────────────────────────────
    def _build_trace_panel(self) -> None:
        box = ttk.LabelFrame(self, text="Under-the-Hood Pipeline Trace", padding=6)
        box.pack(fill="x", pady=(8, 0))
        header = ttk.Frame(box); header.pack(fill="x")
        ttk.Label(
            header,
            text="Live events (sorting, enrichment, LLM calls, JSON writes):",
            foreground="#475569",
        ).pack(side="left")
        ttk.Button(header, text="Clear", command=self.clear_trace).pack(side="right")
        self.trace_text = tk.Text(box, height=6, wrap="word",
                                  font=("Consolas", 9), state="disabled")
        self.trace_text.pack(fill="x", pady=(4, 0))

    # ── Book / chapter selection ──────────────────────────────────────
    def _refresh_book_list(self, select_latest: bool = False) -> None:
        self.saved_books = list_saved_books()
        titles = [b["title"] for b in self.saved_books]
        self.book_combo["values"] = titles
        if titles and select_latest:
            self.book_var.set(titles[-1])
            self._on_book_selected()

    def _on_book_selected(self, _evt=None) -> None:
        title = self.book_var.get()
        for meta in self.saved_books:
            if meta["title"] == title:
                try:
                    self.book = load_book_index(meta["json_path"])
                    self.trace(f"Loaded book '{title}' ({meta['book_id']})")
                    self._refresh_chapters()
                except Exception as exc:
                    self.trace(f"Failed to load book: {exc}")
                break

    def _refresh_chapters(self) -> None:
        if not self.book:
            self.chapter_combo["values"] = []
            return
        numbers = [str(c.number) for c in self.book.chapters]
        self.chapter_combo["values"] = numbers
        if numbers:
            self.chapter_var.set(numbers[0])
            self._on_chapter_selected()

    def _on_chapter_selected(self, _evt=None) -> None:
        if not self.book:
            return
        try:
            n = int(self.chapter_var.get())
        except ValueError:
            return
        self.chapter = self.book.chapter_by_number(n)
        if self.chapter:
            r = chapter_readiness(self.chapter, self.book, self.progress)
            self.readiness_var.set(
                f"Readiness: {r['readiness_ratio']:.0%} "
                f"({r['ready_words']}/{r['total_words']})"
            )
            self.trace(f"Chapter {n}: readiness {r['readiness_ratio']:.0%}")

    # ── Ingest actions ────────────────────────────────────────────────
    def _refresh_pdf_list(self) -> None:
        for row in self.ingest_list.get_children():
            self.ingest_list.delete(row)
        for pdf in discover_pdfs():
            from ingestion.loader import is_ingested, infer_language
            self.ingest_list.insert(
                "", "end",
                values=(
                    pdf.name,
                    infer_language(pdf),
                    "yes" if is_ingested(pdf) else "no",
                    round(pdf.stat().st_size / 1024, 1),
                ),
            )

    def _ingest_one(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        dest = PDFS_DIR / Path(path).name
        PDFS_DIR.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != dest.resolve():
            dest.write_bytes(Path(path).read_bytes())
        try:
            self._progress_begin(f"Ingesting {dest.name}", 5)
            book = ingest_pdf(
                dest, force=True,
                on_progress=self._progress_update,
            )
            self._progress_end(f"Ingested {book.book_id}")
            self._refresh_pdf_list()
            self._refresh_book_list(select_latest=True)
        except Exception as exc:
            self.trace(f"Ingest failed: {exc}")
            messagebox.showerror("Ingest failed", str(exc))

    def _ingest_all(self) -> None:
        pdfs = discover_pdfs()
        self._progress_begin(f"Scanning {len(pdfs)} PDF(s)", max(1, len(pdfs)))
        try:
            for idx, pdf in enumerate(pdfs, start=1):
                self._progress_update(f"ingest {pdf.name}", idx - 1, len(pdfs))
                try:
                    ingest_pdf(pdf, force=True)
                except Exception as exc:
                    self.trace(f"{pdf.name}: {exc}")
                self._progress_update(f"done {pdf.name}", idx, len(pdfs))
            self._progress_end("Scan complete")
            self._refresh_pdf_list()
            self._refresh_book_list(select_latest=True)
        except Exception as exc:
            self.trace(f"Scan failed: {exc}")

    # ── Practice actions ─────────────────────────────────────────────
    def _practice_start(self) -> None:
        if not self.book:
            messagebox.showinfo("No book", "Load a book first.")
            return
        self.practice_session = PracticeSession(self.book, self.progress, self.ollama)
        try:
            n = int(self.chapter_var.get())
        except ValueError:
            n = 1
        queue = self.practice_session.build_queue(chapter_number=n, limit=20)
        self.trace(f"Practice queue: {len(queue)} words")
        self._practice_next()

    def _practice_next(self) -> None:
        if not self.practice_session:
            return
        item = self.practice_session.next_question()
        if not item:
            self._write_details("Session complete.")
            return
        self.practice_current = item
        self.p_lemma_var.set(item["lemma"])
        wk = item["knowledge"]
        self.p_meta_var.set(
            f"{wk['known_status']} · {wk['attempts']} attempts · "
            f"{int(wk['accuracy'] * 100)}%"
        )
        self._write_details("")

    def _practice_reveal(self) -> None:
        if not self.practice_current:
            return
        entry = self.practice_current.get("entry") or {}
        lines: list[str] = []
        for e in entry.get("entries", []):
            lines.append(f"• {e['form']} ({e['relation']}): {', '.join(e['translations'])}")
        ling = entry.get("linguistics") or {}
        if ling.get("explanation"):
            lines.append("")
            lines.append(ling["explanation"])
        for ec in ling.get("edge_cases", []):
            lines.append(f"  – {ec}")
        self._write_details("\n".join(lines) or "(no enriched data — click Enrich)")

    def _write_details(self, text: str) -> None:
        self.p_details.configure(state="normal")
        self.p_details.delete("1.0", "end")
        self.p_details.insert("1.0", text)
        self.p_details.configure(state="disabled")

    def _practice_answer(self, correct: bool) -> None:
        if not self.practice_session:
            return
        self.practice_session.submit_answer(correct)
        self._practice_next()

    def _practice_skip(self) -> None:
        if not (self.practice_session and self.practice_current):
            return
        from learning import mastery
        mastery.mark_word_skipped(self.progress, self.practice_current["lemma"])
        save_user_progress(self.progress)
        self._practice_next()

    def _practice_enrich(self) -> None:
        if not self.practice_session:
            return
        self._progress_begin("Enriching…", 1)
        ok = self.practice_session.enrich_current()
        self._progress_end("Enriched" if ok else "Enrich returned nothing")
        self._practice_reveal()

    # ── Reader actions ───────────────────────────────────────────────
    def _reader_open(self) -> None:
        if not self.book:
            return
        try:
            n = int(self.chapter_var.get())
        except ValueError:
            n = 1
        self.reader_session = ReaderSession(self.book, self.progress, self.ollama)
        ch = self.reader_session.open_chapter(n)
        if ch is None:
            messagebox.showerror("Reader", f"Chapter {n} not found.")
            return
        self.reader_text.delete("1.0", "end")
        self.reader_text.insert("1.0", self.reader_session.chapter_text())
        self.trace(f"Reader opened chapter {n}")
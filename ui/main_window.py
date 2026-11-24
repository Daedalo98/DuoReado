"""
ui/main_window.py
Top-level Tkinter application window coordinating book selection, chapter readiness,
Ollama model choice, and the three main views.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ai.ollama_client import OllamaClient
from config import DEFAULT_MODEL
from ingestion.json_builder import list_saved_books, load_book_index
from learning.mastery import chapter_readiness
from models.schemas import BookIndex, ChapterInfo
from ui.ingest_view import IngestView
from ui.practice_view import PracticeView
from ui.reader_view import ReaderView


class MainWindow(ttk.Frame):
    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root, padding=10)
        self.root = root
        self.pack(fill="both", expand=True)

        self.ollama = OllamaClient(model=DEFAULT_MODEL)
        self.book: BookIndex | None = None
        self.chapter: ChapterInfo | None = None
        self.saved_books: list[dict[str, str]] = []

        self.book_var = tk.StringVar()
        self.chapter_var = tk.StringVar()
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.readiness_var = tk.StringVar(value="Readiness: —")

        self._build_top_bar()
        self._build_notebook()
        self._build_trace_panel()
        self._refresh_book_list(select_latest=True)


    def _build_trace_panel(self) -> None:
        """Create a transparent log pane showing every internal step & model call."""
        trace_box = ttk.LabelFrame(self, text="Under-the-Hood Pipeline Trace", padding=6)
        trace_box.pack(fill="x", pady=(8, 0))

        header = ttk.Frame(trace_box)
        header.pack(fill="x")
        ttk.Label(header, text="Live events (sorting, enrichment, verification, LLM calls, JSON writes):", foreground="#475569").pack(
            side="left"
        )
        ttk.Button(header, text="Clear Trace", command=self.clear_trace).pack(side="right")

        self.trace_text = tk.Text(trace_box, height=5, wrap="word", font=("Consolas", 9), state="disabled")
        self.trace_text.pack(fill="x", pady=(4, 0))


    def clear_trace(self) -> None:
        self.trace_text.configure(state="normal")
        self.trace_text.delete("1.0", "end")
        self.trace_text.configure(state="disabled")


    def trace(self, message: str) -> None:
        """Append a timestamped log line safely from the main thread."""
        from datetime import datetime

        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"

        def _append() -> None:
            self.trace_text.configure(state="normal")
            self.trace_text.insert("end", line)
            self.trace_text.see("end")
            self.trace_text.configure(state="disabled")

        self.after(0, _append)

    def _open_new_pdf_tab(self) -> None:
        self.notebook.select(self.ingest_tab)
        self.ingest_tab.prepare_new_import()
        
    def _build_top_bar(self) -> None:
        bar = ttk.LabelFrame(self, text="Active Book, Chapter & Ollama Model", padding=10)
        bar.pack(fill="x", pady=(0, 8))

        ttk.Label(bar, text="Book:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.book_combo = ttk.Combobox(bar, textvariable=self.book_var, state="readonly", width=28)
        self.book_combo.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.book_combo.bind("<<ComboboxSelected>>", self._on_book_selected)
        ttk.Button(bar, text="+ New PDF", command=self._open_new_pdf_tab).grid(row=0, column=1, sticky="e", padx=(0, 10))

        ttk.Label(bar, text="Chapter:").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.chapter_combo = ttk.Combobox(bar, textvariable=self.chapter_var, state="readonly", width=28)
        self.chapter_combo.grid(row=0, column=3, sticky="ew", padx=(0, 10))
        self.chapter_combo.bind("<<ComboboxSelected>>", self._on_chapter_selected)

        ttk.Label(bar, text="Ollama Model:").grid(row=0, column=4, sticky="w", padx=(0, 4))
        models = self.ollama.list_models()
        if self.model_var.get() not in models:
            models.insert(0, self.model_var.get())
        self.model_combo = ttk.Combobox(bar, textvariable=self.model_var, values=models, width=16)
        self.model_combo.grid(row=0, column=5, sticky="w")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)
        self.model_combo.bind("<Return>", self._on_model_changed)

        # Readiness row
        ttk.Label(bar, textvariable=self.readiness_var, foreground="#0f172a").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )
        self.readiness_bar = ttk.Progressbar(bar, maximum=100, length=240)
        self.readiness_bar.grid(row=1, column=4, columnspan=2, sticky="e", pady=(8, 0))

        bar.columnconfigure(1, weight=1)
        bar.columnconfigure(3, weight=1)

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.ingest_tab = IngestView(self.notebook, on_book_ready=self._on_book_built)
        self.practice_tab = PracticeView(
            self.notebook,
            ollama=self.ollama,
            on_stats_updated=self.refresh_readiness,
            trace=self.trace,
        )
        self.reader_tab = ReaderView(
            self.notebook,
            ollama=self.ollama,
            on_stats_updated=self.refresh_readiness,
            trace=self.trace,
        )
        self.notebook.add(self.ingest_tab, text="1. Import PDF & Index")
        self.notebook.add(self.practice_tab, text="2. Practice Words & Sentences")
        self.notebook.add(self.reader_tab, text="3. Read Chapter")

    def _refresh_book_list(self, select_latest: bool = False, target_book_id: str | None = None) -> None:
        self.saved_books = list_saved_books()
        labels = [f"{b['title']} ({b['language']})" for b in self.saved_books]
        self.book_combo["values"] = labels

        if not self.saved_books:
            self.book = None
            self.chapter = None
            self.book_var.set("")
            self.chapter_var.set("")
            self.refresh_readiness()
            return

        chosen_idx = 0
        if target_book_id:
            for i, b in enumerate(self.saved_books):
                if b["book_id"] == target_book_id:
                    chosen_idx = i
                    break
        elif select_latest:
            chosen_idx = len(self.saved_books) - 1

        self.book_combo.current(chosen_idx)
        self._load_book_at_index(chosen_idx)

    def _on_book_selected(self, _event=None) -> None:
        idx = self.book_combo.current()
        if idx >= 0:
            self._load_book_at_index(idx)

    def _load_book_at_index(self, idx: int) -> None:
        info = self.saved_books[idx]
        self.book = load_book_index(info["json_path"])

        ch_labels = [f"{i+1}. {c.title} (p.{c.start_page}–{c.end_page})" for i, c in enumerate(self.book.chapters)]
        self.chapter_combo["values"] = ch_labels

        if self.book.chapters:
            self.chapter_combo.current(0)
            self.chapter = self.book.chapters[0]
        else:
            self.chapter = None
            self.chapter_var.set("")

        self._propagate_context()

    def _on_chapter_selected(self, _event=None) -> None:
        if not self.book:
            return
        idx = self.chapter_combo.current()
        if 0 <= idx < len(self.book.chapters):
            self.chapter = self.book.chapters[idx]
            self._propagate_context()

    def _on_model_changed(self, _event=None) -> None:
        model_name = self.model_var.get().strip()
        if model_name:
            self.ollama.model = model_name

    def _on_book_built(self, book: BookIndex) -> None:
        self._refresh_book_list(target_book_id=book.book_id)
        self.notebook.select(self.practice_tab)

    def _propagate_context(self) -> None:
        self.refresh_readiness()
        self.practice_tab.set_context(self.book, self.chapter)
        self.reader_tab.set_chapter(self.book, self.chapter)

    def refresh_readiness(self) -> None:
        if not self.book or not self.chapter:
            self.readiness_var.set("Readiness: —")
            self.readiness_bar["value"] = 0
            return

        stats = chapter_readiness(self.book, self.chapter)
        pct = int(round(stats["readiness_ratio"] * 100))
        target_pct = int(round(stats["target_ratio"] * 100))
        needed = stats["words_needed_for_target"]

        self.readiness_var.set(
            f"Chapter Readiness: {pct}% ({stats['ready_words']}/{stats['total_words']} ready)  |  "
            f"Target: {target_pct}% ({needed} more word(s) to goal)"
        )
        self.readiness_bar["value"] = pct
        if self.reader_tab:
            self.reader_tab._apply_word_highlights()
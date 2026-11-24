"""ui/ingest_view.py
Tab 1: Import a PDF book, adjust chapter start pages, and build the TXT + JSON index.
Includes a live progress bar with estimated remaining time and a trace log.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import ProgressCallback, log
from ingestion.json_builder import build_book_index
from ingestion.pdf_extractor import PdfExtractionResult, extract_pdf_pages
from models.schemas import BookIndex


class IngestView(ttk.Frame):
    def __init__(self, parent: tk.Misc, on_book_ready: callable) -> None:
        super().__init__(parent, padding=16)
        self.on_book_ready = on_book_ready
        self.extraction: PdfExtractionResult | None = None
        self.pdf_path_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self.lang_var = tk.StringVar(value="en")
        self.status_var = tk.StringVar(value="Select a PDF book to begin.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.eta_var = tk.StringVar(value="")
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Step 1: File Picker
        file_box = ttk.LabelFrame(self, text="1. Select Book PDF", padding=12)
        file_box.pack(fill="x", pady=(0, 10))
        ttk.Entry(file_box, textvariable=self.pdf_path_var).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ttk.Button(file_box, text="Browse PDF…", command=self._choose_pdf).pack(
            side="left"
        )

        # Step 2: Book Metadata
        meta_box = ttk.LabelFrame(self, text="2. Book Metadata", padding=12)
        meta_box.pack(fill="x", pady=(0, 10))

        ttk.Label(meta_box, text="Title:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(meta_box, textvariable=self.title_var, width=36).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(meta_box, text="Author:").grid(row=0, column=2, sticky="w", padx=(14, 6), pady=4)
        ttk.Entry(meta_box, textvariable=self.author_var, width=28).grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(meta_box, text="Language code (e.g. en, it, es, fr, de):").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=4
        )
        ttk.Entry(meta_box, textvariable=self.lang_var, width=10).grid(row=1, column=2, sticky="w", pady=4)

        meta_box.columnconfigure(1, weight=1)
        meta_box.columnconfigure(3, weight=1)

        # Step 3: Chapters Editor
        ch_box = ttk.LabelFrame(
            self, text="3. Chapters & Start Pages (Editable)", padding=12
        )
        ch_box.pack(fill="both", expand=True, pady=(0, 10))
        columns = ("chapter_title", "start_page")
        self.ch_tree = ttk.Treeview(
            ch_box, columns=columns, show="headings", height=7, selectmode="extended"
        )
        self.ch_tree.bind("<<TreeviewSelect>>", lambda _e: self._remove_chapter())
        self.ch_tree.heading("chapter_title", text="Chapter Title")
        self.ch_tree.heading("start_page", text="Start Page (1‑based)")
        self.ch_tree.pack(fill="both", expand=True)

        # ── Progress bar + ETA ──────────────────────────────────────
        progress_frame = ttk.LabelFrame(
            self, text="Ingestion Progress", padding=10
        )
        progress_frame.pack(fill="x", pady=(6, 0))
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(progress_frame, textvariable=self.eta_var).pack(anchor="w")

    # ── PDF selection ───────────────────────────────────────────────
    def _choose_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a PDF book", filetypes=[("PDF files", "*.pdf")]
        )
        if not path:
            return
        self.pdf_path_var.set(path)
        self.status_var.set("Extracting PDF metadata…")
        log.info("Selected PDF: %s", path)

        # Run extraction in a background thread to keep the UI responsive
        threading.Thread(
            target=self._run_extraction, args=(path,), daemon=True
        ).start()

    def _run_extraction(self, path: str) -> None:
        """Background PDF extraction with progress updates."""
        try:
            def on_progress(msg: str, cur: int, total: int) -> None:
                self._update_progress(msg, cur, total)

            extraction = extract_pdf_pages(path, on_progress=on_progress)
            self.extraction = extraction
            self.title_var.set(extraction.title)
            self.author_var.set(extraction.author)
            self._populate_chapter_tree(extraction.suggested_chapters)
            self.status_var.set(
                f"Detected {len(extraction.suggested_chapters)} chapter(s). "
                "Review and click Build."
            )
            log.info("Extraction complete: %d pages", extraction.total_pages)
        except Exception as exc:
            log.exception("PDF extraction failed")
            messagebox.showerror("Extraction Error", str(exc))

    # ── Progress helper ─────────────────────────────────────────────
    def _update_progress(self, message: str, current: int, total: int) -> None:
        """Update progress bar and ETA label from a worker thread."""
        def _apply() -> None:
            pct = (current / total * 100) if total else 0
            self.progress_var.set(pct)
            self.eta_var.set(message)
            self.status_var.set(message)
            self.update_idletasks()

        self.after(0, _apply)

    # ── Build index ─────────────────────────────────────────────────
    def _build_index(self) -> None:
        if self.extraction is None:
            messagebox.showwarning("No PDF", "Please select a PDF first.")
            return

        specs = self._collect_chapter_specs()
        self.status_var.set("Building index…")
        log.info("Starting index build")

        def worker() -> None:
            try:
                def on_progress(msg: str, cur: int, total: int) -> None:
                    self._update_progress(msg, cur, total)

                book = build_book_index(
                    pdf_path=self.pdf_path_var.get(),
                    title=self.title_var.get().strip() or "Untitled",
                    author=self.author_var.get().strip() or "Unknown",
                    language=self.lang_var.get().strip() or "en",
                    chapter_specs=specs,
                    on_progress=on_progress,
                    extraction=self.extraction,
                )
                self.on_book_ready(book)
                self.status_var.set("Index built successfully.")
                log.info("Index build finished")
            except Exception as exc:
                log.exception("Index build failed")
                messagebox.showerror("Build Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()
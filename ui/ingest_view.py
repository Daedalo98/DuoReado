"""
ui/ingest_view.py
Tab 1: Import a PDF book, adjust chapter start pages, and build the TXT + JSON index.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ingestion.json_builder import build_book_index
from ingestion.pdf_extractor import PdfExtractionResult, extract_pdf_pages
from models.schemas import BookIndex

import time


class IngestView(ttk.Frame):
    def __init__(self, parent: tk.Misc, on_book_ready) -> None:
        super().__init__(parent, padding=16)
        self.on_book_ready = on_book_ready
        self.extraction: PdfExtractionResult | None = None
        self.pdf_path_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self.lang_var = tk.StringVar(value="en")
        self.status_var = tk.StringVar(value="Select a PDF book to begin.")

        self._build_ui()

    def prepare_new_import(self) -> None:
        self.extraction = None
        self.pdf_path_var.set("")
        self.title_var.set("")
        self.author_var.set("")
        for row in self.ch_tree.get_children():
            self.ch_tree.delete(row)
        self.status_var.set("Ready for a new PDF — click 'Browse PDF...' to select a file.")

    def _build_ui(self) -> None:
        # Step 1: File Picker
        file_box = ttk.LabelFrame(self, text="1. Select Book PDF", padding=12)
        file_box.pack(fill="x", pady=(0, 10))

        ttk.Entry(file_box, textvariable=self.pdf_path_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(file_box, text="Browse PDF...", command=self._choose_pdf).pack(side="left")

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
        ch_box = ttk.LabelFrame(self, text="3. Chapters & Start Pages (Editable)", padding=12)
        ch_box.pack(fill="both", expand=True, pady=(0, 10))

        columns = ("chapter_title", "start_page")
        self.ch_tree = ttk.Treeview(ch_box, columns=columns, show="headings", height=7, selectmode="extended")
        self.ch_tree.bind("<Delete>", lambda _e: self._remove_chapter())
        self.ch_tree.heading("chapter_title", text="Chapter Title")
        self.ch_tree.heading("start_page", text="Start Page (1-based)")
        self.ch_tree.column("chapter_title", width=360)
        self.ch_tree.column("start_page", width=130, anchor="center")
        self.ch_tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(ch_box, orient="vertical", command=self.ch_tree.yview)
        self.ch_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="left", fill="y", padx=(0, 8))

        controls = ttk.Frame(ch_box)
        controls.pack(side="left", fill="y")

        ttk.Label(controls, text="Edit Selected:").pack(anchor="w")
        ttk.Label(controls, text="Title:").pack(anchor="w", pady=(6, 0))
        self.edit_ch_title = tk.StringVar()
        ttk.Entry(controls, textvariable=self.edit_ch_title, width=22).pack(fill="x")

        ttk.Label(controls, text="Start Page:").pack(anchor="w", pady=(6, 0))
        self.edit_ch_page = tk.StringVar()
        ttk.Entry(controls, textvariable=self.edit_ch_page, width=10).pack(anchor="w")

        ttk.Button(controls, text="Update Row", command=self._update_selected_chapter).pack(fill="x", pady=(8, 4))
        ttk.Button(controls, text="Add Chapter", command=self._add_chapter).pack(fill="x", pady=2)
        ttk.Button(controls, text="Remove Row", command=self._remove_chapter).pack(fill="x", pady=2)

        self.ch_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Step 4: Build Action
        action_bar = ttk.Frame(self)
        action_bar.pack(fill="x")

        self.progress = ttk.Progressbar(action_bar, mode="determinate", maximum=4)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.build_btn = ttk.Button(action_bar, text="Start Loading Process", command=self._start_build)
        self.build_btn.pack(side="right")

        ttk.Label(self, textvariable=self.status_var, foreground="#334155").pack(anchor="w", pady=(8, 0))
        self.detail_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.detail_var, foreground="#64748b", font=("TkDefaultFont", 9)).pack(
            anchor="w", pady=(2, 0)
        )

    def _choose_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a PDF book",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        self.pdf_path_var.set(path)
        self.status_var.set("Scanning PDF metadata and suggested chapters...")

        def worker():
            try:
                res = extract_pdf_pages(path)
                self.after(0, lambda: self._populate_from_extraction(res))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("PDF Error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _populate_from_extraction(self, res: PdfExtractionResult) -> None:
        self.extraction = res
        self.title_var.set(res.title)
        self.author_var.set(res.author)

        for row in self.ch_tree.get_children():
            self.ch_tree.delete(row)

        for ch_title, start_p in res.suggested_chapters:
            self.ch_tree.insert("", "end", values=(ch_title, start_p))

        self.status_var.set(
            f"Loaded {res.total_pages} pages. Review chapters and click 'Build TXT & JSON Index'."
        )

    def _on_tree_select(self, _event=None) -> None:
        sel = self.ch_tree.selection()
        if not sel:
            return
        vals = self.ch_tree.item(sel[0], "values")
        if len(vals) >= 2:
            self.edit_ch_title.set(vals[0])
            self.edit_ch_page.set(str(vals[1]))

    def _update_selected_chapter(self) -> None:
        sel = self.ch_tree.selection()
        if not sel:
            return
        title = self.edit_ch_title.get().strip() or "Chapter"
        try:
            page = int(self.edit_ch_page.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid Page", "Start page must be a whole number.")
            return
        self.ch_tree.item(sel[0], values=(title, max(1, page)))

    def _add_chapter(self) -> None:
        count = len(self.ch_tree.get_children()) + 1
        self.ch_tree.insert("", "end", values=(f"Chapter {count}", 1))

    def _remove_chapter(self) -> None:
        selected = self.ch_tree.selection()
        if not selected:
            self.status_var.set("Select a chapter row first to delete it.")
            return
        for item in selected:
            next_item = self.ch_tree.next(item) or self.ch_tree.prev(item)
            self.ch_tree.delete(item)
            if next_item and self.ch_tree.exists(next_item):
                self.ch_tree.selection_set(next_item)
        self.status_var.set(f"Removed {len(selected)} chapter row(s). Remaining: {len(self.ch_tree.get_children())}.")

    def _collect_chapters(self) -> list[tuple[str, int]]:
        specs: list[tuple[str, int]] = []
        for row_id in self.ch_tree.get_children():
            vals = self.ch_tree.item(row_id, "values")
            if len(vals) >= 2:
                try:
                    specs.append((str(vals[0]), int(vals[1])))
                except ValueError:
                    continue
        return specs

    def _start_build(self) -> None:
        pdf_path = self.pdf_path_var.get().strip()
        if not pdf_path:
            messagebox.showwarning("Missing PDF", "Please choose a PDF file first.")
            return

        title = self.title_var.get().strip() or "Untitled Book"
        author = self.author_var.get().strip() or "Unknown Author"
        lang = self.lang_var.get().strip().lower() or "en"
        chapter_specs = self._collect_chapters()
        extraction = self.extraction

        self.build_start_time = time.monotonic()
        self.detail_var.set("0%  |  Elapsed: 0.0s  |  Remaining: estimating...")

        self.build_btn.state(["disabled"])
        self.progress["value"] = 0

        def progress_cb(msg: str, step: int, total: int) -> None:
            self.after(0, lambda: self._on_progress(msg, step, total))

        def worker():
            try:
                book: BookIndex = build_book_index(
                    pdf_path=pdf_path,
                    title=title,
                    author=author,
                    language=lang,
                    chapter_specs=chapter_specs,
                    on_progress=progress_cb,
                    extraction=extraction,
                )
                self.after(0, lambda: self._on_build_done(book))
            except Exception as exc:
                self.after(0, lambda: self._on_build_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, msg: str, step: int, total: int) -> None:
        self.status_var.set(msg)
        self.progress["maximum"] = total
        self.progress["value"] = step

        pct = int((step / max(1, total)) * 100)
        elapsed = max(0.0, time.monotonic() - self.build_start_time)

        if step > 0 and step < total and elapsed > 0.2:
            rate = step / elapsed
            remaining_steps = total - step
            eta_sec = remaining_steps / rate
            eta_str = f"{eta_sec:.1f}s" if eta_sec < 60 else f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
        elif step >= total:
            eta_str = "0.0s"
        else:
            eta_str = "estimating..."

        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        self.detail_var.set(f"{pct}%  |  Elapsed: {elapsed_str}  |  Remaining: ~{eta_str}")

    def _on_build_done(self, book: BookIndex) -> None:
        self.build_btn.state(["!disabled"])
        self.status_var.set(
            f"Built '{book.title}': {len(book.chapters)} chapters, {len(book.words)} unique words. "
            f"Saved JSON & TXT."
        )
        messagebox.showinfo(
            "Book Ready",
            f"Indexed '{book.title}'!\n"
            f"Chapters: {len(book.chapters)}\n"
            f"Unique words: {len(book.words)}\n"
            f"Clean TXT: {book.clean_txt_path}",
        )
        self.on_book_ready(book)

    def _on_build_error(self, err: str) -> None:
        self.build_btn.state(["!disabled"])
        self.status_var.set("Build failed.")
        messagebox.showerror("Build Error", err)
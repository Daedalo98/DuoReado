"""
ui/reader_view.py
Tab 3: Chapter Reader view with word lookup, accuracy stats, and Ollama sentence help.
"""

from __future__ import annotations

import re
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from ai.ollama_client import OllamaClient
from ingestion.json_builder import save_book_index
from ingestion.text_normalizer import extract_tokens
from learning.mastery import mark_word_known
from models.schemas import BookIndex, ChapterInfo, WordEntry
from ui.red_flag_dialog import RedFlagWordDialog

from typing import Callable

class ReaderView(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        ollama: OllamaClient,
        on_stats_updated: Callable[[], None],
        trace: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent, padding=16)
        self.ollama = ollama
        self.on_stats_updated = on_stats_updated
        self.trace = trace or (lambda _msg: None)

        self.book: BookIndex | None = None
        self.chapter: ChapterInfo | None = None
        self.selected_lemma: str | None = None

        self.word_info_var = tk.StringVar(value="Click any word in the chapter text to inspect it.")

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, textvariable=self.word_info_var, font=("TkDefaultFont", 10, "bold")).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(top, text="Translate Word", command=self._translate_selected_word).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Mark Known", command=self._mark_selected_known).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Explain Selected Sentence", command=self._explain_selected_sentence).pack(side="left")
        ttk.Button(top, text="🚩 Red Flag (JSON)", command=self._open_red_flag_selected_word).pack(side="left", padx=(6, 0))

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)

        reader_frame = ttk.LabelFrame(paned, text="Chapter Text", padding=10)
        self.reader_text = tk.Text(reader_frame, wrap="word", font=("Georgia", 12), padx=8, pady=8)
        self.reader_text.pack(fill="both", expand=True)
        self.reader_text.bind("<ButtonRelease-1>", self._on_text_click)

        # Visual styling for words based on mastery status
        self.reader_text.tag_configure("mastered", foreground="#15803d")
        self.reader_text.tag_configure("skipped", foreground="#15803d")
        self.reader_text.tag_configure("learning", foreground="#b45309")
        self.reader_text.tag_configure("new", foreground="#1e293b")
        self.reader_text.tag_configure("highlighted", background="#fef08a")

        notes_frame = ttk.LabelFrame(paned, text="Ollama Translation & Grammar Notes", padding=10)
        self.notes_text = tk.Text(notes_frame, height=8, wrap="word", font=("TkDefaultFont", 10))
        self.notes_text.pack(fill="both", expand=True)

        paned.add(reader_frame, weight=3)
        paned.add(notes_frame, weight=1)

    def set_chapter(self, book: BookIndex | None, chapter: ChapterInfo | None) -> None:
        self.book = book
        self.chapter = chapter
        self.selected_lemma = None
        self.reader_text.delete("1.0", "end")
        self.notes_text.delete("1.0", "end")

        if not book or not chapter:
            self.word_info_var.set("No chapter loaded.")
            return

        self.word_info_var.set(
            f"Reading: {chapter.title} (Pages {chapter.start_page}–{chapter.end_page}) — Click any word."
        )
        self.reader_text.insert("1.0", chapter.text or "(No text extracted for this chapter.)")
        self._apply_word_highlights()

    def _apply_word_highlights(self) -> None:
        """Color-code words in the reader according to user mastery status."""
        if not self.book or not self.chapter:
            return

        for tag in ("mastered", "skipped", "learning", "new"):
            self.reader_text.tag_remove(tag, "1.0", "end")

        content = self.reader_text.get("1.0", "end-1c")
        for match in re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ]+)*", content):
            surface = match.group(0)
            toks = extract_tokens(surface, language=self.book.language)
            lemma = toks[0].lemma if toks else surface.lower()
            entry = self.book.words.get(lemma)
            status = entry.known_status if entry else "new"

            start_idx = f"1.0 + {match.start()} chars"
            end_idx = f"1.0 + {match.end()} chars"
            self.reader_text.tag_add(status, start_idx, end_idx)

    def _on_text_click(self, _event=None) -> None:
        if not self.book:
            return
        try:
            idx = self.reader_text.index("@%d,%d" % (self.reader_text.winfo_pointerx() - self.reader_text.winfo_rootx(),
                                                    self.reader_text.winfo_pointery() - self.reader_text.winfo_rooty()))
        except Exception:
            idx = self.reader_text.index("insert")

        # Grab word around cursor
        word_start = self.reader_text.index(f"{idx} wordstart")
        word_end = self.reader_text.index(f"{idx} wordend")
        surface = self.reader_text.get(word_start, word_end).strip()
        if not surface:
            return

        toks = extract_tokens(surface, language=self.book.language)
        lemma = toks[0].lemma if toks else surface.lower()
        self.selected_lemma = lemma

        self.reader_text.tag_remove("highlighted", "1.0", "end")
        self.reader_text.tag_add("highlighted", word_start, word_end)

        entry = self.book.words.get(lemma)
        if entry:
            self.word_info_var.set(
                f"Surface: '{surface}'  |  Lemma: '{entry.lemma}'  |  "
                f"Book uses: {entry.total_count}  |  Pages: {entry.pages[:6]}  |  "
                f"Accuracy: {int(entry.accuracy * 100)}% ({entry.known_status})"
            )
        else:
            self.word_info_var.set(f"Word: '{surface}' (lemma: '{lemma}')")

    def _get_selected_sentence(self) -> str:
        # If user highlighted text, use selection; otherwise grab ~160 chars around cursor
        try:
            sel = self.reader_text.get("sel.first", "sel.last").strip()
            if sel:
                return sel
        except tk.TclError:
            pass

        cursor = self.reader_text.index("insert")
        start = self.reader_text.index(f"{cursor} - 120 chars")
        end = self.reader_text.index(f"{cursor} + 120 chars")
        return re.sub(r"\s+", " ", self.reader_text.get(start, end).strip())

    def _translate_selected_word(self) -> None:
        if not self.book or not self.selected_lemma:
            messagebox.showinfo("Select a Word", "Click a word in the chapter first.")
            return

        lemma = self.selected_lemma
        snippet = self._get_selected_sentence()
        lang = self.book.language
        self._set_notes(f"Translating '{lemma}' via Ollama...")

        def task():
            res = self.ollama.translate_word(lemma, lang, snippet)
            if lemma in self.book.words:
                self.book.words[lemma].cached_translation = res.splitlines()[0] if res else ""
                save_book_index(self.book)
            return res

        self._run_bg(task, on_done=self._set_notes)

    def _mark_selected_known(self) -> None:
        if not self.book or not self.selected_lemma:
            return
        entry = self.book.words.get(self.selected_lemma)
        if entry:
            mark_word_known(entry)
            save_book_index(self.book)
            self._apply_word_highlights()
            self.on_stats_updated()
            self._set_notes(f"Marked '{entry.lemma}' as known.")

    def _explain_selected_sentence(self) -> None:
        if not self.book:
            return
        snippet = self._get_selected_sentence()
        if not snippet:
            return
        lang = self.book.language
        self._set_notes("Asking Ollama to translate and explain this passage...")
        self._run_bg(lambda: self.ollama.explain_sentence(snippet, lang), on_done=self._set_notes)

    def _set_notes(self, text: str) -> None:
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("1.0", text)

    def _run_bg(self, fn, on_done) -> None:
        def worker():
            try:
                res = fn()
                self.after(0, lambda: on_done(res))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Ollama Error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _open_red_flag_selected_word(self) -> None:
        if not self.book or not self.selected_lemma:
            messagebox.showinfo("Select a Word", "Click a word in the chapter text first.")
            return
        entry = self.book.words.get(self.selected_lemma)
        if not entry:
            return
        if self.trace:
            self.trace(f"🚩 Red Flag opened from Reader for '{entry.lemma}'.")

        def on_saved(updated: WordEntry) -> None:
            assert self.book is not None
            self.book.words[updated.lemma] = updated
            save_book_index(self.book)
            self._apply_word_highlights()
            self.on_stats_updated()
            if self.trace:
                self.trace(f"🚩 Saved manual JSON edits for '{updated.lemma}' from Reader.")

        RedFlagWordDialog(self, entry, on_save=on_saved)
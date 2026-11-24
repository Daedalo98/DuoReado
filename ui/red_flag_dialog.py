"""
ui/red_flag_dialog.py
Modal JSON inspector & editor ("Red Flag") for manually fixing or enriching a WordEntry.
"""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from models.schemas import WordEntry


class RedFlagWordDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        word_entry: WordEntry,
        on_save: Callable[[WordEntry], None],
    ) -> None:
        super().__init__(parent)
        self.title(f"🚩 Red Flag — Inspect / Edit JSON: '{word_entry.lemma}'")
        self.geometry("760x560")
        self.transient(parent)
        self.grab_set()

        self.word_entry = word_entry
        self.on_save = on_save

        self._build_ui()
        self._load_json()

    def _build_ui(self) -> None:
        pad = ttk.Frame(self, padding=12)
        pad.pack(fill="both", expand=True)

        ttk.Label(
            pad,
            text=(
                f"Editing source JSON for lemma '{self.word_entry.lemma}'. "
                "Modify translations, false friends, edge meanings, sentences, or stats, then click Save."
            ),
            wraplength=720,
        ).pack(anchor="w", pady=(0, 8))

        text_frame = ttk.Frame(pad)
        text_frame.pack(fill="both", expand=True)

        self.editor = tk.Text(text_frame, wrap="none", font=("Consolas", 10), undo=True)
        y_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.editor.yview)
        x_scroll = ttk.Scrollbar(text_frame, orient="horizontal", command=self.editor.xview)
        self.editor.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.editor.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        btn_bar = ttk.Frame(pad)
        btn_bar.pack(fill="x", pady=(10, 0))

        ttk.Button(btn_bar, text="Format JSON", command=self._format_json).pack(side="left", padx=(0, 6))
        ttk.Button(btn_bar, text="Reset Original", command=self._load_json).pack(side="left")
        ttk.Button(btn_bar, text="Cancel", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btn_bar, text="Save Word JSON", command=self._save).pack(side="right")

    def _load_json(self) -> None:
        formatted = json.dumps(self.word_entry.to_dict(), indent=2, ensure_ascii=False)
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", formatted)

    def _format_json(self) -> None:
        raw = self.editor.get("1.0", "end-1c")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid JSON", f"Cannot format invalid JSON:\n{exc}", parent=self)
            return
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", json.dumps(parsed, indent=2, ensure_ascii=False))

    def _save(self) -> None:
        raw = self.editor.get("1.0", "end-1c")
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level word entry must be a JSON object.")
            # Preserve lemma key consistency if user edited inside object
            parsed.setdefault("lemma", self.word_entry.lemma)
            updated_entry = WordEntry.from_dict(parsed)
        except Exception as exc:
            messagebox.showerror("Validation Error", f"Could not save changes:\n{exc}", parent=self)
            return

        self.on_save(updated_entry)
        self.destroy()
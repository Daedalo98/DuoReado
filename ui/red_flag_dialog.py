"""ui/red_flag_dialog.py

Modal JSON inspector & editor ("Red Flag") for manually fixing or
enriching a VocabularyEntry.

Two layers:
    RedFlagEditor  — pure logic: load JSON, validate shape, apply, save.
                     Framework-agnostic; unit-testable.
    RedFlagDialog  — Tkinter (customtkinter-compatible) shell around it.

The dialog **does not** touch user progress. It only mutates the shared
book JSON via the caller-supplied `on_save(entry)` callback — so persistence
is the caller's job (usually `save_book_index(book)`).
"""
from __future__ import annotations

import json
import tkinter as tk
from dataclasses import asdict
from tkinter import messagebox
from typing import Any, Callable

from models.schemas import VocabularyEntry


# ---------------------------------------------------------------------------
# 1. Pure logic layer
# ---------------------------------------------------------------------------

class RedFlagEditor:
    """Load, validate and apply JSON edits for a single VocabularyEntry."""

    # Fields the user must never edit through this dialog (they are derived).
    _READ_ONLY_KEYS = {"occurrences", "first_chapter", "sentences"}

    def __init__(self, entry: VocabularyEntry, on_save: Callable[[VocabularyEntry], None]):
        self._original = entry
        self._on_save = on_save

    # -- JSON round-trip ----------------------------------------------------

    def to_json_text(self) -> str:
        """Pretty JSON for the text widget. Read-only fields are removed
        because they belong to the build phase, not to manual enrichment."""
        payload = asdict(self._original)
        for key in self._READ_ONLY_KEYS:
            payload.pop(key, None)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def apply_json_text(self, text: str) -> VocabularyEntry:
        """Parse and validate. Raise ValueError on bad input.

        Only the *enrichable* fields are taken from the text:
            entries, linguistics, semantic_keywords
        The build-time fields (occurrences, first_chapter, sentences)
        are preserved from the original entry.
        """
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Top-level JSON must be an object.")

        # Merge: start from the original (keeps read-only fields) and
        # override only the editable ones present in the payload.
        merged = asdict(self._original)
        for key in ("entries", "linguistics", "semantic_keywords"):
            if key in parsed:
                merged[key] = parsed[key]

        # Validate the shape via the schema's from_dict — bad shapes raise.
        try:
            new_entry = VocabularyEntry.from_dict(merged)
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Schema mismatch: {exc}") from exc

        return new_entry

    def save_from_text(self, text: str) -> VocabularyEntry:
        new_entry = self.apply_json_text(text)
        self._on_save(new_entry)
        return new_entry


# ---------------------------------------------------------------------------
# 2. Tkinter shell
# ---------------------------------------------------------------------------

class RedFlagDialog(tk.Toplevel):
    """Modal window. Opens, edits, and calls on_save exactly once."""

    def __init__(
        self,
        master: tk.Misc,
        entry: VocabularyEntry,
        on_save: Callable[[VocabularyEntry], None],
        title: str = "Red Flag — Edit Word Entry",
    ) -> None:
        super().__init__(master)
        self.title(title)
        self.geometry("640x520")
        self.minsize(480, 400)
        self.transient(master)
        self.grab_set()                       # modal

        self._editor = RedFlagEditor(entry, on_save)
        self._build_widgets()
        self._load_into_text()

    def _build_widgets(self) -> None:
        header = tk.Frame(self)
        header.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(
            header,
            text="Edit enrichment fields only. Build-time fields "
                 "(occurrences, first_chapter, sentences) are ignored.",
            anchor="w", justify="left", fg="#8b94b3",
        ).pack(fill="x")

        body = tk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        scrollbar = tk.Scrollbar(body)
        scrollbar.pack(side="right", fill="y")

        self._text = tk.Text(
            body, wrap="none", undo=True,
            font=("Menlo", 12), yscrollcommand=scrollbar.set,
        )
        self._text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self._text.yview)

        footer = tk.Frame(self)
        footer.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(footer, text="Cancel", command=self.destroy).pack(side="right")
        tk.Button(footer, text="Save", command=self._on_save_clicked).pack(
            side="right", padx=(0, 8)
        )
        tk.Button(footer, text="Reformat", command=self._on_reformat).pack(
            side="left"
        )

    # -- actions ------------------------------------------------------------

    def _load_into_text(self) -> None:
        self._text.delete("1.0", "end")
        self._text.insert("1.0", self._editor.to_json_text())

    def _on_reformat(self) -> None:
        """Re-indent the current text without validating the schema."""
        raw = self._text.get("1.0", "end").strip()
        try:
            pretty = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except json.JSONDecodeError as exc:
            messagebox.showerror("JSON error", str(exc), parent=self)
            return
        self._text.delete("1.0", "end")
        self._text.insert("1.0", pretty)

    def _on_save_clicked(self) -> None:
        raw = self._text.get("1.0", "end")
        try:
            self._editor.save_from_text(raw)
        except ValueError as exc:
            messagebox.showerror("Cannot save", str(exc), parent=self)
            return
        self.destroy()


# ---------------------------------------------------------------------------
# 3. Convenience opener
# ---------------------------------------------------------------------------

def open_red_flag_dialog(
    master: tk.Misc,
    entry: VocabularyEntry,
    on_save: Callable[[VocabularyEntry], None],
) -> RedFlagDialog:
    """Create and return the dialog (caller may wait_window if desired)."""
    return RedFlagDialog(master, entry, on_save)
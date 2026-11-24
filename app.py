"""
app.py
Application entry point for the Book Language Trainer.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config import ensure_directories
from ui.main_window import MainWindow


def main() -> None:
    ensure_directories()

    root = tk.Tk()
    root.title("Book Language Trainer — PDF to Vocabulary & Ollama Reader")
    root.geometry("980x700")
    root.minsize(820, 560)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
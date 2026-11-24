"""ingestion/pdf_extractor.py
Extract page text, metadata, candidate chapter starts, and light page info.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF

from config import ProgressCallback, log


@dataclass
class ExtractedPage:
    page_number: int
    text: str


@dataclass
class PdfExtractionResult:
    title: str
    author: str
    total_pages: int
    pages: list[ExtractedPage]
    suggested_chapters: list[tuple[str, int]]


CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:chapter|chapitre|capitolo|capítulo|kapitel|part|book)\s+"
    r"(?:\d+|[ivxlcdm]+)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_page_text(raw_text: str) -> str:
    text = (
        raw_text.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u00ad", "")
    )
    text = re.sub(r"([A-Za-zÀ-ÿ]+)-\s*\n\s*([a-zà-ÿ]+)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = text.split("\n\n")
    cleaned: list[str] = []
    for para in paragraphs:
        single = re.sub(r"[ \t]*\n[ \t]*", " ", para.strip())
        single = re.sub(r"[ \t]{2,}", " ", single)
        if single:
            cleaned.append(single)
    return "\n\n".join(cleaned)


def extract_pdf_pages(
    pdf_path: str | Path,
    on_progress: ProgressCallback | None = None,
) -> PdfExtractionResult:
    path = Path(pdf_path)
    log.info("Opening PDF: %s", path)

    doc = pymupdf.open(path)
    pdf_meta = doc.metadata or {}
    title = (pdf_meta.get("title") or "").strip() or path.stem.replace("_", " ").title()
    author = (pdf_meta.get("author") or "").strip() or "Unknown Author"

    pages: list[ExtractedPage] = []
    total_pages = doc.page_count
    for idx, page in enumerate(doc, start=1):
        raw = page.get_text("text") or ""
        pages.append(ExtractedPage(page_number=idx, text=_clean_page_text(raw)))
        if on_progress and idx % 10 == 0:
            on_progress(f"Extracting page {idx}/{total_pages}", idx, total_pages)

    suggested = detect_chapter_starts(doc, pages)
    doc.close()

    return PdfExtractionResult(
        title=title, author=author, total_pages=total_pages,
        pages=pages, suggested_chapters=suggested,
    )


def detect_chapter_starts(
    doc: pymupdf.Document,
    pages: list[ExtractedPage],
) -> list[tuple[str, int]]:
    chapters: list[tuple[str, int]] = []
    seen: set[int] = set()

    toc = doc.get_toc(simple=True) or []
    for level, ch_title, page_num in toc:
        if page_num not in seen:
            chapters.append((ch_title.strip(), page_num))
            seen.add(page_num)

    if not chapters:
        for page in pages:
            for match in CHAPTER_HEADING_RE.finditer(page.text):
                heading = match.group(0).strip()
                if page.page_number not in seen:
                    chapters.append((heading, page.page_number))
                    seen.add(page.page_number)

    if not chapters:
        chapters.append(("Full Text", 1))

    log.info("Detected %d chapter(s)", len(chapters))
    return chapters


# ── NEW: fast page info (no text extraction) ────────────────────────────
def pdf_page_info(pdf_path: str | Path) -> dict:
    """Return ``{total_pages, suggested_chapters}`` without extracting text.

    Used by the phone bridge's chapter editor; runs in milliseconds even
    for 1000-page PDFs because ``get_toc`` and ``page_count`` are O(1).
    """
    path = Path(pdf_path)
    doc = pymupdf.open(path)
    try:
        total = doc.page_count
        toc = doc.get_toc(simple=True) or []
        suggested = [
            {"title": str(t[1]).strip() or f"Part {i+1}",
             "start_page": int(t[2]),
             "end_page": 0}  # 0 = "until next / end of book"
            for i, t in enumerate(toc)
            if len(t) >= 3 and 1 <= int(t[2]) <= total
        ]
    finally:
        doc.close()
    return {"total_pages": total, "suggested_chapters": suggested}
"""
ingestion/pdf_extractor.py
Extracts page text, metadata, and candidate chapter start pages from PDF files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF


@dataclass
class ExtractedPage:
    page_number: int  # 1-based page number
    text: str


@dataclass
class PdfExtractionResult:
    title: str
    author: str
    total_pages: int
    pages: list[ExtractedPage]
    suggested_chapters: list[tuple[str, int]]  # (chapter_title, start_page)


# Common chapter heading patterns across languages
CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:chapter|chapitre|capitolo|capítulo|kapitel|part|book)\s+"
    r"(?:\d+|[ivxlcdm]+)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_page_text(raw_text: str) -> str:
    """
    Normalize whitespace, repair line-break hyphenation, and fix common ligatures.
    """
    # Replace typographic ligatures
    text = (
        raw_text.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u00ad", "")  # soft hyphen
    )

    # Repair words broken across line breaks with a hyphen, e.g.:
    # "im-\nportante" -> "importante"
    text = re.sub(r"([A-Za-zÀ-ÿ]+)-\s*\n\s*([a-zà-ÿ]+)", r"\1\2", text)

    # Collapse multiple blank lines into two newlines (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse single newlines inside paragraphs into spaces
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    for para in paragraphs:
        single_line = re.sub(r"[ \t]*\n[ \t]*", " ", para.strip())
        single_line = re.sub(r"[ \t]{2,}", " ", single_line)
        if single_line:
            cleaned_paragraphs.append(single_line)

    return "\n\n".join(cleaned_paragraphs)


def extract_pdf_pages(pdf_path: str | Path) -> PdfExtractionResult:
    """
    Open a PDF file, extract metadata and cleaned text for each page,
    and detect suggested chapter start pages.
    """
    path = Path(pdf_path)
    doc = pymupdf.open(path)

    pdf_meta = doc.metadata or {}
    title = (pdf_meta.get("title") or "").strip() or path.stem.replace("_", " ").title()
    author = (pdf_meta.get("author") or "").strip() or "Unknown Author"

    pages: list[ExtractedPage] = []
    for idx, page in enumerate(doc, start=1):
        raw = page.get_text("text") or ""
        cleaned = _clean_page_text(raw)
        pages.append(ExtractedPage(page_number=idx, text=cleaned))

    suggested_chapters = detect_chapter_starts(doc, pages)
    total_pages = len(pages)
    doc.close()

    return PdfExtractionResult(
        title=title,
        author=author,
        total_pages=total_pages,
        pages=pages,
        suggested_chapters=suggested_chapters,
    )


def detect_chapter_starts(
    doc: pymupdf.Document,
    pages: list[ExtractedPage],
) -> list[tuple[str, int]]:
    """
    Detect chapter start pages using:
    1) PDF Table of Contents / outline bookmarks (if present)
    2) Regex heading detection on page text
    3) Fallback to a single "Full Text" chapter if no chapters are detected
    """
    chapters: list[tuple[str, int]] = []
    seen_pages: set[int] = set()

    # 1. Try PDF TOC / outline first
    toc = doc.get_toc(simple=True) or []
    for level, ch_title, page_num in toc:
        if level <= 2 and isinstance(page_num, int) and 1 <= page_num <= len(pages):
            clean_title = re.sub(r"\s+", " ", (ch_title or "")).strip()
            if clean_title and page_num not in seen_pages:
                chapters.append((clean_title, page_num))
                seen_pages.add(page_num)

    # 2. Fallback/supplement with heading regex scan
    if not chapters:
        for p in pages:
            # Check first ~300 chars of page for a chapter heading
            header_slice = p.text[:300]
            match = CHAPTER_HEADING_RE.search(header_slice)
            if match and p.page_number not in seen_pages:
                heading = match.group(0).strip()
                chapters.append((heading, p.page_number))
                seen_pages.add(p.page_number)

    # 3. Ultimate fallback: whole book as one chapter
    if not chapters and pages:
        chapters.append(("Chapter 1 (Full Book)", 1))

    chapters.sort(key=lambda item: item[1])
    return chapters
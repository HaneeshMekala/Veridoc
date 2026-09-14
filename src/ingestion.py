"""
src/ingestion.py — PDF loading and page-to-image conversion.

WHY THIS MODULE EXISTS:
Before we can detect layout regions or embed text, we need to extract two
things from every PDF page:
  1. Text WITH bounding boxes  → fed into LayoutLMv3 for region detection
  2. The page as an image      → also fed into LayoutLMv3 (it's multimodal)

This module handles both. It is the entry point for every document that
enters the system. If ingestion is broken, nothing downstream works.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber          # pdfplumber chosen over PyPDF2: extracts text WITH
                           # bounding box coordinates (x, y, width, height).
                           # PyPDF2 extracts text only — no spatial info.
                           # We NEED coordinates for LayoutLMv3 input.

import pdf2image           # pdf2image chosen to convert PDF pages → PIL Images.
                           # LayoutLMv3 requires an actual image tensor as input,
                           # not just text. This bridges PDF → vision model.

from PIL import Image      # PIL (Pillow) is the standard Python image library.
                           # pdf2image returns PIL Images, and HuggingFace
                           # processors expect PIL Images as input.

from config import PDF_DPI, SAMPLES_DIR


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

# WHY A DATACLASS:
# We could return raw dicts ({"text": ..., "image": ...}), but dataclasses
# give us type hints, dot-access, and __repr__ for free. They make the
# contract between modules explicit — downstream code knows exactly what
# fields to expect.

@dataclass
class PageWord:
    """
    A single word extracted from a PDF page, with its bounding box.

    WHY WE TRACK INDIVIDUAL WORDS (not full page text):
    LayoutLMv3 takes word-level bounding boxes as input — not character-level,
    not sentence-level. It needs to know exactly where each word sits on the
    page so it can fuse text + layout + image signals together.
    """
    text: str
    x0: float       # left edge of bounding box (in PDF points)
    y0: float       # top edge
    x1: float       # right edge
    y1: float       # bottom edge
    page_num: int   # 0-indexed page number


@dataclass
class PageData:
    """
    All data extracted from a single PDF page, ready for layout detection.

    Fields:
        page_num:  0-indexed page number.
        words:     List of PageWord — text + bounding boxes for every word.
        image:     PIL Image of the full page (used by LayoutLMv3 vision branch).
        width:     Page width in PDF points (used for bbox normalization).
        height:    Page height in PDF points.
        raw_text:  Full page text as a single string (used as fallback).
    """
    page_num: int
    words: list[PageWord]
    image: Image.Image
    width: float
    height: float
    raw_text: str = ""


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def load_pdf_pages(pdf_path: Path) -> list[PageData]:
    """
    Load a PDF and extract words + images for every page.

    WHY THIS EXISTS:
    This function is the single entry point for all PDF ingestion.
    It merges two parallel extraction paths (text via pdfplumber,
    images via pdf2image) into one list of PageData objects.

    NAIVE APPROACH (don't do this):
        text = pdf.extract_text()   # one big string for the whole doc
        # Problem: you lose ALL spatial structure. You can't tell where
        # on the page any word was, so LayoutLMv3 can't do its job.

    BETTER APPROACH (what we do):
        Extract word-level bounding boxes + page images separately,
        then merge them into PageData so downstream modules have both.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of PageData, one per page, in page order.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
        ValueError: If the PDF has no extractable pages.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Place your PDF in {SAMPLES_DIR} and pass the full path."
        )

    # Convert all pages to PIL Images up front.
    # pdf2image uses poppler under the hood — a C++ PDF renderer.
    # DPI=200 gives sharp images without excessive memory use.
    page_images = _pdf_to_images(pdf_path)

    pages: list[PageData] = []

    # pdfplumber opens the PDF and gives us page objects with word data.
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        if len(pdf.pages) != len(page_images):
            # This can happen if poppler and pdfplumber disagree on page count
            # (rare, but worth catching explicitly).
            raise ValueError(
                f"Page count mismatch: pdfplumber={len(pdf.pages)}, "
                f"pdf2image={len(page_images)} for {pdf_path}"
            )

        for page_num, (plumber_page, image) in enumerate(
            zip(pdf.pages, page_images)
        ):
            words = _extract_words(plumber_page, page_num)
            raw_text = plumber_page.extract_text() or ""

            pages.append(PageData(
                page_num=page_num,
                words=words,
                image=image,
                width=float(plumber_page.width),
                height=float(plumber_page.height),
                raw_text=raw_text,
            ))

    return pages


def _pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    """
    Convert every page of a PDF to a PIL Image.

    WHY THIS EXISTS:
    LayoutLMv3 is a MULTIMODAL model — it reads both the text layout
    AND the visual appearance of the page. This function produces the
    visual input side.

    LEARN THIS — Why do we need images if we already have text?
    Plain text extraction loses visual cues: bold headings, table borders,
    column alignment, logo positions. LayoutLMv3 was trained to use these
    visual signals to improve region classification. A table looks like a
    table visually even if the text is ambiguous.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PIL Images, one per page.

    Raises:
        RuntimeError: If pdf2image/poppler fails to convert the PDF.
    """
    try:
        images = pdf2image.convert_from_path(
            str(pdf_path),
            dpi=PDF_DPI,
            fmt="RGB",        # LayoutLMv3 expects 3-channel RGB, not grayscale
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert PDF to images: {pdf_path}\n"
            f"Is poppler installed? On Windows: install via conda or "
            f"download from https://github.com/oschwartz10612/poppler-windows\n"
            f"Original error: {e}"
        ) from e

    return images


def _extract_words(page, page_num: int) -> list[PageWord]:
    """
    Extract all words and their bounding boxes from a single pdfplumber page.

    WHY WORD-LEVEL (not character or sentence level):
    LayoutLMv3 tokenizer expects words as input — it handles sub-word
    tokenization internally. Giving it characters would fragment meaning;
    giving it sentences would lose positional granularity.

    LEARN THIS — What is a "bounding box"?
    A bounding box (bbox) is a rectangle defined by 4 numbers:
    (x0, y0, x1, y1) where (x0, y0) is the top-left corner and
    (x1, y1) is the bottom-right corner. In PDF coordinate space,
    (0, 0) is the BOTTOM-left of the page — pdfplumber normalizes
    this so (0, 0) is the TOP-left, matching image coordinates.

    Args:
        page:     A pdfplumber Page object.
        page_num: 0-indexed page number, stored on each word for tracing.

    Returns:
        List of PageWord. Empty list if page has no extractable words.
    """
    # pdfplumber.extract_words() returns a list of dicts with keys:
    # 'text', 'x0', 'top', 'x1', 'bottom' (among others).
    #
    # NAIVE APPROACH (what the first version did):
    #   x_tolerance=3  → characters closer than 3pt are glued into one word.
    #   PROBLEM: tightly typeset PDFs (LaTeX papers, dense leaflets) use word
    #   spaces under 3pt at small font sizes, producing "words" like
    #   "Self-supervisedpre-trainingtechniques" that no tokenizer can read.
    # BETTER APPROACH (what we do):
    #   x_tolerance_ratio=0.15 → the gap threshold is 15% of the font size,
    #   so it scales with the text: tight for 8pt footnotes, looser for titles.
    #   On a test paper this took page 1 from 122 glued "words" to 656 real ones.
    raw_words = page.extract_words(
        x_tolerance_ratio=0.15,
        y_tolerance=3,    # words within 3pt vertically are on same line
        keep_blank_chars=False,
    )

    words: list[PageWord] = []
    for w in raw_words:
        text = w.get("text", "").strip()
        if not text:
            continue  # skip empty "words" that can appear in malformed PDFs

        words.append(PageWord(
            text=text,
            x0=float(w["x0"]),
            y0=float(w["top"]),      # pdfplumber uses "top" for y0
            x1=float(w["x1"]),
            y1=float(w["bottom"]),   # and "bottom" for y1
            page_num=page_num,
        ))

    return words


def is_scanned_pdf(page_data: PageData) -> bool:
    """
    Detect whether a page is scanned (image-only, no embedded text).

    WHY THIS EXISTS:
    Scanned PDFs have no extractable text — a scanner just takes a photo.
    pdfplumber returns zero words for these pages. We need to detect this
    so we can warn the user (full OCR integration is a future extension).

    INTERVIEW ALERT — This is a real-world gotcha in document AI:
    "What happens when someone uploads a scanned PDF?"
    Your answer: "We detect it by checking if word extraction returns
    empty — if so, we flag it. Full OCR via Tesseract or Azure Document
    Intelligence would be the production fix."

    Args:
        page_data: An already-extracted PageData object.

    Returns:
        True if the page appears to be scanned (no words found).
    """
    return len(page_data.words) == 0


def get_page_summary(pages: list[PageData]) -> dict:
    """
    Return a human-readable summary of what was extracted from a PDF.

    WHY THIS EXISTS:
    Useful for debugging and for the Gradio UI to show the user what
    was loaded before they run a query.

    Args:
        pages: List of PageData returned by load_pdf_pages().

    Returns:
        Dict with total_pages, total_words, scanned_pages count.
    """
    scanned = [p for p in pages if is_scanned_pdf(p)]
    total_words = sum(len(p.words) for p in pages)

    return {
        "total_pages": len(pages),
        "total_words": total_words,
        "scanned_pages": len(scanned),
        "scanned_page_numbers": [p.page_num for p in scanned],
    }

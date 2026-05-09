"""
ocr.py
------
Stage 2: Extract text from PDF invoices.

Commercial logistics invoices are typically:
  - Scanned documents from third-party vendors (image-based PDFs)
  - Digitally generated but with inconsistent layouts across vendors
  - Multi-page, with relevant data scattered across line items and headers

Strategy:
  1. Attempt direct text extraction via pdfplumber (fast, accurate for digital PDFs)
  2. Fall back to pytesseract OCR for image-based or scanned PDFs

The pipeline intentionally does NOT assume clean, structured input.
Robustness to vendor layout variation is a core design requirement.
"""

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

# Minimum character count to trust pdfplumber extraction.
# Below this threshold we assume the PDF is image-based.
MIN_TEXT_LENGTH = 50


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file.

    Tries pdfplumber first. If the extracted text is too short (indicating
    a scanned/image PDF), falls back to Tesseract OCR via pdf2image.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text across all pages, with page breaks preserved.

    Raises:
        RuntimeError: If both extraction methods fail.
    """
    text = _extract_with_pdfplumber(pdf_path)

    if len(text.strip()) < MIN_TEXT_LENGTH:
        logger.info(
            f"pdfplumber returned sparse text ({len(text.strip())} chars) "
            f"for {pdf_path.name} — falling back to Tesseract OCR."
        )
        text = _extract_with_tesseract(pdf_path)

    if not text.strip():
        raise RuntimeError(f"Both OCR methods returned empty output for {pdf_path.name}")

    return text


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    """
    Direct text extraction. Works on digitally generated PDFs.
    Preserves whitespace layout which helps the LLM identify fields.
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                pages.append(f"[Page {i + 1}]\n{page_text}")
    except Exception as e:
        logger.warning(f"pdfplumber failed on {pdf_path.name}: {e}")
        return ""

    return "\n\n".join(pages)


def _extract_with_tesseract(pdf_path: Path) -> str:
    """
    OCR fallback for scanned/image-based PDFs.
    Converts each page to an image, then runs Tesseract.

    Requires: pdf2image (`pip install pdf2image`) and
              Tesseract binary installed on the system.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as e:
        raise RuntimeError(
            "Tesseract OCR dependencies not installed. "
            "Run: pip install pdf2image pytesseract\n"
            "Also install Tesseract binary: https://github.com/tesseract-ocr/tesseract"
        ) from e

    pages = []
    try:
        images = convert_from_path(pdf_path, dpi=300)
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image, config="--psm 6")
            pages.append(f"[Page {i + 1}]\n{page_text}")
    except Exception as e:
        raise RuntimeError(f"Tesseract OCR failed on {pdf_path.name}: {e}") from e

    return "\n\n".join(pages)

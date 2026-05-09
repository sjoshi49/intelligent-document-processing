"""
pipeline.py
-----------
Orchestrates the four-stage document intelligence pipeline:

    Stage 1 — Ingestion   : Discover and load PDF invoices from a source directory
    Stage 2 — OCR         : Extract raw text from each PDF using pdfplumber
    Stage 3 — Classification : Pass OCR text + structured prompt to GPT-4 for
                              field extraction and movement classification
    Stage 4 — Output       : Parse model response, compute derived fields,
                              write structured CSV for downstream analysis

Domain: Commercial logistics invoice processing.
The pipeline classifies each vendor invoice as a billable service move
(equipment cannot self-transport) or a potentially internalizable transfer
(equipment is operable and could be moved by internal drivers at lower cost).

This is a sanitized portfolio reconstruction demonstrating the same
production architecture pattern applied to publicly available data.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from ocr import extract_text_from_pdf
from classifier import classify_invoice
from postprocess import compute_derived_fields, flatten_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INVOICE_DIR = ROOT / "data" / "sample_invoices"
PROMPT_FILE = ROOT / "prompts" / "classification_prompt.txt"
CACHE_DIR = ROOT / "outputs" / "cache"
OUTPUT_CSV = ROOT / "outputs" / "classified_invoices.csv"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _prompt_hash(prompt_path: Path) -> str:
    """SHA-256 of the prompt file content — used to detect prompt changes."""
    return hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:12]


def _cache_path(pdf_path: Path, prompt_hash: str) -> Path:
    """
    Cache key encodes both the invoice filename and the prompt version.
    If the prompt changes, the hash changes and the invoice is reprocessed.
    This avoids stale classifications when prompt logic is iterated.
    """
    key = hashlib.md5(f"{pdf_path.name}::{prompt_hash}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _load_cache(cache_file: Path) -> dict | None:
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def _save_cache(cache_file: Path, data: dict) -> None:
    cache_file.write_text(json.dumps(data, indent=2))


# ── Core pipeline ────────────────────────────────────────────────────────────

def process_invoice(pdf_path: Path, prompt_text: str, prompt_hash: str) -> dict:
    """
    Run a single invoice through all four pipeline stages.
    Returns a flat dict ready for DataFrame insertion.
    """
    cache_file = _cache_path(pdf_path, prompt_hash)
    cached = _load_cache(cache_file)

    if cached:
        logger.debug(f"Cache hit: {pdf_path.name}")
        return cached

    # Stage 2 — OCR
    logger.info(f"OCR: {pdf_path.name}")
    ocr_text = extract_text_from_pdf(pdf_path)

    if not ocr_text.strip():
        logger.warning(f"Empty OCR output for {pdf_path.name} — skipping.")
        return {"InvoiceFile": pdf_path.name, "Error": "OCR produced no text"}

    # Stage 3 — GPT-4 Classification
    logger.info(f"Classifying: {pdf_path.name}")
    raw_response = classify_invoice(ocr_text, prompt_text)

    # Stage 4 — Postprocessing
    flat = flatten_response(raw_response, invoice_file=pdf_path.name)
    flat = compute_derived_fields(flat)

    _save_cache(cache_file, flat)
    return flat


def run_pipeline() -> pd.DataFrame:
    """
    Discover all PDFs in INVOICE_DIR, process each, and write a consolidated CSV.

    Caching logic:
        - Prompt unchanged + invoice cached  → skip (use cache)
        - Prompt changed   + invoice cached  → reprocess (cache key changes)
        - Invoice not cached                 → process always
    """
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    prompt_text = PROMPT_FILE.read_text()
    prompt_hash = _prompt_hash(PROMPT_FILE)
    logger.info(f"Prompt hash: {prompt_hash}")

    pdf_files = sorted(INVOICE_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {INVOICE_DIR}")
        return pd.DataFrame()

    logger.info(f"Found {len(pdf_files)} invoice(s) to process.")

    results = []
    for pdf_path in tqdm(pdf_files, desc="Processing invoices"):
        try:
            record = process_invoice(pdf_path, prompt_text, prompt_hash)
            results.append(record)
        except Exception as e:
            logger.error(f"Failed on {pdf_path.name}: {e}")
            results.append({"InvoiceFile": pdf_path.name, "Error": str(e)})

    df = pd.DataFrame(results)

    # Reorder columns for readability
    priority_cols = [
        "InvoiceFile", "MoveType", "Vendor", "Reason",
        "ServiceFrom", "ServiceTo",
        "Distance_miles", "DriveLength_minutes",
        "TotalAmountCharged_dollars", "HourlyRate_dollars",
        "EstimatedInternalCost_dollars", "ProjectedSavings_dollars",
        "Confidence", "ProcessedAt",
    ]
    present = [c for c in priority_cols if c in df.columns]
    remainder = [c for c in df.columns if c not in priority_cols]
    df = df[present + remainder]

    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Output written to {OUTPUT_CSV} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    df = run_pipeline()
    if not df.empty:
        print("\n── Sample output ──────────────────────────────────────")
        print(df[["InvoiceFile", "MoveType", "Vendor", "ProjectedSavings_dollars"]].to_string(index=False))
        print(f"\nTotal projected savings: ${df['ProjectedSavings_dollars'].sum():,.2f}")

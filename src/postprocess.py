"""
postprocess.py
--------------
Stage 4: Parse the model's JSON output, compute derived cost fields,
and flatten the response into a row-ready dict for the output CSV.

Derived fields computed here (not by the model):
  - HourlyRate_dollars          : TotalAmountCharged / (DriveLength / 60)
  - EstimatedInternalCost_dollars : distance-based estimate using internal
                                    driver cost benchmarks
  - ProjectedSavings_dollars    : TotalAmountCharged - EstimatedInternalCost
                                  (only meaningful for Transfer classifications)

Design note on reasoning fields:
  The model returns several *Reasoning fields (e.g., MoveTypeReasoning,
  DistanceReasoning) to make its logic auditable during development and QA.
  These are stored in the cache but excluded from the output CSV to keep
  the stakeholder-facing file clean. They can be re-enabled via the
  INCLUDE_REASONING flag below.

  This was an explicit product decision: stakeholders care about the
  classification result and the savings figure, not the model's reasoning chain.
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Set True to include *Reasoning columns in the output CSV (for debugging)
INCLUDE_REASONING = False

# Internal cost benchmark: estimated cost per mile for an internal driver move.
# Used to compute savings opportunity on Transfer-classified invoices.
# Adjust this to match your organisation's internal rate card.
INTERNAL_COST_PER_MILE = 1.85  # dollars/mile (fuel + driver time estimate)

# Fields the model returns that are reasoning/audit fields — hidden by default
REASONING_FIELDS = {
    "MoveTypeReasoning",
    "DistanceReasoning",
    "DriveLengthReasoning",
    "UberFareReasoning",
    "HourlyRateReasoning",
}


def flatten_response(raw: dict[str, Any], invoice_file: str) -> dict[str, Any]:
    """
    Normalise field names from the model response and inject metadata.

    The model uses descriptive key names (e.g. "Distance(miles)").
    We normalise these to snake_case-friendly column names for the DataFrame.

    Args:
        raw:          Parsed JSON dict from the classifier.
        invoice_file: Source filename, used as the row identifier.

    Returns:
        Normalised flat dict.
    """
    # Normalise keys — strip units from parenthetical notation
    normalised = {}
    key_map = {
        "Distance(miles)": "Distance_miles",
        "DriveLength(minutes)": "DriveLength_minutes",
        "TotalAmountCharged(dollars)": "TotalAmountCharged_dollars",
        "HourlyRate(dollars)": "HourlyRate_dollars",
        "UberFare(dollars)": "UberFare_dollars",
    }
    for k, v in raw.items():
        normalised[key_map.get(k, k)] = v

    # Inject metadata
    normalised["InvoiceFile"] = invoice_file
    normalised["ProcessedAt"] = datetime.now(timezone.utc).isoformat()

    # Optionally remove reasoning fields
    if not INCLUDE_REASONING:
        for field in REASONING_FIELDS:
            normalised.pop(field, None)

    return normalised


def compute_derived_fields(record: dict[str, Any]) -> dict[str, Any]:
    """
    Compute cost-analysis fields that are not extracted from the invoice
    but are derived from extracted values.

    These fields are the primary business output — they quantify the savings
    opportunity when an invoice is classified as an internalizable Transfer.

    Args:
        record: Normalised flat dict from flatten_response().

    Returns:
        Record with additional computed fields appended.
    """
    total_charged = _to_float(record.get("TotalAmountCharged_dollars"))
    drive_length_min = _to_float(record.get("DriveLength_minutes"))
    distance_miles = _to_float(record.get("Distance_miles"))
    move_type = str(record.get("MoveType", "")).strip().lower()

    # HourlyRate — compute if not extracted
    if record.get("HourlyRate_dollars") is None and total_charged and drive_length_min:
        try:
            rate = total_charged / (drive_length_min / 60.0)
            record["HourlyRate_dollars"] = round(rate, 2)
        except ZeroDivisionError:
            record["HourlyRate_dollars"] = None

    # EstimatedInternalCost — distance-based internal rate
    if distance_miles:
        record["EstimatedInternalCost_dollars"] = round(
            distance_miles * INTERNAL_COST_PER_MILE, 2
        )
    else:
        record["EstimatedInternalCost_dollars"] = None

    # ProjectedSavings — only meaningful for Transfer classifications
    # Tow movements are legitimately non-internalizable; savings = 0
    if move_type == "transfer" and total_charged and record.get("EstimatedInternalCost_dollars"):
        savings = total_charged - record["EstimatedInternalCost_dollars"]
        record["ProjectedSavings_dollars"] = round(max(savings, 0.0), 2)
    else:
        record["ProjectedSavings_dollars"] = 0.0

    return record


def _to_float(value: Any) -> float | None:
    """Safely coerce a value to float; return None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None

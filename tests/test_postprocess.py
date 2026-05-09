"""
test_postprocess.py
-------------------
Unit tests for the postprocessing module.

Tests cover:
  - Field normalisation (key name mapping)
  - HourlyRate derivation from TotalAmountCharged and DriveLength
  - EstimatedInternalCost calculation from distance
  - ProjectedSavings logic for Transfer vs Tow classifications
  - Edge cases: null fields, zero drive length, missing values
"""

import pytest
from src.postprocess import flatten_response, compute_derived_fields, _to_float


# ── flatten_response ──────────────────────────────────────────────────────────

class TestFlattenResponse:
    def test_normalises_key_names(self):
        raw = {
            "Distance(miles)": 12.5,
            "DriveLength(minutes)": 25,
            "TotalAmountCharged(dollars)": 150.00,
            "HourlyRate(dollars)": None,
            "UberFare(dollars)": 22.50,
            "MoveType": "Transfer",
            "Vendor": "FastHaul Logistics",
            "Reason": "Transfer-only vendor.",
            "Confidence": "High",
        }
        result = flatten_response(raw, invoice_file="test_invoice.pdf")
        assert "Distance_miles" in result
        assert "DriveLength_minutes" in result
        assert "TotalAmountCharged_dollars" in result
        assert "InvoiceFile" in result
        assert result["InvoiceFile"] == "test_invoice.pdf"

    def test_reasoning_fields_excluded_by_default(self):
        raw = {
            "MoveType": "Tow",
            "MoveTypeReasoning": "Engine overheating noted.",
            "DistanceReasoning": "Estimated from addresses.",
            "Vendor": "AnyTow",
        }
        result = flatten_response(raw, invoice_file="x.pdf")
        assert "MoveTypeReasoning" not in result
        assert "DistanceReasoning" not in result

    def test_metadata_injected(self):
        result = flatten_response({"MoveType": "Tow"}, invoice_file="inv.pdf")
        assert "ProcessedAt" in result
        assert "InvoiceFile" in result


# ── compute_derived_fields ────────────────────────────────────────────────────

class TestComputeDerivedFields:
    def _base_transfer(self, distance=20.0, drive_len=25.0, total=180.0):
        return {
            "MoveType": "Transfer",
            "Distance_miles": distance,
            "DriveLength_minutes": drive_len,
            "TotalAmountCharged_dollars": total,
            "HourlyRate_dollars": None,
        }

    def _base_tow(self):
        return {
            "MoveType": "Tow",
            "Distance_miles": 15.0,
            "DriveLength_minutes": 30.0,
            "TotalAmountCharged_dollars": 250.0,
            "HourlyRate_dollars": None,
        }

    def test_hourly_rate_computed_when_missing(self):
        rec = compute_derived_fields(self._base_transfer(drive_len=30.0, total=120.0))
        # 120 / (30/60) = 240
        assert rec["HourlyRate_dollars"] == pytest.approx(240.0, rel=1e-3)

    def test_hourly_rate_not_overwritten_when_present(self):
        rec = self._base_transfer()
        rec["HourlyRate_dollars"] = 99.99
        result = compute_derived_fields(rec)
        assert result["HourlyRate_dollars"] == 99.99

    def test_internal_cost_computed_from_distance(self):
        rec = compute_derived_fields(self._base_transfer(distance=10.0))
        # 10 * 1.85 = 18.50
        assert rec["EstimatedInternalCost_dollars"] == pytest.approx(18.50, rel=1e-3)

    def test_savings_positive_for_transfer(self):
        # Total = 180, internal = 20 * 1.85 = 37.00 → savings = 143.00
        rec = compute_derived_fields(self._base_transfer(distance=20.0, total=180.0))
        assert rec["ProjectedSavings_dollars"] > 0

    def test_savings_zero_for_tow(self):
        rec = compute_derived_fields(self._base_tow())
        assert rec["ProjectedSavings_dollars"] == 0.0

    def test_savings_clamped_at_zero(self):
        # Edge case: internal cost > vendor cost (e.g. short haul at base rate)
        rec = self._base_transfer(distance=200.0, total=50.0)
        result = compute_derived_fields(rec)
        assert result["ProjectedSavings_dollars"] == 0.0

    def test_null_distance_yields_no_cost(self):
        rec = self._base_transfer(distance=None)
        result = compute_derived_fields(rec)
        assert result["EstimatedInternalCost_dollars"] is None
        assert result["ProjectedSavings_dollars"] == 0.0

    def test_zero_drive_length_no_division_error(self):
        rec = self._base_transfer(drive_len=0.0)
        result = compute_derived_fields(rec)
        assert result["HourlyRate_dollars"] is None  # safely handled


# ── _to_float ─────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_plain_number(self):
        assert _to_float(42.5) == 42.5

    def test_string_with_dollar_sign(self):
        assert _to_float("$1,234.56") == pytest.approx(1234.56)

    def test_none_returns_none(self):
        assert _to_float(None) is None

    def test_unparseable_string_returns_none(self):
        assert _to_float("N/A") is None

    def test_integer_string(self):
        assert _to_float("100") == 100.0

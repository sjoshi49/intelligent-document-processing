# Document Intelligence Pipeline

A production-grade document classification pipeline that extracts structured fields from unstructured PDF invoices and classifies each vehicle movement using GPT-4 with iterative prompt engineering.

**Domain:** Commercial logistics cost optimization. Identifies when third-party vendor invoices represent movements that could have been handled internally at lower cost.

## Architecture

```
Stage 1 - Ingestion       Stage 2 - OCR            Stage 3 - Classification     Stage 4 - Output
---------------------     ----------------         ------------------------     ----------------
PDF files from             pdfplumber (digital)  >  GPT-4 + structured prompt >  Parsed JSON
local directory       >    pytesseract (scanned)    iterative prompt versions     Derived fields
                                                     JSON schema enforcement       CSV export
```

The pipeline is **prompt-version-aware**: each invoice is cached against a hash of both the invoice content and the current prompt file. If the prompt is updated, invoices are automatically reprocessed. If neither changes, cached results are served instantly.

## Key Design Decisions

**1. External prompt file with version hashing**
The classification prompt lives in `prompts/classification_prompt.txt` and is loaded at runtime, not hardcoded. The cache key is `md5(invoice_filename + prompt_sha256)`, so prompt iteration never requires manual cache invalidation. This mirrors how production prompt engineering workflows version-control prompt assets separately from application logic.

**2. Dual OCR strategy**
`pdfplumber` handles digitally generated PDFs (fast, structure-preserving). If extracted text falls below a minimum character threshold, the pipeline automatically falls back to `pytesseract` OCR. This handles the reality of commercial invoice datasets: vendor documents arrive in inconsistent formats.

**3. Reasoning fields decoupled from output**
The model returns `*Reasoning` fields (e.g., `MoveTypeReasoning`, `DistanceReasoning`) that explain each extraction decision. These are cached for QA and debugging but excluded from the stakeholder-facing CSV by default. The `INCLUDE_REASONING` flag in `postprocess.py` re-enables them. This separates model auditability from output readability.

**4. Derived cost fields computed in code, not by the model**
`HourlyRate`, `EstimatedInternalCost`, and `ProjectedSavings` are computed in `postprocess.py` from model-extracted values, not by GPT-4. Business calculation logic belongs in deterministic code, not in a stochastic model.

**5. Confidence scoring**
The model returns a `Confidence` field (`High`, `Medium`, `Low`) for each classification. Low-confidence records can be filtered for human review without reprocessing the full dataset.

## Output Schema

| Column | Description |
|--------|-------------|
| `InvoiceFile` | Source PDF filename |
| `MoveType` | `Tow` or `Transfer` |
| `Vendor` | Third-party vendor name |
| `Reason` | Model's one-sentence classification rationale |
| `ServiceFrom` | Origin address |
| `ServiceTo` | Destination address |
| `Distance_miles` | Miles traveled (extracted or estimated) |
| `DriveLength_minutes` | Drive time (extracted or estimated) |
| `TotalAmountCharged_dollars` | Vendor invoice total |
| `HourlyRate_dollars` | Derived: Total / (DriveLength / 60) |
| `EstimatedInternalCost_dollars` | Derived: distance x internal rate card |
| `ProjectedSavings_dollars` | Derived: savings opportunity for Transfer moves |
| `Confidence` | Classification confidence (`High` / `Medium` / `Low`) |
| `ProcessedAt` | UTC timestamp of processing |

## Prompt Engineering Notes

The classification prompt went through multiple iterations before reaching production accuracy. Key refinements:

- **JSON schema enforcement:** Early versions returned inconsistently structured output. Adding an explicit JSON schema to the prompt (not just "respond in JSON") reduced parse errors from ~18% to under 2%.
- **Vendor list injection:** Deterministic transfer classifications for known vendor names were implemented as an explicit lookup table in the prompt, preventing the model from re-reasoning cases with a known ground truth.
- **Reasoning field separation:** Adding dedicated `*Reasoning` fields separated from the classification output improved classification accuracy. The model's chain-of-thought was previously bleeding into the classification field and corrupting downstream parsing.
- **Confidence scoring:** After adding the `Confidence` field, low-confidence records clustered around genuinely ambiguous invoice language. This made human review targeted rather than random sampling.

## Tech Stack

- **LLM:** GPT-4o via OpenAI API
- **OCR:** pdfplumber (digital PDFs), pytesseract (scanned PDFs)
- **Data:** pandas
- **Testing:** pytest
- **Prompt versioning:** SHA-256 hash of prompt file, baked into cache key

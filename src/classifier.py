"""
classifier.py
-------------
Stage 3: Send OCR-extracted invoice text to GPT-4 for structured field
extraction and movement classification.

Design decisions:
  - Prompt is loaded from an external file (prompts/classification_prompt.txt)
    so it can be versioned and iterated independently of the code.
  - The model is instructed to respond ONLY in JSON — no preamble, no markdown.
  - Temperature is set to 0 for deterministic, reproducible output.
  - Max tokens capped at 800; invoice extractions are concise by design.
  - A retry wrapper handles transient API errors with exponential backoff.

Prompt engineering notes:
  The prompt underwent multiple rounds of iteration. Key refinements included:
    1. Explicit JSON schema specification to eliminate hallucinated field names
    2. Confidence scoring instructions to flag ambiguous classifications
    3. Reasoning fields (e.g., MoveTypeReasoning) to make model logic auditable
    4. Vendor list injection as a lookup table for deterministic transfer flags
    5. Fallback instructions for fields not present on the invoice (estimate vs null)
"""

import json
import logging
import time
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# Model and sampling config
MODEL = "gpt-4o"
TEMPERATURE = 0          # Deterministic output — critical for auditable pipelines
MAX_TOKENS = 800
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0      # seconds; doubles on each retry

client = OpenAI()        # Reads OPENAI_API_KEY from environment


def classify_invoice(ocr_text: str, prompt_text: str) -> dict[str, Any]:
    """
    Classify a single invoice using GPT-4.

    Constructs the user message by injecting the OCR text into the
    classification prompt, then parses the model's JSON response.

    Args:
        ocr_text:    Raw text extracted from the invoice PDF.
        prompt_text: Full classification prompt loaded from prompts/.

    Returns:
        Parsed JSON dict with extracted fields and classification.

    Raises:
        ValueError: If the model returns non-parseable output after retries.
        RuntimeError: If the API call fails after all retries.
    """
    user_message = (
        f"{prompt_text}\n\n"
        f"---\n"
        f"INVOICE TEXT:\n{ocr_text}\n"
        f"---\n\n"
        f"Respond ONLY with a valid JSON object. No preamble. No markdown."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document intelligence system that extracts structured "
                            "fields from commercial logistics invoices. You output only valid "
                            "JSON — no commentary, no markdown fences."
                        ),
                    },
                    {"role": "user", "content": user_message},
                ],
            )

            raw = response.choices[0].message.content.strip()
            return _parse_json_response(raw)

        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"JSON parse error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise ValueError(
                    f"Model returned non-JSON output after {MAX_RETRIES} attempts."
                ) from e

        except Exception as e:
            logger.warning(f"API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"API call failed after {MAX_RETRIES} attempts.") from e
            time.sleep(RETRY_BACKOFF * attempt)


def _parse_json_response(raw: str) -> dict[str, Any]:
    """
    Parse the model's raw string output into a Python dict.

    Strips markdown fences if present (defensive; model is instructed not to
    include them, but prompt compliance is not guaranteed at 100%).
    """
    # Strip ```json ... ``` fences if model includes them despite instructions
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    return json.loads(raw)

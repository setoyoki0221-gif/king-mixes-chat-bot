"""Central configuration — paths, providers, model names, env loading.

The app is provider-flexible. Two knobs select the backend:

    EMBED_PROVIDER = huggingface | openai
        (default: huggingface — free, runs locally)
    LLM_PROVIDER   = local | ollama | huggingface | gemini | github | openai | groq
        (default: local — a small model on-device, needs no API key)

Flip these (plus the matching API key, if any) to switch backends — no code
changes required. gemini/groq also need `pip install -r requirements-optional.txt`.

Values are read from the environment (a local .env during development, or
Hugging Face Space secrets in production).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op on HF Spaces (secrets come from the environment)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "products.json"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "king_arthur_mixes"

# Human-readable labels for the raw dietary/attribute badge slugs. Shared by
# ingest (document text) and the frontend (product-card badges).
BADGE_LABELS = {
    "glutenfree": "Gluten-Free",
    "kosherpareve": "Kosher (Pareve)",
    "kosherdairy": "Kosher (Dairy)",
    "nongmo": "Non-GMO",
    "sourcednongmo": "Sourced Non-GMO",
    "organic": "Organic",
    "wholegrain": "Whole Grain",
    "wholegrain50": "50%+ Whole Grain",
    "wholegrain100": "100% Whole Grain",
    "madeintheusa": "Made in the USA",
    "vegan": "Vegan",
}

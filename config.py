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

# --- Provider selection -------------------------------------------------
# LLM_PROVIDER:   local | ollama | huggingface | gemini | github | openai | groq
#                 (default: local — a small model on-device, no token/region issues)
# EMBED_PROVIDER: huggingface | openai   (default: huggingface — free/local)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "huggingface").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()

# --- API keys -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# Hugging Face token: powers the free Inference-API LLM and is the same token
# used to deploy to HF Spaces.
HF_TOKEN = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
# GitHub Models = free, OpenAI-compatible access to real OpenAI models.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_MODELS_BASE_URL = os.getenv(
    "GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com"
)

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

# --- Model names (sensible defaults per provider) -----------------------
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
GITHUB_CHAT_MODEL = os.getenv("GITHUB_CHAT_MODEL", "gpt-4o-mini")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Local on-device model (runs via transformers, CPU-friendly, not gated).
# 1.5B is the deployment default (good quality on HF Spaces free CPU); override
# with LOCAL_CHAT_MODEL=Qwen/Qwen2.5-0.5B-Instruct for a faster, lighter model.
LOCAL_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "220"))
# Ollama: fast quantized local models via the Ollama app (better quality+speed).
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Fast mode: skip the grade/rewrite reasoning loop (retrieve -> generate only).
# Roughly halves latency on CPU at the cost of the self-correction step.
AGENT_FAST = os.getenv("AGENT_FAST", "0") in ("1", "true", "True")


def active_chat_model() -> str:
    """Name of the chat model the selected LLM_PROVIDER will use (for display)."""
    return {
        "local": LOCAL_CHAT_MODEL,
        "ollama": OLLAMA_CHAT_MODEL,
        "huggingface": HF_CHAT_MODEL,
        "gemini": GEMINI_CHAT_MODEL,
        "github": GITHUB_CHAT_MODEL,
        "openai": OPENAI_CHAT_MODEL,
        "groq": GROQ_CHAT_MODEL,
    }.get(LLM_PROVIDER, LLM_PROVIDER)


def require_llm_key() -> None:
    """Validate that the selected LLM provider has its key set."""
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. Get a free key "
            "(no credit card) at https://console.groq.com/keys"
        )
    if LLM_PROVIDER == "github" and not GITHUB_TOKEN:
        raise RuntimeError(
            "LLM_PROVIDER=github but GITHUB_TOKEN is not set. Get a free token "
            "at https://github.com/settings/tokens (fine-grained, no scopes needed)."
        )
    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set. Get a free key "
            "(no credit card) at https://aistudio.google.com/apikey"
        )

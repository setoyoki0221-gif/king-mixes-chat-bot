"""Build the Chroma vector store from scraped product data.

Each product becomes one rich, self-contained document (title + price +
rating + dietary badges + description). Products are short, so one document
per product is the right granularity — it keeps each retrieved chunk fully
coherent and avoids splitting a description mid-sentence.

The document/metadata construction lives in catalog.py so the agent's
structured lookup and this index always agree on what a product looks like.

Run once after scraping:  python ingest.py
"""

from __future__ import annotations

import shutil

from langchain_chroma import Chroma

import config
from catalog import build_documents
from providers import get_embeddings


def main() -> None:
    docs = build_documents()
    print(f"Loaded {len(docs)} product documents from {config.DATA_PATH.name}")

    # Rebuild cleanly so re-runs are deterministic.
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)

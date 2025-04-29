"""Catalog access plus structured (non-semantic) product lookup.

Why this exists
---------------
Embedding search answers "what is *like* X?" well, but it cannot answer
aggregate questions such as "cheapest", "top rated" or "all gluten-free mixes".
The LLM only ever sees the few documents retrieved, so with plain similarity
search "cheapest" really means "cheapest of the six closest matches".

This module parses those intents out of the question with plain rules (no LLM,
so it is fast, deterministic and unit-testable) and answers them exactly by
filtering and sorting the *whole* catalog. Anything it cannot parse falls
through to ordinary semantic search, unchanged.

It also resolves short follow-ups ("What's the cheapest one?") against the
previous questions so they keep their topic and filters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from langchain_core.documents import Document

import config
from config import BADGE_LABELS

STRUCTURED_LIMIT = 8        # max products handed to the LLM on the structured path
SEMANTIC_TOPIC_POOL = 12    # semantic candidates kept when a topic matches no title
MAX_FOLLOWUP_CHAIN = 3      # how many earlier questions a follow-up may inherit from
_PRIOR_REVIEWS = 10         # weight of the prior in the review-weighted rating


# --------------------------------------------------------------------------- #
# Catalog documents (shared by ingest.py and the agent)
# --------------------------------------------------------------------------- #
def product_to_text(p: dict) -> str:
    """Render a product as a readable, retrieval-friendly text block."""
    lines = [f"Product: {p['title']}", f"Category: {p.get('category', 'Mixes')}"]
    if p.get("price"):
        lines.append(f"Price: {p['price']}")
    if p.get("promo"):
        lines.append(f"Offer: {p['promo']}")
    if p.get("rating"):
        reviews = p.get("review_count") or 0

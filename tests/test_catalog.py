"""Rule-based query understanding and exact catalog lookup (no LLM, no embeddings)."""

from __future__ import annotations

import pytest
from conftest import make_doc
from langchain_core.documents import Document

from catalog import (
    badges_of,
    compact_line,
    is_followup,
    parse_query,
    resolve_question,
    structured_search,
)


def titles(result):
    return [d.metadata["title"] for d in result.docs]


# --------------------------------------------------------------------------- #
# parse_query
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, sort",
    [
        ("What's the cheapest mix?", "price_asc"),
        ("least expensive option", "price_asc"),
        ("something budget friendly", "price_asc"),
        ("most expensive mix", "price_desc"),
        ("the priciest one", "price_desc"),
        ("Which mixes have the best reviews?", "rating"),
        ("top rated mixes", "rating"),
        ("highest-rated cookie mix", "rating"),
        ("most popular mix", "reviews"),
        ("most reviews", "reviews"),
        ("What cookie mixes do you have?", None),
        ("Which is best for beginners?", None),  # plain "best" is not a ranking request
    ],
)
def test_sort_detection(text, sort):
    assert parse_query(text).sort == sort

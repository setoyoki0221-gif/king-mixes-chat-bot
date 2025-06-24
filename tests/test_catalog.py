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


@pytest.mark.parametrize(
    "text, group",
    [
        ("gluten-free mixes", ("glutenfree",)),
        ("Gluten Free brownies", ("glutenfree",)),
        ("vegan options", ("vegan",)),
        ("non-GMO mixes", ("nongmo", "sourcednongmo")),
        ("kosher mixes", ("kosherpareve", "kosherdairy")),
        ("pareve mixes", ("kosherpareve",)),
        ("whole grain mixes", ("wholegrain", "wholegrain50", "wholegrain100")),
        ("made in the USA", ("madeintheusa",)),
    ],
)
def test_badge_filters(text, group):
    assert group in parse_query(text).badge_groups


def test_multiple_filters_all_required():
    spec = parse_query("cheapest gluten-free kosher mix")
    assert ("glutenfree",) in spec.badge_groups
    assert ("kosherpareve", "kosherdairy") in spec.badge_groups
    assert spec.sort == "price_asc"


@pytest.mark.parametrize(
    "text, mx, mn",
    [
        ("mixes under $7", 7.0, None),
        ("less than 10 dollars", 10.0, None),
        ("$8 or less", 8.0, None),
        ("cheaper than $5.50", 5.5, None),
        ("over $20", None, 20.0),
        ("at least 15 dollars", None, 15.0),
        # Numbers that are NOT prices must be ignored.
        ("mixes with at least 4 stars", None, None),
        ("under 10 reviews", None, None),
        ("rated over 4.5", None, None),
    ],
)
def test_price_bounds(text, mx, mn):
    spec = parse_query(text)
    assert (spec.max_price, spec.min_price) == (mx, mn)

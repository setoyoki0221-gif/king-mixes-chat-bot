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


def test_topic_words_exclude_filler_and_intent():
    assert parse_query("What's the cheapest chocolate chip cookie mix?").topic == [
        "chocolate", "chip", "cookie",
    ]
    assert parse_query("Do you have gluten-free options?").topic == []


def test_plain_semantic_questions_are_not_structured():
    for q in ["What cookie mixes do you have?", "Tell me about the brownie mix",
              "Recommend a mix for a kid's birthday party"]:
        assert not parse_query(q).structured


# --------------------------------------------------------------------------- #
# structured_search (synthetic catalog)
# --------------------------------------------------------------------------- #
def run(q, catalog, prior=(), rank=None):
    text, spec = resolve_question(q, list(prior))
    assert spec.structured, f"{q!r} should be a structured query"
    return structured_search(spec, catalog, semantic_rank=rank, query_text=text)


def test_cheapest_is_exact_over_whole_catalog(mini_catalog):
    r = run("cheapest mix", mini_catalog)
    assert titles(r)[0] == "Gluten-Free Sugar Cookie Mix"
    assert r.total == len(mini_catalog)
    assert "FIRST product listed is the cheapest" in r.note


def test_most_expensive(mini_catalog):
    assert titles(run("most expensive mix", mini_catalog))[0] == "Cupcake Party Kit"


def test_filter_counts_everything_not_just_top_k(mini_catalog):
    r = run("gluten-free mixes", mini_catalog)
    assert r.total == 2
    assert all("glutenfree" in badges_of(d) for d in r.docs)


def test_unmatched_attribute_returns_nothing_with_clear_note(mini_catalog):
    r = run("vegan options", mini_catalog)
    assert r.total == 0 and r.docs == []
    assert "no products in the catalog match" in r.note


def test_title_topic_narrows_before_sorting(mini_catalog):
    r = run("cheapest cookie mix", mini_catalog)
    assert all("Cookie" in t for t in titles(r))
    assert titles(r)[0] == "Gluten-Free Sugar Cookie Mix"
    assert r.total == 3


def test_combined_filter_and_sort(mini_catalog):
    r = run("cheapest gluten-free mix", mini_catalog)
    assert titles(r) == ["Gluten-Free Sugar Cookie Mix", "Fudge Brownie Mix"]


def test_rating_sort_is_review_weighted(mini_catalog):
    """5.0 from a single review must not outrank 4.9 from 300 reviews."""
    order = titles(run("top rated mixes", mini_catalog))
    assert order.index("Fudge Brownie Mix") < order.index("Confetti Sugar Cookie Mix")
    assert order.index("Fudge Brownie Mix") < order.index("Cupcake Party Kit")


def test_price_cap(mini_catalog):
    r = run("mixes under $7", mini_catalog)
    assert all(d.metadata["price_value"] <= 7 for d in r.docs)
    assert r.total == 3


def test_limit_and_more_not_shown_note(mini_catalog):
    _, spec = resolve_question("cheapest mix", [])
    r = structured_search(spec, mini_catalog, limit=2)
    assert len(r.docs) == 2 and r.total == len(mini_catalog)
    assert "more not shown" in r.note


def test_price_ties_broken_by_weighted_rating():
    tied = (
        make_doc("Mediocre Mix", 5.00, 3.5, 200),
        make_doc("Excellent Mix", 5.00, 4.9, 200),
        make_doc("Pricey Mix", 9.00, 5.0, 900),
    )

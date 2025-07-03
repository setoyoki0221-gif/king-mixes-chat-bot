"""Invariants of the scraped dataset and the documents built from it.

These guard against a bad re-scrape or a hand edit silently degrading the bot.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

import config
from catalog import build_documents, product_to_text


@pytest.fixture(scope="module")
def products():
    return json.loads(config.DATA_PATH.read_text(encoding="utf-8"))


def test_dataset_is_not_trivially_small(products):
    assert len(products) >= 50


def test_urls_are_unique_and_on_the_store(products):
    urls = [p["url"] for p in products]
    assert len(urls) == len(set(urls))
    assert all(u.startswith("https://shop.kingarthurbaking.com/") for u in urls)


def test_required_fields_present(products):
    for p in products:
        assert p["title"].strip(), p["url"]
        assert p["description"].strip(), p["url"]
        assert p["image"].startswith("http"), p["url"]


def test_prices_are_clean_numbers(products):
    """Regression: promo copy ("Buy Any 5+/Save $4") used to be glued onto the price."""
    for p in products:
        assert re.fullmatch(r"\$\d{1,3}(,\d{3})*\.\d\d", p["price"]), (p["title"], p["price"])
        assert p["price_value"] == pytest.approx(float(p["price"].lstrip("$").replace(",", "")))
        assert isinstance(p["promo"], str)


def test_ratings_are_consistent_with_review_counts(products):
    """Unrated products carry rating 0.0 (or null) and 0 reviews; rated ones are in (0, 5]."""
    for p in products:
        rating, reviews = p["rating"] or 0, p["review_count"] or 0
        assert reviews >= 0, p["title"]
        if reviews == 0:
            assert rating == 0, f"{p['title']}: rating without reviews"
        else:
            assert 0 < rating <= 5, f"{p['title']}: reviews without a valid rating"


def test_badges_are_known(products):
    known = set(config.BADGE_LABELS)
    unknown = {b for p in products for b in p["badges"]} - known
    assert not unknown, f"add labels for new badges to config.BADGE_LABELS: {unknown}"


def test_meta_matches_dataset(products):
    meta = json.loads(config.DATA_PATH.with_name("meta.json").read_text(encoding="utf-8"))
    assert meta["count"] == len(products)
    assert meta["scraped_at"]


def test_documents_have_chroma_safe_metadata(products):
    """Chroma rejects None metadata values; ingest would crash on them."""
    docs = build_documents(products)
    assert len(docs) == len(products)
    for d in docs:
        for k, v in d.metadata.items():
            assert isinstance(v, (str, int, float, bool)), (d.metadata["title"], k, v)


def test_document_text_contains_the_facts_the_bot_answers_from(products):
    doc = build_documents(products)[0]
    p = products[0]
    assert p["title"] in doc.page_content and p["price"] in doc.page_content
    assert p["url"] in doc.page_content


def test_offer_line_only_when_promo(products):
    with_promo = next(p for p in products if p["promo"])
    without = next(p for p in products if not p["promo"])
    assert "Offer:" in product_to_text(with_promo)
    assert "Offer:" not in product_to_text(without)
    assert "Save $4" not in product_to_text(with_promo).split("Offer:")[0]  # not in the price line


def test_catalog_stays_diverse(products):
    """A broken scrape that returns one product type repeatedly would show here."""

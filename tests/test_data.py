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

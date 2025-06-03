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

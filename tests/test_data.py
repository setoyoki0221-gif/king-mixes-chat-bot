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

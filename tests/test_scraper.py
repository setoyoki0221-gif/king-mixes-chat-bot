"""Scraper parsing and the guard that stops a bad scrape overwriting good data."""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

import scrape

PRODUCT_HTML = """
<html><head><meta property="og:image" content="https://img.example/x.jpg"></head><body>
  <h1 class="productView-title">Test Scone Mix</h1>
  <span data-product-price-without-tax>$8.95 Buy Any 5+/Save $4</span>
  <div id="tab-description"><p>A   great
  mix.</p></div>
  <div class="kab-product-rating" data-rating="4.8" data-reviews="12" data-product-id="77"></div>
  <dl>
    <dt>SKU:</dt><dd>S-1</dd>
    <dt>UPC:</dt><dd>U-1</dd>
    <dt>_badge_glutenfree</dt><dd>Yes</dd>
    <dt>_badge_vegan</dt><dd>No</dd>
    <dt>_badge_nongmo</dt><dd>yes</dd>
  </dl>
</body></html>
"""


# --------------------------------------------------------------------------- #
# parse_price
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, price, value, promo",
    [
        ("$8.95 Buy Any 5+/Save $4", "$8.95", 8.95, "Buy Any 5+/Save $4"),
        ("$4.99", "$4.99", 4.99, ""),
        ("  $1,234.50 ", "$1,234.50", 1234.5, ""),
        ("$5", "$5.00", 5.0, ""),
        ("Sale! $7.95", "$7.95", 7.95, "Sale!"),
        ("", "", None, ""),
        (None, "", None, ""),
        ("Call for price", "", None, "Call for price"),
    ],
)
def test_parse_price(raw, price, value, promo):
    assert scrape.parse_price(raw) == {"price": price, "price_value": value, "promo": promo}


def test_parse_price_is_idempotent_on_clean_prices():
    once = scrape.parse_price("$8.95 Buy Any 5+/Save $4")
    again = scrape.parse_price(once["price"])
    assert (again["price"], again["price_value"]) == (once["price"], once["price_value"])


# --------------------------------------------------------------------------- #
# parse_badges / parse_product
# --------------------------------------------------------------------------- #
def test_parse_badges_only_keeps_yes_values():
    soup = BeautifulSoup(PRODUCT_HTML, "html.parser")
    assert scrape.parse_badges(soup) == ["glutenfree", "nongmo"]


def test_parse_product_end_to_end(monkeypatch):
    class FakeResponse:
        text = PRODUCT_HTML

    monkeypatch.setattr(scrape, "get", lambda url: FakeResponse())
    p = scrape.parse_product("https://shop.example/items/test")

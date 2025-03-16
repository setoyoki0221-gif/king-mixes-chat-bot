"""
Scraper for the King Arthur Baking "Mixes" category.

Source: https://shop.kingarthurbaking.com/mixes  (BigCommerce Stencil storefront)

Strategy
--------
1. Walk the paginated category listing (?page=N) and collect every product URL.
2. Visit each product detail page and extract the rich, statically-rendered
   fields: title, price, marketing description, image, star rating, review
   count, SKU/UPC, and dietary badges (gluten-free, kosher, vegan, ...).
3. Write a clean, re-runnable JSON file to data/products.json.

The script is polite: a real User-Agent, a small delay between requests, and
retries on transient failures.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

BASE = "https://shop.kingarthurbaking.com"
CATEGORY_URL = f"{BASE}/mixes"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "products.json"
META_PATH = OUT_PATH.with_name("meta.json")  # when/where the data was scraped

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_DELAY = 1.0  # seconds between requests (be polite)
MAX_RETRIES = 3


def get(url: str) -> requests.Response:
    """GET with retries and a polite delay."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  ! attempt {attempt}/{MAX_RETRIES} failed for {url}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def discover_product_urls() -> list[str]:
    """Walk every listing page and return de-duplicated product URLs in order."""
    urls: list[str] = []
    seen: set[str] = set()
    page = 1
    while True:
        listing_url = f"{CATEGORY_URL}?page={page}"
        print(f"[listing] page {page}: {listing_url}")
        soup = BeautifulSoup(get(listing_url).text, "html.parser")
        cards = soup.select("article.card")
        if not cards:
            break

        found_this_page = 0
        for card in cards:
            link = card.select_one(".card-title a") or card.select_one("a[href*='/items/']")
            if not link or not link.get("href"):
                continue
            href = link["href"].split("?")[0]
            if href not in seen:
                seen.add(href)
                urls.append(href)
                found_this_page += 1

        print(f"  found {found_this_page} new products (running total {len(urls)})")
        time.sleep(REQUEST_DELAY)

        # Stop when the listing has no "next page" link beyond the current one.
        next_pages = {
            int(m.group(1))
            for a in soup.select("li.pagination-item a")
            if (m := re.search(r"page=(\d+)", a.get("href", "")))
        }
        if not next_pages or max(next_pages) <= page:
            break
        page += 1

    return urls


def clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


_PRICE_RE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)")


def parse_price(raw: str | None) -> dict[str, Any]:
    """Split a raw price string into a clean price, a number, and any promo text.

    The storefront sometimes appends offer copy to the price element, e.g.
    ``"$8.95 Buy Any 5+/Save $4"``. Left as-is, that text pollutes the embedded
    document, the LLM context and the product card, and blocks numeric
    comparison ("cheapest"). The first dollar amount is the price; whatever
    follows it is the promo.

    Returns ``{"price": "$8.95", "price_value": 8.95, "promo": "Buy Any 5+/Save $4"}``.
    ``price_value`` is ``None`` and ``price`` is ``""`` when no amount is found.
    """
    text = clean(raw)
    m = _PRICE_RE.search(text)
    if not m:
        return {"price": "", "price_value": None, "promo": text}
    value = float(m.group(1).replace(",", ""))
    promo = clean((text[: m.start()] + " " + text[m.end():]))
    return {"price": f"${value:,.2f}", "price_value": value, "promo": promo}


def parse_badges(soup: BeautifulSoup) -> list[str]:
    """Dietary / attribute badges live as _badge_<name> spec rows.

    Every product lists *all* possible badge metafields; only those whose value
    is "Yes" actually apply, so we filter on the adjacent <dd> value.
    """
    badges: set[str] = set()
    for dt in soup.select("dt"):
        key = dt.get_text(strip=True).lower()
        m = re.match(r"_badge_(\w+)", key)
        if not m:
            continue
        dd = dt.find_next_sibling("dd")
        value = clean(dd.get_text()).lower() if dd else ""
        if value in ("yes", "true", "1"):
            badges.add(m.group(1))
    return sorted(badges)


def parse_product(url: str) -> dict[str, Any]:
    soup = BeautifulSoup(get(url).text, "html.parser")

    title_el = soup.select_one("h1.productView-title") or soup.select_one("h1")
    title = clean(title_el.get_text()) if title_el else ""

    price_el = (
        soup.select_one("[data-product-price-without-tax]")
        or soup.select_one(".price--withoutTax")
        or soup.select_one(".productView-price .price")
        or soup.select_one(".productView-price")
    )
    price_info = parse_price(price_el.get_text() if price_el else "")

    desc_el = soup.select_one("#tab-description") or soup.select_one(".productView-description")
    description = clean(desc_el.get_text(" ")) if desc_el else ""

    rating_el = soup.select_one(".kab-product-rating, [data-rating]")
    rating = None
    review_count = None
    product_id = None
    if rating_el:
        try:
            rating = float(rating_el.get("data-rating")) if rating_el.get("data-rating") else None
            review_count = (
                int(rating_el.get("data-reviews")) if rating_el.get("data-reviews") else None
            )
            product_id = rating_el.get("data-product-id")
        except (TypeError, ValueError):
            pass

    # SKU / UPC from the spec definition list
    sku = upc = ""
    for dt in soup.select("dt"):
        label = dt.get_text(strip=True).lower()
        dd = dt.find_next_sibling("dd")
        val = clean(dd.get_text()) if dd else ""
        if label.startswith("sku"):
            sku = val
        elif label.startswith("upc"):
            upc = val

    og_img = soup.find("meta", {"property": "og:image"})
    image = og_img.get("content") if og_img else ""

    return {
        "title": title,
        "url": url,
        **price_info,  # price (clean "$x.xx"), price_value (float), promo (text)
        "description": description,
        "rating": rating,
        "review_count": review_count,
        "badges": parse_badges(soup),
        "sku": sku,
        "upc": upc,
        "product_id": product_id,
        "image": image,
        "category": "Mixes",
    }


MIN_PRODUCTS = 50          # a "Mixes" listing this small means the page layout changed
MIN_SUCCESS_RATIO = 0.9    # share of discovered URLs that must parse
MAX_INCOMPLETE_RATIO = 0.05  # share of products allowed to lack a title or price


def validate(products: list[dict[str, Any]], discovered: int) -> list[str]:
    """Return a list of problems; empty means the scrape is safe to keep.

    The scraper used to skip failing pages with only a printed warning, so a
    site change or outage could silently overwrite good data with a partial set.
    """
    problems: list[str] = []
    if discovered < MIN_PRODUCTS:
        problems.append(
            f"only {discovered} product URLs discovered (expected >= {MIN_PRODUCTS}); "
            "the listing page layout may have changed"
        )

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

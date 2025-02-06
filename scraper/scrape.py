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

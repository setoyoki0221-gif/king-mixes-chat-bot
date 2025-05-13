"""Shared fixtures: a small synthetic catalog so tests don't depend on live prices."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


def make_doc(title, price, rating=0.0, reviews=0, badges=(), promo="") -> Document:
    return Document(
        page_content=f"Product: {title}",
        metadata={
            "title": title,
            "url": f"https://shop.example/{title.lower().replace(' ', '-')}",
            "price": f"${price:.2f}",
            "promo": promo,
            "rating": rating,
            "review_count": reviews,
            "badges": ", ".join(badges),
            "image": "",
            "price_value": float(price),
        },
    )

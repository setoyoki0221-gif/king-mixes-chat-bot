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


@pytest.fixture
def mini_catalog() -> tuple[Document, ...]:
    return (
        make_doc("Gluten-Free Sugar Cookie Mix", 4.99, 4.8, 507, ["glutenfree", "kosherpareve"]),
        make_doc("Chocolate Chip Cookie Mix", 9.95, 4.7, 722, ["kosherdairy"]),
        make_doc("Confetti Sugar Cookie Mix", 6.95, 5.0, 1, ["kosherpareve"]),  # 5.0 from ONE review
        make_doc("Fudge Brownie Mix", 8.95, 4.9, 300, ["glutenfree"]),
        make_doc("Birthday Cake Mix", 7.95, 4.6, 120, ["kosherpareve"]),
        make_doc("Cupcake Party Kit", 66.27, 5.0, 1, []),
        make_doc("Pancake Mix", 6.95, 4.8, 1000, ["kosherdairy"], promo="Buy Any 5+/Save $4"),
    )

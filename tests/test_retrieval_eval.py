"""Retrieval quality against the REAL vector store and embedding model.

Slow (loads the embedding model once), so it carries the `retrieval` marker:
    pytest -m "not retrieval"     # skip these
    pytest -m retrieval           # only these

No chat LLM is used; where the graph runs, a stub stands in for it.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agent
import config

pytestmark = pytest.mark.retrieval


@pytest.fixture(scope="module")
def store():
    if not config.CHROMA_DIR.exists():
        pytest.skip("chroma_db not built — run `python ingest.py`")
    return agent.get_vectorstore()


# --------------------------------------------------------------------------- #
# Index freshness
# --------------------------------------------------------------------------- #
def test_index_matches_the_dataset(store, real_catalog):
    """Fails if products.json changed but `python ingest.py` wasn't re-run."""
    stored = store.get(include=["metadatas"])["metadatas"]
    assert len(stored) == len(real_catalog), "document count differs: re-run ingest.py"

    by_url = {m["url"]: m for m in stored}
    for d in real_catalog:
        m = by_url.get(d.metadata["url"])
        assert m is not None, f"not indexed: {d.metadata['title']}"
        assert m["price"] == d.metadata["price"], f"stale price for {d.metadata['title']}: re-run ingest.py"
        assert m.get("price_value") == d.metadata.get("price_value")


# --------------------------------------------------------------------------- #
# Golden questions (semantic path): the right kind of product must appear in the top 6
# --------------------------------------------------------------------------- #
GOLDEN = [
    # (question, word expected in titles, minimum matches among the top 6)
    ("What cookie mixes do you have?", "cookie", 4),
    ("gluten free pancake mix", "pancake", 2),
    ("brownie mix", "brownie", 2),
    ("scone mix", "scone", 4),
    ("pizza crust mix", "pizza", 1),
    ("muffin mix", "muffin", 4),
    ("a mix for making bread", "bread", 2),
    ("waffle mix", "waffle", 1),
    ("cornbread", "cornbread", 1),
    ("popovers", "popover", 1),
]

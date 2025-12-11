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

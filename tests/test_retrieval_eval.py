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


@pytest.mark.parametrize("question, word, min_hits", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_semantic_search_finds_the_right_kind_of_product(store, question, word, min_hits):
    docs = store.similarity_search(question, k=agent.RETRIEVE_K)
    hits = sum(word in d.metadata["title"].lower() for d in docs)
    assert hits >= min_hits, f"only {hits}/{len(docs)} titles mention {word!r}: " + str(
        [d.metadata["title"] for d in docs]
    )


# --------------------------------------------------------------------------- #
# Hybrid behaviour through the real graph (stub LLM)
# --------------------------------------------------------------------------- #
class StubLLM:
    def __init__(self):
        self.system_prompts = []

    def runnable(self):
        from langchain_core.runnables import RunnableLambda

        def call(pv):
            self.system_prompts.append(pv.to_messages()[0].content)
            return AIMessage(content="stub")

        return RunnableLambda(call)


@pytest.fixture
def graph(monkeypatch, store):
    from langchain_core.runnables import RunnableLambda

    gen = StubLLM()
    monkeypatch.setattr(agent, "_gen_llm", gen.runnable())
    monkeypatch.setattr(agent, "_short_llm", RunnableLambda(lambda pv: AIMessage(content="YES")))
    g = agent.build_graph(fast=False)
    g.gen = gen
    return g


def test_topic_without_a_title_match_ranks_semantically_then_sorts_by_price(graph):
    out = agent.ask(graph, "cheapest mix for a kid's birthday party")
    prices = [float(s["price"].lstrip("$")) for s in out["sources"]]
    assert 3 <= len(prices) <= 8, "a single weak word match must not collapse the result"
    assert prices == sorted(prices)
    assert prices[0] < 20, "the cheapest party-ish mix should be an ordinary mix, not a $60+ kit"


def test_followup_keeps_the_conversation_topic_for_semantic_questions(graph):
    history = [HumanMessage(content="What cookie mixes do you have?"), AIMessage(content="Several.")]

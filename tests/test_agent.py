"""Agent behaviour with stub LLMs and fake vector stores (no model, no embeddings)."""

from __future__ import annotations

import pytest
from conftest import make_doc
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import agent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class Recorder:
    """Stub chat model: records every prompt it receives and returns a reply."""

    def __init__(self, reply="OK"):
        self.reply = reply
        self.prompts: list[list] = []

    def __call__(self, prompt_value):
        msgs = prompt_value.to_messages()
        self.prompts.append(msgs)
        text = self.reply(msgs) if callable(self.reply) else self.reply
        return AIMessage(content=text)

    def runnable(self):
        return RunnableLambda(self)


def make_state(text="q", history=(), **over):
    """Agent state for a user turn `text`; `over` may override any field
    (e.g. question=<a rewritten query>, rewrites=1)."""
    state = {
        "messages": [*history, HumanMessage(content=text)],
        "question": text,
        "original_question": text,
        "documents": [],
        "note": "",
        "structured": False,
        "relevant": True,
        "generation": "",
        "rewrites": 0,
    }
    state.update(over)
    return state


def no_embeddings(monkeypatch):
    def boom():
        raise AssertionError("this path must not touch the vector store / embeddings")

    monkeypatch.setattr(agent, "get_vectorstore", boom)


class FakeStore:
    def __init__(self, docs):
        self.docs, self.queries = docs, []

    def similarity_search(self, text, k=4):
        self.queries.append((text, k))
        return self.docs[:k]


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def test_generate_prompt_passes_braces_through_verbatim():
    """Regression: '{...}' in chat history used to raise KeyError."""
    history = [HumanMessage(content="Can I use {flour} and {{sugar}}? {"), AIMessage(content="Sure {x} }")]
    msgs = agent.GENERATE_PROMPT.invoke(
        {"context": "Product: A {braces} too", "history": history, "question": "And {this}?"}
    ).to_messages()
    assert [m.type for m in msgs] == ["system", "human", "ai", "human"]
    assert msgs[1].content == "Can I use {flour} and {{sugar}}? {"
    assert "A {braces} too" in msgs[0].content
    assert msgs[3].content == "And {this}?"


def test_generate_trims_history_and_prepends_lookup_note(monkeypatch):
    rec = Recorder("final")
    monkeypatch.setattr(agent, "_gen_llm", rec.runnable())
    history = []
    for i in range(10):
        history += [HumanMessage(content=f"q{i} {{x}}"), AIMessage(content=f"a{i}")]
    state = make_state("now", history, documents=[make_doc("A Mix", 5.0)], note="CATALOG LOOKUP: note",
                       structured=True)

    out = agent.generate(state)

    sent = rec.prompts[0]
    assert len(sent) == 1 + agent.MAX_HISTORY_MESSAGES + 1          # system + trimmed history + question
    assert sent[1].content == "q7 {x}"                              # oldest kept = 6th from the end
    assert sent[0].content.split("Context:\n", 1)[1].startswith("CATALOG LOOKUP: note")
    assert out["generation"] == "final"
    assert out["messages"][0].content == "final"


def test_generate_with_no_documents_says_so(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(agent, "_gen_llm", rec.runnable())
    agent.generate(make_state("anything"))
    assert "no matching products were found" in rec.prompts[0][0].content


# --------------------------------------------------------------------------- #
# Grading and routing
# --------------------------------------------------------------------------- #
def test_grade_skips_the_llm_for_exact_lookups(monkeypatch):
    def explode():
        raise AssertionError("grader LLM must not be called for a structured lookup")

    monkeypatch.setattr(agent, "short_llm", explode)
    assert agent.grade_documents(make_state(structured=True, documents=[])) == {"relevant": True}


def test_grade_no_documents_is_irrelevant():
    assert agent.grade_documents(make_state(documents=[])) == {"relevant": False}


@pytest.mark.parametrize(
    "reply, relevant",
    [("YES", True), ("yes.", True), ("NO", False), ("No, these are unrelated", False),
     ("Not sure", True),   # lenient: only a leading "no" rejects
     ("", True)],
)
def test_grade_parsing_is_lenient(monkeypatch, reply, relevant):
    monkeypatch.setattr(agent, "_short_llm", Recorder(reply).runnable())
    state = make_state(documents=[make_doc("A Mix", 5.0)])
    assert agent.grade_documents(state) == {"relevant": relevant}


def test_grade_failure_never_blocks_the_answer(monkeypatch):
    def broken(_pv):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(agent, "_short_llm", RunnableLambda(broken))
    assert agent.grade_documents(make_state(documents=[make_doc("A", 1.0)])) == {"relevant": True}


def test_decide_after_grading():
    assert agent.decide_after_grading(make_state(relevant=True)) == "generate"
    assert agent.decide_after_grading(make_state(relevant=False, rewrites=0)) == "rewrite"
    assert agent.decide_after_grading(make_state(relevant=False, rewrites=agent.MAX_REWRITES)) == "generate"


# --------------------------------------------------------------------------- #
# Rewriting
# --------------------------------------------------------------------------- #
def test_rewrite_is_told_what_failed_and_counts_an_attempt(monkeypatch):
    rec = Recorder('"gluten free baking mix"')
    monkeypatch.setattr(agent, "_short_llm", rec.runnable())
    out = agent.rewrite_query(make_state("gf stuff", question="gf stuff", rewrites=0))
    assert out == {"question": "gluten free baking mix", "rewrites": 1}  # quotes stripped
    human = rec.prompts[0][1].content
    assert "gf stuff" in human and "Previous query (failed)" in human


def test_second_rewrite_sees_the_first_rewrite_not_the_original(monkeypatch):
    rec = Recorder("third try")
    monkeypatch.setattr(agent, "_short_llm", rec.runnable())
    agent.rewrite_query(make_state("orig", question="first rewrite", rewrites=1))
    assert "first rewrite" in rec.prompts[0][1].content


@pytest.mark.parametrize("reply", ["same query", "  SAME QUERY ", ""])
def test_unchanged_rewrite_spends_the_remaining_budget(monkeypatch, reply):
    monkeypatch.setattr(agent, "_short_llm", Recorder(reply).runnable())
    out = agent.rewrite_query(make_state("orig", question="same query", rewrites=0))
    assert out["rewrites"] == agent.MAX_REWRITES and out["question"] == "same query"


# --------------------------------------------------------------------------- #
# Retrieval routing
# --------------------------------------------------------------------------- #
def test_structured_followup_uses_history_and_no_embeddings(monkeypatch, real_catalog):
    no_embeddings(monkeypatch)
    hist = [HumanMessage(content="What cookie mixes do you have?"), AIMessage(content="Several.")]
    out = agent.retrieve(make_state("What's the cheapest one?", hist))

    assert out["structured"] is True and "CATALOG LOOKUP" in out["note"]
    assert out["documents"], "expected cookie mixes"
    assert all("cookie" in d.metadata["title"].lower() for d in out["documents"])
    cookie_prices = [d.metadata["price_value"] for d in real_catalog if "cookie" in d.metadata["title"].lower()]
    assert out["documents"][0].metadata["price_value"] == min(cookie_prices)


def test_semantic_question_uses_vector_store_at_k6(monkeypatch):
    store = FakeStore([make_doc(f"Mix {i}", 5.0 + i) for i in range(10)])
    monkeypatch.setattr(agent, "get_vectorstore", lambda: store)
    out = agent.retrieve(make_state("What cookie mixes do you have?"))
    assert store.queries == [("What cookie mixes do you have?", agent.RETRIEVE_K)]
    assert out["structured"] is False and out["note"] == "" and len(out["documents"]) == agent.RETRIEVE_K


def test_semantic_followup_embeds_earlier_topic_too(monkeypatch):
    store = FakeStore([make_doc("Mix", 5.0)])
    monkeypatch.setattr(agent, "get_vectorstore", lambda: store)
    hist = [HumanMessage(content="What cookie mixes do you have?"), AIMessage(content="Several.")]
    agent.retrieve(make_state("Which is best for beginners?", hist))
    assert "cookie" in store.queries[0][0] and "beginners" in store.queries[0][0]


def test_after_a_rewrite_the_query_goes_straight_to_semantic_search(monkeypatch):
    """A rewritten query is a search string, not a user question: no structured parsing."""
    store = FakeStore([make_doc("Mix", 5.0)])
    monkeypatch.setattr(agent, "get_vectorstore", lambda: store)
    out = agent.retrieve(make_state("orig", question="cheapest budget gluten free", rewrites=1))
    assert store.queries[0][0] == "cheapest budget gluten free"
    assert out["structured"] is False


# --------------------------------------------------------------------------- #
# Whole graph (stub LLMs)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fast", [True, False])
def test_graph_exact_lookup_never_calls_grader_or_rewriter(monkeypatch, real_catalog, fast):
    no_embeddings(monkeypatch)
    gen, short = Recorder("ANSWER"), Recorder("NO")
    monkeypatch.setattr(agent, "_gen_llm", gen.runnable())
    monkeypatch.setattr(agent, "_short_llm", short.runnable())

    out = agent.ask(agent.build_graph(fast=fast), "What's the cheapest mix?")

    assert out["answer"] == "ANSWER"
    assert short.prompts == [] and len(gen.prompts) == 1
    cheapest = min(d.metadata["price_value"] for d in real_catalog)
    assert out["sources"][0]["price"] == f"${cheapest:.2f}"
    assert "CATALOG LOOKUP" in gen.prompts[0][0].content


def test_graph_unmatched_attribute_answers_once_without_retry_loop(monkeypatch):
    no_embeddings(monkeypatch)
    gen, short = Recorder("None found."), Recorder("NO")
    monkeypatch.setattr(agent, "_gen_llm", gen.runnable())
    monkeypatch.setattr(agent, "_short_llm", short.runnable())

    out = agent.ask(agent.build_graph(fast=False), "Do you have vegan options?")

    assert out["sources"] == [] and short.prompts == [] and len(gen.prompts) == 1
    assert "no products in the catalog match" in gen.prompts[0][0].content


def test_graph_smart_mode_retries_then_answers_with_the_documents_it_has(monkeypatch):
    """Grader always says NO: rewrite twice, then still answer from retrieved docs."""
    docs = [make_doc("Sugar Cookie Mix", 8.95), make_doc("Brownie Mix", 8.95)]
    store = FakeStore(docs)
    monkeypatch.setattr(agent, "get_vectorstore", lambda: store)

    def short_reply(msgs):
        if "ON-TOPIC" in msgs[0].content:        # grading prompt
            return "NO"
        return f"rewritten query {len(store.queries)}"   # rewrite prompt: always something new

    short, gen = Recorder(short_reply), Recorder("done")
    monkeypatch.setattr(agent, "_short_llm", short.runnable())
    monkeypatch.setattr(agent, "_gen_llm", gen.runnable())

    out = agent.ask(agent.build_graph(fast=False), "tell me about stuff")

    assert [q for q, _ in store.queries] == [
        "tell me about stuff", "rewritten query 1", "rewritten query 2",
    ]
    assert len(gen.prompts) == 1
    assert [s["title"] for s in out["sources"]] == ["Sugar Cookie Mix", "Brownie Mix"]  # never discarded


def test_graph_fast_mode_skips_grading(monkeypatch):
    store = FakeStore([make_doc("Sugar Cookie Mix", 8.95)])
    monkeypatch.setattr(agent, "get_vectorstore", lambda: store)
    short, gen = Recorder("NO"), Recorder("done")
    monkeypatch.setattr(agent, "_short_llm", short.runnable())
    monkeypatch.setattr(agent, "_gen_llm", gen.runnable())

    graph = agent.build_graph(fast=True)
    assert set(graph.get_graph().nodes) == {"__start__", "__end__", "retrieve", "generate"}
    agent.ask(graph, "cookie mixes")
    assert short.prompts == []


def test_graph_survives_braces_in_the_question_and_history(monkeypatch):
    store = FakeStore([make_doc("Sugar Cookie Mix", 8.95)])
    monkeypatch.setattr(agent, "get_vectorstore", lambda: store)
    monkeypatch.setattr(agent, "_gen_llm", Recorder("ok").runnable())
    hist = [HumanMessage(content="what is {this}"), AIMessage(content="a {thing}")]
    out = agent.ask(agent.build_graph(fast=True), "and {that}?", history=hist)
    assert out["answer"] == "ok"


def test_build_graph_defaults_to_config_but_param_wins(monkeypatch):
    monkeypatch.setattr(agent.config, "AGENT_FAST", True)
    assert "grade" not in agent.build_graph().get_graph().nodes
    assert "grade" in agent.build_graph(fast=False).get_graph().nodes


# --------------------------------------------------------------------------- #
# Source cards
# --------------------------------------------------------------------------- #
def src(title, url=None):
    return {"title": title, "url": url or f"https://x/{title}"}


def names(found):
    return [s["title"] for s in found]


def test_referenced_sources_ignores_punctuation_and_markdown():
    sources = [src("Gluten-Free Sugar Cookie Mix"), src("Pancake Mix")]
    answer = "Try the **gluten free sugar cookie mix** — only $4.99."
    assert names(agent.referenced_sources(answer, sources)) == ["Gluten-Free Sugar Cookie Mix"]


def test_referenced_sources_treats_ampersand_as_and():
    sources = [src("Confetti Cake & Cupcake Mix")]
    assert names(agent.referenced_sources("Go with the Confetti Cake and Cupcake Mix.", sources))


def test_referenced_sources_drops_pack_size_suffix():
    sources = [src("Gluten-Free Single Serve Brownie Mix - 12-Pack")]
    assert names(agent.referenced_sources("The Gluten-Free Single Serve Brownie Mix is handy.", sources))


def test_referenced_sources_needs_whole_words():
    assert agent.referenced_sources("I like the cupcake mix.", [src("Cake Mix")]) == []


def test_referenced_sources_prefers_the_longer_cited_title():
    sources = [src("Cookie Mix"), src("Outrageous Chocolate Chip Cookie Mix")]

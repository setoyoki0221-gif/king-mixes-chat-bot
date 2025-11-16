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

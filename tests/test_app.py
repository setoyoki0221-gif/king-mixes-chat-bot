"""Run the real Streamlit script headlessly (stub LLM) and check what the user would see."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

import agent
import config

ANSWER = "The cheapest is the Gluten-Free Sugar Cookie Mix at $4.99."


@pytest.fixture
def app(monkeypatch):
    def boom():
        raise AssertionError("a plain 'cheapest' question must not need embeddings")

    monkeypatch.setattr(agent, "_gen_llm", RunnableLambda(lambda pv: AIMessage(content=ANSWER)))
    monkeypatch.setattr(agent, "_short_llm", RunnableLambda(lambda pv: AIMessage(content="YES")))
    monkeypatch.setattr(agent, "get_vectorstore", boom)
    return AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120)


def page_html(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_first_load_renders_without_errors(app):
    at = app.run()
    assert not at.exception, [e.value for e in at.exception]
    html = page_html(at)
    assert "King Arthur Mixes Assistant" in html
    assert "Prices and ratings as of" in html            # data freshness is shown
    assert [b.label for b in at.button if "Gluten free" in b.label]   # suggestion chips present


def test_a_question_produces_an_answer_and_a_product_card(app):
    at = app.run()
    at.session_state["pending"] = "What's the cheapest mix?"
    at = at.run()

    assert not at.exception, [e.value for e in at.exception]
    chat = at.session_state["chat"]
    assert [t["role"] for t in chat] == ["user", "assistant"]
    assert chat[1]["content"] == ANSWER
    # Only the product the answer names gets a card.
    assert [s["title"] for s in chat[1]["sources"]] == ["Gluten-Free Sugar Cookie Mix"]

    html = page_html(at)

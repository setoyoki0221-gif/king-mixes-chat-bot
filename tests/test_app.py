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

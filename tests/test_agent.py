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

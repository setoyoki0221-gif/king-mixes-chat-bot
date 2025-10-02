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

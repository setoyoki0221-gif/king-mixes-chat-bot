"""Run the real Streamlit script headlessly (stub LLM) and check what the user would see."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

import agent
import config

ANSWER = "The cheapest is the Gluten-Free Sugar Cookie Mix at $4.99."

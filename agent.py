"""Self-correcting RAG agent for King Arthur Baking mixes, built with LangGraph.

This is an *agentic* retrieval pipeline, not a single retrieve-then-answer
chain. The graph reasons about whether the documents it pulled are actually
relevant and, if not, rewrites the query and tries again before answering:

    retrieve -> grade -> (relevant?) -- yes --> generate -> END
                              |
                              no, and retries left
                              v
                          rewrite -> retrieve (loop)

Grading and rewriting give the agent "advanced knowledge retrieval and
reasoning": it self-checks retrieval quality and reformulates poor queries.

Retrieval is hybrid. Aggregate questions ("cheapest", "top rated", "all
gluten-free mixes", "under $7") are answered exactly by filtering/sorting the
whole catalog (see catalog.py) because top-k similarity search cannot rank or
count. Everything else uses semantic search. Short follow-ups ("What's the
cheapest one?") are resolved against earlier questions before either path runs.

The compiled graph is exposed via build_graph() so the frontend can render
its structure (graph.get_graph().draw_mermaid_png()).
"""

from __future__ import annotations

import re
import threading
from typing import Annotated, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import config
from catalog import QuerySpec, get_catalog, resolve_question, structured_search
from providers import get_embeddings, get_llm

MAX_REWRITES = 2
RETRIEVE_K = 6
MAX_HISTORY_MESSAGES = 6  # prior turns sent to the LLM (keeps small models fast/on-topic)


def _message_text(resp) -> str:
    """Extract plain text from an LLM response (AIMessage or raw string)."""
    return resp.content if isinstance(resp, AIMessage) else str(resp)

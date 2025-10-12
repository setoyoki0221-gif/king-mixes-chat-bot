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


def _format_context(documents: list[Document], sep: str = "\n\n") -> str:
    """Join retrieved product documents into a single context block."""
    return sep.join(d.page_content for d in documents)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]  # full chat history
    question: str            # the (possibly rewritten) query used for retrieval
    original_question: str   # what the user actually asked
    documents: list[Document]
    note: str                # how the documents were chosen (structured lookups only)
    structured: bool         # True when documents came from an exact catalog lookup
    relevant: bool           # grader verdict on the current documents
    generation: str
    rewrites: int


# --------------------------------------------------------------------------- #
# Lazily-constructed singletons (built once, reused across turns)
# --------------------------------------------------------------------------- #
_vectorstore: Chroma | None = None
_gen_llm = None
_short_llm = None
# Streamlit serves each browser session on its own thread, so two first visitors
# can reach these lazy getters at once. The lock (double-checked below) makes sure
# only one of them builds the object.
_init_lock = threading.RLock()


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        with _init_lock:
            if _vectorstore is None:
                _vectorstore = Chroma(
                    collection_name=config.COLLECTION_NAME,
                    persist_directory=str(config.CHROMA_DIR),
                    embedding_function=get_embeddings(),
                )
    return _vectorstore


def gen_llm():
    """LLM for writing the final answer (longer output budget)."""
    global _gen_llm
    if _gen_llm is None:
        with _init_lock:
            if _gen_llm is None:
                _gen_llm = get_llm(temperature=0.2)
    return _gen_llm


def short_llm():
    """LLM for grade/rewrite — greedy and tiny output, so it's fast on CPU."""
    global _short_llm

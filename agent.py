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
    if _short_llm is None:
        with _init_lock:
            if _short_llm is None:
                _short_llm = get_llm(temperature=0.0, max_new_tokens=24)
    return _short_llm


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
def _semantic_rank(text: str) -> list[Document]:
    """Every catalog product ordered by similarity to `text`."""
    return get_vectorstore().similarity_search(text, k=len(get_catalog()))


def retrieve(state: AgentState) -> dict:
    """Hybrid retrieval: exact catalog lookup for aggregate questions, else semantic.

    On the first pass the question is resolved against earlier user turns so a
    follow-up keeps its topic/filters. After a rewrite, `question` is already a
    standalone search string, so it goes straight to semantic search.
    """
    if state["rewrites"] == 0:
        prior = [str(m.content) for m in state["messages"][:-1] if m.type == "human"]
        text, spec = resolve_question(state["original_question"], prior)
    else:
        text, spec = state["question"], QuerySpec()

    if spec.structured:
        found = structured_search(
            spec, get_catalog(), semantic_rank=_semantic_rank, query_text=text
        )
        return {"documents": found.docs, "note": found.note, "structured": True}

    docs = get_vectorstore().similarity_search(text, k=RETRIEVE_K)
    return {"documents": docs, "note": "", "structured": False}


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether a set of retrieved baking-mix products is "
            "ON-TOPIC for the user's question. Judge the TOPIC, not whether the "
            "answer is spelled out.\n"
            "- Answer YES if the products are the kind of thing the user asked "
            "about (e.g. cookie mixes for a question about cookies, or any mixes "
            "for a question about prices/ratings). For 'cheapest', 'best', or "
            "'list' questions, relevant products are always YES.\n"
            "- Answer NO only if the products are about something COMPLETELY "
            "different from the question.\n"
            "Reply with exactly one word: YES or NO.",
        ),
        ("human", "Question: {q}\n\nProducts:\n{ctx}\n\nOn-topic? YES or NO."),
    ]
)


def grade_documents(state: AgentState) -> dict:
    """Ask the LLM whether the retrieved products are relevant to the question.

    Reasoning step: if retrieval missed, we'd rather rewrite than answer from
    irrelevant context. Uses a simple YES/NO completion so it works across all
    providers (no tool-calling required).
    """
    if state.get("structured"):
        # An exact lookup is already correct by construction (an empty result
        # means "nothing matches", not "bad search"), so don't second-guess it.
        return {"relevant": True}
    if not state["documents"]:
        return {"relevant": False}
    context = _format_context(state["documents"])
    try:
        resp = (GRADE_PROMPT | short_llm()).invoke(
            {"q": state["original_question"], "ctx": context}
        )
        text = _message_text(resp)
        # Lenient parse: only treat as irrelevant when the FIRST word is "no".
        # (Avoids "not", "none", etc. being misread as a rejection.)
        words = re.findall(r"[a-z]+", text.lower())
        relevant = not (words and words[0] == "no")
    except Exception:
        # If grading fails for any reason, don't block the answer.
        relevant = True
    # NOTE: never discard the documents — grading only decides whether to try a
    # better search. Generation always keeps the retrieved products as context.
    return {"relevant": relevant}


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write search queries for a catalog of baking mixes. The previous "
            "query did not retrieve relevant products. Write a DIFFERENT query for "
            "the user's question: expand abbreviations, add synonyms for key "
            "concepts, and focus on product attributes (flavor, type, dietary "
            "needs). Return only the new query.",
        ),
        ("human", "Question: {q}\nPrevious query (failed): {prev}"),
    ]
)


def rewrite_query(state: AgentState) -> dict:
    """Reformulate the question to improve retrieval.

    The model is greedy (temperature 0), so asking it to rewrite the same input
    twice would return the same query. Each attempt therefore sees the query that
    just failed and is told to produce a different one. If it still comes back
    unchanged, further retries are pointless, so the rewrite budget is spent.
    """
    previous = state["question"]
    resp = (REWRITE_PROMPT | short_llm()).invoke(
        {"q": state["original_question"], "prev": previous}
    )
    new_q = _message_text(resp).strip().strip("\"'")
    if not new_q or new_q.lower() == previous.strip().lower():
        return {"question": previous, "rewrites": MAX_REWRITES}
    return {"question": new_q, "rewrites": state["rewrites"] + 1}


GENERATE_SYSTEM = """You are the King Arthur Baking mixes assistant. Answer the \
user's question using ONLY the product information in the context below. The \
context is the catalog's "Mixes" category.

Guidelines:
- Keep the answer SHORT: at most 3-4 sentences or a short bulleted list. Do not \
repeat the question or pad the answer.
- Recommend specific products by name with price and dietary attributes (e.g. \
gluten-free, kosher) when relevant.
- For "cheapest/most expensive" questions, COMPARE the Price values in the context \
and pick the correct one. For "best/highest rated" or "most reviews" questions, \
COMPARE the Rating and review counts. Read the numbers carefully.
- If the user asks about dietary needs, only claim an attribute if the product's \
listed attributes confirm it.
- If the context does not contain the answer, say you couldn't find a matching \
mix rather than inventing one.
- If the context starts with a CATALOG LOOKUP note, the products were already \
filtered and sorted over the WHOLE catalog. Trust that order and those counts: \
when the note says the first product is the cheapest/top rated, that is your \
answer. When listing, mention how many matched in total.

Context:
{context}"""

# Prior turns are injected through a MessagesPlaceholder, so they are passed as
# message objects and never parsed as template text. (Inlining them as
# (role, content) tuples made any "{...}" the user typed raise a KeyError.)
GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATE_SYSTEM),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def generate(state: AgentState) -> dict:
    """Produce the grounded final answer."""
    if state["documents"]:
        sep = "\n" if state.get("structured") else "\n\n"  # compact lines vs full docs
        context = _format_context(state["documents"], sep)
    else:
        context = "(no matching products were found in the catalog)"
    if state.get("note"):
        context = f"{state['note']}\n\n{context}"

    answer = (GENERATE_PROMPT | gen_llm()).invoke(
        {
            "context": context,
            # prior turns (the current question is last), newest few only
            "history": state["messages"][:-1][-MAX_HISTORY_MESSAGES:],
            "question": state["original_question"],
        }
    )
    text = _message_text(answer)
    return {"generation": text, "messages": [AIMessage(content=text)]}


# --------------------------------------------------------------------------- #
# Conditional routing
# --------------------------------------------------------------------------- #
def decide_after_grading(state: AgentState) -> str:
    """Decide whether to answer now or try a better search.

    We answer when the documents are judged relevant OR we've exhausted our
    rewrite budget. Crucially, the retrieved documents are always kept, so a
    misfiring grader can only cost an extra search — never an empty answer.
    """
    if state.get("relevant", True):
        return "generate"
    if state["rewrites"] >= MAX_REWRITES:
        return "generate"  # out of retries; answer from what we have
    return "rewrite"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph(fast: bool | None = None):
    """Compile and return the LangGraph agent.

    Smart graph self-corrects: retrieve -> grade -> rewrite/generate. Fast graph
    collapses to retrieve -> generate for lower latency. ``fast`` defaults to the
    AGENT_FAST setting, so callers can choose per graph without touching config.
    """
    fast = config.AGENT_FAST if fast is None else fast
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")

    if fast:
        g.add_edge("retrieve", "generate")
    else:
        g.add_node("grade", grade_documents)
        g.add_node("rewrite", rewrite_query)
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade",
            decide_after_grading,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        g.add_edge("rewrite", "retrieve")

    g.add_edge("generate", END)

"""Streamlit frontend for the King Arthur Baking "Mixes" assistant.

A modern chat-app UI: nav sidebar, greeting, suggestion chips, styled product
cards, and follow-up chips — over a LangGraph self-correcting RAG agent.

Run locally:   streamlit run app.py   (or  .\\run.bat)
Deployed on:   Hugging Face Spaces (Streamlit SDK).
"""

from __future__ import annotations

import html
import json
from datetime import datetime

import markdown as md
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage

import config
from agent import ask, build_graph, referenced_sources
from config import BADGE_LABELS

st.set_page_config(
    page_title="King Arthur · Baking Assistant",
    page_icon="🧁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root{
  --bg:#F7F6F4; --side:#FFFFFF; --ink:#2B2622; --muted:#8C8379;
  --amber:#B5651D; --amber2:#D98A4A; --soft:#FBE7DA; --line:#EEE9E3;
  --star:#F5A623; --green:#2F7D43; --green-bg:#EAF5EC;
  --shadow:0 6px 22px rgba(43,38,34,.07);
}
.stApp{ background:var(--bg); }
header[data-testid="stHeader"]{ display:none; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:1.4rem; padding-bottom:7rem; max-width:1180px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{ background:var(--side); border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container{ padding-top:1rem; }
.side-brand{ display:flex; align-items:center; gap:11px; margin:2px 4px 14px; }
.side-logo{ width:42px;height:42px;border-radius:50%;
  background:radial-gradient(circle at 30% 30%, #FAD9C8, #F3B894);
  display:flex;align-items:center;justify-content:center;font-size:22px; }
.side-brand b{ font-size:1.02rem;color:var(--ink);display:block;line-height:1.1; }
.side-brand span{ font-size:.78rem;color:var(--muted); }
.side-h{ font-size:.72rem;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
  font-weight:700;margin:16px 6px 4px; }
.recent{ display:flex;align-items:center;gap:10px;padding:7px 8px;border-radius:10px;color:#6f665d; }
.recent:hover{ background:#FAF7F2; }
.recent .rc-ic{ font-size:.9rem;opacity:.7; }
.recent .rc-t{ font-size:.84rem;font-weight:500;color:var(--ink);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:165px; }
.side-sep{ height:1px;background:var(--line);margin:16px 4px 6px; }

/* ---------- Welcome / empty state ---------- */
.welcome{ text-align:center;margin:5vh 0 6px; }
.w-logo{ width:74px;height:74px;border-radius:50%;margin:0 auto 14px;
  background:radial-gradient(circle at 30% 30%,#FAD9C8,#F0AE7E);
  display:flex;align-items:center;justify-content:center;font-size:38px;
  box-shadow:var(--shadow); }
.welcome h1{ margin:0;font-size:1.85rem;font-weight:800;color:var(--ink);letter-spacing:-.5px; }
.welcome p{ margin:8px auto 0;color:var(--muted);font-size:1rem;max-width:560px; }
.chips-label{ text-align:center;color:var(--muted);font-size:.82rem;font-weight:600;
  margin:22px 0 10px; }

/* ---------- Chat bubbles ---------- */
.row-user{ display:flex;justify-content:flex-end;margin:12px 0; }
.b-user{ background:var(--soft);color:var(--ink);padding:11px 15px;border-radius:16px 16px 4px 16px;
  max-width:72%;box-shadow:var(--shadow);font-size:.96rem;line-height:1.45; }
.row-bot{ display:flex;gap:11px;margin:12px 0;align-items:flex-start; }
.b-ava{ width:38px;height:38px;border-radius:50%;flex:0 0 38px;
  background:radial-gradient(circle at 30% 30%,#FAD9C8,#F3B894);
  display:flex;align-items:center;justify-content:center;font-size:19px;margin-top:2px; }
.b-bot{ background:#fff;color:var(--ink);padding:12px 16px;border-radius:4px 16px 16px 16px;
  max-width:78%;box-shadow:var(--shadow);font-size:.96rem;line-height:1.5;border:1px solid var(--line); }
.b-bot p:first-child{ margin-top:0; } .b-bot p:last-child{ margin-bottom:0; }
.b-time{ font-size:.7rem;color:var(--muted);margin-top:6px;text-align:right; }
.b-time .rr{ color:#36C26B; }

/* ---------- Product cards (premium / glassmorphism) ---------- */
.pgrid{ display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:20px;
  margin:10px 0 6px 49px; }
.pcard{ position:relative;background:rgba(255,255,255,.88);backdrop-filter:blur(18px);
  border:1px solid rgba(255,255,255,.6);border-radius:26px;overflow:hidden;
  box-shadow:0 18px 50px rgba(17,24,39,.09);display:flex;flex-direction:column;
  transition:transform .35s ease, box-shadow .35s ease; }
.pcard:hover{ transform:translateY(-8px);box-shadow:0 30px 70px rgba(17,24,39,.16); }
.pc-img{ padding:24px;background:linear-gradient(180deg,#fafafa,#ffffff); }
.pc-img img{ width:100%;aspect-ratio:1;object-fit:contain;transition:.4s; }
.pcard:hover .pc-img img{ transform:scale(1.06); }
.pc-content{ padding:20px 22px 22px;display:flex;flex-direction:column;flex:1; }
.pc-badge{ align-self:flex-start;display:inline-flex;padding:6px 12px;border-radius:999px;
  background:#ECFDF3;color:#15803D;font-size:.7rem;font-weight:700;margin-bottom:12px; }
.pc-title{ font-size:1.16rem;font-weight:800;color:#111827;line-height:1.25;margin-bottom:10px; }
.pc-rating{ display:flex;align-items:center;gap:7px;margin-bottom:12px;color:#6B7280;font-size:.84rem; }
.pc-rating strong{ color:#111827; } .pc-star{ color:#F59E0B; }
.pc-desc{ color:#6B7280;line-height:1.6;font-size:.86rem;margin-bottom:16px;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden; }
.pc-price-row{ display:flex;justify-content:space-between;align-items:center;
  margin-top:auto;margin-bottom:16px; }
.pc-price{ font-size:1.7rem;font-weight:800;color:#111827;line-height:1; }
.pc-promo{ color:#B5651D;font-size:.74rem;font-weight:700;margin-top:4px; }
.pc-actions{ display:flex;gap:10px; }
.pc-view{ flex:1;height:48px;display:flex;align-items:center;justify-content:center;
  text-decoration:none;border-radius:14px;color:#fff;font-size:.9rem;font-weight:700;
  background:linear-gradient(135deg,#6366F1,#8B5CF6);transition:.25s; }
.pc-view:hover{ transform:translateY(-2px);color:#fff; }
.pc-cart{ width:48px;height:48px;display:flex;align-items:center;justify-content:center;
  text-decoration:none;border-radius:14px;background:#111827;color:#fff;font-size:18px; }
.pc-cart:hover{ background:#000;color:#fff; }

/* ---------- Chips (Streamlit buttons) ---------- */
div[data-testid="stButton"] > button{
  border-radius:999px;border:1px solid var(--line);background:#fff;color:var(--ink);
  font-weight:600;font-size:.9rem;box-shadow:var(--shadow);padding:.45rem 1rem; }
div[data-testid="stButton"] > button:hover{ border-color:var(--amber);color:var(--amber); }

/* ---------- Chat input ---------- */
[data-testid="stChatInput"]{ border-radius:16px;border:1px solid var(--line);
  box-shadow:var(--shadow);background:#fff; }
.disclaimer{ text-align:center;color:var(--muted);font-size:.78rem;margin-top:8px; }

[data-testid="stSidebar"] [data-testid="stExpander"]{ border:none; }
</style>
"""


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Warming up the AI agent…")
def load_graph(fast: bool):
    return build_graph(fast=fast)


@st.cache_data(show_spinner=False)
def graph_mermaid(_graph) -> str:
    return _graph.get_graph().draw_mermaid()


@st.cache_data(show_spinner=False)
def product_descriptions() -> dict[str, str]:
    """Map product URL -> short description (for the cards)."""
    try:
        data = json.loads(config.DATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for p in data:
        desc = (p.get("description") or "").split(". ")[0].strip()
        if len(desc) > 95:
            desc = desc[:92].rstrip() + "…"
        out[p.get("url", "")] = desc
    return out


@st.cache_data(show_spinner=False)
def data_as_of() -> str:
    """Human-readable date the catalog was scraped (empty if unknown)."""
    try:
        meta = json.loads(config.DATA_PATH.with_name("meta.json").read_text(encoding="utf-8"))
        return datetime.fromisoformat(meta["scraped_at"]).strftime("%b %d, %Y").replace(" 0", " ")
    except Exception:
        return ""


def render_mermaid(src: str, height: int = 300) -> None:
    diagram = f"""
    <div class="mermaid" style="text-align:center;">{src}</div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad:true, theme:'base',
        themeVariables:{{ primaryColor:'#FBE7DA', primaryBorderColor:'#B5651D',
                          lineColor:'#A85812', fontFamily:'sans-serif' }} }});
    </script>"""
    components.html(diagram, height=height, scrolling=False)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def product_card(s: dict, descs: dict[str, str]) -> str:
    title = html.escape(s.get("title", ""))
    url = html.escape(s.get("url", "") or "#")
    price = html.escape(s.get("price", ""))
    img = html.escape(s.get("image", ""))
    desc = html.escape(descs.get(s.get("url", ""), ""))

    try:
        r = float(s.get("rating") or 0)
    except (TypeError, ValueError):
        r = 0.0
    reviews = s.get("review_count") or 0
    rating_html = (
        f'<div class="pc-rating"><span class="pc-star">★</span> '
        f'<strong>{r:.1f}</strong> <span>({reviews} Reviews)</span></div>'
        if r > 0 else ""
    )

    badges = [b.strip() for b in (s.get("badges") or "").split(",") if b.strip()]
    priority = ["glutenfree", "vegan", "organic", "nongmo", "kosherpareve", "kosherdairy"]
    badges.sort(key=lambda b: priority.index(b) if b in priority else 99)
    badge_html = (
        f'<span class="pc-badge">{html.escape(BADGE_LABELS.get(badges[0], badges[0].title()))}</span>'
        if badges else ""
    )

    img_html = f'<img src="{img}" loading="lazy" alt="">' if img else ""
    desc_html = f'<p class="pc-desc">{desc}</p>' if desc else ""
    promo = html.escape(s.get("promo", "") or "")
    price_html = (
        f'<div><div class="pc-price">{price}</div>'
        + (f'<div class="pc-promo">{promo}</div>' if promo else "")
        + "</div>"
        if price else "<div></div>"
    )

    return f"""
    <div class="pcard">
      <div class="pc-img">{img_html}</div>
      <div class="pc-content">
        {badge_html}
        <div class="pc-title">{title}</div>
        {rating_html}
        {desc_html}
        <div class="pc-price-row">{price_html}</div>
        <div class="pc-actions">
          <a class="pc-view" href="{url}" target="_blank" rel="noopener">View Product →</a>
          <a class="pc-cart" href="{url}" target="_blank" rel="noopener"
             title="Open on King Arthur Baking" aria-label="Open on King Arthur Baking">🛒</a>
        </div>
      </div>
    </div>"""


def render_turn(turn: dict, descs: dict[str, str]) -> None:
    if turn["role"] == "user":
        st.markdown(
            f'<div class="row-user"><div class="b-user">{html.escape(turn["content"])}'
            f'<div class="b-time">{turn["time"]} <span class="rr">✓✓</span></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        return

    answer_html = md.markdown(turn["content"], extensions=["nl2br"])
    st.markdown(
        f'<div class="row-bot"><div class="b-ava">🧁</div>'
        f'<div class="b-bot">{answer_html}'
        f'<div class="b-time">{turn["time"]}</div></div></div>',
        unsafe_allow_html=True,
    )
    sources = [s for s in turn.get("sources", []) if s.get("title")]
    if sources:
        cards = "".join(product_card(s, descs) for s in sources)
        st.markdown(f'<div class="pgrid">{cards}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(graph) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="side-brand"><div class="side-logo">🧁</div>'
            '<div><b>King Arthur</b><span>Baking Assistant ✨</span></div></div>',
            unsafe_allow_html=True,
        )
        if st.button("➕  New Chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()

        # Recent chats — derived from this session's questions.
        recents = [t for t in st.session_state.get("chat", []) if t["role"] == "user"][::-1][:6]
        if recents:
            st.markdown('<div class="side-h">Recent</div>', unsafe_allow_html=True)
            rows = "".join(
                f'<div class="recent"><span class="rc-ic">💬</span>'
                f'<span class="rc-t">{html.escape(t["content"])}</span></div>'
                for t in recents
            )
            st.markdown(rows, unsafe_allow_html=True)

        st.markdown('<div class="side-sep"></div>', unsafe_allow_html=True)

        # The required agent-graph visualization + settings.
        with st.expander("🕸️  Agent graph"):
            render_mermaid(graph_mermaid(graph))
        with st.expander("⚙️  Settings"):
            st.toggle("Fast mode", key="fast_mode",
                      help="Fast: retrieve → answer. Smart: adds grade → rewrite self-correction.")
            st.caption(f"Model: `{config.active_chat_model()}` · "
                       f"Embeddings: `{config.HF_EMBED_MODEL if config.EMBED_PROVIDER == 'huggingface' else config.OPENAI_EMBED_MODEL}`")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
SUGGESTIONS = [
    ("🌿", "Gluten free options", "What gluten-free mixes do you have?"),
    ("🎂", "Birthday cake ideas", "Recommend a mix for a kid's birthday party"),
    ("⭐", "Top rated mixes", "Which mixes have the best reviews?"),
    ("🍪", "Cookie recipes", "What cookie mixes do you have?"),
]
FOLLOWUPS = [
    "Which is best for beginners?",
    "Do you have vegan options?",
    "What's the cheapest one?",
]


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("fast_mode", True)
    st.session_state.setdefault("chat", [])  # list of {role, content, time, sources}
    graph = load_graph(st.session_state["fast_mode"])
    sidebar(graph)
    descs = product_descriptions()

    empty = not st.session_state.chat

    if empty:
        # Welcome / empty state: greeting + suggestion chips.
        st.markdown(
            '<div class="welcome"><div class="w-logo">🧁</div>'
            '<h1>King Arthur Mixes Assistant</h1>'
            '<p>Ask about flavors, prices, ratings, or dietary options — '
            'answers come straight from the catalog, with pictures.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chips-label">Try one of these</div>', unsafe_allow_html=True)
        cols = st.columns(len(SUGGESTIONS))
        for col, (emoji, label, query) in zip(cols, SUGGESTIONS):
            if col.button(f"{emoji}  {label}", use_container_width=True, key=f"sug-{label}"):
                st.session_state.pending = query
                st.rerun()
    else:
        # Conversation
        for turn in st.session_state.chat:
            render_turn(turn, descs)

        # Follow-up chips after the latest assistant answer
        if st.session_state.chat[-1]["role"] == "assistant":
            fcols = st.columns(len(FOLLOWUPS))
            for col, q in zip(fcols, FOLLOWUPS):
                if col.button(q, use_container_width=True, key=f"fu-{q}-{len(st.session_state.chat)}"):
                    st.session_state.pending = q
                    st.rerun()

    prompt = st.chat_input("Ask anything about baking…")
    if "pending" in st.session_state:
        prompt = st.session_state.pop("pending")

    if prompt:
        now = datetime.now().strftime("%I:%M %p").lstrip("0")

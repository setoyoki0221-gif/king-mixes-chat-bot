---
title: King Arthur Mixes Assistant
emoji: 🧁
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
short_description: AI assistant for King Arthur Baking's Mixes catalog (RAG + LangGraph)
---

# 🧁 King Arthur Baking — Mixes Assistant

An AI chatbot that answers questions about the **Mixes** category of
[King Arthur Baking](https://shop.kingarthurbaking.com/mixes) — flavors, prices,
dietary options (gluten-free, kosher…), and recommendations — grounded in data
scraped directly from the store.

Built with **LangChain + LangGraph** (a self-correcting RAG agent), **Chroma**
for vector search, and a **Streamlit** frontend, deployed on **Hugging Face
Spaces**.

---

## How it works

```
shop.kingarthurbaking.com/mixes
        │  (1) scraper/scrape.py  — requests + BeautifulSoup
        ▼
   data/products.json   (118 products, clean numeric prices + promo text)
        │  (2) ingest.py  — one document per product, embedded
        ▼
   chroma_db/           (persistent vector store)
        │  (3) agent.py + catalog.py  — LangGraph agent
        ▼
   Streamlit app.py     — chat + live agent-graph plot
```

### The agent (LangGraph)

This is an **agentic** retrieval pipeline, not a plain retrieve-then-answer
chain. It self-checks retrieval quality and reformulates weak queries before
answering:

```
retrieve → grade → (relevant?) ── yes ──► generate → END
                       │
                       no, retries left
                       ▼
                   rewrite → retrieve (loop)
```

- **retrieve** — *hybrid*: aggregate questions (see below) are answered by an
  exact lookup over the whole catalog; everything else uses semantic search
- **grade** — the LLM judges whether semantic results are relevant (skipped for
  exact lookups, which are correct by construction)
- **rewrite** — if not, reformulate the query (told what already failed) and
  retry, up to 2×
- **generate** — answer grounded *only* in the catalog, with sources

The compiled graph is rendered live in the app's sidebar.

### Why retrieval is hybrid

Similarity search returns the few closest documents, so it cannot rank or count:
"cheapest" would mean "cheapest of the six nearest matches". `catalog.py` parses
those intents with plain rules (no LLM) and filters/sorts the *whole* catalog:

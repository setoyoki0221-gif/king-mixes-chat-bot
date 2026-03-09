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

| Question | Handled as |
|---|---|
| "What's the cheapest mix?" / "most expensive" | sort by numeric price |
| "Top rated" / "best reviews" / "most popular" | sort by review-weighted rating (5.0 from 1 review doesn't beat 4.9 from 300) |
| "gluten-free", "kosher", "non-GMO", "vegan"… | filter on the product's attribute badges, with an exact total count |
| "under $7", "$8 or less" | numeric price bounds |
| "What's the cheapest **one**?" (follow-up) | inherits the topic/filters of the earlier question |

If an attribute matches nothing (e.g. there are no vegan-labelled mixes), the bot
says so rather than guessing. Anything it can't parse falls back to semantic search.

---

## Providers (all free, configurable via env)

The app is **provider-flexible** — pick a backend with two env vars. The default
runs a small model **on-device** (no API key, works in any region):

| Knob | Default | Other options |
|---|---|---|
| `LLM_PROVIDER` | `local` (Qwen2.5-1.5B-Instruct, on-device) | `ollama`, `huggingface`, `gemini`, `github`, `openai`, `groq` |
| `EMBED_PROVIDER` | `huggingface` (all-MiniLM-L6-v2, local) | `openai` |

See [.env.example](.env.example) for every variable. `gemini` and `groq` need an
extra package: `pip install -r requirements-optional.txt`.

---

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env          # defaults work with no API key (local model)

# (optional) re-scrape + rebuild the knowledge base
python scraper/scrape.py
python ingest.py

streamlit run app.py
```

The first run downloads the small local model (~3 GB for the 1.5B default; set
`LOCAL_CHAT_MODEL=Qwen/Qwen2.5-0.5B-Instruct` for a lighter one) once.

The scraper refuses to overwrite `data/products.json` if a run looks broken
(too few products, many failed pages, missing prices), so a site change can't
silently degrade the bot. After re-scraping, run `python ingest.py` again.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                        # everything (several minutes: loads the embedding model)
pytest -m "not retrieval"     # fast (~15 s): pure logic, data checks, stub-LLM agent + app tests
```

- `test_catalog.py` — query parsing, exact lookups, follow-ups (synthetic data + invariants on the real catalog)
- `test_scraper.py` / `test_data.py` — price/promo parsing, the scrape guard, dataset sanity
- `test_agent.py` — prompt/brace regression, grading, rewriting, source cards, graph wiring (stub LLM)
- `test_app.py` — runs the real Streamlit script headlessly (stub LLM): page loads, answer + product card render
- `test_retrieval_eval.py` — golden questions against the real vector store (marker `retrieval`)

---

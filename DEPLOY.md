# Deploying to Hugging Face Spaces

The deployed app uses the **in-Space local model** (`LLM_PROVIDER=local`,
Qwen2.5-1.5B) — **no API keys or secrets required**. Ollama is only for local
development; it does not run on Spaces.

Everything needed ships in the repo: `data/products.json` + `chroma_db/` (the
prebuilt knowledge base), so the Space never re-scrapes or re-embeds.

---

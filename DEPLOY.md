# Deploying to Hugging Face Spaces

The deployed app uses the **in-Space local model** (`LLM_PROVIDER=local`,
Qwen2.5-1.5B) — **no API keys or secrets required**. Ollama is only for local
development; it does not run on Spaces.

Everything needed ships in the repo: `data/products.json` + `chroma_db/` (the
prebuilt knowledge base), so the Space never re-scrapes or re-embeds.

---

## Part A — Create the Space

1. Go to **https://huggingface.co** and sign in.
2. Open **https://huggingface.co/new-space**.
3. Fill in:
   - **Owner:** your username
   - **Space name:** `king-mixes-assistant` (or anything)
   - **License:** optional
   - **SDK:** **Streamlit**
   - **Hardware:** **CPU basic · Free**
   - **Visibility:** Public
4. Click **Create Space**. You now have an empty Space at
   `https://huggingface.co/spaces/<your-username>/king-mixes-assistant`.

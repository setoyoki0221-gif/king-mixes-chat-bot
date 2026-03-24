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

## Part B — Get a WRITE token (for pushing code)

1. Go to **https://huggingface.co/settings/tokens**.
2. **Create new token** → **Type: Write** → name it `deploy` → **Create**.
3. Copy the `hf_...` token (you'll paste it as the *password* when git asks).

## Part C — Push your files (run in PowerShell, in the project folder)

Replace `<your-username>` and `<space-name>` below.

```powershell
cd "e:\Development\AI\King Mixes chat bot"

# The project is already a git repo. Commit any pending changes first:
git add -A
git commit -m "Prepare for deploy"

git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

> A brand-new Space already contains a starter commit, so the first push may be
> rejected as non-fast-forward. Only in that case, add `--force` — it replaces
> the Space's starter files with this repo.

When prompted:
- **Username:** your Hugging Face username
- **Password:** paste the **Write token** (`hf_...`) from Part B

> `.env` is git-ignored, so your local secrets are never uploaded. ✅

## Part D — Watch it build

- On the Space page, the status shows **Building** (installs PyTorch, etc. — a
  few minutes), then **Running**.
- The **first question** downloads the model (~3 GB, on HF's fast servers) and
  will take ~1 minute; after that answers are cached and quicker.

## Part E — Done

Your assistant is live at
`https://huggingface.co/spaces/<your-username>/<space-name>`.

---

## Optional: better/faster answers

Set these under the Space's **Settings → Variables and secrets**:

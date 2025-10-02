"""Factory functions that build the embeddings and chat model.

Centralizing provider construction here keeps ingest.py and the agent free of
provider-specific branching — they just call get_embeddings() / get_llm().
"""

from __future__ import annotations

import os
import threading

import config

# Cache the on-device model/tokenizer so multiple get_llm() calls (e.g. one for
# grading with a tiny output budget, one for generation) share a single copy of
# the weights in memory instead of loading the model twice.
_local_model = None
_local_tokenizer = None
# Streamlit runs each session on its own thread; without this lock two first
# visitors would each load the multi-GB model at the same moment.
_local_lock = threading.Lock()


def _load_local_model():
    global _local_model, _local_tokenizer
    if _local_model is None:
        with _local_lock:
            if _local_model is None:  # re-check: another thread may have loaded it
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Use all CPU cores for inference (default can under-utilise the CPU).
                torch.set_num_threads(os.cpu_count() or 4)
                _local_tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_CHAT_MODEL)
                _local_model = AutoModelForCausalLM.from_pretrained(
                    config.LOCAL_CHAT_MODEL, torch_dtype=torch.float32
                )
    return _local_model, _local_tokenizer


def _optional_import(module: str, package: str, provider: str):
    """Import a provider package that is not in the core requirements."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"LLM_PROVIDER={provider} needs the '{package}' package. "
            "Install it with:  pip install -r requirements-optional.txt"
        ) from exc


def get_embeddings():
    """Return an embeddings object for the configured EMBED_PROVIDER."""
    if config.EMBED_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        # check_embedding_ctx_length=False avoids a tiktoken vocab download
        # that is blocked on some networks; product docs are short anyway.
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL,
            api_key=config.OPENAI_API_KEY,
            check_embedding_ctx_length=False,
        )

    if config.EMBED_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        # Local, free, CPU-friendly. Downloads ~90MB model once and caches it.
        return HuggingFaceEmbeddings(
            model_name=config.HF_EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown EMBED_PROVIDER: {config.EMBED_PROVIDER!r}")


def get_llm(temperature: float = 0.2, max_new_tokens: int | None = None):
    """Return a chat model for the configured LLM_PROVIDER.

    max_new_tokens lets callers cap output length per task (e.g. a 4-token budget
    for the YES/NO grader). It is honoured by the local provider and ignored by
    hosted providers that don't expose it cheaply.
    """
    config.require_llm_key()

    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

    if config.LLM_PROVIDER == "local":
        # Small instruct model running on-device via transformers. No API token,
        # no network, no regional restrictions. The model is loaded once and
        # shared; only a lightweight pipeline wrapper is created per call.
        from transformers import pipeline as hf_pipeline
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        model, tokenizer = _load_local_model()
        gen = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens or config.LOCAL_MAX_NEW_TOKENS,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            return_full_text=False,
            repetition_penalty=1.1,
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen))

    if config.LLM_PROVIDER == "ollama":
        # Quantized local models served by the Ollama app — much faster on CPU
        # than fp32 transformers, so a smarter 1.5B/3B model stays responsive.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=max_new_tokens or config.LOCAL_MAX_NEW_TOKENS,
        )

    if config.LLM_PROVIDER == "huggingface":
        # Free Hugging Face Inference API. Uses the same HF token as deployment
        # and works in regions where other providers are blocked.
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=config.HF_CHAT_MODEL,
            huggingfacehub_api_token=config.HF_TOKEN,
            task="text-generation",
            temperature=max(temperature, 0.01),  # HF API rejects temperature=0
            max_new_tokens=700,
        )
        return ChatHuggingFace(llm=endpoint)

    if config.LLM_PROVIDER == "gemini":
        ChatGoogleGenerativeAI = _optional_import(
            "langchain_google_genai", "langchain-google-genai", "gemini"
        ).ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_CHAT_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    if config.LLM_PROVIDER == "github":
        # GitHub Models is OpenAI-compatible: real OpenAI models, free, via a
        # GitHub token and a custom base URL.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.GITHUB_CHAT_MODEL,
            api_key=config.GITHUB_TOKEN,
            base_url=config.GITHUB_MODELS_BASE_URL,
            temperature=temperature,
        )

    if config.LLM_PROVIDER == "groq":
        ChatGroq = _optional_import("langchain_groq", "langchain-groq", "groq").ChatGroq

        return ChatGroq(
            model=config.GROQ_CHAT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=temperature,
        )

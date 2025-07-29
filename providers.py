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

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

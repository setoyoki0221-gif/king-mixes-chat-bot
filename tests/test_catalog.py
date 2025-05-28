"""Rule-based query understanding and exact catalog lookup (no LLM, no embeddings)."""

from __future__ import annotations

import pytest
from conftest import make_doc
from langchain_core.documents import Document

from catalog import (
    badges_of,
    compact_line,
    is_followup,
    parse_query,
    resolve_question,
    structured_search,
)


def titles(result):
    return [d.metadata["title"] for d in result.docs]

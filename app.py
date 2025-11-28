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

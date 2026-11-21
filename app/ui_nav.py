"""Page registry so views can link to each other without importing app.py."""
from __future__ import annotations

import streamlit as st

PAGES = {}


def register(name, page):
    PAGES[name] = page


def go(name):
    page = PAGES.get(name)
    if page is not None:
        st.switch_page(page)
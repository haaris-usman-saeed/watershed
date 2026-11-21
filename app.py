"""WatershedShift frontend.

    WATERSHED_DATA_DIR=app/sample_data streamlit run app.py   # synthetic fixtures
    streamlit run app.py                                       # pipeline outputs in outputs/

Frontend modules live in app/ (deliberately not a package: app/__init__.py would collide
with this file's module name), so that directory is put on sys.path below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

import ui_components as ui  # noqa: E402
import ui_nav  # noqa: E402
import view_backtest  # noqa: E402
import view_explorer  # noqa: E402
import view_overview  # noqa: E402
import view_research  # noqa: E402

st.set_page_config(page_title="WatershedShift", page_icon="💧", layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()

bundle = ui.load_bundle()

pages = {
    "overview": st.Page(view_overview.render, title="Overview", url_path="overview", default=True),
    "explorer": st.Page(view_explorer.render, title="Reservoir explorer", url_path="explorer"),
    "backtest": st.Page(view_backtest.render, title="Backtesting", url_path="backtest"),
    "research": st.Page(view_research.render, title="Model and research", url_path="research"),
}
for name, page in pages.items():
    ui_nav.register(name, page)

try:
    nav = st.navigation(list(pages.values()), position="top")
except TypeError:  # Streamlit < 1.46 has no top navigation
    nav = st.navigation(list(pages.values()))

ui.masthead(bundle)
ui.data_mode_warning(bundle)
nav.run()
ui.footer(bundle)
"""Headless render of the 30-Day Forecast page (Streamlit AppTest). Needs streamlit and plotly."""
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _page():
    import sys
    from pathlib import Path as _P
    root = _P.cwd()
    for p in (str(root), str(root / "app")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ui_components as ui
    import view_forecast
    ui.inject_css()
    view_forecast.render()


def run_page(monkeypatch, forecast_file, data_dir="app/sample_data"):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("WATERSHED_DATA_DIR", str(ROOT / data_dir))
    if forecast_file is None:
        monkeypatch.setenv("WATERSHED_FORECAST_FILE", str(ROOT / "does-not-exist.json"))
    else:
        monkeypatch.setenv("WATERSHED_FORECAST_FILE", str(forecast_file))
    at = AppTest.from_function(_page, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def text(at):
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


def test_synthetic_fixture_renders_with_banner(monkeypatch):
    at = run_page(monkeypatch, ROOT / "app/sample_data/forecast_snapshot.json")
    t = text(at)
    assert "Synthetic forecast fixture. Not a forecast." in t
    assert "Forecast interpretation" in t and "not operational advice" in t
    assert len(at.metric) >= 7


def test_missing_snapshot_shows_explained_state(monkeypatch):
    at = run_page(monkeypatch, None)
    t = text(at)
    assert "No forecast snapshot on this deployment" in t and len(at.metric) == 0


def test_invalid_snapshot_is_not_displayed(monkeypatch, tmp_path):
    import json
    snap = json.loads((ROOT / "app/sample_data/forecast_snapshot.json").read_text())
    snap["is_heldout_test_prediction"] = True
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(snap))
    at = run_page(monkeypatch, p)
    assert "failed validation" in text(at) and len(at.metric) == 0


@pytest.mark.skipif(not (ROOT / "forecasts/latest.json").exists(), reason="no private real snapshot")
def test_real_snapshot_renders(monkeypatch):
    at = run_page(monkeypatch, ROOT / "forecasts/latest.json", data_dir="outputs")
    t = text(at)
    assert ("Forecast interpretation" in t) or ("Forecast expired" in t)
    assert "Synthetic forecast fixture" not in t

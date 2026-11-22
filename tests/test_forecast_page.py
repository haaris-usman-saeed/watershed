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


def _overview():
    import sys
    from pathlib import Path as _P
    root = _P.cwd()
    for p in (str(root), str(root / "app")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ui_components as ui
    import view_overview
    ui.inject_css()
    view_overview.render()


def run_overview(monkeypatch, forecast_file, data_dir):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("WATERSHED_DATA_DIR", str(ROOT / data_dir))
    monkeypatch.setenv("WATERSHED_FORECAST_FILE", str(forecast_file))
    at = AppTest.from_function(_overview, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_forecast_page_explains_who_it_helps(monkeypatch):
    t = text(run_page(monkeypatch, ROOT / "app/sample_data/forecast_snapshot.json"))
    assert "Who this forecast helps" in t and "not</strong> a tool for reservoir operators" in t
    assert "What it cannot see" in t


def test_overview_guide_and_dated_heading_without_live_data(monkeypatch):
    at = run_overview(monkeypatch, ROOT / "does-not-exist.json", "app/sample_data")
    t = text(at)
    assert "How to read this site" in t and "30-Day Forecast" in t
    assert "Storage on the last day of the historical dataset" in t
    assert "not today's levels" in t and "Latest observed storage" not in t


@pytest.mark.skipif(not (ROOT / "forecasts/latest.json").exists() or not (ROOT / "outputs/metrics.json").exists(),
                    reason="no private real snapshot or outputs")
def test_overview_shows_latest_live_storage_when_refresh_exists(monkeypatch):
    t = text(run_overview(monkeypatch, ROOT / "forecasts/latest.json", "outputs"))
    assert "Latest observed storage" in t and "not today's levels" not in t


def _interactive():
    import sys
    from pathlib import Path as _P
    root = _P.cwd()
    for p in (str(root), str(root / "app")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ui_components as ui
    import view_forecast_v3
    ui.inject_css()
    view_forecast_v3.render_interactive(ui.load_bundle())


def test_generate_forecast_button_runs_service_offline(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    import view_forecast_v3
    from tests.test_v3 import NOW, service as synthetic_service
    svc = synthetic_service(tmp_path, NOW)
    monkeypatch.setattr(view_forecast_v3, "service", lambda: svc)
    from src.v3_service import RecordStore
    monkeypatch.setattr(view_forecast_v3, "record_store", lambda: RecordStore(root=tmp_path / "rec"))
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("WATERSHED_DATA_DIR", str(ROOT / "app/sample_data"))
    at = AppTest.from_function(_interactive, default_timeout=60)
    at.run()
    assert "press Generate Forecast" in text(at)
    at.segmented_control(key="v3_station").set_value("FOL").run() if hasattr(at, "segmented_control") else None
    at.button(key="v3_go").click().run()
    assert not at.exception, at.exception
    t = text(at)
    assert "Model comparison" in t and "Data freshness" in t and "Interval caution: Folsom" in t
    assert any("Model A" in m.label for m in at.metric)
    assert len(list((tmp_path / "rec").glob("*.json"))) == 1


def test_public_mode_shows_policy_block(monkeypatch, tmp_path, restricted_policy):
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    import view_forecast_v3
    from tests.test_v3 import NOW, service as synthetic_service
    monkeypatch.setenv("WATERSHED_DEPLOYMENT", "public")
    monkeypatch.setattr(view_forecast_v3, "service", lambda: synthetic_service(tmp_path, NOW))
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("WATERSHED_DATA_DIR", str(ROOT / "app/sample_data"))
    at = AppTest.from_function(_interactive, default_timeout=60)
    at.run()
    at.button(key="v3_go").click().run()
    t = text(at)
    assert "Not available on this public deployment" in t and len(at.metric) == 0


def test_public_mode_hides_real_issued_snapshot(monkeypatch, tmp_path, restricted_policy):
    import json
    snap = json.loads((ROOT / "app/sample_data/forecast_snapshot.json").read_text())
    snap["data_mode"] = "real"
    p = tmp_path / "real_like.json"
    p.write_text(json.dumps(snap))
    monkeypatch.setenv("WATERSHED_DEPLOYMENT", "public")
    at = run_page(monkeypatch, p)
    t = text(at)
    assert "Not available on this public deployment" in t and "Forecast interpretation" not in t


def test_public_mode_blocks_whole_app_on_real_outputs(monkeypatch, tmp_path, restricted_policy):
    """A public deployment pointed at real (CDEC-derived) outputs renders no page content."""
    import json
    import shutil
    d = tmp_path / "real_like"
    shutil.copytree(ROOT / "app/sample_data", d)
    m = json.loads((d / "metrics.json").read_text())
    m["data_mode"] = "real"
    (d / "metrics.json").write_text(json.dumps(m))
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("WATERSHED_DATA_DIR", str(d))
    monkeypatch.setenv("WATERSHED_DEPLOYMENT", "public")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    t = text(at)
    assert "Real data cannot be shown on this public deployment" in t
    assert "How to read this site" not in t and len(at.metric) == 0
    monkeypatch.setenv("WATERSHED_DEPLOYMENT", "private")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert "Real data cannot be shown" not in text(at)


def test_fresh_deployment_uses_latest_public_release(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    import ui_io
    monkeypatch.delenv("WATERSHED_DATA_DIR", raising=False)
    monkeypatch.setattr(ui_io, "REPO_ROOT", tmp_path)
    assert ui_io.resolve_data_dir() == ((tmp_path / "outputs").resolve(), "default")
    rel = tmp_path / "public_data/watershedshift-public-data-2026-11-22.2/ui_outputs"
    rel.mkdir(parents=True)
    assert ui_io.resolve_data_dir() == (rel.resolve(), "public release")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/.gitkeep").write_text("")
    assert ui_io.resolve_data_dir()[1] == "public release"          # an empty outputs/ (as in a clone) is not data
    (tmp_path / "outputs/metrics.json").write_text("{}")
    assert ui_io.resolve_data_dir()[1] == "default"                 # private outputs win when present


@pytest.mark.parametrize("page", ["overview", "explorer", "backtest", "research", "forecast"])
def test_every_page_renders_on_the_published_release_in_public_mode(monkeypatch, page):
    """The public deployment's data (newest public release) must render every page without an exception."""
    rel = sorted(ROOT.glob("public_data/watershedshift-public-data-*/ui_outputs"))
    if not rel:
        pytest.skip("no public release with UI outputs")
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("WATERSHED_DATA_DIR", str(rel[-1]))
    monkeypatch.setenv("WATERSHED_DEPLOYMENT", "public")
    monkeypatch.setenv("WATERSHED_FORECAST_FILE", str(rel[-1] / "forecast_snapshot.json"))

    def _render(name):
        import sys as _s
        from pathlib import Path as _P
        for p in (str(_P.cwd()), str(_P.cwd() / "app")):
            if p not in _s.path:
                _s.path.insert(0, p)
        import importlib
        import ui_components as ui
        ui.inject_css()
        importlib.import_module({"overview": "view_overview", "explorer": "view_explorer", "backtest": "view_backtest",
                                 "research": "view_research", "forecast": "view_forecast"}[name]).render()

    at = AppTest.from_function(_render, default_timeout=60, args=(page,))
    at.run()
    assert not at.exception, at.exception

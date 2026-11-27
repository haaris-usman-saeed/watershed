"""Website: HGB-only Generate Forecast and the Model playground. Offline; synthetic data or tracked aggregates only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v5 import playground as pg
from tests.test_v3 import NOW, mock_models, synthetic_cdec, synthetic_power

ROOT = Path(__file__).resolve().parents[1]
HAVE_V5_RESULTS = (ROOT / "results/v5/metrics/HELD_posthoc.csv").exists()
needs_results = pytest.mark.skipif(not HAVE_V5_RESULTS, reason="tracked v5 results not in this checkout")


# ---------------------------------------------------------------- the service runs only the frozen v2 HGB

def _refuse(*_a, **_k):
    raise AssertionError("this source must not be fetched when only the HGB runs")


def test_hgb_only_service_never_fetches_weather_or_gefs(tmp_path):
    from src.v3_service import ForecastService
    svc = ForecastService(cache_dir=tmp_path / "c", art_v3=tmp_path / "none", clock=lambda: NOW, model_keys=("A",),
                          fetchers={"cdec": synthetic_cdec, "power": _refuse, "gefs": _refuse})
    svc._models = mock_models()
    r = svc.forecast("FOL")
    assert r["status"] == "ok" and list(r["models"]) == ["A"] and r["models"]["A"]["available"]
    assert set(r["source_status"]) == {"cdec"} and r["sources"]["power"] is None
    assert r["target_date"] == "2026-12-16" and pd.Timestamp(r["origin_date"]).dayofweek == 0


def test_default_service_still_runs_all_four_models(tmp_path):
    from src.v3_service import ForecastService
    from tests.test_v3 import synthetic_gefs
    svc = ForecastService(cache_dir=tmp_path / "c", art_v3=tmp_path / "none", clock=lambda: NOW,
                          fetchers={"cdec": synthetic_cdec, "power": synthetic_power, "gefs": synthetic_gefs})
    svc._models = mock_models()
    r = svc.forecast("FOL")
    assert list(r["models"]) == ["A", "B", "B2", "C"] and {"cdec", "power", "gefs"} <= set(r["source_status"])


@pytest.mark.parametrize("keys", [("B",), ("A", "D"), ()])
def test_model_keys_must_include_the_hgb_and_be_known(keys):
    from src.v3_service import ForecastService
    with pytest.raises(ValueError):
        ForecastService(model_keys=keys)


# ---------------------------------------------------------------- playground data

def test_ten_models_in_report_order_with_the_hgb_as_model_6():
    assert [m["number"] for m in pg.TEN] == list(range(1, 11))
    assert [m["number"] for m in pg.OTHERS] == [1, 2, 3, 4, 5, 7, 8, 9, 10]
    main = pg.by_number(6)
    assert pg.run_id(main, "held") == pg.run_id(main, "t26") == "REF-V2-ARTIFACT"
    assert pg.run_id(main, "confirmation") == "B-HGB-SNCR"      # the artifact was trained on those years
    assert all(pg.FLOWS in m["sources"] for m in pg.TEN if m["number"] in (5, 7, 8, 9, 10))
    assert all(pg.FLOWS not in m["sources"] for m in pg.TEN if m["number"] <= 4 or m["number"] == 6)


def test_flow_sources_are_not_cleared_by_the_release_policy():
    pol = json.loads((ROOT / "config/data_release_policy.json").read_text())
    assert pg.FLOWS not in pol["sources"]
    assert pg.display_block(pg.by_number(7), mode="public") is not None
    assert pg.display_block(pg.by_number(4), mode="public") is None
    assert pg.display_block(pg.by_number(6), mode="public") is None
    assert pg.display_block(pg.by_number(7), mode="private") is None


@needs_results
@pytest.mark.parametrize("window,n", [("confirmation", 2500), ("held", 593), ("t26", 251)])
def test_summary_reads_tracked_metrics_on_the_same_rows(window, n):
    s = pg.summary(window)
    assert list(s.number) == list(range(1, 11)) and set(s.n) == {n}
    t = pd.read_csv(ROOT / f"results/v5/metrics/{pg.WINDOWS[window]['file']}.csv").set_index("model")
    for r in s.itertuples():
        assert r.mae == pytest.approx(t.loc[r.run, "mae"])
    main = s.set_index("number").loc[6]
    assert main.delta_mae_vs_main == 0
    assert (s.delta_mae_vs_main == s.mae - main.mae).all()


@needs_results
def test_frozen_artifact_numbers_match_the_frozen_results():
    """Model 6 on 2024-2025 and 2026 is the saved artifact: its MAE equals the frozen v2 and v3 result files."""
    v2 = json.loads((ROOT / "results/v2/final/metrics.json").read_text())
    v3 = json.loads((ROOT / "results/v3/final/metrics.json").read_text())
    held, t26 = pg.summary("held").set_index("number"), pg.summary("t26").set_index("number")
    assert held.loc[6, "mae"] == pytest.approx(v2["ML"], abs=5e-4)
    assert t26.loc[6, "mae"] == pytest.approx(v3["mae"]["pred_A_frozen"]["mae"], abs=5e-4)


@needs_results
def test_comparisons_are_tracked_rows_for_the_window_only():
    c = pg.comparisons("t26", pg.by_number(10))
    assert len(c) and set(c.population) == {"T26 (post-hoc retrospective)"}
    assert c.contrast.str.startswith("+1*A-N4 ").all()
    assert pg.comparisons("confirmation", pg.by_number(1)).empty     # no paired test was run with persistence first


@needs_results
def test_breakdowns_cover_the_ten_models():
    bd = pg.breakdown("held", "station_id")
    assert set(bd.number) == set(range(1, 11)) and set(bd.station_id) == {"DNP", "EXC", "FOL", "NML", "ORO", "SHA"}


def _fake_preds(path):
    dates = pd.date_range("2024-01-01", periods=4, freq="7D")
    rows = []
    for fold, ds in (("HELD", dates), ("WY2016", dates - pd.Timedelta(days=3000))):
        for st in ("SHA", "FOL"):
            for d in ds:
                r = {"station_id": st, "date": d, "target_date": d + pd.Timedelta(days=30), "fold": fold,
                     "target_delta_pp": 2.0, "storage_pct": 50.0, "regime": "other", "unused": 9.0}
                r.update({m: float(i) for i, m in enumerate(["BL-PERS", "BL-TREND", "BL-MONTH", "BL-DOY", "A-RIDGE",
                                                              "REF-V2-ARTIFACT", "B-HGB-SNCR", "A-N1", "A-N2", "A-N3", "A-N4"])})
                rows.append(r)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_rows_filter_the_window_and_hide_flow_models_on_public(tmp_path):
    pytest.importorskip("pyarrow")
    p = tmp_path / "preds.parquet"
    _fake_preds(p)
    fr, hidden = pg.rows("held", mode="private", path=p)
    assert len(fr) == 8 and set(fr.station_id) == {"SHA", "FOL"} and not hidden
    assert "REF-V2-ARTIFACT" in fr and "B-HGB-SNCR" not in fr and "unused" not in fr
    fr, hidden = pg.rows("held", mode="public", path=p)
    assert set(hidden) == {5, 7, 8, 9, 10}
    assert not {"A-RIDGE", "A-N1", "A-N2", "A-N3", "A-N4"} & set(fr.columns)
    assert {"BL-PERS", "BL-DOY", "REF-V2-ARTIFACT"} <= set(fr.columns)
    fr, _ = pg.rows("confirmation", mode="private", path=p)
    assert len(fr) == 8 and "B-HGB-SNCR" in fr and "REF-V2-ARTIFACT" not in fr


def test_missing_prediction_file_is_explained(tmp_path):
    fr, hidden = pg.rows("t26", path=tmp_path / "absent.parquet")
    assert fr is None and "private" in hidden["all"]


def test_replay_gives_day30_storage_and_errors(tmp_path):
    pytest.importorskip("pyarrow")
    p = tmp_path / "preds.parquet"
    _fake_preds(p)
    fr, _ = pg.rows("held", mode="private", path=p)
    rp = pg.replay(fr, "held", "SHA", "2024-01-08")
    assert list(rp.number) == list(range(1, 11))
    r6 = rp.set_index("number").loc[6]
    assert r6.pred_change_pp == 5.0 and r6.pred_day30_pct == 55.0 and r6.obs_day30_pct == 52.0 and r6.error_pp == 3.0
    assert pg.replay(fr, "held", "SHA", "2023-01-02").empty


# ---------------------------------------------------------------- the page

def _playground():
    import sys
    from pathlib import Path as _P
    for p in (str(_P.cwd()), str(_P.cwd() / "app")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ui_components as ui
    import view_playground
    ui.inject_css()
    view_playground.render()


@needs_results
def test_playground_page_renders_without_private_rows(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(pg, "PREDS", tmp_path / "absent.parquet")
    at = AppTest.from_function(_playground, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    t = " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
    assert "Research comparison only" in t and "Replays are not available here" in t
    assert any("HGB (main model)" in m.label for m in at.metric)
    at.selectbox(key="pg_model").set_value(4).run()
    assert not at.exception and at.metric[0].value == f"{pg.summary('confirmation').mae[3]:.3f} pp"
    at.button_group(key="pg_window").set_value("t26").run()
    assert not at.exception, at.exception


def test_model_6_specification_names_the_refit_on_confirmation():
    main = pg.by_number(6)
    assert pg.spec(main, "held") == pg.spec(main, "t26") == "HGB[S,N,C,R] · frozen v2 artifact"
    assert pg.spec(main, "confirmation") == "HGB[S,N,C,R] · frozen hyperparameters, per-fold refit"
    assert pg.spec(pg.by_number(10), "confirmation") == pg.by_number(10)["spec"]

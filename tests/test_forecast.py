"""Tests for the 30-day research forecast pipeline (src.forecast_live, src.forecast_model, src.forecast_schema).

Synthetic tests always run. Tests marked `needs_data` use the private processed data and the
reconstructed model artifact and are skipped when those are absent (as in CI).
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecast_live import (FEATURES_15, MAX_ORIGIN_AGE_DAYS, basin_index, candidate_origins, design_row,
                               forecast_station, inference_features)
from src.forecast_schema import FEATURE_ORDER, expired, validate

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["DNP", "EXC", "FOL", "NML", "ORO", "SHA"]
HAVE_DATA = (ROOT / "data/processed/weekly_features.csv").exists() and (ROOT / "data/processed/snow_basin_index.csv").exists()
HAVE_ART = (ROOT / "models/v2_forecast/model_card.json").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="private processed data not present")
needs_art = pytest.mark.skipif(not (HAVE_DATA and HAVE_ART), reason="reconstructed model artifact not present")
ISSUE = pd.Timestamp("2026-11-22")       # a Sunday; latest Monday origin is 2026-11-16


def storage_series(end="2026-11-20", start="2026-06-01", base=500_000.0, slope=1_000.0):
    d = pd.date_range(start, end, freq="D")
    return pd.Series(base + slope * np.arange(len(d)), index=d)


def snow_series(end="2026-11-21", start="2026-06-01"):
    d = pd.date_range(start, end, freq="D")
    return pd.Series(np.arange(len(d)) / 100.0, index=d)    # value encodes the day


class ConstModel:
    def __init__(self, v):
        self.v = v
        self.seen = None

    def predict(self, x):
        self.seen = x.copy()
        return np.full(len(x), self.v)


def card(reliable_fol=False):
    return {"stations": STATIONS, "feature_order": FEATURE_ORDER,
            "snow_climatology_in": {s: [1.0] * 366 for s in STATIONS},
            "interval_offsets_pp": {"DJF": [-10.0, 10.0], "MAM": [-8.0, 8.0], "JJA": [-3.0, 3.0], "SON": [-4.0, 4.0]},
            "interval": {"method": "sym_season", "nominal": 0.9},
            "interval_reliable": {s: (reliable_fol or s != "FOL") for s in STATIONS},
            "historical_interval_coverage": {s: (0.752 if s == "FOL" else 0.95) for s in STATIONS},
            "core_snow_pillows": {s: ["X"] for s in STATIONS}}


def snow_df(st="FOL", series=None):
    s = snow_series() if series is None else series
    return pd.DataFrame({"station_id": st, "date": s.index, "swe_basin_in": s.to_numpy(), "n_core_reporting": 3,
                         "n_core": 3, "share_revised_flag": 0.5})


def run_station(storage, st="FOL", cap=1_000_000.0, model=None, snow=None, issue=ISSUE, breaks=(), c=None):
    sdf = pd.DataFrame({"storage_af": storage, "cdec_flag": ""})
    return forecast_station(st, st, model or ConstModel(2.0), c or card(), sdf, snow if snow is not None else snow_df(st),
                            lambda d: cap, issue, list(breaks))


# ---------------------------------------------------------------- 1, 2: no target, no future data

def test_inference_never_reads_storage_after_origin():
    s = storage_series(end="2026-12-31")                 # includes t+30 = 2026-12-16
    t = pd.Timestamp("2026-11-16")
    f1, m1 = inference_features(s, lambda d: 1e6, snow_series(), [1.0] * 366, t)
    poisoned = s.copy()
    poisoned[poisoned.index > t] = np.nan                # remove everything after t, including the target
    f2, m2 = inference_features(poisoned, lambda d: 1e6, snow_series(), [1.0] * 366, t)
    assert f1 == f2 and not m1 and not m2


def test_future_observations_do_not_change_earlier_features():
    t = pd.Timestamp("2026-10-12")
    a, _ = inference_features(storage_series(end="2026-10-12"), lambda d: 1e6, snow_series(end="2026-10-11"), [1.0] * 366, t)
    longer_storage = storage_series(end="2026-11-20") * 1.0
    longer_storage[longer_storage.index > t] = 9e9      # absurd future values must have no effect
    longer_snow = snow_series()
    longer_snow[longer_snow.index >= t] = 999.0
    b, _ = inference_features(longer_storage, lambda d: 1e6, longer_snow, [1.0] * 366, t)
    assert a == b


def test_forecast_works_without_any_storage_after_origin():
    r = run_station(storage_series(end="2026-11-16"))
    assert r["status"] == "ok" and r["origin_date"] == "2026-11-16"


# ---------------------------------------------------------------- 3, 4, 5: feature definitions

def test_exact_storage_changes():
    t = pd.Timestamp("2026-11-16")
    f, _ = inference_features(storage_series(slope=1_000.0), lambda d: 1_000_000.0, snow_series(), [1.0] * 366, t)
    assert f["prev7_pct"] == pytest.approx(0.7)
    assert f["prev30_pct"] == pytest.approx(3.0)
    assert f["prev60_pct"] == pytest.approx(6.0)
    s = storage_series()
    assert f["storage_pct"] == pytest.approx(100 * s[t] / 1e6)


def test_swe_uses_t_minus_1_and_t_minus_15():
    t = pd.Timestamp("2026-11-16")
    sn = snow_series()
    f, _ = inference_features(storage_series(), lambda d: 1e6, sn, [0.25] * 366, t)
    assert f["swe_basin_in"] == sn[pd.Timestamp("2026-11-15")]
    assert f["swe_change_14d_in"] == pytest.approx(sn[pd.Timestamp("2026-11-15")] - sn[pd.Timestamp("2026-11-01")])
    assert f["swe_anom_in"] == pytest.approx(sn[pd.Timestamp("2026-11-15")] - 0.25)
    sn2 = sn.copy()
    sn2[pd.Timestamp("2026-11-16")] = np.nan            # SWE on the origin day itself is never needed
    assert inference_features(storage_series(), lambda d: 1e6, sn2, [0.25] * 366, t)[0] == f


def test_snow_anomaly_uses_climatology_of_t_minus_1_day_of_year():
    t = pd.Timestamp("2026-03-01")                       # t-1 = Feb 28, day of year 59
    clim = list(np.arange(1, 367, dtype=float))
    f, _ = inference_features(storage_series(start="2025-12-01", end="2026-03-02"), lambda d: 1e6,
                              snow_series(start="2025-12-01", end="2026-03-01"), clim, t)
    assert f["swe_anom_in"] == pytest.approx(f["swe_basin_in"] - 59.0)


def test_snow_climatology_ignores_data_from_cutoff_on():
    from src.models import snow_climatology
    d = pd.date_range("2018-01-01", "2025-12-31", freq="D")
    idx = pd.DataFrame({"station_id": "FOL", "date": d, "swe_basin_in": np.sin(np.arange(len(d)) / 50.0) + 2})
    a = snow_climatology(idx, pd.Timestamp("2024-01-01"))
    idx2 = idx.copy()
    idx2.loc[idx2.date >= "2024-01-01", "swe_basin_in"] = 500.0
    b = snow_climatology(idx2, pd.Timestamp("2024-01-01"))
    assert np.allclose(a["FOL"], b["FOL"])


def test_season_encoding_matches_historical():
    t = pd.Timestamp("2026-11-16")
    f, _ = inference_features(storage_series(), lambda d: 1e6, snow_series(), [1.0] * 366, t)
    assert f["season_sin"] == pytest.approx(np.sin(2 * np.pi * t.dayofyear / 365.25))
    assert f["season_cos"] == pytest.approx(np.cos(2 * np.pi * t.dayofyear / 365.25))


def test_basin_index_requires_two_thirds_of_core_pillows():
    d = pd.Timestamp("2026-11-15")
    snow = pd.DataFrame({"snow_station_id": ["A", "B", "C", "A"], "date": [d, d, d, d + pd.Timedelta(days=1)],
                         "swe_in": [1.0, 2.0, 6.0, 4.0], "flag": ["r", "", "r", ""]})
    idx = basin_index(snow, {"FOL": ["A", "B", "C"]}).set_index("date")
    assert idx.loc[d, "swe_basin_in"] == pytest.approx(3.0)
    assert np.isnan(idx.loc[d + pd.Timedelta(days=1), "swe_basin_in"])   # 1 of 3 reporting: missing, never zero


# ---------------------------------------------------------------- 6, 7: encoding and capacity

def test_one_hot_order_matches_frozen_design():
    from src.models import FEATURE_SETS, design
    f, _ = inference_features(storage_series(), lambda d: 1e6, snow_series(), [1.0] * 366, pd.Timestamp("2026-11-16"))
    x = design_row(f, "ORO", STATIONS)
    assert list(x.columns) == FEATURE_ORDER
    frame = pd.DataFrame([{**f, "station_id": "ORO"}])
    ref = design(frame, FEATURE_SETS["storage_snow"], STATIONS)
    assert list(ref.columns) == FEATURE_ORDER and np.allclose(ref.to_numpy(), x.to_numpy())
    assert x["station_ORO"].iloc[0] == 1.0 and x.filter(like="station_").sum(axis=1).iloc[0] == 1.0


def test_capacity_in_force_at_origin_and_series_break():
    from src import build_dataset as bd
    hist = pd.read_csv(ROOT / "data/capacity_history.csv")
    cap_at = lambda d: float(bd.capacity_series("ORO", pd.DatetimeIndex([pd.Timestamp(d)]), hist).iloc[0])  # noqa: E731
    assert cap_at("2024-07-01") == 3537577 and cap_at("2024-07-02") == 3424753 and cap_at("2026-11-16") == 3424753
    brk = [pd.Timestamp("2026-10-01")]                   # a documented break inside the 60-day window
    s = storage_series()
    f, missing = inference_features(s, cap_at, snow_series(), [1.0] * 366, pd.Timestamp("2026-11-16"), brk)
    assert np.isnan(f["prev60_pct"]) and not np.isnan(f["prev30_pct"])
    assert any("series break" in m for m in missing)
    r = run_station(s, st="ORO", snow=snow_df("ORO"), breaks=brk)
    assert r["status"] == "unavailable"


# ---------------------------------------------------------------- 9, 12: conversion and dates

def test_pp_to_acre_feet_and_projection():
    r = run_station(storage_series(), cap=977_000.0, model=ConstModel(4.3))
    f, ob = r["forecast"], r["observed"]
    assert f["change_af"] == pytest.approx(0.043 * 977_000.0)
    assert f["projected_af"] == pytest.approx(ob["storage_af"] + 0.043 * 977_000.0)
    assert f["projected_pct"] == pytest.approx(ob["storage_pct"] + 4.3)
    assert f["lower_change_pp"] == pytest.approx(4.3 - 4.0) and f["upper_change_pp"] == pytest.approx(4.3 + 4.0)


def test_target_is_origin_plus_30_days_and_in_the_future():
    r = run_station(storage_series())
    o, t = pd.Timestamp(r["origin_date"]), pd.Timestamp(r["target_date"])
    assert o.dayofweek == 0 and (t - o).days == 30 and t > ISSUE
    assert candidate_origins(pd.Timestamp("2026-11-17"))[0] == pd.Timestamp("2026-11-16")
    assert candidate_origins(pd.Timestamp("2026-11-16"))[0] == pd.Timestamp("2026-11-09")   # Monday not yet complete
    assert all((pd.Timestamp("2026-11-22") - o).days <= MAX_ORIGIN_AGE_DAYS for o in candidate_origins(ISSUE))


# ---------------------------------------------------------------- 10, 11: missing and stale data

def test_missing_inputs_block_the_forecast():
    s = storage_series()
    s2 = s.drop(index=[pd.Timestamp("2026-11-16"), pd.Timestamp("2026-11-09")])
    r = run_station(s2)
    assert r["status"] == "unavailable" and "forecast" not in r and r["reasons"]
    sn = snow_series()
    sn[(sn.index >= "2026-11-01")] = np.nan
    r = run_station(s, snow=snow_df("FOL", sn))
    assert r["status"] == "unavailable"
    assert any("SWE" in m for o in r["origins_considered"] for m in o["missing"])
    r = run_station(pd.Series(dtype=float))
    assert r["status"] == "unavailable" and "CDEC" in r["reasons"][0]


def test_model_never_sees_missing_values():
    m = ConstModel(1.0)
    run_station(storage_series(), model=m)
    assert m.seen is not None and np.isfinite(m.seen.to_numpy()).all()


def test_stale_origin_flagged_and_old_origins_refused():
    r = run_station(storage_series(end="2026-11-10"))     # newest Monday 11-16 missing; 11-09 is 13 days old
    assert r["status"] == "ok" and r["origin_date"] == "2026-11-09"
    assert any(w.startswith("Stale origin") for w in r["warnings"])
    assert any("storage observation" in w for w in r["warnings"])
    r = run_station(storage_series(end="2026-11-06"))     # nothing within 14 days
    assert r["status"] == "unavailable" and "forecast" not in r


def test_implausible_projection_is_flagged_not_clipped():
    r = run_station(storage_series(base=990_000.0, slope=0.0), cap=1_000_000.0, model=ConstModel(5.0))
    assert r["forecast"]["projected_pct"] == pytest.approx(104.0)
    assert any("outside 0-100%" in w for w in r["warnings"])
    assert r["plausibility"]["projected_within_0_100_pct"] is False


# ---------------------------------------------------------------- 13: intervals

def test_interval_uses_origin_season_and_keeps_folsom_warning():
    r = run_station(storage_series())                     # November origin: SON offsets
    assert r["interval"]["season"] == "SON" and r["interval"]["offsets_pp"] == [-4.0, 4.0]
    assert r["interval"]["future_coverage_established"] is False
    assert r["interval"]["interval_reliable"] is False
    assert any(w.startswith("Interval caution") for w in r["warnings"])
    r2 = run_station(storage_series(), st="SHA", snow=snow_df("SHA"))
    assert not any(w.startswith("Interval caution") for w in r2["warnings"])


# ---------------------------------------------------------------- 14: records are not test rows; validation

def snapshot(recs):
    return {"schema": "watershedshift-forecast/1.0", "record_type": "prospective_research_forecast",
            "is_heldout_test_prediction": False, "data_mode": "real", "forecast_id": "x", "issue_date": "2026-11-22",
            "horizon_days": 30, "model": {"model_version": "v2-frozen", "feature_order": FEATURE_ORDER},
            "stations": recs}


def test_validation_accepts_good_and_rejects_bad_records():
    good = snapshot([run_station(storage_series())])
    assert validate(good, today=ISSUE.date())[0] == []
    for mutate in (lambda s: s.update(is_heldout_test_prediction=True),
                   lambda s: s.update(record_type="test_prediction"),
                   lambda s: s["stations"][0].update(target_date="2026-12-17"),
                   lambda s: s["stations"][0]["forecast"].update(change_af=1.0),
                   lambda s: s["stations"][0].update(warnings=[]),                   # Folsom warning removed
                   lambda s: s["stations"][0]["features"].update(rain_7d_mm=1.0),
                   lambda s: s["model"].update(model_version="retrained-2025"),
                   lambda s: s["stations"][0]["interval"].update(future_coverage_established=True)):
        bad = copy.deepcopy(good)
        mutate(bad)
        assert validate(bad, today=ISSUE.date())[0], mutate


def test_expired_and_future_issue_rejected():
    good = snapshot([run_station(storage_series())])
    assert expired(good, pd.Timestamp("2026-12-16").date())
    assert validate(good, today=pd.Timestamp("2026-12-20").date())[0]
    assert validate(good, today=pd.Timestamp("2026-11-21").date())[0]          # issued "tomorrow"
    assert any("days ago" in w for w in validate(good, today=pd.Timestamp("2026-12-05").date())[1])


def test_forecast_records_do_not_look_like_test_predictions():
    r = run_station(storage_series())
    test_cols = {"target_delta_pp", "pred_delta_pp", "lower_90_pp", "upper_90_pp", "wy"}
    assert not test_cols & set(r)
    assert "target_delta_pp" not in json.dumps(r)


def test_synthetic_fixture_is_valid_and_labelled():
    p = ROOT / "app/sample_data/forecast_snapshot.json"
    snap = json.loads(p.read_text())
    assert snap["data_mode"] == "synthetic" and "NOT the v2 model" in snap["model"]["model_name"]
    assert validate(snap)[0] == []
    assert all(r["station_id"].startswith("SYN") for r in snap["stations"])


def test_sample_data_contract_unaffected_by_forecast_fixture():
    from src.contract import validate_dir
    assert validate_dir(ROOT / "app/sample_data") == []


def test_live_downloads_never_touch_tracked_raw_manifest():
    import src.forecast_live as fl
    assert fl.LIVE.relative_to(ROOT).parts[:2] == ("data", "live")
    ignored = (ROOT / ".gitignore").read_text()
    assert "data/live/" in ignored and "forecasts/" in ignored and "models/*" in ignored


# ---------------------------------------------------------------- 8 and equivalence with the historical pipeline (private data)

@needs_data
def test_inference_features_equal_historical_features_on_all_test_origins():
    from src import build_dataset as bd
    from src.features import load_breaks
    from src.models import add_snow_anomaly, snow_climatology
    w = pd.read_csv(ROOT / "data/processed/weekly_features.csv", parse_dates=["date", "target_date"])
    test = w[(w.date >= "2024-01-01") & (w.target_date <= "2025-12-31")]
    idx = pd.read_csv(ROOT / "data/processed/snow_basin_index.csv", parse_dates=["date"])
    clim = snow_climatology(idx, pd.Timestamp("2024-01-01"))
    test = add_snow_anomaly(test, clim)
    daily = pd.read_csv(ROOT / "data/processed/daily_joined.csv", parse_dates=["date"])
    hist = pd.read_csv(ROOT / "data/capacity_history.csv")
    breaks = load_breaks()
    n = 0
    for st, g in test.groupby("station_id"):
        s = daily[daily.station_id == st].set_index("date").storage_af.dropna()
        sw = idx[idx.station_id == st].set_index("date").swe_basin_in.dropna()
        cap_at = lambda d, st=st: float(bd.capacity_series(st, pd.DatetimeIndex([pd.Timestamp(d)]), hist).iloc[0])  # noqa: E731
        cl = clim[st].reindex(range(1, 367)).to_numpy()
        for row in g.itertuples():
            f, missing = inference_features(s, cap_at, sw, cl, row.date, breaks.get(st, []))
            assert not missing, (st, row.date, missing)
            for k in FEATURES_15:
                assert f[k] == pytest.approx(getattr(row, k), abs=1e-9), (st, row.date, k)
            n += 1
    assert n == 593


@needs_art
def test_reconstructed_model_reproduces_frozen_predictions():
    from src.forecast_model import TEST_END, TEST_START, TRAIN_END, load_artifact, verify_against_archive
    from src.models import FEATURE_SETS
    from src.train_v2 import load
    model, c = load_artifact()
    _, test, _ = load(TRAIN_END, TEST_START, TEST_END)
    check = verify_against_archive(model, test, c["stations"], FEATURE_SETS["storage_snow"])
    assert check["n_rows"] == 593 and check["max_abs_diff_pp"] <= 1e-9
    assert c["feature_order"] == FEATURE_ORDER and c["training_cutoff_exclusive"] == "2024-01-01"


@needs_art
def test_card_interval_offsets_equal_archived_intervals():
    from src.forecast_model import load_artifact
    _, c = load_artifact()
    arc = pd.read_csv(ROOT / "results/v2/final/test_predictions_public.csv")
    for season, g in arc.groupby("season"):
        lo, hi = c["interval_offsets_pp"][season]
        assert np.allclose(g.lower_90_pp - g.pred_delta_pp, lo) and np.allclose(g.upper_90_pp - g.pred_delta_pp, hi)
    assert c["interval_reliable"]["FOL"] is False and c["historical_interval_coverage"]["FOL"] < 0.85


@needs_art
def test_artifact_refuses_tampered_model(tmp_path):
    import shutil
    from src.forecast_model import ART_DIR, load_artifact
    d = tmp_path / "art"
    shutil.copytree(ART_DIR, d)
    with open(d / "model.joblib", "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(ValueError):
        load_artifact(d)
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "missing")


@needs_art
def test_end_to_end_offline_forecast_from_cached_cdec_files(tmp_path):
    """Uses the most recent cached live download (no network). Skips if none exists."""
    import src.forecast_live as fl
    runs = sorted(p for p in fl.LIVE.glob("*") if (p / "fetch_log.jsonl").exists())
    if not runs:
        pytest.skip("no cached live CDEC download")
    issue = runs[-1].name
    snap = fl.run(issue, offline=runs[-1], out_dir=tmp_path)
    assert validate(snap, today=pd.Timestamp(issue).date())[0] == []
    ok = [r for r in snap["stations"] if r["status"] == "ok"]
    assert ok and snap["sources"]["all_files_match_logged_sha256"]
    for r in ok:
        assert (pd.Timestamp(r["target_date"]) - pd.Timestamp(r["origin_date"])).days == 30
        hist_used = [h for h in r["recent_history"] if h["used_by_forecast"]]
        assert max(h["date"] for h in hist_used) == r["origin_date"]

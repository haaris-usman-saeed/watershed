"""Offline tests for src/v5 (synthetic data only; no network, no private data, no PyTorch)."""
import numpy as np
import pandas as pd
import pytest

from src.v5 import STATIONS


def synthetic_panel(start="2010-01-01", days=500, seed=0, gefs=True):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    frames = []
    for i, st in enumerate(STATIONS):
        cap = 1e6 * (i + 1)
        s = cap * (0.5 + 0.2 * np.sin(np.arange(days) / 40 + i)) + rng.normal(0, 1e3, days)
        g = pd.DataFrame({"station_id": st, "date": dates, "seg": 0, "storage_af": s, "capacity_af": cap,
                          "swe_in": np.clip(10 + rng.normal(0, 3, days), 0, None), "rain_mm": rng.gamma(0.5, 4, days),
                          "temp_c": 10 + rng.normal(0, 5, days), "rh_pct": 50 + rng.normal(0, 10, days),
                          "rain_box_mm": rng.gamma(0.5, 4, days), "temp_box_c": 8 + rng.normal(0, 5, days),
                          "rh_box_pct": 55 + rng.normal(0, 10, days), "inflow_cfs": 1000 + rng.normal(0, 100, days),
                          "outflow_cfs": np.abs(1000 + rng.normal(0, 100, days))})
        g.loc[rng.choice(days, 20, replace=False), "storage_af"] = np.nan
        g.loc[rng.choice(days, 30, replace=False), "inflow_cfs"] = np.nan
        if gefs:
            g["gefs_p_d1_7_mm"] = np.where(rng.random(days) < 0.9, rng.gamma(1, 10, days), np.nan)
            g["gefs_p_d8_16_mm"] = np.where(g.gefs_p_d1_7_mm.notna(), rng.gamma(1, 15, days), np.nan)
            g["gefs_age_days"] = np.where(g.gefs_p_d1_7_mm.notna(), (dates.dayofweek.to_numpy()) % 7, np.nan)
            g["gefs_source"] = "reforecast"
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ data

def test_align_gefs_age_rule():
    from src.v5.data import align_gefs
    cal = pd.date_range("2020-01-01", "2020-02-10", freq="D")
    cyc = pd.DataFrame({"init_date": pd.to_datetime(["2019-12-30", "2020-01-06", "2020-01-20"]), "source": "operational",
                        "station_id": "SHA", "gefs_p_d1_7_mm": [1.0, 2.0, 3.0], "gefs_p_d8_16_mm": [4.0, 5.0, 6.0]})
    a = align_gefs(cal, cyc)
    assert a.loc["2020-01-06", "gefs_p_d1_7_mm"] == 2.0 and a.loc["2020-01-06", "gefs_age_days"] == 0
    assert a.loc["2020-01-12", "gefs_p_d1_7_mm"] == 2.0 and a.loc["2020-01-12", "gefs_age_days"] == 6
    assert np.isnan(a.loc["2020-01-13", "gefs_p_d1_7_mm"])             # 2020-01-13 cycle missing; 7-day-old one not used
    assert a.loc["2020-01-20", "gefs_p_d1_7_mm"] == 3.0
    assert np.isnan(a.loc["2020-02-03", "gefs_p_d1_7_mm"])
    assert a.loc["2020-01-01", "gefs_p_d1_7_mm"] == 1.0                 # never a cycle issued after the day


def test_daily_features_information_rule():
    from src.v5.data import daily_features
    p = synthetic_panel(seed=1)
    f0 = daily_features(p)
    t = pd.Timestamp("2010-08-02")
    q = p.copy()
    fut = q.date > t
    for c in ("storage_af", "inflow_cfs", "outflow_cfs", "swe_in", "rain_mm", "temp_c", "rain_box_mm"):
        q.loc[fut, c] = q.loc[fut, c] * 3 + 7
    q.loc[(q.date > t - pd.Timedelta(days=7)) & (q.date <= t), ["rain_mm", "temp_c", "rh_pct", "rain_box_mm"]] = 999.0
    q.loc[q.date == t, "swe_in"] = 999.0                                    # snow on day t is not known at t
    f1 = daily_features(q)
    cols = [c for c in f0.columns if c not in ("target_delta_pp", "target_date", "station_id", "date", "v2_row",
                                               "storage_af", "capacity_af", "gefs_source")]
    a = f0[f0.date == t].set_index("station_id")[cols]
    b = f1[f1.date == t].set_index("station_id")[cols]
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(f0[f0.date == t].target_delta_pp.fillna(0), f1[f1.date == t].target_delta_pp.fillna(0))


def test_flow_features_need_complete_window():
    from src.v5.data import daily_features
    p = synthetic_panel(seed=2)
    p["inflow_cfs"] = 100.0
    p.loc[(p.station_id == "SHA") & (p.date == "2010-05-10"), "inflow_cfs"] = np.nan
    f = daily_features(p).set_index(["station_id", "date"])
    assert np.isnan(f.loc[("SHA", pd.Timestamp("2010-05-12")), "inflow_7d_pp"])
    assert np.isnan(f.loc[("SHA", pd.Timestamp("2010-06-08")), "inflow_30d_pp"])
    v = f.loc[("SHA", pd.Timestamp("2010-05-20")), "inflow_7d_pp"]
    cap = p[p.station_id == "SHA"].capacity_af.iloc[0]
    assert v == pytest.approx(100 * 100 * 1.983471 / cap)


# ------------------------------------------------------------------ sequences

def test_seqdata_matches_v4_without_gefs():
    from src.v4.sequences import SequenceData, fold_split
    from src.v5.sequences import SeqData, split_at
    p = synthetic_panel(seed=3)
    end = pd.Timestamp("2024-01-01")
    a = SequenceData(p.drop(columns=["gefs_p_d1_7_mm", "gefs_p_d8_16_mm", "gefs_age_days", "gefs_source"]), lmax=60)
    b = SeqData(p, end, lmax=60, gefs=False)
    assert len(a.samples) == len(b.samples)
    idx = np.arange(0, len(a.samples), 7)
    for u, v in zip(a.batch(idx, 30), b.batch(idx, 30)):
        np.testing.assert_array_equal(u, v)
    for k in ("y30", "traj", "traj_mask", "cum_in", "cum_in_ok", "s0_pct"):
        np.testing.assert_array_equal(a.targets(idx)[k], b.targets(idx)[k])
    np.testing.assert_array_equal(a.future(idx), b.future(idx))
    # fold split equals the v4 rule for a water-year start
    p2 = synthetic_panel(start="2008-01-01", days=1400, seed=4)
    a2 = SequenceData(p2.drop(columns=["gefs_p_d1_7_mm", "gefs_p_d8_16_mm", "gefs_age_days", "gefs_source"]), lmax=40)
    b2 = SeqData(p2, end, lmax=40)
    fa, va = fold_split(a2.samples, 2011, stride=2)
    fb, vb = split_at(b2.samples, "2010-10-01", stride=2)
    np.testing.assert_array_equal(fa, fb)
    np.testing.assert_array_equal(va, vb)


def test_seqdata_gefs_static_and_drop():
    from src.v5.sequences import SeqData
    p = synthetic_panel(seed=5)
    d = SeqData(p, pd.Timestamp("2024-01-01"), lmax=40, gefs=True)
    assert d.n_static == 5
    idx = np.arange(0, len(d.samples), 5)
    x, s, st = d.batch(idx, 20)
    miss = d.samples.gefs_p_d1_7_mm.isna().to_numpy()[idx]
    assert (s[miss, 1:] == 0).all() and (s[~miss, 3] == 1).all()
    _, s2, _ = d.batch(idx, 20, drop_sources=("gefs",))
    assert (s2[:, 1:] == 0).all() and np.array_equal(s2[:, 0], s[:, 0])


def test_split_at_purges():
    from src.v5.sequences import SeqData, split_at
    p = synthetic_panel(start="2008-01-01", days=1400, seed=6)
    d = SeqData(p, pd.Timestamp("2024-01-01"), lmax=40)
    fit, val = split_at(d.samples, "2011-01-01")
    s = d.samples
    assert s.target_date.loc[fit].max() < s.date.loc[val].min()
    assert s.target_date.loc[np.concatenate([fit, val])].max() < pd.Timestamp("2011-01-01")
    assert ((s.date.loc[fit] - s.date.min()).dt.days % 2 == 0).all()


# ------------------------------------------------------------------ evaluation

def _pred_frame(seed=7, n_dates=60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-04", periods=n_dates, freq="7D")
    rows = [{"date": d, "station_id": s, "target_delta_pp": rng.normal(0, 5)} for d in dates for s in STATIONS]
    df = pd.DataFrame(rows)
    df["a"] = df.target_delta_pp + rng.normal(0, 2, len(df))
    df["b"] = df.target_delta_pp + rng.normal(0, 2.4, len(df))
    df["c"] = df.target_delta_pp + rng.normal(0.3, 2, len(df))
    return df


def test_contrast_bootstrap_matches_src_evaluation():
    from src.evaluation import block_bootstrap_diff
    from src.v5.evaluate import contrast_bootstrap
    df = _pred_frame()
    for block in (5, 9, 13):
        ref = block_bootstrap_diff(df, "a", "b", block=block)
        mine = contrast_bootstrap(df, {"a": 1, "b": -1}, block=block)
        assert mine["estimate"] == pytest.approx(ref["estimate"])
        assert mine["ci95_low"] == pytest.approx(ref["ci95_low"])
        assert mine["ci95_high"] == pytest.approx(ref["ci95_high"])


def test_contrast_did_and_pvalue():
    from src.v5.evaluate import contrast_bootstrap, test_contrast
    df = _pred_frame(seed=8)
    r = contrast_bootstrap(df, {"a": 1, "b": -1, "c": -1}, block=5)
    mae = {k: np.mean(np.abs(df[k] - df.target_delta_pp)) for k in "abc"}
    assert r["estimate"] == pytest.approx(mae["a"] - mae["b"] - mae["c"])
    t = test_contrast(df, {"a": 1, "b": -1})
    assert t["p_max"] == max(v["p_two_sided"] for v in t["blocks"].values())
    same = test_contrast(df.assign(a2=df.a), {"a": 1, "a2": -1})
    assert same["estimate"] == 0 and same["p_max"] == 1.0


def test_holm():
    from src.v5.evaluate import holm
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm([0.5]), [0.5])


def test_conformal_uses_only_earlier_folds():
    from src.v5.evaluate import conformal90
    df = pd.DataFrame({"fold": ["A"] * 10 + ["B"] * 10 + ["C"] * 10, "target_delta_pp": 0.0,
                       "m": list(np.arange(10.0)) + [100.0] * 10 + [0.0] * 10})
    hw = conformal90(df, "m", ["A", "B", "C"])
    assert hw[df.fold == "A"].isna().all()
    assert hw[df.fold == "B"].iloc[0] == np.quantile(np.arange(10.0), 0.9, method="higher")
    assert hw[df.fold == "C"].iloc[0] == 100.0


def test_point_metrics():
    from src.v5.evaluate import point_metrics
    m = point_metrics([0, 1, 2, 3], [0, 1, 2, 5])
    assert m["mae"] == 0.5 and m["max_ae"] == 2 and m["bias"] == 0.5 and m["median_ae"] == 0
    assert m["rmse"] == pytest.approx(1.0) and m["r2"] == pytest.approx(1 - 4 / 5)


# ------------------------------------------------------------------ registry and tabular

def test_search_space_deterministic_and_frozen_first():
    from src.v5.registry import FROZEN_HGB, NN_DEFAULT, N_CONFIGS, TUNED, search_space
    for t in TUNED:
        a, b = search_space(t), search_space(t)
        assert a == b and len(a) == N_CONFIGS
    assert search_space("T-RIDGE")[0] == {"alpha": "ridgecv"}
    h0 = search_space("T-HGB-V5")[0]
    assert all(h0[k] == v for k, v in FROZEN_HGB.items()) and h0["max_leaf_nodes"] == 31 and h0["max_features"] == 1.0
    assert search_space("T-N1")[0] == NN_DEFAULT
    assert search_space("T-N4")[0] == dict(NN_DEFAULT, cand="wb_noaux")


def test_hgb_v5_has_29_columns_and_codes():
    from src.v5.registry import TABULAR
    from src.v5.tabular import matrix
    p = synthetic_panel(seed=9)
    from src.v5.data import daily_features
    f = daily_features(p)
    f["swe_anom_in"] = 0.0
    assert matrix(f.head(5), "HGB-V5").shape[1] == 29
    assert matrix(f.head(5), "A-HGB-DAILY", daily=True).shape[1] == 30
    assert matrix(f.head(5), "B-HGB-SNCR").shape[1] == 15
    assert set(TABULAR["HGB-V5"]["codes"]) == {"S", "N", "Wd", "Wu", "G", "Q", "C", "R"}


def test_fold_frames_training_precedes_fold():
    from src.v5.data import daily_features
    from src.v5.tabular import fold_frames
    p = synthetic_panel(start="2008-01-01", days=1500, seed=10)
    f = daily_features(p)
    f["v2_row"] = (f.date.dt.dayofweek == 0) & f.target_delta_pp.notna() & f.prev60_pct.notna()
    snow = p[["station_id", "date", "swe_in"]].rename(columns={"swe_in": "swe_basin_in"})
    start = pd.Timestamp("2010-10-01")
    tr, te, daily = fold_frames(f, snow, start, start, pd.Timestamp("2011-09-30"), pd.Timestamp("2011-12-31"))
    assert tr.target_date.max() < start and daily.target_date.max() < start
    assert te.date.min() >= start and "swe_anom_in" in tr and "swe_anom_in" in daily
    assert ((daily.date - pd.Timestamp("2001-01-01")).dt.days % 2 == 0).all()

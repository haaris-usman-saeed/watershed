"""v4 (NeuralHydro) offline tests: daily panel, leakage-safe sequences, masks, chronological folds, flow parsing
and audit; and, when PyTorch is installed (requirements-v4.txt), the networks and training loop.
Synthetic data only: no network, no private files."""
import numpy as np
import pandas as pd
import pytest

from src.v4 import DEV_END
from src.v4.dataset import build_panel
from src.v4.sequences import CHANNELS, HORIZON, SNOW_LAG, WEATHER_LAG, SequenceData, fold_split, monday_rows

ST = ("AAA", "BBB")
CH = {name: i for i, (name, _, _) in enumerate(CHANNELS)}


def synthetic_inputs(start="2010-01-01", days=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    daily, snow, power, flows = [], [], [], []
    for k, st in enumerate(ST):
        cap = 1_000_000.0 * (k + 1)
        inflow = 2000 + 1500 * np.sin(np.arange(days) / 58) + rng.normal(0, 50, days)
        outflow = np.full(days, 2200.0)
        storage = 500_000.0 * (k + 1) + np.cumsum((inflow - outflow) * 1.983471)
        daily.append(pd.DataFrame({"station_id": st, "date": dates, "storage_af": storage, "capacity_af": cap}))
        snow.append(pd.DataFrame({"station_id": st, "date": dates, "swe_basin_in": 10 + np.sin(np.arange(days) / 40)}))
        for src in ("dam_point", "upstream_box_mean"):
            power.append(pd.DataFrame({"station_id": st, "source": src, "date": dates, "rain_mm": rng.gamma(1, 2, days),
                                       "temp_c": 10 + rng.normal(0, 3, days), "humidity_pct": 60 + rng.normal(0, 5, days)}))
        flows.append(pd.DataFrame({"station_id": st, "date": dates, "inflow_cfs": inflow, "outflow_cfs": outflow}))
    return pd.concat(daily), pd.concat(snow), pd.concat(power), pd.concat(flows)


def panel(days=900, end=None, breaks=None, **kw):
    daily, snow, power, flows = synthetic_inputs(days=days, **kw)
    end = end or pd.Timestamp("2010-01-01") + pd.Timedelta(days=days)
    return build_panel(daily, snow, power, flows, breaks=breaks or {}, start=pd.Timestamp("2010-01-01"), end=end,
                       stations=ST)


def seqdata(p, lmax=120):
    return SequenceData(p, lmax=lmax, stations=ST)


def sample_at(d, st, date):
    s = d.samples
    return int(s.index[(s.station_id == st) & (s.date == pd.Timestamp(date))][0])


# ---------------------------------------------------------------- panel and targets

def test_panel_is_a_full_calendar_and_never_fills_missing_values():
    daily, snow, power, flows = synthetic_inputs(days=200)
    daily = daily[~((daily.station_id == "AAA") & (daily.date == "2010-03-01"))]
    p = build_panel(daily, snow, power, flows, breaks={}, start=pd.Timestamp("2010-01-01"),
                    end=pd.Timestamp("2010-07-20"), stations=ST)
    assert len(p) == 2 * 200 and p.groupby("station_id").date.apply(lambda d: d.diff().dropna().nunique()).eq(1).all()
    assert np.isnan(p.loc[(p.station_id == "AAA") & (p.date == "2010-03-01"), "storage_af"]).all()


def test_targets_match_definitions():
    p = panel()
    d = seqdata(p)
    g = p[p.station_id == "BBB"].set_index("date")
    i = sample_at(d, "BBB", "2011-02-07")
    t = pd.Timestamp("2011-02-07")
    s0, s30, cap = g.storage_af[t], g.storage_af[t + pd.Timedelta(days=30)], g.capacity_af[t]
    row = d.samples.loc[i]
    assert row.y30 == pytest.approx(100 * (s30 - s0) / cap)
    assert row.traj[29] == pytest.approx(row.y30, rel=1e-5) and row.traj_mask.all()
    days = pd.date_range(t + pd.Timedelta(days=1), periods=30)
    assert row.cum_in == pytest.approx(100 * (g.inflow_cfs[days] * 1.983471).sum() / cap)
    assert row.target_date == t + pd.Timedelta(days=HORIZON)
    # synthetic storage obeys the water balance exactly, so inflow minus outflow reproduces the target
    assert row.cum_in - row.cum_out == pytest.approx(row.y30, rel=1e-6)


def test_no_sample_when_target_storage_is_missing_or_crosses_a_break():
    daily, snow, power, flows = synthetic_inputs(days=400)
    daily.loc[(daily.station_id == "AAA") & (daily.date == "2010-08-01"), "storage_af"] = np.nan
    p = build_panel(daily, snow, power, flows, breaks={"BBB": [pd.Timestamp("2010-09-15")]},
                    start=pd.Timestamp("2010-01-01"), end=pd.Timestamp("2011-02-05"), stations=ST)
    d = seqdata(p)
    s = d.samples
    assert not ((s.station_id == "AAA") & (s.date == "2010-07-02")).any()          # t+30 missing
    bbb = s[s.station_id == "BBB"]
    crosses = (bbb.date < "2010-09-15") & (bbb.target_date >= "2010-09-15")
    assert not crosses.any() and len(bbb) > 0


# ---------------------------------------------------------------- leakage

def _inputs(p, st, date, length=60):
    d = seqdata(p)
    x, static, station = d.batch([sample_at(d, st, date)], length)
    return x[0]


def test_nothing_after_the_origin_and_lagged_sources_cannot_change_inputs():
    p0 = panel()
    t = pd.Timestamp("2011-03-14")
    base = _inputs(p0, "AAA", t)

    def changed(mask_fn, cols):
        p = p0.copy()
        m = (p.station_id == "AAA") & mask_fn(p.date)
        p.loc[m, cols] = p.loc[m, cols] * 3 + 7
        return not np.allclose(_inputs(p, "AAA", t), base)

    allcols = ["storage_af", "swe_in", "rain_mm", "temp_c", "rh_pct", "rain_box_mm", "temp_box_c", "rh_box_pct",
               "inflow_cfs", "outflow_cfs"]
    wx = ["rain_mm", "temp_c", "rh_pct", "rain_box_mm", "temp_box_c", "rh_box_pct"]
    assert not changed(lambda dt: dt > t, allcols)                                       # the future never enters
    assert not changed(lambda dt: (dt > t - pd.Timedelta(days=WEATHER_LAG)) & (dt <= t), wx)
    assert changed(lambda dt: dt == t - pd.Timedelta(days=WEATHER_LAG), wx)
    assert not changed(lambda dt: dt == t, ["swe_in"])                                  # snow lag 1 day
    assert changed(lambda dt: dt == t - pd.Timedelta(days=SNOW_LAG), ["swe_in"])
    assert changed(lambda dt: dt == t, ["storage_af"]) and changed(lambda dt: dt == t, ["inflow_cfs"])


def test_masks_mark_missing_withheld_and_padded_days():
    p = panel()
    p.loc[(p.station_id == "AAA") & (p.date == "2011-03-10"), "swe_in"] = np.nan
    d = seqdata(p, lmax=120)
    c = len(CHANNELS)
    x = d.batch([sample_at(d, "AAA", "2011-03-14")], 60)[0][0]
    last = 59
    assert x[last - 4, c + CH["swe_in"]] == 0 and x[last - 4, CH["swe_in"]] == 0         # missing: value 0, mask 0
    assert (x[last - WEATHER_LAG + 1:, c + CH["rain_mm"]] == 0).all()                     # withheld weather days
    assert x[last - WEATHER_LAG, c + CH["rain_mm"]] == 1
    early = d.batch([sample_at(d, "AAA", "2010-01-20")], 60)[0][0]                       # before the record starts
    assert (early[:40, c:2 * c] == 0).all() and (early[:40, :c] == 0).all()


def test_storage_before_a_series_break_is_masked():
    p = panel(breaks={"AAA": [pd.Timestamp("2011-03-01")]})
    d = seqdata(p)
    x = d.batch([sample_at(d, "AAA", "2011-03-14")], 30)[0][0]
    c = len(CHANNELS)
    assert (x[:16, c + CH["storage_pct"]] == 0).all() and (x[16:, c + CH["storage_pct"]] == 1).all()
    assert (x[:16, c + CH["inflow_pp"]] == 1).all()                                        # flows are not table-based


def test_source_removal_and_dropout():
    d = seqdata(panel())
    idx = np.arange(300, 500)
    c = len(CHANNELS)
    x = d.batch(idx, 30, drop_sources=("flow",))[0]
    assert (x[:, :, c + CH["inflow_pp"]] == 0).all() and (x[:, :, CH["outflow_pp"]] == 0).all()
    assert x[:, :, c + CH["storage_pct"]].any()
    x = d.batch(idx, 30, source_dropout=0.5, rng=np.random.default_rng(1))[0]
    dropped = (x[:, :, c + CH["swe_in"]].sum(1) == 0).mean()
    assert 0.3 < dropped < 0.7 and x[:, :, c + CH["storage_pct"]].all(axis=1).mean() > 0.99


# ---------------------------------------------------------------- folds and dates

def test_chronological_folds_never_overlap_target_windows():
    daily, snow, power, flows = synthetic_inputs(start="2003-01-01", days=2500)
    p = build_panel(daily, snow, power, flows, breaks={}, start=pd.Timestamp("2003-01-01"),
                    end=pd.Timestamp("2003-01-01") + pd.Timedelta(days=2500), stations=ST)
    d = seqdata(p)
    s = d.samples
    fit, val = fold_split(s, 2009)
    start = pd.Timestamp("2008-10-01")
    assert s.target_date[fit].max() < s.date[val].min()
    assert s.target_date[val].max() < start and s.date[val].min() >= start - pd.Timedelta(days=365)
    fit2, _ = fold_split(s, 2009, stride=3)
    assert 0.3 < len(fit2) / len(fit) < 0.37


def test_development_data_must_end_before_2024():
    with pytest.raises(AssertionError):
        panel(days=60, start="2023-12-15", end=pd.Timestamp("2024-02-13"))
    p = panel(days=60, start="2023-10-01", end=pd.Timestamp("2023-11-30"))
    p.loc[p.index[-1], "date"] = DEV_END
    with pytest.raises(AssertionError):
        SequenceData(p, lmax=30, stations=ST)


def test_monday_rows_must_reproduce_the_v2_target():
    d = seqdata(panel(days=900))
    s = d.samples
    mondays = s[(s.date.dt.dayofweek == 0) & (s.date >= "2010-10-01") & (s.date < "2011-10-01")]
    weekly = mondays[["station_id", "date", "target_date"]].assign(target_delta_pp=mondays.y30.to_numpy())
    idx, rows = monday_rows(s, weekly, 2011)
    assert len(idx) == len(weekly) and (s.date.to_numpy()[idx] == rows.date.to_numpy()).all()
    weekly.loc[weekly.index[0], "target_delta_pp"] += 0.5
    with pytest.raises(AssertionError):
        monday_rows(s, weekly, 2011)


# ---------------------------------------------------------------- flows

def test_cdec_flow_parsing_cleaning_and_fetch_window():
    from src.v4.flows import clean, fetch, parse_cdec
    text = ("STATION_ID,DURATION,SENSOR_NUMBER,SENSOR_TYPE,DATE TIME,OBS DATE,VALUE,DATA_FLAG,UNITS\n"
            "FOL,D,23,OUTFLOW,20050101 0000,20050101 0000,1200,,CFS\n"
            "FOL,D,23,OUTFLOW,20050102 0000,20050102 0000,---,,CFS\n"
            "FOL,D,23,OUTFLOW,20050103 0000,20050103 0000,-5,,CFS\n")
    d = clean(parse_cdec(text), "outflow_cfs")
    assert d.date.iloc[0] == pd.Timestamp("2005-01-01") and d.value.iloc[0] == 1200
    assert d.value.iloc[1:].isna().all()                                 # missing and negative outflow
    inflow = clean(parse_cdec(text.replace("23,OUTFLOW", "76,INFLOW")), "inflow_cfs")
    assert inflow.value.iloc[2] == -5                                    # computed inflow may be negative
    with pytest.raises(ValueError):
        fetch(end="2024-01-31", get=lambda params: None)


def test_flow_audit_closes_on_an_exact_water_balance():
    from src.v4.flows import audit
    res = audit(panel(days=500))
    for st in ST:
        r = res["stations"][st]
        assert r["daily_residual_median_abs_pct_capacity"] < 1e-6 and r["thirty_day_mae_pp_flow_sum_vs_storage"] < 1e-6


def test_hgb_flow_features_use_only_days_up_to_the_origin():
    from src.v4.experiment import FLOW_FEATURES, flow_features
    p = panel(days=200)
    t = pd.Timestamp("2010-05-03")
    rows = pd.DataFrame({"station_id": ["AAA"], "date": [t]})
    base = flow_features(p, rows)
    g = p[p.station_id == "AAA"].set_index("date")
    expect = (100 * g.inflow_cfs * 1.983471 / g.capacity_af)[t - pd.Timedelta(days=6):t].mean()
    assert base[0, FLOW_FEATURES.index("inflow_7d_pp")] == pytest.approx(expect)
    future = p.copy()
    future.loc[(future.station_id == "AAA") & (future.date > t), ["inflow_cfs", "outflow_cfs"]] *= 5
    assert np.allclose(flow_features(future, rows), base)
    gap = p.copy()
    gap.loc[(gap.station_id == "AAA") & (gap.date == t - pd.Timedelta(days=3)), "inflow_cfs"] = np.nan
    f = flow_features(gap, rows)[0]
    assert np.isnan(f[FLOW_FEATURES.index("inflow_7d_pp")]) and not np.isnan(f[FLOW_FEATURES.index("outflow_7d_pp")])


# ---------------------------------------------------------------- networks and training (PyTorch only)

def test_quantile_heads_never_cross_and_models_have_expected_shapes():
    torch = pytest.importorskip("torch")
    from src.v4.nets import QUANTILES, QuantileHead, build, n_params
    q = QuantileHead(8)(torch.randn(64, 8) * 10)
    assert (q[:, 1:] >= q[:, :-1]).all()
    for kind in ("lstm", "gru", "tcn"):
        m = build(kind, 22, 1, 6, 90, hidden=16)
        out = m(torch.randn(5, 90, 22), torch.randn(5, 1), torch.tensor([0, 1, 2, 3, 5]))
        assert out.shape == (5, len(QUANTILES)) and (out[:, 1:] >= out[:, :-1]).all()
        assert 1_000 < n_params(m) < 50_000
    from src.v4.experiment import DEFAULTS
    for kind in ("lstm", "gru", "tcn"):                  # the experiment's own defaults must build every model
        m = build(kind, 22, 1, 6, 30, **DEFAULTS)
        assert m(torch.randn(2, 30, 22), torch.randn(2, 1), torch.tensor([0, 5])).shape == (2, len(QUANTILES))


def test_training_is_seeded_reduces_loss_and_restores_the_best_epoch():
    pytest.importorskip("torch")
    from src.v4.nets import QUANTILES, build
    from src.v4.sequences import N_STATIC
    from src.v4.train import fit, predict, set_seed
    daily, snow, power, flows = synthetic_inputs(start="2003-01-01", days=1500)
    p = build_panel(daily, snow, power, flows, breaks={}, start=pd.Timestamp("2003-01-01"),
                    end=pd.Timestamp("2003-01-01") + pd.Timedelta(days=1500), stations=ST)
    d = seqdata(p, lmax=30)
    fit_idx, val_idx = fold_split(d.samples, 2007, stride=2)
    runs = []
    for _ in range(2):
        set_seed(3)
        m = build("lstm", d.n_inputs, N_STATIC, len(ST), 30, hidden=16)
        info = fit(m, d, fit_idx, val_idx, 30, seed=3, max_epochs=4, patience=10, batch=64)
        runs.append((info, predict(m, d, val_idx[:20], 30)))
    (a, pa), (b, pb) = runs
    assert a["curves"]["train_pinball"] == b["curves"]["train_pinball"] and np.allclose(pa, pb)
    assert a["curves"]["train_pinball"][-1] < a["curves"]["train_pinball"][0]
    assert a["best_val_pinball"] == pytest.approx(min(a["curves"]["val_pinball"]))
    assert pa.shape == (20, len(QUANTILES))

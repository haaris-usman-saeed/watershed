import numpy as np
import pandas as pd

from src.features import FEATURES, construct_features


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "station_id": "X", "reservoir_name": "Test", "capacity_af": 1000,
                         "storage_af": np.arange(n) * 2.0 + 300, "rain_mm": rng.gamma(1, 3, n),
                         "temp_c": rng.normal(15, 3, n), "humidity_pct": rng.uniform(30, 90, n)})


def test_target_definition_and_weather_window():
    df = toy()
    df["rain_mm"] = 1.0
    r = construct_features(df, breaks={})
    assert len(r) > 0
    assert np.allclose(r.target_delta_pp, 6.0)          # 30 days * 2 AF / 1000 AF * 100
    assert np.allclose(r.rain_30d_mm, 30.0)
    assert (r.target_date - r.date == pd.Timedelta(days=30)).all()


def test_features_do_not_use_future_data():
    """Perturb everything after origin t0 -> features at t0 must be unchanged."""
    df = toy()
    base = construct_features(df, breaks={}).set_index("date")
    t0 = base.index[2]
    pert = df.copy()
    after = pert.date > t0
    pert.loc[after, ["storage_af", "rain_mm", "temp_c", "humidity_pct"]] *= 3.7
    new = construct_features(pert, breaks={}).set_index("date")
    assert np.allclose(base.loc[t0, FEATURES].astype(float), new.loc[t0, FEATURES].astype(float))


def test_weather_lag_excludes_last_seven_days():
    df = toy()
    base = construct_features(df, breaks={}).set_index("date")
    t0 = base.index[2]
    pert = df.copy()
    recent = (pert.date > t0 - pd.Timedelta(days=7)) & (pert.date <= t0)
    pert.loc[recent, ["rain_mm", "temp_c", "humidity_pct"]] += 100
    new = construct_features(pert, breaks={}).set_index("date")
    wcols = ["rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"]
    assert np.allclose(base.loc[t0, wcols].astype(float), new.loc[t0, wcols].astype(float))


def test_missing_storage_is_not_interpolated():
    df = toy()
    r0 = construct_features(df, breaks={})
    t = r0.date.iloc[3]
    df.loc[df.date == t + pd.Timedelta(days=30), "storage_af"] = np.nan
    r1 = construct_features(df, breaks={})
    assert t not in set(r1.date)                          # row dropped, not filled


def test_no_window_spans_series_break():
    df = toy(n=300)
    brk = pd.Timestamp("2020-06-01")
    r = construct_features(df, breaks={"X": [brk]})
    spans_target = (r.date < brk) & (r.target_date >= brk)
    spans_prev60 = (r.date >= brk) & (r.date - pd.Timedelta(days=60) < brk)
    assert not spans_target.any()
    assert not spans_prev60.any()
    assert len(r) > 0

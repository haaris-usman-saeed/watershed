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


def snow_for(df, value=10.0):
    return pd.DataFrame({"station_id": "X", "date": df.date, "swe_basin_in": value})


def test_snow_features_use_only_t_minus_1_and_earlier():
    df = toy()
    si = snow_for(df)
    si["swe_basin_in"] = np.arange(len(si), dtype=float)
    base = construct_features(df, breaks={}, snow_index=si).set_index("date")
    t0 = base.index[2]
    pert = si.copy()
    pert.loc[pert.date >= t0, "swe_basin_in"] += 500       # change origin-day and later snow
    new = construct_features(df, breaks={}, snow_index=pert).set_index("date")
    assert base.loc[t0, "swe_basin_in"] == new.loc[t0, "swe_basin_in"]
    assert base.loc[t0, "swe_change_14d_in"] == new.loc[t0, "swe_change_14d_in"]
    assert base.loc[t0, "swe_change_14d_in"] == 14.0


def test_missing_snow_stays_missing_not_zero():
    df = toy()
    si = snow_for(df)
    si.loc[si.date < pd.Timestamp("2020-04-01"), "swe_basin_in"] = np.nan
    r = construct_features(df, breaks={}, snow_index=si)
    early = r[r.date < pd.Timestamp("2020-04-01")]
    assert len(early) > 0 and early.swe_basin_in.isna().all()


def test_excluded_station_never_enters_features(tmp_path):
    from src.features import eligible_stations, filter_eligible
    cov = pd.DataFrame({"station_id": ["X", "Y"], "status": ["included", "excluded"]})
    cov.to_csv(tmp_path / "coverage.csv", index=False)
    a, b = toy(), toy()
    b["station_id"] = "Y"
    data = filter_eligible(pd.concat([a, b]), eligible_stations(tmp_path / "coverage.csv"))
    assert set(construct_features(data, breaks={}).station_id) == {"X"}


def test_basin_index_requires_two_thirds_of_core_pillows():
    from src.features import build_snow_index
    days = pd.date_range("2001-10-01", "2023-09-30", freq="D")
    snow = pd.concat([pd.DataFrame({"snow_station_id": s, "date": days, "swe_in": v})
                      for s, v in [("A", 10.0), ("B", 20.0), ("C", 30.0)]])
    meta = pd.DataFrame({"snow_station_id": ["A", "B", "C"], "reservoir_id": "X"})
    d0 = pd.Timestamp("2010-01-01")
    drop = (snow.date == d0) & snow.snow_station_id.isin(["B", "C"])   # only 1 of 3 reports
    idx = build_snow_index(snow[~drop], meta).set_index("date")
    assert np.isnan(idx.loc[d0, "swe_basin_in"])
    assert idx.loc[pd.Timestamp("2010-01-02"), "swe_basin_in"] == 20.0

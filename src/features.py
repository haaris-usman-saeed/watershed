"""Weekly reservoir-date table with leakage-safe features and 30-day targets (v2).

Run: python -m src.features   (reads data/processed/daily_joined.csv and snow_daily.csv)

Rules
- Forecast origin t = Monday; storage features use end-of-day storage up to and including t.
- Weather features use only days <= t - WEATHER_LAG (7 d; observed POWER latency about 3 d).
- Snow features use only basin SWE dated <= t - SNOW_LAG (1 d).
- Target y = 100 * (storage[t+30] - storage[t]) / capacity in force at t, exact calendar-day
  match, never interpolated; rows whose t+30 (or lagged) storage is missing are dropped.
- No difference (feature or target) may span a documented series break
  (data/series_breaks.csv), e.g. the ORO elevation-storage table change on 2024-07-02.
- Only stations with status "included" in data/processed/coverage.csv are used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 30
WEATHER_LAG = 7
SNOW_LAG = 1
STORAGE = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct"]
WEATHER = ["rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"]
SEASON = ["season_sin", "season_cos"]
SNOW = ["swe_basin_in", "swe_change_14d_in"]          # swe_anom_in is added per fold (src/experiments.py)
FEATURES = STORAGE + WEATHER + SEASON                  # v1 feature set (always required)
CORE_SHARE, CORE_AVAIL, CORE_LAST_WY = 0.75, 0.80, 2023
MIN_CORE_FRACTION = 2 / 3


def load_breaks(path=ROOT / "data" / "series_breaks.csv"):
    if not Path(path).exists():
        return {}
    b = pd.read_csv(path, parse_dates=["break_date"])
    return {s: sorted(g.break_date) for s, g in b.groupby("station_id")}


def eligible_stations(coverage_path=ROOT / "data" / "processed" / "coverage.csv"):
    cov = pd.read_csv(coverage_path)
    return sorted(cov.loc[cov.status == "included", "station_id"])


def filter_eligible(data, eligible):
    return data[data.station_id.isin(eligible)].copy()


def core_snow_stations(snow, meta, first_wy=2002, last_wy=CORE_LAST_WY):
    """Predetermined availability rule (no outcomes used, WY<=2023 only): a pillow is core if
    >= 75% of Dec-May snow seasons have >= 80% daily availability."""
    s = snow.copy()
    s["wy"] = s.date.dt.year + (s.date.dt.month >= 10)
    s = s[s.date.dt.month.isin([12, 1, 2, 3, 4, 5]) & s.wy.between(first_wy, last_wy)]
    days = pd.Series({wy: len(pd.date_range(f"{wy - 1}-12-01", f"{wy}-05-31")) for wy in range(first_wy, last_wy + 1)})
    av = s.groupby(["snow_station_id", "wy"]).size().unstack(fill_value=0).reindex(columns=days.index, fill_value=0) / days
    share = (av >= CORE_AVAIL).mean(axis=1)
    core = set(share[share >= CORE_SHARE].index)
    return meta[meta.snow_station_id.isin(core)][["snow_station_id", "reservoir_id"]]


def build_snow_index(snow, meta):
    """Daily basin index: mean SWE (inches) over core pillows reporting that day, only when at
    least 2/3 of the basin's core pillows report; otherwise NaN. Missing is never zero."""
    core = core_snow_stations(snow, meta)
    s = snow.merge(core, on="snow_station_id")
    n_core = core.groupby("reservoir_id").size()
    g = s.groupby(["reservoir_id", "date"]).swe_in.agg(["mean", "count"]).reset_index()
    g["n_core"] = g.reservoir_id.map(n_core)
    g["swe_basin_in"] = g["mean"].where(g["count"] >= MIN_CORE_FRACTION * g.n_core)
    return g.rename(columns={"reservoir_id": "station_id", "count": "n_core_reporting"})[
        ["station_id", "date", "swe_basin_in", "n_core_reporting", "n_core"]]


def construct_features(data: pd.DataFrame, horizon=HORIZON, weather_lag=WEATHER_LAG,
                       breaks=None, weekday=0, snow_index=None) -> pd.DataFrame:
    breaks = load_breaks() if breaks is None else breaks
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    frames = []
    for station, g in data.sort_values("date").groupby("station_id"):
        g = g.set_index("date")
        full = pd.date_range(g.index.min(), g.index.max(), freq="D", name="date")
        g = g.reindex(full)  # missing days stay NaN, so shift(k) is always exactly k calendar days
        g["station_id"] = station
        g["capacity_af"] = g.capacity_af.ffill().bfill()
        g["reservoir_name"] = g.reservoir_name.ffill().bfill()
        seg = pd.Series(0, index=full)
        for b in breaks.get(station, []):
            seg[full >= b] += 1
        cap, s = g.capacity_af, g.storage_af

        def diff_back(k):  # storage[t] - storage[t-k], same segment only, capacity in force at t
            same = seg == seg.shift(k)
            return (100 * (s - s.shift(k)) / cap).where(same)

        g["storage_pct"] = 100 * s / cap
        g["prev7_pct"], g["prev30_pct"], g["prev60_pct"] = diff_back(7), diff_back(30), diff_back(60)
        past = g[["rain_mm", "temp_c", "humidity_pct"]].shift(weather_lag)
        g["rain_7d_mm"] = past.rain_mm.rolling(7, min_periods=7).sum()
        g["rain_30d_mm"] = past.rain_mm.rolling(30, min_periods=30).sum()
        g["temp_30d_c"] = past.temp_c.rolling(30, min_periods=30).mean()
        g["humidity_30d_pct"] = past.humidity_pct.rolling(30, min_periods=30).mean()
        if snow_index is not None:
            si = snow_index[snow_index.station_id == station].set_index("date").swe_basin_in.reindex(full)
            g["swe_basin_in"] = si.shift(SNOW_LAG)
            g["swe_change_14d_in"] = si.shift(SNOW_LAG) - si.shift(SNOW_LAG + 14)
        doy = g.index.dayofyear
        g["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
        g["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
        g["future_storage_af"] = s.shift(-horizon).where(seg == seg.shift(-horizon))
        g["target_delta_pp"] = 100 * (g.future_storage_af - s) / cap
        g["target_date"] = g.index.shift(horizon, freq="D")
        frames.append(g.reset_index())
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.loc[rows.date.dt.dayofweek == weekday]
    return rows.dropna(subset=FEATURES + ["target_delta_pp"]).reset_index(drop=True)


def main():
    source = ROOT / "data/processed/daily_joined.csv"
    if not source.exists():
        raise FileNotFoundError("Run python -m src.build_dataset first")
    data = filter_eligible(pd.read_csv(source), eligible_stations())
    snow_path = ROOT / "data/processed/snow_daily.csv"
    snow_index = None
    if snow_path.exists():
        snow = pd.read_csv(snow_path, parse_dates=["date"])
        snow_index = build_snow_index(snow, pd.read_csv(ROOT / "data" / "snow_stations.csv"))
        snow_index.to_csv(ROOT / "data/processed/snow_basin_index.csv", index=False, date_format="%Y-%m-%d")
    out = construct_features(data, snow_index=snow_index)
    cols = ["station_id", "reservoir_name", "date", "target_date", "storage_af", "capacity_af",
            "future_storage_af", "target_delta_pp"] + FEATURES + [c for c in SNOW if c in out]
    out[cols].to_csv(ROOT / "data/processed/weekly_features.csv", index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(out)} weekly samples; stations={out.station_id.nunique()}")
    print(out.groupby("station_id").agg(samples=("date", "size"), start=("date", "min"), end=("date", "max"),
                                        swe_available=("swe_basin_in", lambda x: round(x.notna().mean(), 3))
                                        if "swe_basin_in" in out else ("date", "size")).to_string())


if __name__ == "__main__":
    main()

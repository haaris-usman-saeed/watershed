"""Weekly reservoir-date table with leakage-safe features and 30-day targets.

Run: python -m src.features   (reads data/processed/daily_joined.csv)

Rules
- Forecast origin t = Monday; storage features use end-of-day storage up to and including t.
- Weather features use only days <= t - WEATHER_LAG (7 d; observed POWER latency ~3 d).
- Target y = 100 * (storage[t+30] - storage[t]) / capacity, exact calendar-day match,
  never interpolated; rows whose t+30 (or lag) storage is missing are dropped.
- No difference (feature or target) may span a documented series break
  (data/series_breaks.csv), e.g. ORO elevation-storage table change on 2024-07-02.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 30
WEATHER_LAG = 7
FEATURES = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct",
            "rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct",
            "season_sin", "season_cos"]


def load_breaks(path=ROOT / "data" / "series_breaks.csv"):
    if not Path(path).exists():
        return {}
    b = pd.read_csv(path, parse_dates=["break_date"])
    return {s: sorted(g.break_date) for s, g in b.groupby("station_id")}


def construct_features(data: pd.DataFrame, horizon=HORIZON, weather_lag=WEATHER_LAG,
                       breaks=None, weekday=0) -> pd.DataFrame:
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

        def diff_back(k):  # storage[t] - storage[t-k], same segment only
            same = seg == seg.shift(k)
            return (100 * (s - s.shift(k)) / cap).where(same)

        g["storage_pct"] = 100 * s / cap
        g["prev7_pct"], g["prev30_pct"], g["prev60_pct"] = diff_back(7), diff_back(30), diff_back(60)
        past = g[["rain_mm", "temp_c", "humidity_pct"]].shift(weather_lag)
        g["rain_7d_mm"] = past.rain_mm.rolling(7, min_periods=7).sum()
        g["rain_30d_mm"] = past.rain_mm.rolling(30, min_periods=30).sum()
        g["temp_30d_c"] = past.temp_c.rolling(30, min_periods=30).mean()
        g["humidity_30d_pct"] = past.humidity_pct.rolling(30, min_periods=30).mean()
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
    out = construct_features(pd.read_csv(source))
    cols = ["station_id", "reservoir_name", "date", "target_date", "storage_af", "capacity_af",
            "future_storage_af", "target_delta_pp"] + FEATURES
    out[cols].to_csv(ROOT / "data/processed/weekly_features.csv", index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(out)} weekly samples; stations={out.station_id.nunique()}")
    print(out.groupby("station_id").agg(samples=("date", "size"), start=("date", "min"),
                                        end=("date", "max")).to_string())


if __name__ == "__main__":
    main()

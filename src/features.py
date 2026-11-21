"""Create leakage-aware historical features and independent 30-day targets."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]


def construct_features(data: pd.DataFrame, horizon=30, weather_lag=7) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["station_id", "date"])
    frames = []
    for station, group in data.groupby("station_id"):
        group = group.set_index("date").sort_index()
        # Reindex daily: missing reservoir dates must not silently become different horizons.
        full = pd.date_range(group.index.min(), group.index.max(), freq="D", name="date")
        group = group.reindex(full)
        group["station_id"] = station
        group["capacity_af"] = group.capacity_af.ffill().bfill()
        group["reservoir_name"] = group.reservoir_name.ffill().bfill()
        # NO interpolation of storage labels or features across missing days.
        group["storage_pct"] = 100 * group.storage_af / group.capacity_af
        group["prev7_pct"] = 100 * (group.storage_af - group.storage_af.shift(7)) / group.capacity_af
        group["prev30_pct"] = 100 * (group.storage_af - group.storage_af.shift(30)) / group.capacity_af
        group["prev60_pct"] = 100 * (group.storage_af - group.storage_af.shift(60)) / group.capacity_af
        # Weather only up through t-7. A conservative retrospective availability buffer.
        past = group[["rain_mm", "temp_c", "humidity_pct"]].shift(weather_lag)
        group["rain_7d_mm"] = past.rain_mm.rolling(7, min_periods=7).sum()
        group["rain_30d_mm"] = past.rain_mm.rolling(30, min_periods=30).sum()
        group["temp_30d_c"] = past.temp_c.rolling(30, min_periods=30).mean()
        group["humidity_30d_pct"] = past.humidity_pct.rolling(30, min_periods=30).mean()
        doy = group.index.dayofyear
        group["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
        group["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
        group["future_storage_af"] = group.storage_af.shift(-horizon)
        group["target_delta_pp"] = 100 * (group.future_storage_af - group.storage_af) / group.capacity_af
        group["target_date"] = group.index + pd.Timedelta(days=horizon)
        frames.append(group.reset_index())
    all_rows = pd.concat(frames, ignore_index=True)
    # Weekly sampling; keep same weekday and drop rows with insufficient history.
    all_rows = all_rows.loc[all_rows.date.dt.dayofweek == 0].copy()
    feature_cols = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct",
                    "rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct",
                    "season_sin", "season_cos"]
    all_rows = all_rows.dropna(subset=feature_cols + ["target_delta_pp"])
    return all_rows


def main():
    source = ROOT / "data/processed/daily_joined.csv"
    if not source.exists():
        raise FileNotFoundError("Run python -m src.fetch first")
    out = construct_features(pd.read_csv(source))
    file = ROOT / "data/processed/weekly_features.csv"
    out.to_csv(file, index=False)
    print(f"Saved {len(out)} weekly samples; stations={out.station_id.nunique()}")
    print(out.groupby("station_id").agg(samples=("date", "size"), start=("date", "min"), end=("date", "max")).to_string())

if __name__ == "__main__":
    main()

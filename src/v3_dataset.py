"""Build the v3 weekly table (private: contains CDEC values) for Models A, B, B2 and C.

Run: python -m src.v3_dataset [--live-dir data/live/2026-11-22]

Reuses the frozen v2 functions unchanged:
- src.features.construct_features (targets, storage changes, 7-day weather lag, SWE t-1/t-15, season)
- src.features.core_snow_stations (core pillow rule, WY<=2023)
- src.build_dataset.load_cdec / load_snow (cleaning), via src.forecast_live.clean
Inputs
- storage and snow through 2025-12-31: data/processed (v2); 2026: the CDEC download in --live-dir
- weather: data/v3/public/power_daily.csv (dam point and upstream-box means)
- GEFS: data/v3/gefs_basin_precip.csv (complete cycles only)
Output: data/v3/private/weekly_v3.csv and data/v3/private/weekly_v3_manifest.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.features import construct_features, core_snow_stations, eligible_stations, load_breaks
from src.forecast_live import basin_index, clean

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "v3" / "private"
WEATHER_COLS = ["rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"]
BOX_COLS = [c + "_box" for c in WEATHER_COLS]
GEFS_COLS = ["gefs_p_d1_7_mm", "gefs_p_d8_16_mm"]
SPLIT = pd.Timestamp("2026-01-01")
EARLIEST_ISSUE_AFTER_INIT = pd.Timedelta(hours=32)   # end of the origin Monday in Pacific Standard Time (Tue 08:00 UTC)


def usable_gefs(g):
    """Complete cycles only. Operational cycles must have every file published on S3 before the earliest moment a
    Monday-origin forecast could be issued (end of Monday, PST); later uploads are archive backfills with no evidence
    of real-time availability and are excluded. Reforecasts are retrospective by definition and are labelled so."""
    g = g[g.complete.astype(bool)].copy()
    pub = pd.to_datetime(g.published_utc_max, utc=True)
    limit = pd.to_datetime(g.init_date).dt.tz_localize("UTC") + EARLIEST_ISSUE_AFTER_INIT
    ok = (g.source == "reforecast") | ((g.source == "operational") & (pub <= limit))
    return g[ok]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def daily_storage(live_dir, stations):
    hist = pd.read_csv(ROOT / "data/processed/daily_joined.csv", parse_dates=["date"])
    hist = hist[hist.station_id.isin(stations)][["station_id", "reservoir_name", "date", "storage_af", "capacity_af"]]
    storage, _, snow, _, cap_hist = clean(live_dir, "2025-01-01", Path(live_dir).name, stations)
    names = hist.groupby("station_id").reservoir_name.first()
    from src import build_dataset as bd
    rows = []
    for st, df in storage.items():
        d = df[df.index >= SPLIT]
        rows.append(pd.DataFrame({"station_id": st, "reservoir_name": names[st], "date": d.index,
                                  "storage_af": d.storage_af.to_numpy(),
                                  "capacity_af": bd.capacity_series(st, d.index, cap_hist).to_numpy()}))
    return pd.concat([hist[hist.date < SPLIT]] + rows, ignore_index=True), snow


def snow_index(live_snow):
    snow = pd.read_csv(ROOT / "data/processed/snow_daily.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/snow_stations.csv")
    core = core_snow_stations(snow, meta)
    core_map = {s: sorted(g.snow_station_id) for s, g in core.groupby("reservoir_id")}
    both = pd.concat([snow[snow.date < SPLIT], live_snow[live_snow.date >= SPLIT]], ignore_index=True)
    return basin_index(both, core_map)[["station_id", "date", "swe_basin_in", "n_core_reporting", "n_core"]]


def build(live_dir):
    stations = eligible_stations()
    daily, live_snow = daily_storage(live_dir, stations)
    w = pd.read_csv(ROOT / "data/v3/public/power_daily.csv", parse_dates=["date"])
    idx = snow_index(live_snow)
    breaks = load_breaks()

    def with_weather(src):
        ww = w[w.source == src].drop(columns="source")
        return daily.merge(ww, on=["station_id", "date"], how="left")

    rows = construct_features(with_weather("dam_point"), breaks=breaks, snow_index=idx)
    box = construct_features(with_weather("upstream_box_mean"), breaks=breaks, snow_index=idx)
    box = box[["station_id", "date"] + WEATHER_COLS].rename(columns=dict(zip(WEATHER_COLS, BOX_COLS)))
    rows = rows.merge(box, on=["station_id", "date"], how="left")
    g = pd.read_csv(ROOT / "data/v3/gefs_basin_precip.csv", parse_dates=["init_date"])
    g = usable_gefs(g).rename(columns={"init_date": "date", "source": "gefs_source",
                                       "published_utc_max": "gefs_published_utc"})
    rows = rows.merge(g[["station_id", "date", "gefs_source", "gefs_published_utc"] + GEFS_COLS],
                      on=["station_id", "date"], how="left")
    keep = ["station_id", "reservoir_name", "date", "target_date", "storage_af", "capacity_af", "future_storage_af",
            "target_delta_pp", "storage_pct", "prev7_pct", "prev30_pct", "prev60_pct", "swe_basin_in",
            "swe_change_14d_in", "season_sin", "season_cos"] + WEATHER_COLS + BOX_COLS + GEFS_COLS + \
           ["gefs_source", "gefs_published_utc"]
    rows = rows[keep].sort_values(["station_id", "date"]).reset_index(drop=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    rows.to_csv(PRIVATE / "weekly_v3.csv", index=False, date_format="%Y-%m-%d")
    idx.to_csv(PRIVATE / "snow_index_v3.csv", index=False, date_format="%Y-%m-%d")
    inputs = ["data/processed/daily_joined.csv", "data/processed/snow_daily.csv", "data/v3/public/power_daily.csv",
              "data/v3/gefs_basin_precip.csv", "data/capacity_history.csv", "data/series_breaks.csv", "data/spike_review.csv"]
    live_log = Path(live_dir) / "fetch_log.jsonl"
    man = {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "live_dir": str(live_dir),
           "inputs_sha256": {f: sha(ROOT / f) for f in inputs}, "live_fetch_log_sha256": sha(live_log),
           "rows": len(rows), "rows_2026_origins": int((rows.date >= SPLIT).sum()),
           "gefs_available_share": float(rows.gefs_p_d1_7_mm.notna().mean()),
           "output_sha256": sha(PRIVATE / "weekly_v3.csv")}
    (PRIVATE / "weekly_v3_manifest.json").write_text(json.dumps(man, indent=1))
    return rows, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default=str(ROOT / "data/live/2026-11-22"))
    a = ap.parse_args()
    rows, man = build(Path(a.live_dir))
    print(json.dumps({k: v for k, v in man.items() if k != "inputs_sha256"}, indent=1))
    print(rows.groupby(rows.date.dt.year >= 2026).size())


if __name__ == "__main__":
    main()

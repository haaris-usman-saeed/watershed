"""v4 daily panel: one row per reservoir-day, from the underlying daily observations (not the weekly table).

Run: python -m src.v4.dataset            # -> data/v4/daily_panel.csv (git-ignored; CDEC-derived)

Sources (all read-only):
- data/processed/daily_joined.csv       CDEC daily storage (sensor 15) and capacity in force
- data/processed/snow_basin_index.csv   basin snow index (core pillows, sensor 82), inches
- data/v3/public/power_daily.csv        NASA POWER daily weather at the dam point and upstream-box mean
- data/v4/flows_daily.csv               CDEC daily inflow (76) and outflow (23), if fetched (src.v4.flows)

Every day of the calendar is present for every reservoir; a missing value stays NaN (never filled).
`seg` counts documented series breaks (data/series_breaks.csv); storage differences never cross a break.
Only dates before DEV_END are written.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.v4 import DEV_END, STATIONS

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "v4" / "daily_panel.csv"
START = pd.Timestamp("2001-01-01")
COLUMNS = ["station_id", "date", "seg", "storage_af", "capacity_af", "swe_in", "rain_mm", "temp_c", "rh_pct",
           "rain_box_mm", "temp_box_c", "rh_box_pct", "inflow_cfs", "outflow_cfs"]


def build_panel(daily, snow, power, flows=None, breaks=None, start=START, end=DEV_END, stations=STATIONS):
    from src.features import load_breaks
    assert end <= DEV_END, "the v4 development panel must end before 2024-01-01"
    breaks = load_breaks() if breaks is None else breaks
    for f in (daily, snow, power) + ((flows,) if flows is not None else ()):
        f["date"] = pd.to_datetime(f["date"])
    cal = pd.date_range(start, end - pd.Timedelta(days=1), freq="D", name="date")
    wx = {}
    for src, suf in (("dam_point", ""), ("upstream_box_mean", "_box")):
        w = power[power.source == src].rename(columns={"rain_mm": f"rain{suf}_mm", "temp_c": f"temp{suf}_c",
                                                       "humidity_pct": f"rh{suf}_pct"})
        wx[src] = w.drop(columns=["source"])
    frames = []
    for st in stations:
        g = pd.DataFrame(index=cal)
        d = daily[daily.station_id == st].set_index("date")
        g["storage_af"] = d.storage_af.reindex(cal)
        g["capacity_af"] = d.capacity_af.reindex(cal).ffill().bfill()
        g["swe_in"] = snow[snow.station_id == st].set_index("date").swe_basin_in.reindex(cal)
        for src in wx:
            w = wx[src][wx[src].station_id == st].set_index("date").drop(columns=["station_id"])
            for c in w.columns:
                g[c] = w[c].reindex(cal)
        for c in ("inflow_cfs", "outflow_cfs"):
            g[c] = (flows[flows.station_id == st].set_index("date")[c].reindex(cal)
                    if flows is not None and c in flows else np.nan)
        seg = np.zeros(len(cal), dtype=int)
        for b in breaks.get(st, []):
            seg += (cal >= b).astype(int)
        g["seg"] = seg
        g["station_id"] = st
        frames.append(g.reset_index())
    panel = pd.concat(frames, ignore_index=True)[COLUMNS]
    assert panel.date.max() < end
    return panel


def load_inputs():
    daily = pd.read_csv(ROOT / "data/processed/daily_joined.csv")
    snow = pd.read_csv(ROOT / "data/processed/snow_basin_index.csv")
    power = pd.read_csv(ROOT / "data/v3/public/power_daily.csv")
    fp = ROOT / "data/v4/flows_daily.csv"
    flows = pd.read_csv(fp) if fp.exists() else None
    return daily, snow, power, flows


def load_panel(path=PANEL):
    p = pd.read_csv(path, parse_dates=["date"])
    assert p.date.max() < DEV_END, "the v4 development panel must end before 2024-01-01"
    return p


def main():
    panel = build_panel(*load_inputs())
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL, index=False, date_format="%Y-%m-%d")
    cov = panel.groupby("station_id")[COLUMNS[3:]].apply(lambda g: g.notna().mean().round(3))
    print(f"{len(panel)} reservoir-days, {panel.date.min().date()} to {panel.date.max().date()} -> "
          f"{PANEL.relative_to(ROOT)}\nshare of days observed:\n{cov.drop(columns=['capacity_af']).to_string()}")


if __name__ == "__main__":
    main()

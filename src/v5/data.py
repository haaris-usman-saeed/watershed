"""v5 consolidated dataset: the daily panel, eligible GEFS aligned to every day, and the tabular feature table.

Run: python -m src.v5.data          # -> data/v5/daily_panel.parquet, data/v5/features_daily.parquet,
                                    #    results/v5/dataset/ (accounting and checks; aggregates only)

Sources (all read-only)
- storage and capacity: data/processed/daily_joined.csv through 2025-12-31; 2026 from the CDEC download used by v3
  (data/live/2026-11-22) via src.v3_dataset.daily_storage (unchanged)
- basin snow index: data/v3/private/snow_index_v3.csv (equal to data/processed/snow_basin_index.csv where both exist)
- weather: data/v3/public/power_daily.csv (NASA POWER, dam point and upstream-box mean)
- flows: data/v4/flows_daily.csv (2001-2023) + data/v5/flows_daily_2024_2026.csv (src.v5.flows_ext)
- GEFS: data/v3/gefs_basin_precip.csv filtered by src.v3_dataset.usable_gefs (complete cycles; operational cycles
  published by init + 32 h). Reforecast cycles (2001-2019) are retrospective.

GEFS alignment: for day t, the latest usable Monday 00Z cycle with init <= t and t - init <= 6 days (age 0 on Mondays);
otherwise missing. The tabular features use the v2 definitions (src.features.construct_features) computed on every day.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from src.v5 import DATA, DEV_END, PANEL_END, RESULTS, ROOT, STATIONS

START = pd.Timestamp("2001-01-01")
CFS_DAY_AF = 1.983471
PANEL_COLS = ["station_id", "date", "seg", "storage_af", "capacity_af", "swe_in", "rain_mm", "temp_c", "rh_pct",
              "rain_box_mm", "temp_box_c", "rh_box_pct", "inflow_cfs", "outflow_cfs"]
GEFS_COLS = ["gefs_p_d1_7_mm", "gefs_p_d8_16_mm"]
GEFS_DAILY = GEFS_COLS + ["gefs_age_days", "gefs_source"]
STORAGE = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct"]
SNOW = ["swe_basin_in", "swe_change_14d_in", "swe_anom_in"]
WD = ["rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"]
WU = [c + "_box" for c in WD]
FLOW = ["inflow_7d_pp", "inflow_30d_pp", "outflow_7d_pp", "outflow_30d_pp"]
SEASON = ["season_sin", "season_cos"]
GROUPS = {"S": STORAGE, "N": SNOW, "Wd": WD, "Wu": WU, "G": GEFS_COLS, "Q": FLOW, "C": SEASON}
PANEL_FILE = DATA / "daily_panel.parquet"
FEATURES_FILE = DATA / "features_daily.parquet"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources():
    from src.v3_dataset import daily_storage, usable_gefs
    from src.v5.flows_ext import load_all_flows
    daily, _ = daily_storage(ROOT / "data" / "live" / "2026-11-22", list(STATIONS))
    snow = pd.read_csv(ROOT / "data/v3/private/snow_index_v3.csv", parse_dates=["date"])
    power = pd.read_csv(ROOT / "data/v3/public/power_daily.csv", parse_dates=["date"])
    flows = load_all_flows()
    g = pd.read_csv(ROOT / "data/v3/gefs_basin_precip.csv", parse_dates=["init_date"])
    gefs = usable_gefs(g)[["init_date", "source", "station_id"] + GEFS_COLS]
    return daily, snow, power, flows, gefs


def align_gefs(cal, gefs_st, max_age=6):
    """Latest usable cycle with init <= t and age <= max_age days, for every day in `cal`."""
    g = gefs_st.sort_values("init_date").set_index("init_date")
    pos = g.index.searchsorted(cal, side="right") - 1
    ok = pos >= 0
    init = pd.DatetimeIndex(np.where(ok, g.index.to_numpy()[np.clip(pos, 0, None)], np.datetime64("NaT")))
    age = (cal - init).days.to_numpy(float)
    ok &= np.isfinite(age) & (age <= max_age)
    out = pd.DataFrame(index=cal)
    for c in GEFS_COLS:
        out[c] = np.where(ok, g[c].to_numpy()[np.clip(pos, 0, None)], np.nan)
    out["gefs_age_days"] = np.where(ok, age, np.nan)
    out["gefs_source"] = np.where(ok, g.source.to_numpy()[np.clip(pos, 0, None)], None)
    return out


def build_panel(daily, snow, power, flows, gefs, end=PANEL_END, start=START):
    """One row per reservoir-day in [start, end). Same columns and rules as src.v4.dataset.build_panel, plus GEFS."""
    from src.features import load_breaks
    breaks = load_breaks()
    cal = pd.date_range(start, end - pd.Timedelta(days=1), freq="D", name="date")
    wx = {}
    for src, suf in (("dam_point", ""), ("upstream_box_mean", "_box")):
        w = power[power.source == src].rename(columns={"rain_mm": f"rain{suf}_mm", "temp_c": f"temp{suf}_c",
                                                       "humidity_pct": f"rh{suf}_pct"})
        wx[src] = w.drop(columns=["source"])
    frames = []
    for st in STATIONS:
        g = pd.DataFrame(index=cal)
        d = daily[daily.station_id == st].set_index("date")
        g["storage_af"] = d.storage_af.reindex(cal)
        g["capacity_af"] = d.capacity_af.reindex(cal).ffill().bfill()
        g["swe_in"] = snow[snow.station_id == st].set_index("date").swe_basin_in.reindex(cal)
        for src in wx:
            w = wx[src][wx[src].station_id == st].set_index("date").drop(columns=["station_id"])
            for c in w.columns:
                g[c] = w[c].reindex(cal)
        fl = flows[flows.station_id == st].set_index("date")
        for c in ("inflow_cfs", "outflow_cfs"):
            g[c] = fl[c].reindex(cal)
        seg = np.zeros(len(cal), dtype=int)
        for b in breaks.get(st, []):
            seg += (cal >= b).astype(int)
        g["seg"] = seg
        g["station_id"] = st
        g = g.join(align_gefs(cal, gefs[gefs.station_id == st]))
        frames.append(g.reset_index())
    panel = pd.concat(frames, ignore_index=True)[PANEL_COLS + GEFS_DAILY]
    assert panel.date.max() < end
    return panel


def daily_features(panel):
    """v2 feature definitions (src.features.construct_features) on every reservoir-day, plus upstream-box weather,
    four flow features and aligned GEFS. No row is dropped and nothing is filled; swe_anom_in is added per fold."""
    out = []
    for st, g in panel.groupby("station_id", sort=True):
        g = g.set_index("date").sort_index()
        cap, s, seg = g.capacity_af, g.storage_af, g.seg
        f = pd.DataFrame(index=g.index)
        f["station_id"] = st
        f["storage_af"], f["capacity_af"] = s, cap

        def diff_back(k):
            return (100 * (s - s.shift(k)) / cap).where(seg == seg.shift(k))
        f["storage_pct"] = 100 * s / cap
        f["prev7_pct"], f["prev30_pct"], f["prev60_pct"] = diff_back(7), diff_back(30), diff_back(60)
        for suf in ("", "_box"):
            past = g[[f"rain{suf}_mm", f"temp{suf}_c", f"rh{suf}_pct"]].shift(7)
            f[f"rain_7d_mm{suf}"] = past[f"rain{suf}_mm"].rolling(7, min_periods=7).sum()
            f[f"rain_30d_mm{suf}"] = past[f"rain{suf}_mm"].rolling(30, min_periods=30).sum()
            f[f"temp_30d_c{suf}"] = past[f"temp{suf}_c"].rolling(30, min_periods=30).mean()
            f[f"humidity_30d_pct{suf}"] = past[f"rh{suf}_pct"].rolling(30, min_periods=30).mean()
        f["swe_basin_in"] = g.swe_in.shift(1)
        f["swe_change_14d_in"] = g.swe_in.shift(1) - g.swe_in.shift(15)
        for c, name in (("inflow_cfs", "inflow"), ("outflow_cfs", "outflow")):
            pp = 100 * g[c] * CFS_DAY_AF / cap
            f[f"{name}_7d_pp"] = pp.rolling(7, min_periods=7).mean()
            f[f"{name}_30d_pp"] = pp.rolling(30, min_periods=30).mean()
        for c in GEFS_DAILY:
            f[c] = g[c]
        doy = f.index.dayofyear
        f["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
        f["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
        fut = s.shift(-30).where(seg == seg.shift(-30))
        f["target_delta_pp"] = 100 * (fut - s) / cap
        f["target_date"] = f.index + pd.Timedelta(days=30)
        out.append(f.reset_index())
    feat = pd.concat(out, ignore_index=True)
    req = STORAGE + WD + ["target_delta_pp"]
    feat["v2_row"] = (feat.date.dt.dayofweek == 0) & feat[req].notna().all(axis=1)
    return feat


V3_SHARED = STORAGE + ["swe_basin_in", "swe_change_14d_in"] + WD + WU + GEFS_COLS + SEASON + ["target_delta_pp"]


def use_v3_values(feat):
    """On v2 Monday rows, take the shared columns from data/v3/private/weekly_v3.csv, which is bit-identical to the v2
    table (data/processed/weekly_features.csv) on every shared row. Recomputing them differs by up to ~1e-13 (floating
    point order), and HGB bin thresholds are sensitive to such ties, so this keeps the historical configurations exactly
    reproducible. Rows and missingness are unchanged (checked in `checks`)."""
    w = pd.read_csv(ROOT / "data/v3/private/weekly_v3.csv", parse_dates=["date"])
    key = pd.MultiIndex.from_frame(feat.loc[feat.v2_row, ["station_id", "date"]])
    w = w.set_index(["station_id", "date"]).reindex(key)
    assert w.target_delta_pp.notna().all()
    for c in V3_SHARED:
        feat.loc[feat.v2_row, c] = w[c].to_numpy()
    return feat


def load_features(end=DEV_END):
    """The daily feature table restricted to origins AND targets before `end` (development: DEV_END)."""
    f = pd.read_parquet(FEATURES_FILE)
    f = f[(f.date < end) & (f.target_date < end)].copy() if end is not None else f
    return f.reset_index(drop=True)


def load_panel(end=DEV_END):
    p = pd.read_parquet(PANEL_FILE)
    return p[p.date < end].reset_index(drop=True)


def load_snow(end=DEV_END):
    s = pd.read_csv(ROOT / "data/v3/private/snow_index_v3.csv", parse_dates=["date"])
    return s[s.date < end][["station_id", "date", "swe_basin_in"]].reset_index(drop=True)


def checks(panel, feat):
    """Equality with the v4 panel (before DEV_END) and with the v3 weekly table (every shared column)."""
    res = {}
    v4 = pd.read_csv(ROOT / "data/v4/daily_panel.csv", parse_dates=["date"])
    mine = panel[panel.date < DEV_END][PANEL_COLS].reset_index(drop=True)
    v4 = v4.sort_values(["station_id", "date"]).reset_index(drop=True)
    mine = mine.sort_values(["station_id", "date"]).reset_index(drop=True)
    assert len(mine) == len(v4) and (mine.station_id.to_numpy() == v4.station_id.to_numpy()).all()
    diffs = {}
    for c in PANEL_COLS[2:]:
        a, b = mine[c].to_numpy(float), v4[c].to_numpy(float)
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        diffs[c] = {"nan_mismatch": nan_mismatch, "max_abs_diff": float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0}
    res["v4_panel_equal"] = {"rows": int(len(v4)), "columns": diffs,
                             "identical": all(d["nan_mismatch"] == 0 and d["max_abs_diff"] == 0 for d in diffs.values())}
    w = pd.read_csv(ROOT / "data/v3/private/weekly_v3.csv", parse_dates=["date", "target_date"])
    m = feat[feat.v2_row].merge(w, on=["station_id", "date"], how="outer", suffixes=("", "_v3"), indicator=True)
    res["v3_weekly_rows"] = {"v3": int(len(w)), "v5_v2_rows": int(feat.v2_row.sum()),
                             "only_v5": int((m._merge == "left_only").sum()), "only_v3": int((m._merge == "right_only").sum())}
    both = m[m._merge == "both"]
    cols = STORAGE + ["swe_basin_in", "swe_change_14d_in"] + WD + WU + GEFS_COLS + SEASON + ["target_delta_pp"]
    res["v3_weekly_max_abs_diff"] = {}
    for c in cols:
        a, b = both[c].to_numpy(float), both[f"{c}_v3"].to_numpy(float)
        res["v3_weekly_max_abs_diff"][c] = {
            "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum()),
            "max_abs_diff": float(np.nanmax(np.abs(a - b))) if (~np.isnan(a) & ~np.isnan(b)).any() else 0.0}
    return res


def main():
    daily, snow, power, flows, gefs = load_sources()
    panel = build_panel(daily, snow, power, flows, gefs)
    feat = daily_features(panel)
    res = checks(panel, feat)               # recomputed values against v4 and v3 (before replacement)
    feat = use_v3_values(feat)
    DATA.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_FILE, index=False)
    feat.to_parquet(FEATURES_FILE, index=False)
    res["monday_rows_use_v3_csv_values"] = V3_SHARED
    out = RESULTS / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    inputs = ["data/processed/daily_joined.csv", "data/v3/private/snow_index_v3.csv", "data/v3/public/power_daily.csv",
              "data/v4/flows_daily.csv", "data/v5/flows_daily_2024_2026.csv", "data/v3/gefs_basin_precip.csv",
              "data/live/2026-11-22/fetch_log.jsonl", "data/series_breaks.csv", "data/capacity_history.csv"]
    manifest = {"inputs_sha256": {f: sha(ROOT / f) for f in inputs},
                "outputs_sha256": {str(PANEL_FILE.relative_to(ROOT)): sha(PANEL_FILE),
                                   str(FEATURES_FILE.relative_to(ROOT)): sha(FEATURES_FILE)},
                "panel_rows": int(len(panel)), "panel_period": [str(panel.date.min().date()), str(panel.date.max().date())],
                "checks": res}
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

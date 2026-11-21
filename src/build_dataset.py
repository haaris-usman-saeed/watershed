"""Build the processed daily joined dataset + audit tables from unmodified raw files.

Run: python -m src.build_dataset [--start 2018-01-01 --end 2025-12-31]
Offline: reads data/raw/cdec/*.csv and data/raw/power/*.json written by src.fetch.

Outputs
  data/processed/daily_joined.csv   one row per station x calendar day (missing storage left blank)
  data/processed/coverage.csv       per-station audit (coverage, duplicates, invalid values, gaps)
  outputs/stations.csv              UI contract: station metadata + status
  outputs/history_daily.csv         UI contract: observed daily history

Date convention (verified 2026-11-21 against CDEC hourly sensor 15): the CDEC daily
`DATE TIME` field D carries the END-OF-DAY (midnight) storage of day D. The
`OBS DATE` field is inconsistent across operators (= D for USBR stations,
= D+1 for DWR/others) and is therefore NOT used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
SPIKE_PP = 2.0        # isolated one-day spike threshold, pp of capacity
MAX_MISSING_PCT = 10  # station inclusion rule
PLAUSIBLE = {"temp_c": (-40, 55), "rain_mm": (0, 500), "humidity_pct": (0, 100)}


def load_cdec(station, capacity):
    files = sorted((RAW / "cdec").glob(f"{station}_sensor15_D_*.csv"))
    if not files:
        return None, {"raw_files": 0}
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    df.columns = [c.strip().upper() for c in df.columns]
    audit = {"raw_files": len(files), "raw_rows": len(df)}
    wrong = df[(df.STATION_ID != station) | (df.SENSOR_NUMBER.str.strip() != "15")
               | (df.DURATION.str.strip() != "D") | (df.UNITS.str.strip() != "AF")]
    if len(wrong):
        raise ValueError(f"{station}: {len(wrong)} rows with unexpected station/sensor/duration/units")
    df["date"] = pd.to_datetime(df["DATE TIME"].str.strip(), format="%Y%m%d %H%M").dt.normalize()
    df["value"] = pd.to_numeric(df.VALUE.str.strip(), errors="coerce")
    audit["non_numeric_or_blank"] = int(df.value.isna().sum())
    audit["data_flags"] = {k: int(v) for k, v in df.DATA_FLAG.str.strip().value_counts().items() if k}
    bad = (df.value <= 0) | (df.value > 1.10 * capacity)
    audit["out_of_range"] = int(bad.sum())
    df.loc[bad, "value"] = np.nan
    # duplicates: identical values collapse; conflicting values -> missing (never pick one)
    dup = df.duplicated("date", keep=False)
    audit["duplicate_date_rows"] = int(dup.sum())
    agg = df.groupby("date").value.agg(["nunique", "first", "size"])
    conflicting = agg["nunique"] > 1
    audit["conflicting_duplicate_dates"] = int(conflicting.sum())
    s = agg["first"].where(~conflicting)
    # isolated one-day spikes (jump away and straight back): treat as invalid
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    d_prev = 100 * (full - full.shift(1)) / capacity
    d_next = 100 * (full.shift(-1) - full) / capacity
    spike = (d_prev.abs() > SPIKE_PP) & (d_next.abs() > SPIKE_PP) & (np.sign(d_prev) != np.sign(d_next))
    audit["isolated_spikes_removed"] = int(spike.sum())
    audit["spike_dates"] = [d.strftime("%Y-%m-%d") for d in full.index[spike]][:20]
    full[spike] = np.nan
    daily_step = (100 * full.diff().abs() / capacity)
    audit["max_abs_daily_change_pp"] = round(float(daily_step.max()), 3)
    return full.rename("storage_af"), audit


def load_power(station):
    files = sorted((RAW / "power").glob(f"{station}_daily_*.json"))
    if not files:
        return None, {}
    obj = json.loads(files[-1].read_text())
    hdr, par = obj["header"], obj["properties"]["parameter"]
    fill = hdr.get("fill_value", -999.0)
    units = {k: v["units"] for k, v in obj["parameters"].items()}
    assert units == {"T2M": "C", "PRECTOTCORR": "mm/day", "RH2M": "%"}, units
    w = pd.DataFrame({"temp_c": par["T2M"], "rain_mm": par["PRECTOTCORR"], "humidity_pct": par["RH2M"]})
    w.index = pd.to_datetime(w.index, format="%Y%m%d")
    audit = {"power_api_version": hdr["api"]["version"], "power_sources": hdr["sources"],
             "power_time_standard": hdr["time_standard"],
             "power_grid_elev_m": obj["geometry"]["coordinates"][2]}
    w = w.mask(w <= fill + 1e-9)
    for c, (lo, hi) in PLAUSIBLE.items():
        bad = (w[c] < lo) | (w[c] > hi)
        audit[f"implausible_{c}"] = int(bad.sum())
        w.loc[bad, c] = np.nan
    audit["weather_missing_days"] = int(w.isna().any(axis=1).sum())
    return w, audit


def longest_gap(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()
    cal = pd.date_range(args.start, args.end, freq="D", name="date")
    stations = pd.read_csv(ROOT / "stations.csv")
    rows, audits = [], []
    for st in stations.itertuples():
        storage, a = load_cdec(st.station_id, st.capacity_af)
        weather, wa = load_power(st.station_id)
        a.update(wa)
        a["station_id"] = st.station_id
        if storage is None or weather is None:
            a.update(status="excluded", exclusion_reason="raw CDEC or POWER file missing")
            audits.append(a)
            continue
        df = pd.DataFrame(index=cal).join(storage).join(weather)
        df["station_id"], df["reservoir_name"], df["capacity_af"] = st.station_id, st.reservoir_name, st.capacity_af
        df["lat"], df["lon"] = st.lat, st.lon
        df["storage_pct"] = 100 * df.storage_af / st.capacity_af
        valid = df.storage_af.notna()
        a.update(first_date=df.index[valid].min().strftime("%Y-%m-%d") if valid.any() else None,
                 last_date=df.index[valid].max().strftime("%Y-%m-%d") if valid.any() else None,
                 calendar_days=len(cal), n_storage_days=int(valid.sum()),
                 missing_storage_days=int((~valid).sum()),
                 missing_storage_pct=round(100 * (~valid).mean(), 3),
                 longest_storage_gap_days=longest_gap(~valid),
                 storage_min_pct=round(float(df.storage_pct.min()), 2),
                 storage_max_pct=round(float(df.storage_pct.max()), 2))
        ok = a["missing_storage_pct"] <= MAX_MISSING_PCT
        a["status"] = "included" if ok else "excluded"
        a["exclusion_reason"] = "" if ok else f"missing storage > {MAX_MISSING_PCT}%"
        audits.append(a)
        rows.append(df.reset_index())
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    joined = pd.concat(rows, ignore_index=True)
    cols = ["station_id", "reservoir_name", "date", "storage_af", "storage_pct", "capacity_af",
            "rain_mm", "temp_c", "humidity_pct", "lat", "lon"]
    joined[cols].to_csv(PROC / "daily_joined.csv", index=False, date_format="%Y-%m-%d")
    cov = pd.DataFrame(audits)
    cov.to_csv(PROC / "coverage.csv", index=False)
    meta = stations.merge(cov[["station_id", "status", "exclusion_reason", "first_date", "last_date",
                               "n_storage_days", "missing_storage_pct"]], on="station_id", how="left")
    meta[["station_id", "reservoir_name", "lat", "lon", "capacity_af", "capacity_source", "status",
          "first_date", "last_date", "n_storage_days", "missing_storage_pct", "exclusion_reason",
          "operator"]].to_csv(OUT / "stations.csv", index=False)
    joined[["station_id", "date", "storage_af", "storage_pct", "rain_mm", "temp_c", "humidity_pct"]] \
        .to_csv(OUT / "history_daily.csv", index=False, date_format="%Y-%m-%d")
    show = ["station_id", "status", "first_date", "last_date", "n_storage_days", "missing_storage_pct",
            "longest_storage_gap_days", "duplicate_date_rows", "conflicting_duplicate_dates",
            "out_of_range", "isolated_spikes_removed", "max_abs_daily_change_pp", "weather_missing_days",
            "storage_min_pct", "storage_max_pct"]
    table = cov[[c for c in show if c in cov] + ["data_flags", "spike_dates"]]
    breaks = pd.read_csv(ROOT / "data" / "series_breaks.csv")
    md = ["# Per-station coverage / audit table", "",
          f"Generated by `python -m src.build_dataset --start {args.start} --end {args.end}` from raw files in data/raw/.",
          "Storage = CDEC sensor 15, daily (end-of-day). Weather = NASA POWER daily point (LST).",
          "`data_flags`: counts of CDEC DATA_FLAG values (values kept; e = estimated, r = revised).",
          "`isolated_spikes_removed`: one-day jumps > 2 pp of capacity that immediately reverse (set to missing).", "",
          "| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)]
    md += ["", "## Documented series breaks (no target or change-feature may span these)", "",
           "| station | first day on new basis | description |", "|---|---|---|"]
    md += [f"| {b.station_id} | {b.break_date} | {b.description} |" for b in breaks.itertuples()]
    (ROOT / "data" / "coverage_table.md").write_text("\n".join(md) + "\n")
    # tracked manifest of raw downloads (URLs, times, SHA-256) - raw data itself stays local
    if (RAW / "fetch_log.jsonl").exists():
        (ROOT / "data" / "raw_manifest.jsonl").write_text((RAW / "fetch_log.jsonl").read_text())
    print(cov[[c for c in show if c in cov]].to_string(index=False))
    print(f"Saved {len(joined)} station-days to {PROC / 'daily_joined.csv'}")


if __name__ == "__main__":
    main()

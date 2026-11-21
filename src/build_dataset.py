"""Build the processed daily joined dataset and audit tables from unmodified raw files (v2).

Run: python -m src.build_dataset [--start 2001-01-01 --end 2025-12-31]
Offline: reads data/raw/cdec/*.csv, data/raw/power/*.json and data/raw/snow/*.csv.

Outputs
  data/processed/daily_joined.csv     station x calendar day, storage (blank if missing), weather,
                                      capacity in force that day, basin snow index columns
  data/processed/snow_daily.csv       QC'd daily SWE per snow pillow (long format)
  data/processed/coverage.csv         per-station audit
  data/processed/station_year_audit.csv
  data/processed/discontinuity_scan.csv  dry-day level shifts flagged for manual review
  data/coverage_table.md              tracked markdown audit
  outputs/stations.csv, outputs/history_daily.csv   UI contract files

Date convention (verified against CDEC hourly sensor 15): the CDEC daily `DATE TIME`
field D carries the END-OF-DAY storage of day D. `OBS DATE` is operator dependent and unused.
Storage values are never modified except to mark invalid values as missing; capacity
denominators follow data/capacity_history.csv (ORO changed storage tables on 2024-07-02).
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
SPIKE_PP = 2.0          # isolated one-day spike threshold, pp of capacity
MAX_MISSING_PCT = 10    # station eligibility: max % missing storage days in the eligible period
DRY_SHIFT_PP = 2.0      # discontinuity scan: one-day change > 2 pp ...
DRY_RAIN_MM = 5.0       # ... with < 5 mm of rain over that day and the 3 days before
PLAUSIBLE = {"temp_c": (-40, 55), "rain_mm": (0, 500), "humidity_pct": (0, 100)}
SWE_MAX_IN = 250.0      # physically implausible pillow SWE (inches) above this


def capacity_series(station, dates, hist):
    h = hist[hist.station_id == station]
    cap = pd.Series(np.nan, index=dates)
    for r in h.itertuples():
        cap[(dates >= pd.Timestamp(r.valid_from)) & (dates <= pd.Timestamp(r.valid_to))] = r.capacity_af
    if cap.isna().any():
        raise ValueError(f"{station}: capacity_history does not cover every date")
    return cap


def load_cdec(station, cap_hist, start, end):
    files = sorted((RAW / "cdec").glob(f"{station}_sensor15_D_*.csv"))
    files = [f for f in files if start[:4] <= f.stem[-4:] <= end[:4]]
    if not files:
        return None, None, {"raw_files": 0}
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    df.columns = [c.strip().upper() for c in df.columns]
    audit = {"raw_files": len(files), "raw_rows": len(df)}
    wrong = df[(df.STATION_ID != station) | (df.SENSOR_NUMBER.str.strip() != "15")
               | (df.DURATION.str.strip() != "D") | (df.UNITS.str.strip() != "AF")]
    if len(wrong):
        raise ValueError(f"{station}: {len(wrong)} rows with unexpected station/sensor/duration/units")
    df["date"] = pd.to_datetime(df["DATE TIME"].str.strip(), format="%Y%m%d %H%M").dt.normalize()
    df["flag"] = df.DATA_FLAG.str.strip()
    df["value"] = pd.to_numeric(df.VALUE.str.strip(), errors="coerce")
    audit["non_numeric_or_blank"] = int(df.value.isna().sum())
    audit["data_flags"] = {k: int(v) for k, v in df.flag.value_counts().items() if k}
    cap_row = capacity_series(station, pd.DatetimeIndex(df.date), cap_hist).to_numpy()
    bad = (df.value <= 0) | (df.value > 1.10 * cap_row)
    audit["out_of_range"] = int(bad.sum())
    df.loc[bad, "value"] = np.nan
    dup = df.duplicated("date", keep=False)
    audit["duplicate_date_rows"] = int(dup.sum())
    agg = df.groupby("date").agg(n=("value", "nunique"), value=("value", "first"), flag=("flag", "first"))
    conflicting = agg.n > 1
    audit["conflicting_duplicate_dates"] = int(conflicting.sum())
    s = agg.value.where(~conflicting)
    full_idx = pd.date_range(start, end, freq="D")
    full = s.reindex(full_idx)
    flags = agg.flag.reindex(full_idx).fillna("")
    cap = capacity_series(station, full_idx, cap_hist)
    d_prev = 100 * (full - full.shift(1)) / cap
    d_next = 100 * (full.shift(-1) - full) / cap
    cand = (d_prev.abs() > SPIKE_PP) & (d_next.abs() > SPIKE_PP) & (np.sign(d_prev) != np.sign(d_next))
    review = pd.read_csv(ROOT / "data" / "spike_review.csv", parse_dates=["date"])
    review = review[review.station_id == station].set_index("date").decision
    decision = pd.Series([review.get(d, "unreviewed") for d in full.index[cand]], index=full.index[cand], dtype=object)
    remove = decision.index[decision != "physical"]   # artifacts and unreviewed candidates -> missing
    audit["spike_candidates"] = int(cand.sum())
    audit["spikes_kept_physical"] = int((decision == "physical").sum())
    audit["spikes_removed_artifact"] = int((decision == "artifact").sum())
    audit["spikes_removed_unreviewed"] = [d.strftime("%Y-%m-%d") for d in decision.index[decision == "unreviewed"]]
    full[remove] = np.nan
    return full.rename("storage_af"), flags.rename("cdec_flag"), audit


def load_power(station, start, end):
    path = RAW / "power" / f"{station}_daily_{start}_{end}.json"
    if not path.exists():
        return None, {}
    obj = json.loads(path.read_text())
    hdr, par = obj["header"], obj["properties"]["parameter"]
    fill = hdr.get("fill_value", -999.0)
    units = {k: v["units"] for k, v in obj["parameters"].items()}
    assert units == {"T2M": "C", "PRECTOTCORR": "mm/day", "RH2M": "%"}, units
    w = pd.DataFrame({"temp_c": par["T2M"], "rain_mm": par["PRECTOTCORR"], "humidity_pct": par["RH2M"]})
    w.index = pd.to_datetime(w.index, format="%Y%m%d")
    audit = {"power_file": path.name, "power_api_version": hdr["api"]["version"],
             "power_sources": hdr["sources"], "power_time_standard": hdr["time_standard"],
             "power_grid_elev_m": obj["geometry"]["coordinates"][2]}
    w = w.mask(w <= fill + 1e-9)
    for c, (lo, hi) in PLAUSIBLE.items():
        bad = (w[c] < lo) | (w[c] > hi)
        audit[f"implausible_{c}"] = int(bad.sum())
        w.loc[bad, c] = np.nan
    audit["weather_missing_days"] = int(w.isna().any(axis=1).sum())
    return w, audit


def load_snow(start, end):
    """Long-format QC'd SWE. Missing stays missing (never zero). Negative values are set to 0
    only when the raw value is within -0.5 in of zero (pillow noise near bare ground); larger
    negatives and values above SWE_MAX_IN are set to missing."""
    files = sorted((RAW / "snow").glob("snow_sensor82_D_*.csv"))
    files = [f for f in files if start[:4] <= f.stem[-4:] <= end[:4]]
    if not files:
        return None, {}
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    df.columns = [c.strip().upper() for c in df.columns]
    assert set(df.SENSOR_NUMBER.str.strip()) == {"82"} and set(df.UNITS.str.strip()) == {"INCHES"}
    df["date"] = pd.to_datetime(df["DATE TIME"].str.strip(), format="%Y%m%d %H%M").dt.normalize()
    df["swe_in"] = pd.to_numeric(df.VALUE.str.strip(), errors="coerce")
    df["flag"] = df.DATA_FLAG.str.strip()
    a = {"snow_rows": len(df), "snow_missing_or_text": int(df.swe_in.isna().sum()),
         "snow_small_negative_to_zero": int(((df.swe_in < 0) & (df.swe_in >= -0.5)).sum()),
         "snow_invalid_to_missing": int(((df.swe_in < -0.5) | (df.swe_in > SWE_MAX_IN)).sum()),
         "snow_flags": {k: int(v) for k, v in df.flag.value_counts().items() if k}}
    df.loc[(df.swe_in < -0.5) | (df.swe_in > SWE_MAX_IN), "swe_in"] = np.nan
    df.loc[df.swe_in < 0, "swe_in"] = 0.0
    df = df.rename(columns={"STATION_ID": "snow_station_id"})
    df = df.dropna(subset=["swe_in"]).drop_duplicates(["snow_station_id", "date"])
    return df[["snow_station_id", "date", "swe_in", "flag"]], a


def longest_gap(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2001-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()
    cal = pd.date_range(args.start, args.end, freq="D", name="date")
    stations = pd.read_csv(ROOT / "stations.csv")
    cap_hist = pd.read_csv(ROOT / "data" / "capacity_history.csv")
    breaks = pd.read_csv(ROOT / "data" / "series_breaks.csv")
    rows, audits, year_rows, scan_rows = [], [], [], []
    for st in stations.itertuples():
        storage, flags, a = load_cdec(st.station_id, cap_hist, args.start, args.end)
        weather, wa = load_power(st.station_id, args.start, args.end)
        a.update(wa)
        a["station_id"] = st.station_id
        if storage is None or weather is None:
            a.update(status="excluded", exclusion_reason="raw CDEC or POWER file missing")
            audits.append(a)
            continue
        df = pd.DataFrame(index=cal).join(storage).join(flags).join(weather)
        df["station_id"], df["reservoir_name"] = st.station_id, st.reservoir_name
        df["capacity_af"] = capacity_series(st.station_id, cal, cap_hist).to_numpy()
        df["lat"], df["lon"] = st.lat, st.lon
        df["storage_pct"] = 100 * df.storage_af / df.capacity_af
        valid = df.storage_af.notna()
        for y, g in df.groupby(df.index.year):
            v = g.storage_af.notna()
            year_rows.append({"station_id": st.station_id, "year": y, "days": len(g), "valid_days": int(v.sum()),
                              "missing_days": int((~v).sum()), "longest_gap_days": longest_gap(~v),
                              "flag_e": int((g.cdec_flag == "e").sum()), "flag_r": int((g.cdec_flag == "r").sum()),
                              "other_flags": int((~g.cdec_flag.isin(["", "e", "r"])).sum()),
                              "min_pct": round(float(g.storage_pct.min()), 2), "max_pct": round(float(g.storage_pct.max()), 2),
                              "weather_missing_days": int(g[["rain_mm", "temp_c", "humidity_pct"]].isna().any(axis=1).sum())})
        ch = df.storage_pct.diff()
        rain4 = df.rain_mm.rolling(4, min_periods=1).sum()
        for t in df.index[(ch.abs() > DRY_SHIFT_PP) & (rain4 < DRY_RAIN_MM)]:
            known = breaks[(breaks.station_id == st.station_id) & (pd.to_datetime(breaks.break_date) == t)]
            scan_rows.append({"station_id": st.station_id, "date": t.strftime("%Y-%m-%d"),
                              "change_pp": round(float(ch[t]), 3), "rain_4d_mm": round(float(rain4[t]), 2),
                              "documented_break": bool(len(known))})
        a.update(first_date=df.index[valid].min().strftime("%Y-%m-%d"),
                 last_date=df.index[valid].max().strftime("%Y-%m-%d"),
                 calendar_days=len(cal), n_storage_days=int(valid.sum()),
                 missing_storage_days=int((~valid).sum()), missing_storage_pct=round(100 * (~valid).mean(), 3),
                 longest_storage_gap_days=longest_gap(~valid),
                 storage_min_pct=round(float(df.storage_pct.min()), 2),
                 storage_max_pct=round(float(df.storage_pct.max()), 2),
                 max_abs_daily_change_pp=round(float(ch.abs().max()), 3))
        ok = a["missing_storage_pct"] <= MAX_MISSING_PCT
        a["status"] = "included" if ok else "excluded"
        a["exclusion_reason"] = "" if ok else f"missing storage > {MAX_MISSING_PCT}%"
        audits.append(a)
        rows.append(df.reset_index())

    joined = pd.concat(rows, ignore_index=True)
    snow, sa = load_snow(args.start, args.end)
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if snow is not None:
        snow.to_csv(PROC / "snow_daily.csv", index=False, date_format="%Y-%m-%d")
    cols = ["station_id", "reservoir_name", "date", "storage_af", "storage_pct", "capacity_af", "cdec_flag",
            "rain_mm", "temp_c", "humidity_pct", "lat", "lon"]
    joined[cols].to_csv(PROC / "daily_joined.csv", index=False, date_format="%Y-%m-%d")
    cov = pd.DataFrame(audits)
    cov.to_csv(PROC / "coverage.csv", index=False)
    years = pd.DataFrame(year_rows)
    years.to_csv(PROC / "station_year_audit.csv", index=False)
    scan = pd.DataFrame(scan_rows, columns=["station_id", "date", "change_pp", "rain_4d_mm", "documented_break"])
    scan.to_csv(PROC / "discontinuity_scan.csv", index=False)

    meta = stations.merge(cov[["station_id", "status", "exclusion_reason", "first_date", "last_date",
                               "n_storage_days", "missing_storage_pct"]], on="station_id", how="left")
    meta[["station_id", "reservoir_name", "lat", "lon", "capacity_af", "capacity_source", "status",
          "first_date", "last_date", "n_storage_days", "missing_storage_pct", "exclusion_reason",
          "operator"]].to_csv(OUT / "stations.csv", index=False)
    joined[["station_id", "date", "storage_af", "storage_pct", "rain_mm", "temp_c", "humidity_pct"]] \
        .to_csv(OUT / "history_daily.csv", index=False, date_format="%Y-%m-%d")

    show = ["station_id", "status", "first_date", "last_date", "n_storage_days", "missing_storage_pct",
            "longest_storage_gap_days", "duplicate_date_rows", "conflicting_duplicate_dates", "out_of_range",
            "spike_candidates", "spikes_kept_physical", "spikes_removed_artifact", "spikes_removed_unreviewed", "max_abs_daily_change_pp", "weather_missing_days", "storage_min_pct",
            "storage_max_pct", "data_flags"]
    table = cov[[c for c in show if c in cov]]
    md = ["# Per-station coverage and audit table", "",
          f"Generated by `python -m src.build_dataset --start {args.start} --end {args.end}` from raw files in data/raw/.",
          "Storage = CDEC sensor 15, daily (end-of-day). Weather = NASA POWER daily point (LST). "
          "storage_pct uses the capacity in force on each date (data/capacity_history.csv).",
          "`data_flags`: counts of CDEC DATA_FLAG values (values kept; e = estimated, r = revised).",
          "Spike candidates (one-day jumps > 2 pp that immediately reverse) are adjudicated in data/spike_review.csv using "
          "the independent CDEC elevation sensor: artifacts and unreviewed candidates are set to missing, physical ones kept.", "",
          table.to_markdown(index=False), "",
          "## Station-year missing storage days", "",
          years.pivot(index="year", columns="station_id", values="missing_days").to_markdown(), "",
          "## Dry-day level shifts (> 2 pp in one day with < 5 mm rain over 4 days)", "",
          (scan.to_markdown(index=False) if len(scan) else "None found."), "",
          "## Documented series breaks (no target or change-feature may span these)", "",
          breaks[["station_id", "break_date", "description"]].to_markdown(index=False), ""]
    if snow is not None:
        md += ["## Snow pillows (CDEC sensor 82, adjusted SWE, inches)", "",
               "```", json.dumps(sa, indent=1), "```", ""]
    (ROOT / "data" / "coverage_table.md").write_text("\n".join(md) + "\n")
    if (RAW / "fetch_log.jsonl").exists():
        (ROOT / "data" / "raw_manifest.jsonl").write_text((RAW / "fetch_log.jsonl").read_text())
    print(table.drop(columns=["data_flags"]).to_string(index=False))
    print("dry-day shifts:\n", scan.to_string(index=False))
    print("snow:", sa)
    print(f"Saved {len(joined)} station-days to {PROC / 'daily_joined.csv'}")


if __name__ == "__main__":
    main()

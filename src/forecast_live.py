"""30-day research storage forecasts from recent CDEC observations with the frozen v2 model.

Run: python -m src.forecast_live [--as-of YYYY-MM-DD] [--refresh] [--offline DIR]

Steps (all local and private; Streamlit only reads the snapshot written at the end)
1. Refresh: download CDEC daily storage (sensor 15) for the six reservoirs and daily adjusted SWE
   (sensor 82) for the candidate pillows, from Jan 1 of the previous year to the issue date.
   Raw responses are saved unmodified under data/live/<issue date>/ with their own fetch log
   (never data/raw/, never the tracked raw manifest). A same-day cache is reused unless --refresh.
2. Clean with the unchanged historical code (src.build_dataset.load_cdec / load_snow): capacity in
   force, out-of-range values to missing, unreviewed one-day spikes to missing, no interpolation.
3. Basin SWE index from the frozen core pillows (WY<=2023 rule, stored in the model card).
4. Prediction-time features (no target, no storage after t, no SWE after t-1), identical in
   definition to src.features.construct_features plus models.add_snow_anomaly.
5. Score with the hash-checked reconstructed model (src.forecast_model) and add the frozen
   sym_season 90% interval offsets.
6. Write a versioned record forecasts/records/<forecast_id>.json (never overwritten) and copy it
   to forecasts/latest.json. Both are git-ignored and contain CDEC-derived values.
"""
import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src import build_dataset as bd
from src.features import HORIZON, MIN_CORE_FRACTION, SNOW_LAG, load_breaks
from src.forecast_schema import SCHEMA, validate

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
FORECASTS = ROOT / "forecasts"
CDEC_URL = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
FEATURES_15 = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct", "swe_basin_in", "swe_change_14d_in",
               "swe_anom_in", "season_sin", "season_cos"]
MAX_ORIGIN_AGE_DAYS = 14      # older Mondays are never used
STALE_ORIGIN_DAYS = 7         # an origin older than this is flagged stale
STALE_OBS_DAYS = 5            # last storage or SWE observation older than this (vs issue date) is flagged
HISTORY_DAYS = 180            # recent observations carried in the snapshot for the chart
PLAUSIBLE_PCT = (0.0, 100.0)  # projections outside this are flagged, never clipped


# ------------------------------------------------------------------ 1. refresh

def _sha(b):
    return hashlib.sha256(b).hexdigest()


def fetch(run_dir, start, end, stations, pillows, refresh=False):
    """Download one file per (station, year) for storage and one per year for SWE. Returns the fetch log."""
    from src.fetch import get   # retries with backoff, timeout, header validation
    run_dir = Path(run_dir)
    log_p = run_dir / "fetch_log.jsonl"
    if refresh and run_dir.exists():
        shutil.rmtree(run_dir)
    jobs = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        s, e = max(start, f"{y}-01-01"), min(end, f"{y}-12-31")
        for st in stations:
            jobs.append((run_dir / "cdec" / f"{st}_sensor15_D_{y}.csv",
                         {"Stations": st, "SensorNums": 15, "dur_code": "D", "Start": s, "End": e}))
        jobs.append((run_dir / "snow" / f"snow_sensor82_D_{y}.csv",
                     {"Stations": ",".join(pillows), "SensorNums": 82, "dur_code": "D", "Start": s, "End": e}))
    for path, params in jobs:
        if path.exists() and path.stat().st_size > 200:
            continue
        res = get(CDEC_URL, params, lambda r: r.text.startswith("STATION_ID") and r.text.count("\n") > 1,
                  tries=4, timeout=90)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(res.content)
        with open(log_p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"file": str(path.relative_to(run_dir)), "url": res.url, "status": res.status_code,
                                 "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "bytes": len(res.content), "sha256": _sha(res.content)}) + "\n")
    return read_log(run_dir)


def read_log(run_dir):
    p = Path(run_dir) / "fetch_log.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    latest = {r["file"]: r for r in rows}          # last download of each file wins
    for f, r in latest.items():                    # files on disk must still be the logged bytes
        fp = Path(run_dir) / f
        r["sha256_ok"] = fp.exists() and _sha(fp.read_bytes()) == r["sha256"]
    return sorted(latest.values(), key=lambda r: r["file"])


# ------------------------------------------------------------------ 2. clean (historical code, unchanged)

@contextmanager
def raw_dir(path):
    """Point src.build_dataset at a live raw folder for the duration of a call."""
    old = bd.RAW
    bd.RAW = Path(path)
    try:
        yield
    finally:
        bd.RAW = old


def clean(run_dir, start, end, stations):
    cap_hist = pd.read_csv(ROOT / "data" / "capacity_history.csv")
    storage, audits = {}, {}
    with raw_dir(run_dir):
        for st in stations:
            s, flags, a = bd.load_cdec(st, cap_hist, start, end)
            if s is None:
                audits[st] = {"raw_files": 0}
                continue
            storage[st] = pd.DataFrame({"storage_af": s, "cdec_flag": flags})
            audits[st] = a
        snow, snow_audit = bd.load_snow(start, end)
    return storage, audits, snow, snow_audit, cap_hist


def basin_index(snow, core):
    """Same rule as src.features.build_snow_index, with the frozen core pillow list:
    mean SWE over core pillows reporting that day, only if >= 2/3 of them report."""
    rows = []
    for st, ids in core.items():
        s = snow[snow.snow_station_id.isin(ids)]
        g = s.groupby("date").swe_in.agg(["mean", "count"])
        n = len(ids)
        rev = s.assign(r=s.flag.eq("r")).groupby("date").r.mean()
        rows.append(pd.DataFrame({"station_id": st, "date": g.index,
                                  "swe_basin_in": g["mean"].where(g["count"] >= MIN_CORE_FRACTION * n).to_numpy(),
                                  "n_core_reporting": g["count"].to_numpy(), "n_core": n,
                                  "share_revised_flag": rev.reindex(g.index).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ 3-4. prediction-time features

def segment(dates, breaks):
    dates = pd.DatetimeIndex(dates)
    return np.array([sum(d >= b for b in breaks) for d in dates])


def inference_features(storage, cap_at, snow_idx, clim, t, breaks=()):
    """Feature values for origin t using storage on dates <= t and SWE index on dates <= t-1 only.

    storage: Series of storage_af indexed by date (missing days absent or NaN).
    cap_at:  function date -> capacity in force (acre-feet).
    snow_idx: Series of basin SWE index indexed by date.
    clim:    sequence of 366 smoothed day-of-year SWE means (doy 1..366) from pre-2024 days.
    Returns (features dict, list of missing-input reasons). Nothing at or after t+1 is read for
    storage and nothing at or after t is read for SWE.
    """
    t = pd.Timestamp(t)
    storage = storage[storage.index <= t]
    snow_idx = snow_idx[snow_idx.index <= t - pd.Timedelta(days=SNOW_LAG)]

    def s_at(d):
        v = storage.get(d, np.nan)
        return float(v) if pd.notna(v) else np.nan

    def w_at(d):
        v = snow_idx.get(d, np.nan)
        return float(v) if pd.notna(v) else np.nan

    cap = float(cap_at(t))
    seg_t = segment([t], breaks)[0]
    st_t = s_at(t)
    f = {"storage_pct": 100 * st_t / cap}
    for k in (7, 30, 60):
        d = t - pd.Timedelta(days=k)
        same = segment([d], breaks)[0] == seg_t
        f[f"prev{k}_pct"] = 100 * (st_t - s_at(d)) / cap if same else np.nan
    t1 = t - pd.Timedelta(days=SNOW_LAG)
    f["swe_basin_in"] = w_at(t1)
    f["swe_change_14d_in"] = w_at(t1) - w_at(t1 - pd.Timedelta(days=14))
    f["swe_anom_in"] = f["swe_basin_in"] - float(clim[t1.dayofyear - 1])
    doy = t.dayofyear
    f["season_sin"], f["season_cos"] = float(np.sin(2 * np.pi * doy / 365.25)), float(np.cos(2 * np.pi * doy / 365.25))
    missing = []
    if np.isnan(st_t):
        missing.append(f"storage on origin {t.date()} missing")
    for k in (7, 30, 60):
        d = t - pd.Timedelta(days=k)
        if segment([d], breaks)[0] != seg_t:
            missing.append(f"{k}-day storage change would span a documented series break")
        elif np.isnan(s_at(d)):
            missing.append(f"storage on {d.date()} (t-{k}) missing")
    if np.isnan(w_at(t1)):
        missing.append(f"basin SWE index on {t1.date()} (t-1) missing")
    if np.isnan(w_at(t1 - pd.Timedelta(days=14))):
        missing.append(f"basin SWE index on {(t1 - pd.Timedelta(days=14)).date()} (t-15) missing")
    return f, missing


def design_row(f, station, stations):
    row = {k: f[k] for k in FEATURES_15}
    row.update({f"station_{s}": float(s == station) for s in stations})
    return pd.DataFrame([row])


def candidate_origins(issue_date):
    """Mondays t with t <= issue_date - 1 day and issue_date - t <= MAX_ORIGIN_AGE_DAYS, newest first."""
    last = pd.Timestamp(issue_date) - pd.Timedelta(days=1)
    t = last - pd.Timedelta(days=last.dayofweek)
    out = []
    while (pd.Timestamp(issue_date) - t).days <= MAX_ORIGIN_AGE_DAYS:
        out.append(t)
        t -= pd.Timedelta(days=7)
    return out


def season_of(d):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA"}.get(
        pd.Timestamp(d).month, "SON")


# ------------------------------------------------------------------ 5. score one reservoir

def forecast_station(st, name, model, card, storage_df, snow_idx_df, cap_at, issue_date, breaks, cap_source=""):
    issue = pd.Timestamp(issue_date)
    storage = storage_df.storage_af.dropna() if storage_df is not None else pd.Series(dtype=float)
    si = snow_idx_df[snow_idx_df.station_id == st].set_index("date") if snow_idx_df is not None else pd.DataFrame()
    swe = si.swe_basin_in.dropna() if len(si) else pd.Series(dtype=float)
    rec = {"station_id": st, "reservoir_name": name, "status": "unavailable", "reasons": [], "warnings": [],
           "issue_date": str(issue.date()),
           "storage_last_obs_date": str(storage.index.max().date()) if len(storage) else None,
           "snow_last_obs_date": str(swe.index.max().date()) if len(swe) else None,
           "origins_considered": []}
    if not len(storage):
        rec["reasons"].append("No CDEC storage values were retrieved for this reservoir (CDEC unavailable or no data).")
        return rec
    if not len(swe):
        rec["reasons"].append("No basin snowpack index could be formed (snow pillows unavailable or too few reporting).")
    clim = card["snow_climatology_in"][st]
    chosen = None
    for t in candidate_origins(issue):
        f, missing = inference_features(storage, cap_at, swe, clim, t, breaks)
        rec["origins_considered"].append({"origin_date": str(t.date()), "eligible": not missing, "missing": missing})
        if not missing:
            chosen = (t, f)
            break
    if chosen is None:
        rec["reasons"].append(f"No Monday origin within {MAX_ORIGIN_AGE_DAYS} days of the issue date has every "
                              "required input. Older origins are never used.")
        return rec
    t, f = chosen
    target = t + pd.Timedelta(days=HORIZON)
    if target <= issue:
        rec["reasons"].append(f"Target date {target.date()} is not after the issue date.")
        return rec
    x = design_row(f, st, card["stations"])
    if list(x.columns) != card["feature_order"]:
        raise ValueError("feature order mismatch")
    pp = float(model.predict(x)[0])
    cap = float(cap_at(t))
    s_af = float(storage[t])
    lo_off, hi_off = card["interval_offsets_pp"][season_of(t)]
    lo, hi = pp + lo_off, pp + hi_off
    proj_pct = f["storage_pct"] + pp
    age = (issue - t).days
    if age > STALE_ORIGIN_DAYS:
        rec["warnings"].append(f"Stale origin: the newest eligible Monday is {age} days before the issue date; "
                               "a more recent Monday lacked required observations.")
    for label, d in (("storage", rec["storage_last_obs_date"]), ("snowpack", rec["snow_last_obs_date"])):
        if d is None or (issue - pd.Timestamp(d)).days > STALE_OBS_DAYS:
            rec["warnings"].append(f"Latest {label} observation ({d}) is more than {STALE_OBS_DAYS} days before the issue date.")
    plaus_ok = PLAUSIBLE_PCT[0] <= proj_pct <= PLAUSIBLE_PCT[1]
    if not plaus_ok:
        rec["warnings"].append(f"Projected storage {proj_pct:.1f}% of documented capacity is outside 0-100%. "
                               "The value is shown unclipped and is unsuitable for operational interpretation.")
    reliable = bool(card["interval_reliable"][st])
    cov = card["historical_interval_coverage"][st]
    if not reliable:
        rec["warnings"].append(f"Interval caution: nominal 90% intervals covered only {cov:.0%} of {st} outcomes in the "
                               "2024-2025 backtest.")
    t1 = t - pd.Timedelta(days=SNOW_LAG)
    srow = si.loc[t1] if t1 in si.index else None
    rec.update({
        "status": "ok", "origin_date": str(t.date()), "target_date": str(target.date()),
        "horizon_days": HORIZON, "origin_age_days": age, "origin_season": season_of(t),
        "capacity_af": cap, "capacity_source": cap_source,
        "observed": {"storage_af": s_af, "storage_pct": f["storage_pct"], "date": str(t.date()),
                     "cdec_flag": str(storage_df.cdec_flag.get(t, "") or "")},
        "forecast": {"change_pp": pp, "change_af": pp / 100 * cap,
                     "projected_af": s_af + pp / 100 * cap, "projected_pct": proj_pct,
                     "lower_change_pp": lo, "upper_change_pp": hi,
                     "lower_projected_af": s_af + lo / 100 * cap, "upper_projected_af": s_af + hi / 100 * cap,
                     "lower_projected_pct": f["storage_pct"] + lo, "upper_projected_pct": f["storage_pct"] + hi},
        "interval": {"method": card["interval"]["method"], "nominal": card["interval"]["nominal"],
                     "season": season_of(t), "offsets_pp": [lo_off, hi_off],
                     "historical_coverage": cov, "interval_reliable": reliable,
                     "calibration": "absolute out-of-fold residuals of the primary model, development folds WY2008-2023",
                     "future_coverage_established": False},
        "plausibility": {"projected_within_0_100_pct": plaus_ok,
                         "bounds_within_0_100_pct": bool(0 <= f["storage_pct"] + lo and f["storage_pct"] + hi <= 100),
                         "suitable_for_operational_interpretation": False},
        "features": {**{k: f[k] for k in FEATURES_15}, **{c: float(x[c].iloc[0]) for c in x.columns if c.startswith("station_")}},
        "snow": {"swe_date": str(t1.date()), "swe_basin_in": f["swe_basin_in"], "swe_change_14d_in": f["swe_change_14d_in"],
                 "swe_anom_in": f["swe_anom_in"], "climatology_swe_in": float(clim[t1.dayofyear - 1]),
                 "n_core_reporting": int(srow.n_core_reporting) if srow is not None else None,
                 "n_core": int(srow.n_core) if srow is not None else None,
                 "share_revised_flag": (float(srow.share_revised_flag) if srow is not None and pd.notna(srow.share_revised_flag) else None),
                 "core_pillows": card["core_snow_pillows"][st]},
    })
    return rec


def recent_history(storage_df, snow_idx_df, st, cap_at, issue, origin):
    start = pd.Timestamp(issue) - pd.Timedelta(days=HISTORY_DAYS)
    d = storage_df[(storage_df.index >= start) & (storage_df.index <= pd.Timestamp(issue))].copy()
    si = snow_idx_df[snow_idx_df.station_id == st].set_index("date").swe_basin_in if snow_idx_df is not None else None
    out = []
    for day, r in d.iterrows():
        v = r.storage_af
        out.append({"date": str(day.date()), "storage_af": float(v) if pd.notna(v) else None,
                    "storage_pct": float(100 * v / cap_at(day)) if pd.notna(v) else None,
                    "swe_basin_in": (float(si[day]) if si is not None and day in si.index and pd.notna(si[day]) else None),
                    "used_by_forecast": bool(origin is not None and day <= pd.Timestamp(origin))})
    return out


# ------------------------------------------------------------------ revision check against the archive

def revision_check(storage, start, end):
    p = ROOT / "data/processed/daily_joined.csv"
    if not p.exists():
        return {"available": False}
    h = pd.read_csv(p, parse_dates=["date"], usecols=["station_id", "date", "storage_af"])
    h = h[(h.date >= start) & (h.date <= end)]
    out = {"available": True, "period": [str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())], "by_station": {}}
    for st, df in storage.items():
        m = h[h.station_id == st].set_index("date").storage_af.to_frame("archived").join(df.storage_af.rename("live"), how="inner").dropna()
        if len(m):
            d = (m.live - m.archived).abs()
            out["by_station"][st] = {"overlap_days": int(len(m)), "days_changed": int((d > 0.5).sum()),
                                     "max_abs_change_af": float(d.max())}
    return out


# ------------------------------------------------------------------ 6. snapshot

def run(issue_date=None, refresh=False, offline=None, out_dir=FORECASTS, art_dir=None):
    from src.forecast_model import ART_DIR, load_artifact, sha_file, source_commit
    model, card = load_artifact(art_dir or ART_DIR)
    issued_utc = datetime.now(timezone.utc)
    issue = pd.Timestamp(issue_date or issued_utc.date())
    if issue.date() > issued_utc.date():
        raise SystemExit("REFUSED: the issue date cannot be in the future.")
    stations_meta = pd.read_csv(ROOT / "stations.csv").set_index("station_id")
    stations = card["stations"]
    pillows = sorted({p for v in card["core_snow_pillows"].values() for p in v})
    start, end = f"{issue.year - 1}-01-01", str(issue.date())
    run_dir = Path(offline) if offline else LIVE / str(issue.date())
    log = read_log(run_dir) if offline else None
    fetch_error = None
    if not offline:
        try:
            log = fetch(run_dir, start, end, stations, pillows, refresh=refresh)
        except Exception as exc:  # noqa: BLE001  reported in the snapshot, never filled in
            fetch_error = f"{type(exc).__name__}: {exc}"
            log = read_log(run_dir)
    storage, audits, snow, snow_audit, cap_hist = clean(run_dir, start, end, stations)
    snow_idx = basin_index(snow, card["core_snow_pillows"]) if snow is not None else None
    breaks = load_breaks()
    recs = []
    for st in stations:
        cap_at = (lambda d, st=st: float(bd.capacity_series(st, pd.DatetimeIndex([pd.Timestamp(d)]), cap_hist).iloc[0]))
        r = forecast_station(st, stations_meta.loc[st, "reservoir_name"], model, card, storage.get(st), snow_idx,
                             cap_at, issue, breaks.get(st, []), stations_meta.loc[st, "capacity_source"])
        if fetch_error and r["status"] != "ok":
            r["reasons"].insert(0, f"CDEC refresh failed: {fetch_error}")
        r["cleaning_audit"] = {k: v for k, v in audits.get(st, {}).items() if k != "data_flags"}
        r["recent_history"] = recent_history(storage[st], snow_idx, st, cap_at, issue, r.get("origin_date")) if st in storage else []
        recs.append(r)
    fid = f"wsf-v2-{issue.date()}-{issued_utc.strftime('%Y%m%dT%H%M%SZ')}"
    snap = {
        "schema": SCHEMA, "record_type": "prospective_research_forecast", "is_heldout_test_prediction": False,
        "data_mode": "real", "forecast_id": fid, "issued_utc": issued_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue_date": str(issue.date()), "horizon_days": HORIZON, "origin_rule": (
            f"latest Monday on or before issue date minus 1 day with all 15 inputs present, at most "
            f"{MAX_ORIGIN_AGE_DAYS} days before the issue date"),
        "target_quantity": "30-day change in reservoir storage, percentage points of the capacity in force at the origin "
                           "(not water-surface elevation)",
        "model": {k: card[k] for k in ("model_version", "model_name", "estimator", "hyperparameters", "training_rule",
                                       "training_cutoff_exclusive", "training_rows", "feature_order", "config_sha256",
                                       "artifact_sha256", "created_utc", "preregistered_run_commit", "source_commit",
                                       "reproduction_check")},
        "pipeline_commit": source_commit(),
        "sources": {"provider": "California Data Exchange Center (CDEC), CA Department of Water Resources",
                    "endpoint": CDEC_URL, "storage_sensor": "15 (reservoir storage, daily, acre-feet; end of day)",
                    "snow_sensor": "82 (snow water content, adjusted/revised, daily, inches; morning reading)",
                    "window": [start, end], "raw_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                    "refresh_error": fetch_error, "files": log or [],
                    "all_files_match_logged_sha256": bool(log) and all(r.get("sha256_ok") for r in log),
                    "snow_cleaning_audit": {k: v for k, v in (snow_audit or {}).items() if k != "snow_flags"},
                    "snow_flags": (snow_audit or {}).get("snow_flags", {}),
                    "revision_check_vs_archive": revision_check(storage, start, f"{issue.year - 1}-12-31")},
        "assumptions": [
            "Storage inputs use CDEC daily sensor 15 end-of-day values through the origin date only.",
            "Snow inputs use the basin index of core pillows (frozen WY<=2023 rule) through origin minus 1 day only.",
            "Recent sensor 82 values may still be revised; the model was trained mostly on fully revised values.",
            "The capacity denominator is the capacity in force on the origin date (data/capacity_history.csv) and is "
            "assumed unchanged through the target date.",
            "NASA POWER weather is not an input of the frozen v2 primary model and is not used.",
            "No planned releases, operations, forecasts of precipitation or interventions are represented."],
        "disclaimers": [
            "Experimental research forecast. Not operational water-management advice, not a water-supply outlook, "
            "and not a flood or dam-safety product.",
            "The model has not been prospectively validated.",
            "90% intervals are retrospectively calibrated on historical out-of-fold residuals; their future coverage "
            "is not established.",
            "Folsom (FOL) intervals covered only about 75% of outcomes in the 2024-2025 backtest.",
            "These records are new forecasts, not held-out test predictions, and are not part of the frozen v2 evaluation."],
        "stations": recs}
    errs, _ = validate(snap, today=issued_utc.date())
    if errs:
        raise SystemExit("REFUSED: snapshot failed validation: " + "; ".join(errs))
    rec_dir = Path(out_dir) / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rp = rec_dir / f"{fid}.json"
    if rp.exists():
        raise SystemExit(f"REFUSED: {rp} already exists; records are never overwritten.")
    body = json.dumps(snap, indent=2, allow_nan=False)
    rp.write_text(body)
    (Path(out_dir) / "latest.json").write_text(body)
    snap["_record_path"] = str(rp)
    snap["_record_sha256"] = sha_file(rp)
    return snap


def main():
    import warnings   # pandas on Python 3.14 emits a NumPy 'generic' timedelta notice; results are unaffected
    warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta", category=DeprecationWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="issue date (default: today, UTC); never in the future")
    ap.add_argument("--refresh", action="store_true", help="re-download even if today's raw files are cached")
    ap.add_argument("--offline", default=None, help="use raw files already in this folder; no network")
    a = ap.parse_args()
    s = run(a.as_of, refresh=a.refresh, offline=a.offline)
    print(f"forecast {s['forecast_id']} written to {s['_record_path']}")
    for r in s["stations"]:
        if r["status"] == "ok":
            f = r["forecast"]
            print(f"  {r['station_id']}: origin {r['origin_date']} -> target {r['target_date']}: "
                  f"{r['observed']['storage_pct']:.1f}% {f['change_pp']:+.2f} pp -> {f['projected_pct']:.1f}% "
                  f"(90% {f['lower_change_pp']:+.2f} to {f['upper_change_pp']:+.2f} pp)"
                  + (f"  WARN: {'; '.join(r['warnings'])}" if r["warnings"] else ""))
        else:
            print(f"  {r['station_id']}: UNAVAILABLE: {'; '.join(r['reasons'])}")


if __name__ == "__main__":
    main()

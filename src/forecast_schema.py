"""Schema and validation for 30-day research forecast snapshots (pure Python, no ML imports).

Used by src.forecast_live before a snapshot is written and by the Streamlit page before anything
is displayed. validate() returns (errors, warnings): errors mean the snapshot must not be shown
as a forecast; warnings are shown next to it.
"""
from datetime import date, timedelta
import math

SCHEMA = "watershedshift-forecast/1.0"
RECORD_TYPE = "prospective_research_forecast"
FEATURE_ORDER = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct", "swe_basin_in", "swe_change_14d_in",
                 "swe_anom_in", "season_sin", "season_cos", "station_DNP", "station_EXC", "station_FOL",
                 "station_NML", "station_ORO", "station_SHA"]
WEATHER = {"rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"}
STALE_ISSUE_DAYS = 7
TOL = 1e-6


def _d(s):
    return date.fromisoformat(str(s))


def _finite(*v):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)


def validate_station(r, issue, horizon, errs, warns, today, real=True):
    sid = r.get("station_id", "?")
    if r.get("status") != "ok":
        if not r.get("reasons"):
            errs.append(f"{sid}: unavailable without a stated reason")
        return
    try:
        o, t = _d(r["origin_date"]), _d(r["target_date"])
    except (KeyError, ValueError):
        errs.append(f"{sid}: origin or target date missing or malformed")
        return
    if (t - o).days != horizon:
        errs.append(f"{sid}: target {t} is not origin {o} + {horizon} days")
    if o.weekday() != 0:
        errs.append(f"{sid}: origin {o} is not a Monday")
    if not o < issue:
        errs.append(f"{sid}: origin {o} is not before the issue date {issue}")
    if not t > issue:
        errs.append(f"{sid}: target {t} is not after the issue date {issue}")
    if today is not None and t <= today:
        (errs if real else warns).append(f"{sid}: target date {t} has already passed (today {today}); "
                                         "this forecast has expired")
    ob, f, cap = r.get("observed", {}), r.get("forecast", {}), r.get("capacity_af")
    keys = ["change_pp", "change_af", "projected_af", "projected_pct", "lower_change_pp", "upper_change_pp",
            "lower_projected_af", "upper_projected_af", "lower_projected_pct", "upper_projected_pct"]
    if not _finite(cap, ob.get("storage_af"), ob.get("storage_pct"), *[f.get(k) for k in keys]):
        errs.append(f"{sid}: non-finite or missing forecast values")
        return
    if cap <= 0:
        errs.append(f"{sid}: capacity must be positive")
        return
    checks = {"storage_pct": (ob["storage_pct"], 100 * ob["storage_af"] / cap),
              "change_af": (f["change_af"], f["change_pp"] / 100 * cap),
              "projected_af": (f["projected_af"], ob["storage_af"] + f["change_af"]),
              "projected_pct": (f["projected_pct"], ob["storage_pct"] + f["change_pp"]),
              "lower_projected_pct": (f["lower_projected_pct"], ob["storage_pct"] + f["lower_change_pp"]),
              "upper_projected_pct": (f["upper_projected_pct"], ob["storage_pct"] + f["upper_change_pp"]),
              "lower_projected_af": (f["lower_projected_af"], ob["storage_af"] + f["lower_change_pp"] / 100 * cap),
              "upper_projected_af": (f["upper_projected_af"], ob["storage_af"] + f["upper_change_pp"] / 100 * cap)}
    for k, (a, b) in checks.items():
        if abs(a - b) > TOL * max(1.0, abs(b)):
            errs.append(f"{sid}: {k} inconsistent ({a} vs {b})")
    if not f["lower_change_pp"] <= f["change_pp"] <= f["upper_change_pp"]:
        errs.append(f"{sid}: point forecast outside its interval")
    feats = r.get("features", {})
    expected = FEATURE_ORDER if real else FEATURE_ORDER[:9] + [k for k in feats if k.startswith("station_")]
    if list(feats) != expected:
        errs.append(f"{sid}: feature names or order differ from the frozen primary model")
    elif not _finite(*feats.values()):
        errs.append(f"{sid}: a model input is missing")
    if WEATHER & set(feats):
        errs.append(f"{sid}: weather appears as a model input")
    iv = r.get("interval", {})
    if iv.get("future_coverage_established") is not False:
        errs.append(f"{sid}: interval must state that future coverage is not established")
    if iv.get("interval_reliable") is False and not any("Interval caution" in w for w in r.get("warnings", [])):
        errs.append(f"{sid}: interval marked unreliable without its warning")
    if r.get("plausibility", {}).get("suitable_for_operational_interpretation") is not False:
        errs.append(f"{sid}: must be marked unsuitable for operational interpretation")
    if not 0 <= f["projected_pct"] <= 100 and not any("outside 0-100%" in w for w in r.get("warnings", [])):
        errs.append(f"{sid}: implausible projection without a warning")
    for w in r.get("warnings", []):
        warns.append(f"{sid}: {w}")


def validate(snap, today=None):
    """today: date used for expiry and staleness checks (None skips them)."""
    errs, warns = [], []
    if not isinstance(snap, dict):
        return ["snapshot is not a JSON object"], warns
    if snap.get("schema") != SCHEMA:
        errs.append(f"schema is {snap.get('schema')!r}, expected {SCHEMA!r}")
    if snap.get("record_type") != RECORD_TYPE or snap.get("is_heldout_test_prediction") is not False:
        errs.append("record is not marked as a prospective research forecast (or is marked as a held-out test row)")
    if snap.get("data_mode") not in ("real", "synthetic"):
        errs.append("data_mode must be 'real' or 'synthetic'")
    model = snap.get("model", {})
    real = snap.get("data_mode") == "real"
    if real and model.get("feature_order") != FEATURE_ORDER:
        errs.append("model feature order differs from the frozen primary model")
    if snap.get("data_mode") == "real" and model.get("model_version") != "v2-frozen":
        errs.append("real-data snapshot does not come from the frozen v2 model")
    try:
        issue = _d(snap["issue_date"])
    except (KeyError, ValueError):
        return errs + ["issue_date missing or malformed"], warns
    horizon = snap.get("horizon_days")
    if horizon != 30:
        errs.append("horizon_days must be 30")
    if today is not None:
        if issue > today and real:
            errs.append(f"issue date {issue} is in the future")
        elif (today - issue).days > STALE_ISSUE_DAYS:
            warns.append(f"Snapshot issued {issue}, {(today - issue).days} days ago: observations may be stale.")
    st = snap.get("stations")
    if not isinstance(st, list) or not st:
        errs.append("no station records")
        return errs, warns
    for r in st:
        validate_station(r, issue, horizon or 30, errs, warns, today, real)
    return errs, warns


def expired(snap, today):
    """True when every available station's target date is on or before today."""
    ok = [r for r in snap.get("stations", []) if r.get("status") == "ok"]
    return bool(ok) and all(_d(r["target_date"]) <= today for r in ok)


def next_origin_after(issue):
    """The Monday on which a new origin becomes possible (one day after it, data permitting)."""
    d = _d(issue)
    return d + timedelta(days=(7 - d.weekday()) % 7 or 7)

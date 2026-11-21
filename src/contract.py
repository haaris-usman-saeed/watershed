"""Shared backend -> frontend output contract (schema v1).

Single source of truth for the files the UI reads. The engineering pipeline
writes these files; `validate_dir` is used by tests on both the real outputs
and the synthetic UI fixtures so the two cannot drift apart.
See docs/UI_CONTRACT.md for the human-readable specification.
"""
from pathlib import Path
import json
import sys

import pandas as pd

SCHEMA_VERSION = "1.0"

# file name -> (required columns, date columns)
CSV_FILES = {
    "stations.csv": (
        ["station_id", "reservoir_name", "lat", "lon", "capacity_af", "capacity_source",
         "status", "first_date", "last_date", "n_storage_days", "missing_storage_pct"],
        ["first_date", "last_date"],
    ),
    "history_daily.csv": (
        ["station_id", "date", "storage_af", "storage_pct", "rain_mm", "temp_c", "humidity_pct"],
        ["date"],
    ),
    "test_predictions.csv": (
        ["station_id", "reservoir_name", "date", "target_date", "target_delta_pp",
         "pred_delta_pp", "lower_90_pp", "upper_90_pp", "baseline_persistence_pp",
         "baseline_trend_pp", "baseline_seasonal_pp"],
        ["date", "target_date"],
    ),
}
JSON_FILES = {
    "metrics.json": ["schema_version", "data_mode", "generated_utc", "ML", "Persistence",
                     "30-day trend", "Historical seasonal", "interval_90_coverage",
                     "interval_90_halfwidth_pp", "train_count", "val_count", "test_count",
                     "test_station_count", "note", "split", "per_station", "bootstrap"],
    "feature_catalog.json": ["schema_version", "features"],
}
DATA_MODES = {"real", "SYNTHETIC_SAMPLE"}
STATUS_VALUES = {"included", "excluded"}


def validate_dir(path) -> list:
    """Return a list of human-readable contract violations (empty list = valid)."""
    path = Path(path)
    errors = []
    frames = {}
    for name, (cols, date_cols) in CSV_FILES.items():
        f = path / name
        if not f.exists():
            errors.append(f"missing file {name}")
            continue
        df = pd.read_csv(f)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            errors.append(f"{name}: missing columns {missing}")
            continue
        for c in date_cols:
            parsed = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
            if parsed.isna().any():
                errors.append(f"{name}: column {c} has non-ISO (YYYY-MM-DD) values")
        frames[name] = df
    for name, keys in JSON_FILES.items():
        f = path / name
        if not f.exists():
            errors.append(f"missing file {name}")
            continue
        obj = json.loads(f.read_text())
        missing = [k for k in keys if k not in obj]
        if missing:
            errors.append(f"{name}: missing keys {missing}")
        frames[name] = obj
    if "metrics.json" in frames and frames["metrics.json"].get("data_mode") not in DATA_MODES:
        errors.append("metrics.json: data_mode must be 'real' or 'SYNTHETIC_SAMPLE'")
    if "stations.csv" in frames:
        bad = set(frames["stations.csv"].status) - STATUS_VALUES
        if bad:
            errors.append(f"stations.csv: unknown status values {bad}")
    if "test_predictions.csv" in frames:
        p = frames["test_predictions.csv"]
        if (pd.to_datetime(p.target_date) <= pd.to_datetime(p.date)).any():
            errors.append("test_predictions.csv: target_date must be after date")
        if (p.lower_90_pp > p.upper_90_pp).any():
            errors.append("test_predictions.csv: lower_90_pp > upper_90_pp")
        if "stations.csv" in frames:
            unknown = set(p.station_id) - set(frames["stations.csv"].station_id)
            if unknown:
                errors.append(f"test_predictions.csv: station_id not in stations.csv {unknown}")
    if "metrics.json" in frames:
        errors += _check_additive(frames["metrics.json"], frames.get("feature_catalog.json"))
    return errors


def _check_additive(m, catalog):
    """Optional additive v2 handoff fields (docs/UI_CONTRACT.md section 3.8). Checked only when present."""
    errors = []
    for name, obj in (("metrics.json", m), ("feature_catalog.json", catalog or {})):
        pmf = obj.get("primary_model_features")
        if pmf is not None and not (isinstance(pmf, list) and pmf and all(isinstance(x, str) for x in pmf)):
            errors.append(f"{name}: primary_model_features must be a non-empty list of strings")
    if catalog and "primary_model_features" in catalog and "primary_model_features" in m \
            and catalog["primary_model_features"] != m["primary_model_features"]:
        errors.append("primary_model_features differs between metrics.json and feature_catalog.json")
    bt = m.get("backtest")
    if bt is not None and bt.get("is_live_forecast") is not False:
        errors.append("metrics.json: backtest.is_live_forecast must be false")
    rel = m.get("interval_reliability")
    if rel is not None:
        thr = rel.get("threshold_min_coverage")
        for sid, st in m.get("per_station", {}).items():
            flag = st.get("interval_reliable")
            if not isinstance(flag, bool):
                errors.append(f"metrics.json: per_station.{sid}.interval_reliable must be true or false")
            elif isinstance(thr, (int, float)) and flag != (st.get("interval_90_coverage", -1) >= thr):
                errors.append(f"metrics.json: per_station.{sid}.interval_reliable disagrees with the threshold rule")
            elif rel.get("by_station", {}).get(sid) != flag:
                errors.append(f"metrics.json: interval_reliability.by_station.{sid} disagrees with per_station")
    return errors


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    problems = validate_dir(target)
    print("\n".join(problems) if problems else f"{target}: contract v{SCHEMA_VERSION} OK")
    sys.exit(1 if problems else 0)

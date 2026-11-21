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
    return errors


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    problems = validate_dir(target)
    print("\n".join(problems) if problems else f"{target}: contract v{SCHEMA_VERSION} OK")
    sys.exit(1 if problems else 0)

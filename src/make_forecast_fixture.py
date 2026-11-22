"""SYNTHETIC forecast snapshot for UI development and tests. NOT A FORECAST.

Run: python -m src.make_forecast_fixture   (writes app/sample_data/forecast_snapshot.json)

Uses the synthetic stations and storage in app/sample_data (SYN1..SYN3), invented snowpack, and a
placeholder arithmetic "model" in place of the v2 model. It exercises the real feature builder,
origin rule, interval arithmetic, record layout and validation, and is labelled data_mode
"synthetic" so the page shows it only under a synthetic-data banner.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_live import FEATURES_15, forecast_station, recent_history
from src.forecast_schema import SCHEMA, validate

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "app" / "sample_data"
ISSUE = pd.Timestamp("2025-12-31")
NOTE = "SYNTHETIC FIXTURE for UI development only. Not real observations and not a forecast from the v2 model."


class PlaceholderModel:
    """Not a trained model: change = 0.4 * prev30_pct + 0.5 * swe_change_14d_in."""
    def predict(self, x):
        return (0.4 * x["prev30_pct"] + 0.5 * x["swe_change_14d_in"]).to_numpy()


def build():
    stations = pd.read_csv(SAMPLE / "stations.csv")
    hist = pd.read_csv(SAMPLE / "history_daily.csv", parse_dates=["date"])
    ids = stations.station_id.tolist()
    days = pd.date_range("2025-01-01", ISSUE, freq="D")
    doy = days.dayofyear.to_numpy()
    snow = pd.concat([pd.DataFrame({"station_id": s, "date": days, "n_core_reporting": 3, "n_core": 3,
                                    "share_revised_flag": 0.0,
                                    "swe_basin_in": np.round(np.clip(12 * np.cos(2 * np.pi * (doy - 70) / 365.25) + 3 + i, 0, None), 2)})
                      for i, s in enumerate(ids)], ignore_index=True)
    card = {"stations": ids, "feature_order": FEATURES_15 + [f"station_{s}" for s in ids],
            "snow_climatology_in": {s: [6.0] * 366 for s in ids},
            "interval_offsets_pp": {k: [-4.0, 4.0] for k in ("DJF", "MAM", "JJA", "SON")},
            "interval": {"method": "sym_season", "nominal": 0.9},
            "interval_reliable": {s: s != "SYN3" for s in ids},
            "historical_interval_coverage": {s: (0.75 if s == "SYN3" else 0.92) for s in ids},
            "core_snow_pillows": {s: ["SYNTHETIC"] for s in ids}}
    recs = []
    for r in stations.itertuples():
        g = hist[hist.station_id == r.station_id].set_index("date")
        g = g[g.index <= ISSUE - pd.Timedelta(days=2)]       # mimic a 1 to 2 day reporting lag
        sdf = pd.DataFrame({"storage_af": g.storage_af, "cdec_flag": ""})
        cap_at = (lambda d, c=float(r.capacity_af): c)
        rec = forecast_station(r.station_id, r.reservoir_name, PlaceholderModel(), card, sdf, snow, cap_at, ISSUE, [],
                               "SYNTHETIC")
        rec["cleaning_audit"] = {}
        rec["recent_history"] = recent_history(sdf, snow, r.station_id, cap_at, ISSUE, rec.get("origin_date"))
        recs.append(rec)
    return {"schema": SCHEMA, "record_type": "prospective_research_forecast", "is_heldout_test_prediction": False,
            "data_mode": "synthetic", "note": NOTE, "forecast_id": "wsf-synthetic-fixture", "issued_utc": "2025-12-31T12:00:00Z",
            "issue_date": str(ISSUE.date()), "horizon_days": 30,
            "origin_rule": "latest Monday on or before issue date minus 1 day with all inputs present",
            "target_quantity": "30-day change in reservoir storage, percentage points of capacity (SYNTHETIC)",
            "model": {"model_version": "synthetic-placeholder", "model_name": "placeholder arithmetic, NOT the v2 model",
                      "training_cutoff_exclusive": "n/a", "feature_order": card["feature_order"],
                      "artifact_sha256": "n/a", "config_sha256": "n/a", "created_utc": "n/a"},
            "sources": {"provider": "SYNTHETIC (no agency data)", "files": [], "refresh_error": None,
                        "all_files_match_logged_sha256": False, "revision_check_vs_archive": {"available": False}},
            "assumptions": [NOTE], "disclaimers": [NOTE], "stations": recs}


def main():
    snap = build()
    errs, _ = validate(snap)
    if errs:
        raise SystemExit("fixture invalid: " + "; ".join(errs))
    (SAMPLE / "forecast_snapshot.json").write_text(json.dumps(snap, indent=2, allow_nan=False) + "\n")
    print("wrote", (SAMPLE / "forecast_snapshot.json").relative_to(ROOT),
          [(r["station_id"], r["status"]) for r in snap["stations"]])


if __name__ == "__main__":
    main()

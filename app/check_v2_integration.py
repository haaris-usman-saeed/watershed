"""Read-only smoke test for the integrated v2 UI data contract.

Run from the repository root after unpacking the private real outputs:
    python app/check_v2_integration.py outputs
This script NEVER retrains, rescores, or modifies any prediction files.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_io
import ui_metrics as M


def main():
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs").resolve()
    d = ui_io.load_dir(data_dir)
    assert d["data_mode"] == "real", "Must point to real v2 outputs, not synthetic fixtures"
    metrics, preds = d["metrics"], d["preds"]
    assert metrics.get("model_version") == "v2"
    assert len(preds) == metrics["test_count"] == 593
    nm = M.normalize_metrics(metrics)
    assert nm.selected_baseline_key == "doy", "Wrong preregistered baseline"
    assert len([b for b in nm.bootstrap if b.model_key == "doy"]) == 3
    assert all(b.lo > 0 and not b.sign_suspect for b in nm.bootstrap if b.model_key == "doy")
    assert M.verdict(nm).kind == "better"
    assert "baseline_doy_pp" in preds.columns
    assert "capacity_af" in preds.columns and "storage_pct" in preds.columns
    assert (preds["target_date"] - preds["date"]).dt.days.eq(30).all()
    assert "storage_pct" in d["history"].columns
    assert "swe_basin_in" in d["history"].columns
    assert metrics["per_station"]["FOL"]["interval_90_coverage"] < .85
    assert metrics["per_station"]["FOL"].get("interval_reliable", False) is False
    model_cols = metrics.get("primary_model_features", [])
    assert "swe_basin_in" in model_cols and "station_ORO" in model_cols
    assert not any(x in model_cols for x in ("rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"))
    hist = d["history"]
    old = hist[(hist.station_id == "ORO") & (hist.date < pd.Timestamp("2024-07-02"))].dropna(subset=["storage_af", "storage_pct"])
    assert len(old) > 5000
    expected = old.storage_af / 3537577 * 100
    assert np.allclose(old.storage_pct, expected, atol=1e-7), "Oroville date-specific capacity percentages not retained"
    expected_ci = metrics["bootstrap"]["by_comparison_and_block"]["ML minus Smoothed DOY seasonal"]
    for b in (5, 9, 13):
        assert expected_ci[f"{b}_weeks"]["ci95_high"] < 0
    problems = M.cross_checks(nm, M.mae_table(preds), M.coverage(preds), preds)
    assert not problems, f"v2 UI data inconsistency: {problems}"
    print("V2 INTEGRATION PASS: baseline, confidence intervals, 593 forecasts, SWE inputs, Oroville capacity and Folsom caution")


if __name__ == "__main__":
    main()
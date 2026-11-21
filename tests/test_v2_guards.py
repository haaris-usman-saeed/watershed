"""Tests for the frozen v2 evaluation guards and v1/v2 comparison logic."""
import json

import numpy as np
import pandas as pd
import pytest

import src.train_v2 as tv


def test_run_record_refuses_different_configuration(tmp_path):
    rec = {"frozen_hashes": {"config/v2_frozen.json": "deadbeef"}, "headline": {}}
    (tmp_path / "RUN_RECORD.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        tv.check_run_record(tmp_path, allow_new=False)


def test_run_record_allows_identical_configuration(tmp_path):
    rec = {"frozen_hashes": tv.frozen_hashes(), "headline": {}}
    (tmp_path / "RUN_RECORD.json").write_text(json.dumps(rec))
    assert tv.check_run_record(tmp_path, allow_new=False) is not None


def test_frozen_hashes_cover_config_and_code():
    h = tv.frozen_hashes()
    assert "config/v2_frozen.json" in h and "src/train_v2.py" in h and "src/models.py" in h


def _rows(n_weeks=30):
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    rows = []
    for s in ["ORO", "SHA"]:
        for d in dates:
            rows.append({"station_id": s, "date": d, "target_delta_pp": float(d.dayofyear % 7)})
    return pd.DataFrame(rows)


def test_v1_comparison_excludes_oro_old_table_rows_and_checks_targets():
    out = _rows()
    out["pred_delta_pp"] = out.target_delta_pp + 1.0
    v1 = out[["station_id", "date", "target_delta_pp"]].copy()
    v1["pred_delta_pp"] = v1.target_delta_pp + 2.0
    r = tv.v1_comparison(out, v1)
    n_oro_old = ((out.station_id == "ORO") & (out.date < tv.ORO_BREAK)).sum()
    assert r["n_common_rows"] == len(out) - n_oro_old
    assert np.isclose(r["v2_primary_mae"], 1.0) and np.isclose(r["v1_gbm_mae"], 2.0)
    bad = v1.copy()
    bad.loc[bad.station_id == "SHA", "target_delta_pp"] += 5
    with pytest.raises(ValueError):
        tv.v1_comparison(out, bad)

from pathlib import Path
import json

from src.contract import validate_dir

ROOT = Path(__file__).resolve().parents[1]


def test_sample_fixtures_follow_contract():
    sample = ROOT / "app" / "sample_data"
    assert validate_dir(sample) == []
    assert json.loads((sample / "metrics.json").read_text())["data_mode"] == "SYNTHETIC_SAMPLE"


def test_real_outputs_follow_contract_if_present():
    out = ROOT / "outputs"
    if (out / "metrics.json").exists() and (out / "stations.csv").exists():
        assert validate_dir(out) == []
        assert json.loads((out / "metrics.json").read_text())["data_mode"] == "real"


def _fixture_copy(tmp_path):
    import shutil
    d = tmp_path / "fx"
    shutil.copytree(ROOT / "app" / "sample_data", d)
    return d


def test_additive_handoff_fields_are_optional_and_checked(tmp_path):
    d = _fixture_copy(tmp_path)
    m = json.loads((d / "metrics.json").read_text())
    c = json.loads((d / "feature_catalog.json").read_text())
    thr = 0.85
    flags = {s: v["interval_90_coverage"] >= thr for s, v in m["per_station"].items()}
    for s, f in flags.items():
        m["per_station"][s]["interval_reliable"] = f
    m["interval_reliability"] = {"threshold_min_coverage": thr, "by_station": flags}
    m["backtest"] = {"is_live_forecast": False}
    m["primary_model_features"] = c["primary_model_features"] = ["storage_pct", "station_SYN1"]
    (d / "metrics.json").write_text(json.dumps(m))
    (d / "feature_catalog.json").write_text(json.dumps(c))
    assert validate_dir(d) == []
    s0 = next(iter(flags))
    m["per_station"][s0]["interval_reliable"] = not flags[s0]
    m["backtest"]["is_live_forecast"] = True
    (d / "metrics.json").write_text(json.dumps(m))
    errs = validate_dir(d)
    assert any("interval_reliable" in e for e in errs) and any("is_live_forecast" in e for e in errs)

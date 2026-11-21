"""report_facts.json must agree with the frozen, tracked evaluation files (runs without local raw data)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    facts = json.loads((ROOT / "results/v2/report_facts.json").read_text())
    return {f["id"]: f for f in facts["facts"]}, json.loads((ROOT / "results/v2/final/metrics.json").read_text())


def test_every_fact_has_value_kind_and_source():
    facts, _ = load()
    assert sorted(facts, key=int) == [str(i) for i in range(1, 16)]
    for f in facts.values():
        assert f["value"] not in (None, "") and f["sources"]
        assert f["kind"] in ("observed", "computed", "documented configuration")
        assert all((ROOT / s["path"]).exists() or not s["tracked_in_git"] for s in f["sources"])


def test_facts_match_frozen_metrics_and_config():
    facts, m = load()
    cfg = json.loads((ROOT / "config/v2_frozen.json").read_text())
    for name, v in facts["9"]["value"].items():
        assert abs(v - m[name]) < 1e-4
    assert facts["4"]["value"]["held_out_test_rows"] == m["test_count"] == 593
    assert facts["4"]["value"]["final_training_rows"] == m["train_count"]
    assert facts["6"]["value"] == cfg["snow_lag_days"] == 1
    assert facts["8"]["value"]["method"] == cfg["interval"]["method"]
    assert facts["7"]["value"][:9] == [c for c in cfg["primary_model"]["features"] if c != "station one-hot"]
    assert not any(c.startswith(("rain_", "temp_", "humidity_")) for c in facts["7"]["value"])
    for k, v in facts["12"]["value"].items():
        if k != "all":
            assert abs(v["coverage"] - m["per_station"][k]["interval_90_coverage"]) < 1e-4
    run = json.loads((ROOT / "results/v2/final/RUN_RECORD.json").read_text())
    assert facts["15"]["value"]["config_sha256"] == run["frozen_hashes"]["config/v2_frozen.json"]

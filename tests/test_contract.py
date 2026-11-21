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

"""Shared fixtures."""
import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def restricted_policy_dict():
    """The release policy as it stood before CDEC authorization: CDEC unresolved, NOAA and NASA cleared."""
    pol = copy.deepcopy(json.loads((ROOT / "config/data_release_policy.json").read_text()))
    for k in ("cdec_storage_sensor15", "cdec_snow_sensor82"):
        pol["sources"][k] = {"status": "unresolved", "publish_raw": False, "publish_derived": False, "public_display": False,
                             "attribution": "CDEC"}
    return pol


@pytest.fixture
def restricted_policy(monkeypatch, tmp_path):
    """Make every policy-reading code path see the restricted policy. Returns the policy file path."""
    import src.v3_service as svc
    pol = restricted_policy_dict()
    monkeypatch.setattr(svc, "load_policy", lambda: pol)
    p = tmp_path / "restricted_policy.json"
    p.write_text(json.dumps(pol))
    return p

"""Prospective scoring protocol: registered hashes, record extraction, realised-change scoring, stage timing,
ledger checks, write-once score files and the preregistered analysis gate. Offline: CDEC is replaced by a fake."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src import prospective as pr

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "public_data" / "watershedshift-public-data-2026-11-22.2" / "issued_forecasts"
V2_ID, V3_ID = "wsf-v2-2026-11-22-20261122T052508Z", "wsf-v3-ORO-20261122T064412Z"


def cdec_csv(station, values):
    lines = ["STATION_ID,DURATION,SENSOR_NUMBER,SENSOR_TYPE,DATE TIME,OBS DATE,VALUE,DATA_FLAG,UNITS"]
    for d, v in values.items():
        stamp = pd.Timestamp(d).strftime("%Y%m%d 0000")
        lines.append(f"{station},D,15,STORAGE,{stamp},{stamp},{'---' if v is None else v},,AF")
    return "\n".join(lines) + "\n"


class FakeResponse:
    def __init__(self, text):
        self.text, self.content, self.url = text, text.encode(), "https://cdec.example/CSVDataServlet"


def v3_record(station="FOL", origin="2026-11-16", target="2026-12-16", cap=977000.0, issue_storage=400000.0):
    m = lambda c: {"available": True, "change_pp": c, "projected_pct": 100 * issue_storage / cap + c,  # noqa: E731
                   "interval": {"lower_change_pp": c - 4, "upper_change_pp": c + 4}}
    return {"schema": "watershedshift-forecast/2.0", "record_type": "prospective_research_forecast", "station_id": station,
            "status": "ok", "origin_date": origin, "target_date": target, "capacity_af": cap,
            "observed": {"storage_af": issue_storage},
            "models": {"A": m(2.0), "B": m(1.0), "B2": {"available": False, "reasons": ["POWER"]}, "C": m(3.0)}}


def write_record(tmp_path, rec, rid, listed=True):
    p = tmp_path / f"{rid}.json"
    p.write_text(json.dumps(rec))
    ledger = tmp_path / "ledger.jsonl"
    if listed:
        ledger.write_text(json.dumps({"forecast_id": rid, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}) + "\n")
    else:
        ledger.write_text("")
    return p, {"official_weekly": ledger}


# ---------------------------------------------------------------- protocol registration

def test_registered_protocol_and_baseline_hashes_match():
    cfg = pr.load_protocol()
    assert cfg["version"] == "1.0" and cfg["scoring"]["stage_wait_days"] == {"provisional": 2, "settled": 30}
    assert cfg["primary"]["min_weekly_origins"] == 52 and [s["comparison"] for s in cfg["secondary"]] == \
        ["C vs A", "B vs A", "B2 vs B"]


def test_edited_protocol_or_baseline_is_refused(tmp_path):
    doc, base = tmp_path / "PROTOCOL.md", tmp_path / "baseline.csv"
    shutil.copy(pr.DOC, doc)
    shutil.copy(pr.BASELINE, base)
    pr.load_protocol(doc=doc, baseline=base)
    doc.write_text(doc.read_text().replace("52 official weekly origins", "40 official weekly origins"))
    with pytest.raises(pr.ProtocolError, match="PROTOCOL.md"):
        pr.load_protocol(doc=doc, baseline=base)
    shutil.copy(pr.DOC, doc)
    base.write_text(base.read_text().replace("SHA,1,", "SHA,1,9"))
    with pytest.raises(pr.ProtocolError, match="baseline"):
        pr.load_protocol(doc=doc, baseline=base)


def test_fixed_baseline_table_is_complete():
    t = pd.read_csv(pr.BASELINE)
    assert len(t) == 6 * 366 and t.baseline_doy_pp.notna().all()
    assert set(t.station_id) == {"SHA", "ORO", "FOL", "NML", "DNP", "EXC"}


@pytest.mark.skipif(not (ROOT / "data/v3/private/weekly_v3.csv").exists(), reason="private weekly table not present")
def test_fixed_baseline_equals_src_models_on_rows_before_2026():
    from src.models import baselines
    w = pd.read_csv(ROOT / "data/v3/private/weekly_v3.csv", parse_dates=["date", "target_date"])
    train, target = w[w.target_date < "2026-01-01"], w[w.target_date >= "2026-01-01"]
    ref = baselines(train, target)["baseline_doy_pp"].to_numpy()
    tab = pr.load_baseline()
    mine = np.array([tab[(s, d.dayofyear)] for s, d in zip(target.station_id, target.date)])
    assert np.abs(ref - mine).max() < 1e-5


# ---------------------------------------------------------------- record extraction and scoring

def test_both_record_schemas_extract():
    v2 = json.loads((RELEASE / f"{V2_ID}.json").read_text())
    e2 = pr.extract(v2)
    assert [e["station_id"] for e in e2] == ["DNP", "EXC", "FOL", "NML", "ORO", "SHA"]
    assert all(set(e["models"]) == {"A"} and e["capacity_af"] > 0 for e in e2)
    assert e2[0]["target_date"] == "2026-12-09" and e2[1]["target_date"] == "2026-12-16"
    e3 = pr.extract(json.loads((RELEASE / f"{V3_ID}.json").read_text()))
    assert len(e3) == 1 and set(e3[0]["models"]) == {"A", "B", "B2", "C"} and e3[0]["target_date"] == "2026-12-16"


def test_score_entry_realised_change_errors_hits_and_baselines():
    cfg = pr.load_protocol()
    e = pr.extract(v3_record())[0]
    storage = pd.Series({pd.Timestamp("2026-11-16"): 410000.0, pd.Timestamp("2026-12-16"): 430000.0})
    s = pr.score_entry(e, storage, pr.load_baseline(), cfg)
    y = 100 * (430000 - 410000) / 977000
    assert s["status"] == "scored" and s["realised_change_pp"] == pytest.approx(y)
    assert s["models"]["A"]["error_pp"] == pytest.approx(2.0 - y) and s["models"]["A"]["interval_hit"]
    assert "B2" not in s["models"] and set(s["models"]) == {"A", "B", "C"}
    assert s["origin_revision_af"] == pytest.approx(10000) and s["origin_revised"]   # > 1% of 977,000 AF
    assert s["baselines"]["persistence"]["abs_error_pp"] == pytest.approx(abs(y))
    b = float(pr.load_baseline()[("FOL", pd.Timestamp("2026-11-16").dayofyear)])
    assert s["baselines"]["doy_seasonal"]["abs_error_pp"] == pytest.approx(abs(b - y))
    assert s["models"]["C"]["level_error_pp"] == pytest.approx((100 * 400000 / 977000 + 3.0) - 100 * 430000 / 977000)


def test_missing_observation_series_break_and_no_forecast_are_never_scored():
    cfg, base = pr.load_protocol(), pr.load_baseline()
    e = pr.extract(v3_record())[0]
    s = pr.score_entry(e, pd.Series({pd.Timestamp("2026-11-16"): 410000.0}), base, cfg)
    assert s["status"] == "unscorable" and "target" in s["reasons"][0] and "models" not in s
    oro = pr.extract(v3_record("ORO", "2024-06-24", "2024-07-24", 3424753.0))[0]       # spans the 2024-07-02 break
    full = pd.Series({pd.Timestamp("2024-06-24"): 3.3e6, pd.Timestamp("2024-07-24"): 3.0e6})
    assert pr.score_entry(oro, full, base, cfg)["status"] == "unscorable"
    nf = dict(v3_record(), status="unavailable", reasons=["CDEC refresh failed"], models={})
    s = pr.score_entry(pr.extract(nf)[0], pd.Series(dtype=float), base, cfg)
    assert s["status"] == "no_forecast" and s["reasons"] == ["CDEC refresh failed"]


def test_stage_waiting_periods():
    cfg = pr.load_protocol()
    at = lambda d: datetime.fromisoformat(d).replace(tzinfo=timezone.utc)  # noqa: E731
    assert not pr.ready("provisional", "2026-12-09", at("2026-12-10T23:00"), cfg)
    assert pr.ready("provisional", "2026-12-09", at("2026-12-11T00:01"), cfg)
    assert not pr.ready("settled", "2026-12-09", at("2027-01-07T12:00"), cfg)
    assert pr.ready("settled", "2026-12-09", at("2027-01-08T00:00"), cfg)


# ---------------------------------------------------------------- end to end, ledgers, write-once

def test_score_record_writes_once_and_refuses_unlisted_or_altered_records(tmp_path):
    rec = v3_record()
    p, ledgers = write_record(tmp_path, rec, "wsf-test-FOL")
    calls = []

    def get(params):
        calls.append(params)
        return FakeResponse(cdec_csv("FOL", {"2026-11-16": 400000, "2026-12-15": 401000, "2026-12-16": 402000}))
    now = datetime(2026, 12, 19, tzinfo=timezone.utc)
    written, skipped = pr.score_record(p, "provisional", now=now, get=get, scores=tmp_path / "scores",
                                       raw=tmp_path / "raw", ledgers=ledgers)
    assert len(written) == 1 and skipped == [] and calls[0]["SensorNums"] == 15
    s = json.loads(written[0].read_text())
    assert s["record_class"] == "official_weekly" and s["stage"] == "provisional" and s["status"] == "scored"
    assert s["cdec_response"]["sha256"] == hashlib.sha256(get({}).content).hexdigest()
    assert s["protocol_version"] == "1.0" and len(s["record_sha256"]) == 64
    written2, skipped2 = pr.score_record(p, "provisional", now=now, get=get, scores=tmp_path / "scores",
                                         raw=tmp_path / "raw", ledgers=ledgers)
    assert written2 == [] and skipped2 == [("FOL", "already scored")]
    with pytest.raises(FileExistsError):
        pr.write_once(written[0], {"x": 1})
    w3, s3 = pr.score_record(p, "settled", now=now, get=get, scores=tmp_path / "scores", raw=tmp_path / "raw",
                             ledgers=ledgers)
    assert w3 == [] and "not ready" in s3[0][1]
    (tmp_path / "v").mkdir()
    visitor, vl = write_record(tmp_path / "v", rec, "visitor", listed=False)
    with pytest.raises(pr.ProtocolError, match="not in an issued or official ledger"):
        pr.score_record(visitor, "provisional", now=now, get=get, scores=tmp_path / "s2", raw=tmp_path / "r2", ledgers=vl)
    p.chmod(0o644)
    p.write_text(json.dumps(dict(rec, capacity_af=1.0)))
    with pytest.raises(pr.ProtocolError, match="does not match its ledger"):
        pr.score_record(p, "provisional", now=now, get=get, scores=tmp_path / "s3", raw=tmp_path / "r3", ledgers=ledgers)


def test_pre_protocol_records_are_listed_and_cannot_be_scored_before_their_dates(tmp_path):
    def no_network(params):
        raise AssertionError("nothing is ready, so CDEC must not be contacted")
    now = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
    for rid, n in ((V2_ID, 6), (V3_ID, 1)):
        written, skipped = pr.score_record(RELEASE / f"{rid}.json", "provisional", now=now, get=no_network,
                                           scores=tmp_path / "scores", raw=tmp_path / "raw")
        assert written == [] and len(skipped) == n and all("not ready" in why for _, why in skipped)
    assert pr.classify(V2_ID, hashlib.sha256((RELEASE / f"{V2_ID}.json").read_bytes()).hexdigest()) == "pre_protocol_issued"


# ---------------------------------------------------------------- analyses

def synthetic_scores(n_origins, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.date_range("2026-12-07", periods=n_origins, freq="7D")):
        for st in ("SHA", "FOL", "ORO"):
            y = rng.normal(0, 4)
            rows.append({"record_id": f"wk{i}", "record_class": "official_weekly", "station_id": st,
                         "origin_date": str(d.date()), "status": "scored", "realised_change_pp": y,
                         "origin_revised": False, "A": y + rng.normal(0, 1), "A_hit": True, "B": y + rng.normal(0, 1.2),
                         "B2": y + rng.normal(0, 1.3), "C": y + rng.normal(0, 0.9), "doy_seasonal": y + rng.normal(0, 3),
                         "persistence": 0.0})
    return pd.DataFrame(rows)


def test_primary_analysis_waits_for_52_official_origins_and_uses_only_the_first_52():
    cfg = pr.load_protocol()
    with pytest.raises(pr.ProtocolError, match="52"):
        pr.primary_analysis(synthetic_scores(51), cfg)
    df = synthetic_scores(60)
    df.loc[0, "record_class"] = "pre_protocol_issued"
    res = pr.primary_analysis(df, cfg)
    assert res["n_origins"] == 52 and res["last_origin"] == str((pd.Timestamp("2026-12-07") + pd.Timedelta(weeks=51)).date())
    assert res["P1"]["verdict"] == "supported" and set(res) >= {"P1", "P2", "S1", "S2", "S3"}
    assert res["P2"]["coverage_90"] == 1.0 and res["P2"]["ci95"] == [1.0, 1.0]
    assert res["P1"]["n_rows"] == 52 * 3 - 1                       # the pre-protocol row is excluded


def test_summary_separates_record_classes():
    df = synthetic_scores(3)
    df.loc[df.index[:3], "record_class"] = "pre_protocol_issued"
    df.loc[df.index[3], "status"] = "no_forecast"
    out = pr.summarize(df)
    assert out["pre_protocol_issued"]["scored"] == 3 and out["official_weekly"]["no_forecast"] == 1
    assert out["official_weekly"]["models"]["A"]["n"] == 5

"""Official weekly issuance (src/official.py): issue window and origin, no older-origin substitution, persistent-store
requirement, one record per week, write-once copies and ledger, missed weeks, and scoring of an official record.
Offline: synthetic CDEC, POWER and GEFS, and a fake GitHub contents API."""
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd
import pytest

from src import official as of
from src import prospective as pr
from src.v3_service import RecordStore
from tests.test_v3 import FakeGitHub, mock_models, service, synthetic_cdec

TUESDAY = datetime(2026, 11, 24, 18, 0, tzinfo=timezone.utc)          # week of Monday 2026-11-23


def fresh_cdec(n):
    return synthetic_cdec(n, end="2026-11-25")                           # FOL storage through Monday 2026-11-23


def github_store(gh=None):
    return RecordStore(github_repo="o/records", github_token="t0ken", session=gh or FakeGitHub(), prefix=of.PREFIX)


def run_issue(tmp_path, clock=lambda: TUESDAY, cdec=fresh_cdec, store=None, dry_run=False):
    svc = service(tmp_path / "svc", TUESDAY, cdec=cdec)
    return of.issue(clock=clock, service=svc, store=store if store is not None else github_store(),
                    ledger=tmp_path / "ledger.jsonl", local_dir=tmp_path / "official", dry_run=dry_run)


def test_issue_window_and_origin_monday():
    assert of.week_origin(TUESDAY) == pd.Timestamp("2026-11-23")
    assert of.week_origin(datetime(2026, 11, 25, 23, 59, tzinfo=timezone.utc)) == pd.Timestamp("2026-11-23")
    for d in (datetime(2026, 11, 23, 23, 0, tzinfo=timezone.utc), datetime(2026, 11, 26, 0, 1, tzinfo=timezone.utc)):
        with pytest.raises(of.IssueError, match="outside the issue window"):
            of.week_origin(d)


def test_dry_run_builds_the_record_and_writes_nothing(tmp_path):
    gh = FakeGitHub()
    rec, line = run_issue(tmp_path, store=github_store(gh), dry_run=True)
    assert line is None and gh.files == {} and not (tmp_path / "ledger.jsonl").exists()
    assert not (tmp_path / "official").exists()
    assert rec["schema"] == of.SCHEMA and rec["record_type"] == "official_weekly_forecast"
    assert rec["forecast_id"] == "wsf-official-2026-11-23" and rec["origin_date"] == "2026-11-23"
    st = {e["station_id"]: e for e in rec["stations"]}
    assert len(st) == 6 and st["FOL"]["status"] == "ok" and st["FOL"]["origin_date"] == "2026-11-23"
    assert st["FOL"]["models"]["A"]["available"]
    assert all(st[s]["status"] == "no_forecast" and st[s]["reasons"] for s in ("SHA", "ORO", "NML", "DNP", "EXC"))
    assert rec["stations_with_forecast"] == 1 and rec["stations_without_forecast"] == 5


def test_an_older_origin_becomes_an_explicit_no_forecast_entry(tmp_path):
    rec, _ = run_issue(tmp_path, cdec=lambda n: synthetic_cdec(n), dry_run=True)   # FOL data ends before Monday
    fol = next(e for e in rec["stations"] if e["station_id"] == "FOL")
    assert fol["status"] == "no_forecast" and "2026-11-23" in fol["reasons"][0] and "older origin" in fol["reasons"][0]


def test_issue_requires_the_persistent_github_store(tmp_path):
    with pytest.raises(of.IssueError, match="persistent GitHub records repository"):
        run_issue(tmp_path, store=RecordStore(root=tmp_path / "local"))
    assert not (tmp_path / "ledger.jsonl").exists()


def test_issue_writes_once_everywhere_and_only_once_per_week(tmp_path):
    gh = FakeGitHub()
    rec, line = run_issue(tmp_path, store=github_store(gh))
    local = tmp_path / "official" / "wsf-official-2026-11-23.json"
    body = local.read_bytes()
    assert line["sha256"] == hashlib.sha256(body).hexdigest() and json.loads(body) == rec
    assert oct(local.stat().st_mode)[-3:] == "444"
    assert list(gh.files) == ["https://api.github.com/repos/o/records/contents/official-records/2026/11/"
                              "wsf-official-2026-11-23.json"]
    assert line["location"].endswith("official-records/2026/11/wsf-official-2026-11-23.json")
    ledger = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert ledger == [line] and line["type"] == "official_weekly_forecast" and line["records_repo_commit"] == "c0ffee"
    with pytest.raises(of.IssueError, match="already has a ledger entry"):
        run_issue(tmp_path, store=github_store(gh))


def test_nothing_is_written_if_the_window_closes_while_building(tmp_path):
    times = iter([datetime(2026, 11, 25, 23, 59, 59, tzinfo=timezone.utc), datetime(2026, 11, 26, 0, 0, 1, tzinfo=timezone.utc)])
    gh = FakeGitHub()
    with pytest.raises(of.IssueError, match="window closed"):
        run_issue(tmp_path, clock=lambda: next(times), store=github_store(gh))
    assert gh.files == {} and not (tmp_path / "ledger.jsonl").exists()


def test_missed_weeks_are_recorded_only_after_the_window(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(of.IssueError, match="has not closed"):
        of.record_missed("2026-11-23", "operator unavailable", clock=lambda: TUESDAY, ledger=ledger)
    with pytest.raises(of.IssueError, match="not a Monday"):
        of.record_missed("2026-11-24", "x", clock=lambda: TUESDAY, ledger=ledger)
    thursday = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)
    line = of.record_missed("2026-11-23", "operator unavailable", clock=lambda: thursday, ledger=ledger)
    assert line == {"type": "missed_week", "origin_date": "2026-11-23", "reason": "operator unavailable",
                    "recorded_utc": "2026-11-26T09:00:00Z"}
    with pytest.raises(of.IssueError, match="already has a ledger entry"):
        of.record_missed("2026-11-23", "again", clock=lambda: thursday, ledger=ledger)
    svc = service(tmp_path / "svc", TUESDAY, cdec=fresh_cdec)
    with pytest.raises(of.IssueError, match="already has a ledger entry"):
        of.issue(clock=lambda: TUESDAY, service=svc, store=github_store(), ledger=ledger, local_dir=tmp_path / "o")


def test_official_records_are_classified_and_scored(tmp_path):
    rec, line = run_issue(tmp_path)
    path = tmp_path / "official" / f"{rec['forecast_id']}.json"
    entries = pr.extract(rec)
    assert [e["station_id"] for e in entries] == ["DNP", "EXC", "FOL", "NML", "ORO", "SHA"]
    fol = next(e for e in entries if e["station_id"] == "FOL")
    assert fol["target_date"] == "2026-12-23" and set(fol["models"]) == {"A"} and fol["capacity_af"] > 0

    class R:
        def __init__(self, text):
            self.text, self.content, self.url = text, text.encode(), "https://cdec.example"

    def get(params):
        return R("STATION_ID,DURATION,SENSOR_NUMBER,SENSOR_TYPE,DATE TIME,OBS DATE,VALUE,DATA_FLAG,UNITS\n"
                 "FOL,D,15,STORAGE,20261123 0000,20261123 0000,450000,,AF\n"
                 "FOL,D,15,STORAGE,20261223 0000,20261223 0000,470000,,AF\n")
    written, skipped = pr.score_record(path, "provisional", now=datetime(2026, 12, 26, tzinfo=timezone.utc), get=get,
                                       scores=tmp_path / "scores", raw=tmp_path / "raw",
                                       ledgers={"official_weekly": tmp_path / "ledger.jsonl"})
    scores = {json.loads(p.read_text())["station_id"]: json.loads(p.read_text()) for p in written}
    assert scores["FOL"]["status"] == "scored" and scores["FOL"]["record_class"] == "official_weekly"
    assert scores["FOL"]["realised_change_pp"] == pytest.approx(100 * 20000 / fol["capacity_af"])
    assert all(scores[s]["status"] == "no_forecast" for s in ("SHA", "ORO", "NML", "DNP", "EXC"))
    assert skipped == []


def test_record_store_accepts_an_explicit_record_id(tmp_path):
    s = RecordStore(root=tmp_path)
    out = s.save({"station_id": "ALL", "issued_utc": "2026-11-24T18:00:00Z"}, record_id="wsf-official-2026-11-23")
    assert out["id"] == "wsf-official-2026-11-23" and (tmp_path / "wsf-official-2026-11-23.json").exists()
    with pytest.raises(FileExistsError):
        s.save({"station_id": "ALL", "issued_utc": "2026-11-24T18:00:00Z"}, record_id="wsf-official-2026-11-23")

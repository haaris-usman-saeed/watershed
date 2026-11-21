"""Regression tests for data-integrity corrections in src/build_dataset.py."""
import numpy as np
import pandas as pd
import pytest

import src.build_dataset as bd

HEADER = "STATION_ID,DURATION,SENSOR_NUMBER,SENSOR_TYPE,DATE TIME,OBS DATE,VALUE,DATA_FLAG,UNITS\n"
CAP = pd.DataFrame([{"station_id": "ORO", "valid_from": "1900-01-01", "valid_to": "2024-07-01", "capacity_af": 3537577},
                    {"station_id": "ORO", "valid_from": "2024-07-02", "valid_to": "2100-01-01", "capacity_af": 3424753},
                    {"station_id": "XXX", "valid_from": "1900-01-01", "valid_to": "2100-01-01", "capacity_af": 1000}])


def write_raw(tmp_path, station, year, rows, review=None):
    (tmp_path / "data" / "raw" / "cdec").mkdir(parents=True, exist_ok=True)
    text = HEADER + "".join(rows)
    (tmp_path / "data" / "raw" / "cdec" / f"{station}_sensor15_D_{year}.csv").write_text(text)
    rv = review if review is not None else pd.DataFrame(columns=["station_id", "date", "decision", "evidence"])
    rv.to_csv(tmp_path / "data" / "spike_review.csv", index=False)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(bd, "ROOT", tmp_path)
    return tmp_path


def row(st, d, obs, v, flag=" ", units="AF", sensor="15"):
    return f"{st},D,{sensor},STORAGE,{d} 0000,{obs} 0000,{v},{flag},{units}\n"


def test_date_comes_from_date_time_not_obs_date(sandbox):
    # DWR-style rows: OBS DATE is D+1. The value must be assigned to DATE TIME (= D).
    write_raw(sandbox, "XXX", 2020, [row("XXX", "20200101", "20200102", 500), row("XXX", "20200102", "20200103", 510)])
    s, _, _ = bd.load_cdec("XXX", CAP, "2020-01-01", "2020-01-02")
    assert s[pd.Timestamp("2020-01-01")] == 500 and s[pd.Timestamp("2020-01-02")] == 510


def test_wrong_units_or_sensor_rejected(sandbox):
    write_raw(sandbox, "XXX", 2020, [row("XXX", "20200101", "20200101", 500, units="FEET")])
    with pytest.raises(ValueError):
        bd.load_cdec("XXX", CAP, "2020-01-01", "2020-01-01")


def test_missing_and_text_values_stay_missing(sandbox):
    write_raw(sandbox, "XXX", 2020, [row("XXX", "20200101", "20200101", 500), row("XXX", "20200102", "20200102", "---"),
                                     row("XXX", "20200104", "20200104", 520)])
    s, _, a = bd.load_cdec("XXX", CAP, "2020-01-01", "2020-01-04")
    assert np.isnan(s[pd.Timestamp("2020-01-02")]) and np.isnan(s[pd.Timestamp("2020-01-03")])
    assert a["non_numeric_or_blank"] == 1


def test_conflicting_duplicates_become_missing_identical_collapse(sandbox):
    write_raw(sandbox, "XXX", 2020, [row("XXX", "20200101", "20200101", 500), row("XXX", "20200101", "20200101", 501),
                                     row("XXX", "20200102", "20200102", 505), row("XXX", "20200102", "20200102", 505)])
    s, _, a = bd.load_cdec("XXX", CAP, "2020-01-01", "2020-01-02")
    assert np.isnan(s[pd.Timestamp("2020-01-01")]) and s[pd.Timestamp("2020-01-02")] == 505
    assert a["conflicting_duplicate_dates"] == 1


def test_spike_review_removes_artifacts_and_unreviewed_keeps_physical(sandbox):
    vals = [500, 500, 900, 500, 500, 500, 100, 500, 500]       # spikes on day 3 and day 7
    rows = [row("XXX", f"202001{d:02d}", f"202001{d:02d}", v) for d, v in enumerate(vals, start=1)]
    review = pd.DataFrame([{"station_id": "XXX", "date": "2020-01-03", "decision": "physical", "evidence": "test"}])
    write_raw(sandbox, "XXX", 2020, rows, review)
    s, _, a = bd.load_cdec("XXX", CAP, "2020-01-01", "2020-01-09")
    assert s[pd.Timestamp("2020-01-03")] == 900            # adjudicated physical -> kept
    assert np.isnan(s[pd.Timestamp("2020-01-07")])          # unreviewed -> missing (conservative)
    assert a["spikes_removed_unreviewed"] == ["2020-01-07"]


def test_oroville_capacity_is_historically_appropriate():
    dates = pd.DatetimeIndex(["2017-02-14", "2024-07-01", "2024-07-02", "2025-01-01"])
    cap = bd.capacity_series("ORO", dates, CAP)
    assert list(cap) == [3537577, 3537577, 3424753, 3424753]


def test_capacity_history_must_cover_all_dates():
    gap = CAP[CAP.station_id == "ORO"].iloc[:1]
    with pytest.raises(ValueError):
        bd.capacity_series("ORO", pd.DatetimeIndex(["2025-01-01"]), gap)


def test_repo_capacity_and_breaks_are_consistent():
    hist = pd.read_csv(bd.ROOT / "data" / "capacity_history.csv")
    breaks = pd.read_csv(bd.ROOT / "data" / "series_breaks.csv")
    for b in breaks.itertuples():
        h = hist[hist.station_id == b.station_id]
        assert b.break_date in set(h.valid_from), "each series break must start a capacity regime"
    stations = pd.read_csv(bd.ROOT / "stations.csv")
    for s in stations.itertuples():   # current capacity in stations.csv == latest regime
        latest = hist[hist.station_id == s.station_id].sort_values("valid_from").iloc[-1]
        assert latest.capacity_af == s.capacity_af

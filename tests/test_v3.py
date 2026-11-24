"""v3 tests: GEFS provenance and issue-time rules, weather lags, service cache, coalescing, rate limits,
failures, corrupted cache, release policy, and preservation of the frozen v2 work.
Network is never used: upstream fetchers are replaced by synthetic functions."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import pytest

from src import v3_gefs as g
from src.v3_basins import STATIONS, basin_boxes, box_mask
from src.v3_service import (APPROVED_STATIONS, ForecastService, PolicyBlocked, RateLimited, RecordStore, RecordStoreError,
                            SourceCache, SourceError, TokenBucket, check_display_allowed, days_remaining, weather_features)

ROOT = Path(__file__).resolve().parents[1]
HAVE_V3 = (ROOT / "data/v3/private/weekly_v3.csv").exists()
HAVE_ART = (ROOT / "models/v2_forecast/model_card.json").exists()


# ---------------------------------------------------------------- 1. frozen v2 preserved

def test_frozen_v2_hashes_unchanged():
    from src.train_v2 import frozen_hashes
    rec = json.loads((ROOT / "results/v2/final/RUN_RECORD.json").read_text())["frozen_hashes"]
    assert frozen_hashes() == rec


def test_issued_forecast_ledger_is_intact_if_record_present():
    led = [json.loads(l) for l in (ROOT / "docs/v3/issued_forecast_ledger.jsonl").read_text().splitlines() if l.strip()]
    assert led[0]["forecast_id"] == "wsf-v2-2026-11-22-20261122T052508Z"
    import hashlib
    for e in led:
        sub = "records" if e["forecast_id"].startswith("wsf-v2") else "records_v3"
        for p in [ROOT / "forecasts" / sub / f"{e['forecast_id']}.json"] + \
                 sorted(ROOT.glob(f"public_data/*/issued_forecasts/{e['forecast_id']}.json")):
            if p.exists():
                assert hashlib.sha256(p.read_bytes()).hexdigest() == e["sha256"], p


# ---------------------------------------------------------------- 2. Model B weather lag

def daily_weather(start="2026-01-01", end="2026-03-31"):
    d = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"rain_mm": np.arange(len(d), dtype=float), "temp_c": 10.0, "humidity_pct": 50.0}, index=d)


def test_weather_windows_end_seven_days_before_origin():
    w = daily_weather()
    t = pd.Timestamp("2026-03-02")
    f = weather_features(w, t)
    end = t - pd.Timedelta(days=7)
    assert f["rain_7d_mm"] == pytest.approx(w.rain_mm[end - pd.Timedelta(days=6):end].sum())
    assert f["rain_30d_mm"] == pytest.approx(w.rain_mm[end - pd.Timedelta(days=29):end].sum())
    w2 = w.copy()
    w2.loc[w2.index > end, "rain_mm"] = 1e6          # observed weather after t-7 must not matter
    assert weather_features(w2, t) == f


def test_weather_window_with_gap_is_missing_not_filled():
    w = daily_weather()
    w.loc[pd.Timestamp("2026-02-20"), "rain_mm"] = np.nan
    assert np.isnan(weather_features(w, pd.Timestamp("2026-03-02"))["rain_7d_mm"])


@pytest.mark.skipif(not HAVE_V3, reason="private v3 table not present")
def test_weather_features_match_historical_builder():
    from src.v3_dataset import WEATHER_COLS
    rows = pd.read_csv(ROOT / "data/v3/private/weekly_v3.csv", parse_dates=["date"])
    pw = pd.read_csv(ROOT / "data/v3/public/power_daily.csv", parse_dates=["date"])
    for st in STATIONS:
        d = pw[(pw.station_id == st) & (pw.source == "dam_point")].set_index("date")
        for r in rows[rows.station_id == st].sample(20, random_state=1).itertuples():
            f = weather_features(d, r.date)
            for c in WEATHER_COLS:
                assert f[c] == pytest.approx(getattr(r, c), abs=1e-6)


# ---------------------------------------------------------------- 3, 4, 10. GEFS issue-time rules

def test_gefs_source_periods():
    assert g.source_for(date(2019, 12, 30)) == "reforecast"
    assert g.source_for(date(2020, 5, 4)) is None                 # GEFSv11 era: no product used
    assert g.source_for(date(2020, 9, 28)) == "operational"
    assert all(g.source_for(d) for d in g.history_dates(date(2026, 11, 22)))
    assert all(d.weekday() == 0 for d in g.history_dates(date(2026, 11, 22)))


def test_idx_parsing_selects_only_six_hour_accumulations():
    idx = "\n".join(["1:0:d=2010010400:APCP:surface:0-3 hour acc fcst:ENS=low-res ctl",
                     "2:430:d=2010010400:APCP:surface:0-6 hour acc fcst:ENS=low-res ctl",
                     "3:900:d=2010010400:APCP:surface:6-9 hour acc fcst:ENS=low-res ctl",
                     "4:1300:d=2010010400:APCP:surface:6-12 hour acc fcst:ENS=low-res ctl",
                     "5:1700:d=2010010400:TMP:2 m above ground:12 hour fcst:ENS=low-res ctl"])
    msgs = g.parse_idx(idx)
    six = [(a, b) for _, _, a, b, _ in msgs if b - a == 6]
    assert six == [(0, 6), (6, 12)]
    assert msgs[1][1] == 899                                     # byte range ends before the next message


def test_incomplete_cycle_produces_no_feature(monkeypatch):
    class R:
        status_code, text, ok = 200, "1:0:d=x:APCP:surface:0-6 hour acc fcst:ENS=low-res ctl", True
    monkeypatch.setattr(g, "get", lambda *a, **k: R())
    plan, why = g.message_plan(date(2010, 1, 4))
    assert plan is None and "incomplete" in why
    rows = g.extract_cycle(date(2010, 1, 4))
    assert all(r["complete"] is False and "gefs_p_d1_7_mm" not in r for r in rows)


def test_gefs_lead_split_has_no_days_after_16():
    assert g.LEADS[0] == 6 and g.LEADS[-1] == 384 and len(g.LEADS) == 64
    assert sum(1 for h in g.LEADS if h <= g.D1_7) == 28


GEFS_FILES = [ROOT / "data/v3/gefs_basin_precip.csv"] + sorted(ROOT.glob("public_data/*/gefs_basin_precip.csv"))


@pytest.mark.parametrize("path", [p for p in GEFS_FILES if p.exists()] or [pytest.param(None, marks=pytest.mark.skip("no GEFS file"))])
def test_gefs_history_uses_forecast_cycles_only_and_operational_published_before_issue(path):
    d = pd.read_csv(path, parse_dates=["init_date"])
    ok = d[d.complete.astype(bool)]
    assert set(ok.source) <= {"reforecast", "operational"}
    assert (ok.init_date.dt.dayofweek == 0).all()                 # cycle date == the Monday origin
    assert (ok.n_messages == 64).all()
    from src.v3_dataset import EARLIEST_ISSUE_AFTER_INIT, usable_gefs
    used = usable_gefs(d)
    op = used[used.source == "operational"]
    pub = pd.to_datetime(op.published_utc_max, utc=True)
    assert (pub <= op.init_date.dt.tz_localize("UTC") + EARLIEST_ISSUE_AFTER_INIT).all()
    late = ok[(ok.source == "operational") & (pd.to_datetime(ok.published_utc_max, utc=True) >
                                              ok.init_date.dt.tz_localize("UTC") + EARLIEST_ISSUE_AFTER_INIT)]
    assert set(late.init_date).isdisjoint(set(op.init_date))     # archive backfills are never used
    assert (ok[ok.source == "reforecast"].note.str.contains("not issued in real time")).all()


@pytest.mark.skipif(not HAVE_V3, reason="private v3 table not present")
def test_model_c_rows_use_cycle_of_their_own_origin():
    rows = pd.read_csv(ROOT / "data/v3/private/weekly_v3.csv", parse_dates=["date"])
    gefs = pd.read_csv(ROOT / "data/v3/gefs_basin_precip.csv", parse_dates=["init_date"])
    m = rows.dropna(subset=["gefs_p_d1_7_mm"]).merge(gefs, left_on=["station_id", "date"],
                                                     right_on=["station_id", "init_date"], suffixes=("", "_g"))
    assert len(m) == rows.gefs_p_d1_7_mm.notna().sum()
    assert np.allclose(m.gefs_p_d1_7_mm, m.gefs_p_d1_7_mm_g)
    gap = rows[(rows.date >= "2020-01-01") & (rows.date < "2020-09-24")]
    assert gap.gefs_p_d1_7_mm.isna().all()                       # never filled from another product


def test_box_mask_selects_cells_inside_box():
    boxes = basin_boxes()
    lats = np.arange(90, -90.01, -0.25)
    lons = np.arange(0, 360, 0.25)
    m = box_mask(lats, lons, boxes["FOL"])
    la, lo = np.where(m)
    assert m.sum() > 0
    assert lats[la].min() >= boxes["FOL"]["lat_min"] and (lons[lo] - 360).max() <= boxes["FOL"]["lon_max"]


# ---------------------------------------------------------------- 11-16. cache, coalescing, limits, failures, corruption

def test_cache_hit_and_new_bucket_invalidates(tmp_path):
    c = SourceCache(tmp_path)
    calls = []
    f = lambda: (calls.append(1) or ({"v": len(calls)}, {}))  # noqa: E731
    a, hit_a = c.get_or_fetch("cdec_20261122T00", f)
    b, hit_b = c.get_or_fetch("cdec_20261122T00", f)
    d, hit_d = c.get_or_fetch("cdec_20261122T06", f)          # next 6-hour bucket: new observations eligible
    assert (hit_a, hit_b, hit_d) == (False, True, False) and len(calls) == 2


def test_concurrent_requests_coalesce_into_one_fetch(tmp_path):
    c = SourceCache(tmp_path)
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.3)
        return {"v": 1}, {}

    with ThreadPoolExecutor(20) as ex:
        res = list(ex.map(lambda _: c.get_or_fetch("k", slow)[0]["payload"], range(20)))
    assert len(calls) == 1 and all(r == {"v": 1} for r in res)


def test_rate_limit_blocks_excess_upstream_fetches(tmp_path):
    now = [0.0]
    c = SourceCache(tmp_path, bucket=TokenBucket(2, clock=lambda: now[0]))
    c.get_or_fetch("a", lambda: ({}, {}))
    c.get_or_fetch("b", lambda: ({}, {}))
    with pytest.raises(RateLimited):
        c.get_or_fetch("c", lambda: ({}, {}))
    now[0] = 60.0
    c.get_or_fetch("c", lambda: ({}, {}))                     # budget refills over time


def test_corrupted_cache_entry_is_discarded_and_refetched(tmp_path):
    c = SourceCache(tmp_path)
    c.get_or_fetch("k", lambda: ({"v": 1}, {}))
    p = tmp_path / "k.json"
    obj = json.loads(p.read_text())
    obj["payload"]["v"] = 999                                  # tamper without updating the hash
    p.write_text(json.dumps(obj))
    e, hit = c.get_or_fetch("k", lambda: ({"v": 2}, {}))
    assert hit is False and e["payload"] == {"v": 2}
    p.write_text("{not json")
    assert c.read("k") is None and not p.exists()


def synthetic_cdec(now, station="FOL", end=None, missing=()):
    end = pd.Timestamp(end or now.date()) - pd.Timedelta(days=2)
    d = pd.date_range(end - pd.Timedelta(days=200), end, freq="D")
    storage = {str(x.date()): (None if str(x.date()) in missing else 400_000.0 + 500 * i) for i, x in enumerate(d)}
    snow = {str(x.date()): 2.0 + 0.01 * i for i, x in enumerate(d)}
    cap = pd.read_csv(ROOT / "data/capacity_history.csv").to_dict(orient="list")
    return ({"stations": {station: {"name": "Folsom Lake", "storage": storage}}, "snow_index": {station: snow},
             "capacity_history": cap}, {"provider": "synthetic"})


def synthetic_power(now):
    d = pd.date_range(now.date() - timedelta(days=60), now.date(), freq="D")
    w = {"rain_mm": {str(x.date()): 1.0 for x in d}, "temp_c": {str(x.date()): 9.0 for x in d},
         "humidity_pct": {str(x.date()): 70.0 for x in d}}
    return {"series": {"dam_point": {s: w for s in APPROVED_STATIONS}, "upstream_box_mean": {s: w for s in APPROVED_STATIONS}}}, {}


def synthetic_gefs(cycle, published="2026-11-16T06:00:00Z", complete=True):
    return ({"cycle": cycle.isoformat(), "complete": complete, "source": "operational", "published_utc_max": published,
             "values": {s: {"gefs_p_d1_7_mm": 20.0, "gefs_p_d8_16_mm": 30.0} for s in APPROVED_STATIONS}}, {})


class MockModel:
    """Controlled stand-in for a fitted model: change = 0.1 * prev30_pct + 0.5; records what it was given."""
    def __init__(self):
        self.seen = []

    def predict(self, x):
        self.seen.append(x.copy())
        return (0.1 * x["prev30_pct"] + 0.5).to_numpy()


def mock_models(model=None):
    """Model A as a mock with the frozen feature order; B, B2, C reported as not built (as on a fresh deployment)."""
    from src.forecast_schema import FEATURE_ORDER
    a = {"model": model or MockModel(), "features": FEATURE_ORDER, "cutoff": "2024-01-01",
         "offsets": {s: [-4.0, 4.0] for s in ("DJF", "MAM", "JJA", "SON")},
         "clim": {s: [1.0] * 366 for s in APPROVED_STATIONS}, "coverage": {s: 0.9 for s in APPROVED_STATIONS},
         "sha256": "mock", "version": "mock"}
    return {"A": a, **{k: {"error": f"v3 artifact for Model {k} not built (python -m src.v3_experiment --final)"}
                       for k in ("B", "B2", "C")}}


def service(tmp_path, now, models="mock", **fetchers):
    """models="mock": controlled mock models (no private artifacts needed); None: load real artifacts; dict: as given."""
    base = {"cdec": lambda n: synthetic_cdec(n), "power": synthetic_power, "gefs": synthetic_gefs}
    base.update(fetchers)
    svc = ForecastService(cache_dir=tmp_path / "cache", art_v3=tmp_path / "no_v3", fetchers=base, clock=lambda: now)
    if models is not None:
        svc._models = mock_models() if models == "mock" else models
    return svc


NOW = datetime(2026, 11, 22, 18, 0, tzinfo=timezone.utc)


def test_service_forecast_dates_days_remaining_and_unbuilt_models(tmp_path):
    r = service(tmp_path, NOW).forecast("FOL")
    assert r["status"] == "ok" and r["origin_date"] == "2026-11-16" and r["target_date"] == "2026-12-16"
    assert r["days_remaining"] == pytest.approx(24.25)          # from 18:00 UTC on 11-22 to the end of 12-16
    assert r["models"]["A"]["available"] and r["storage_last_obs_date"] == "2026-11-20"
    a = r["models"]["A"]
    assert a["interval"]["lower_change_pp"] < a["change_pp"] < a["interval"]["upper_change_pp"]
    assert a["information_cutoff"] == {"storage": "2026-11-16", "snow": "2026-11-15"}
    for k in ("B", "B2", "C"):
        assert not r["models"][k]["available"] and "not built" in r["models"][k]["reasons"][0]
    assert all(h["used_by_forecast"] == (h["date"] <= "2026-11-16") for h in r["recent_history"])


def test_missing_cdec_records_block_forecast(tmp_path):
    miss = {"2026-11-16", "2026-11-09"}
    r = service(tmp_path, NOW, cdec=lambda n: synthetic_cdec(n, missing=miss)).forecast("FOL")
    assert r["status"] == "unavailable" and not r["models"]


def test_missing_snowpack_blocks_forecast(tmp_path):
    def no_snow(n):
        p, m = synthetic_cdec(n)
        p["snow_index"] = {}
        return p, m
    r = service(tmp_path, NOW, cdec=no_snow).forecast("FOL")
    assert r["status"] == "unavailable"


def test_stale_observations_flagged(tmp_path):
    r = service(tmp_path, NOW, cdec=lambda n: synthetic_cdec(n, end="2026-11-12")).forecast("FOL")
    assert r["status"] == "ok" and r["origin_date"] == "2026-11-09"
    assert any("Stale origin" in w for w in r["warnings"]) and any("more than 5 days" in w for w in r["warnings"])


def test_external_api_failure_gives_no_forecast(tmp_path):
    calls = []

    def boom(n):
        calls.append(n)
        raise SourceError("CDEC timed out")
    mock = MockModel()
    r = service(tmp_path, NOW, models=mock_models(mock), cdec=boom).forecast("FOL")
    assert calls == [NOW]                                      # the CDEC fetch was really attempted
    assert r["status"] == "unavailable" and "CDEC refresh failed: CDEC timed out" in r["reasons"][0]
    assert r["models"] == {} and mock.seen == []               # no model was run on missing data


def test_power_and_gefs_failures_disable_only_dependent_models(tmp_path):
    def no_power(n):
        raise SourceError("POWER timed out")

    def no_gefs(c):
        raise SourceError("S3 503")
    models = mock_models()
    for k in ("B", "C"):
        models[k] = dict(models["A"], features=models["A"]["features"][:9] +
                         (["rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"] if k == "B" else
                          ["gefs_p_d1_7_mm", "gefs_p_d8_16_mm"]) + models["A"]["features"][9:])
    r = service(tmp_path, NOW, models=models, power=no_power, gefs=no_gefs).forecast("FOL")
    assert r["status"] == "ok" and r["models"]["A"]["available"]
    assert not r["models"]["B"]["available"] and "NASA POWER" in r["models"]["B"]["reasons"][0]
    assert not r["models"]["C"]["available"] and "GEFS" in r["models"]["C"]["reasons"][0]
    assert r["source_status"]["power"]["ok"] is False and r["source_status"]["gefs"]["ok"] is False


def test_gefs_cycle_published_after_issue_is_rejected(tmp_path):
    models = mock_models()
    models["C"] = dict(models["A"], features=models["A"]["features"][:9] + ["gefs_p_d1_7_mm", "gefs_p_d8_16_mm"] +
                       models["A"]["features"][9:])
    late = lambda c: synthetic_gefs(c, published="2026-11-23T00:00:00Z")  # noqa: E731  after NOW
    r = service(tmp_path, NOW, models=models, gefs=late).forecast("FOL")
    assert not r["models"]["C"]["available"] and "not published before the issue time" in r["models"]["C"]["reasons"][0]
    incomplete = lambda c: synthetic_gefs(c, complete=False)  # noqa: E731
    r = service(tmp_path / "b", NOW, models=models, gefs=incomplete).forecast("FOL")
    assert not r["models"]["C"]["available"] and "incomplete" in r["models"]["C"]["reasons"][0]


def test_unknown_station_rejected(tmp_path):
    with pytest.raises(ValueError):
        service(tmp_path, NOW).forecast("../../etc/passwd")


def test_forced_refresh_has_cooldown(tmp_path):
    calls = []
    svc = service(tmp_path, NOW, cdec=lambda n: (calls.append(1), synthetic_cdec(n))[1])
    svc.cdec(NOW)
    svc.cdec(NOW, force=True)
    svc.cdec(NOW, force=True)                                   # within cooldown: served from cache
    assert len(calls) == 2


# ---------------------------------------------------------------- 19. days remaining

def test_days_remaining_from_actual_issue_time():
    assert days_remaining("2026-12-16", datetime(2026, 11, 22, 0, 0, tzinfo=timezone.utc)) == 25.0
    assert days_remaining("2026-12-16", datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)) == 24.5


# ---------------------------------------------------------------- 20. release policy

def test_public_deployment_blocks_unresolved_cdec_display():
    from tests.conftest import restricted_policy_dict
    policy = restricted_policy_dict()
    with pytest.raises(PolicyBlocked):
        check_display_allowed(["cdec_storage_sensor15"], policy, mode="public")
    check_display_allowed(["cdec_storage_sensor15"], policy, mode="private")
    check_display_allowed(["noaa_gefs_operational", "nasa_power"], policy, mode="public")


def test_public_service_refuses_before_fetching(tmp_path, monkeypatch, restricted_policy):
    monkeypatch.setenv("WATERSHED_DEPLOYMENT", "public")
    calls = []
    svc = service(tmp_path, NOW, cdec=lambda n: (calls.append(1), synthetic_cdec(n))[1])
    with pytest.raises(PolicyBlocked):
        svc.forecast("FOL")
    assert calls == []


def test_release_pipeline_never_packages_unresolved_sources(tmp_path, restricted_policy):
    from src.v3_release import build_release, check_releases
    man = build_release(tmp_path, version="test", policy_path=restricted_policy)
    for f in man["files"]:
        assert not set(f["sources"]) & {"cdec_storage_sensor15", "cdec_snow_sensor82"}, f["path"]
    assert any("cdec_daily_storage" in e for e in man["excluded_pending_clearance"])
    assert check_releases(tmp_path, policy_path=restricted_policy) == []
    with pytest.raises(FileExistsError):
        build_release(tmp_path, version="test", policy_path=restricted_policy)   # never overwritten


def test_record_store_is_write_once(tmp_path):
    s = RecordStore(root=tmp_path)
    rec = {"station_id": "FOL", "issued_utc": "2026-11-22T18:00:00Z"}
    s.save(rec)
    with pytest.raises(FileExistsError):
        s.save(rec)

    fake = FakeS3()
    s3 = RecordStore(client=fake, bucket="b")
    out = s3.save(rec)
    assert out["location"].startswith("s3://b/") and s3.persistent
    assert fake.puts[0]["IfNoneMatch"] == "*"                  # conditional write: S3 itself refuses an overwrite
    with pytest.raises(FileExistsError):
        s3.save(rec)
    assert len(fake.puts) == 1


class S3ClientError(Exception):
    """Shaped like botocore.exceptions.ClientError: the code is in .response["Error"]["Code"]."""
    def __init__(self, code, status):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3:
    def __init__(self, head_error=None, put_error=None):
        self.objs, self.puts, self.head_error, self.put_error = {}, [], head_error, put_error

    def head_object(self, Bucket, Key):
        if self.head_error:
            raise self.head_error
        if Key not in self.objs:
            raise S3ClientError("404", 404)

    def put_object(self, Bucket, Key, Body, ContentType, IfNoneMatch=None):
        if self.put_error:
            raise self.put_error
        if IfNoneMatch == "*" and Key in self.objs:
            raise S3ClientError("PreconditionFailed", 412)
        self.puts.append({"Key": Key, "IfNoneMatch": IfNoneMatch})
        self.objs[Key] = Body


def test_s3_record_store_never_writes_after_an_ambiguous_check():
    rec = {"station_id": "FOL", "issued_utc": "2026-11-22T18:00:00Z"}
    denied = FakeS3(head_error=S3ClientError("403", 403))      # e.g. missing permission: not proof of absence
    with pytest.raises(S3ClientError):
        RecordStore(client=denied, bucket="b").save(rec)
    assert denied.puts == []
    raced = FakeS3(put_error=S3ClientError("PreconditionFailed", 412))   # another writer won the race
    with pytest.raises(FileExistsError):
        RecordStore(client=raced, bucket="b").save(rec)


def test_record_ids_do_not_collide_within_one_second(tmp_path):
    s = RecordStore(root=tmp_path)
    a = s.save({"station_id": "FOL", "issued_utc": "2026-11-22T18:00:00Z", "models": {"A": 1}})
    b = s.save({"station_id": "FOL", "issued_utc": "2026-11-22T18:00:00Z", "models": {"A": 2}})
    assert a["id"] != b["id"] and len(list(tmp_path.glob("*.json"))) == 2
    assert a["id"].startswith("wsf-v3-FOL-20261122T180000Z-")
    assert not s.persistent and "lost when the app restarts" in s.describe()


@pytest.mark.skipif(not (ROOT / "models/v3/model_cards.json").exists() or not HAVE_V3, reason="v3 artifacts not built")
def test_v3_artifacts_reproduce_recorded_test_predictions():
    import hashlib
    import joblib
    from src.models import add_snow_anomaly, design
    cards = json.loads((ROOT / "models/v3/model_cards.json").read_text())["models"]
    pred = pd.read_csv(ROOT / "data/v3/private/test_predictions_v3.csv", parse_dates=["date"])
    rows = pd.read_csv(ROOT / "data/v3/private/weekly_v3.csv", parse_dates=["date", "target_date"])
    test = rows.merge(pred[["station_id", "date"]], on=["station_id", "date"])
    for k, c in cards.items():
        p = ROOT / f"models/v3/model_{k}.joblib"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c["artifact_sha256"]
        clim = {s: pd.Series(v, index=range(1, 367)) for s, v in c["snow_climatology_in"].items()}
        t = add_snow_anomaly(test, clim)
        got = joblib.load(p).predict(design(t, [f for f in c["feature_order"] if not f.startswith("station_")], STATIONS))
        ref = test[["station_id", "date"]].merge(pred, on=["station_id", "date"])[f"pred_{k}"].to_numpy()
        assert np.allclose(got, ref, atol=1e-9), k


@pytest.mark.skipif(not (ROOT / "results/v3/final/RUN_RECORD.json").exists(), reason="v3 final not run")
def test_v3_final_ran_once_against_the_committed_plan():
    rec = json.loads((ROOT / "results/v3/final/RUN_RECORD.json").read_text())
    plan = json.loads((ROOT / "config/v3_plan.json").read_text())
    assert rec["code_sha256"] == plan["code_sha256"]
    m = json.loads((ROOT / "results/v3/final/metrics.json").read_text())
    for h, v in m["hypotheses"].items():
        sup = all(b["ci95_high"] < 0 for b in v["bootstrap"].values())
        assert v["verdict"] == ("supported" if sup else "not established") == rec["verdicts"][h]


def test_missing_model_artifacts_refuse_before_any_fetch(tmp_path, monkeypatch):
    """Real loading path (ForecastService._load_models) with no artifacts installed, as on a clean deployment."""
    import src.forecast_model as fm

    def missing():
        raise FileNotFoundError(f"model artifact missing in {tmp_path / 'models'}: run python -m src.forecast_model --build")
    monkeypatch.setattr(fm, "load_artifact", missing)
    calls = []
    counting = {k: (lambda *a, k=k: calls.append(k)) for k in ("cdec", "power", "gefs")}
    svc = service(tmp_path, NOW, models=None, **counting)
    r = svc.forecast("FOL")
    assert r["status"] == "unavailable" and "artifacts are not installed" in r["reasons"][0]
    assert calls == [] and r["models"] == {}
    loaded = svc.models()
    assert all("error" in loaded[k] for k in ("A", "B", "B2", "C"))


def test_tampered_v3_artifact_is_not_loaded(tmp_path, monkeypatch):
    import src.forecast_model as fm
    monkeypatch.setattr(fm, "load_artifact", lambda: (_ for _ in ()).throw(FileNotFoundError("absent")))
    art = tmp_path / "v3"
    art.mkdir()
    (art / "model_B.joblib").write_bytes(b"not the model")
    (art / "model_cards.json").write_text(json.dumps({"models": {"B": {"artifact_sha256": "0" * 64}}}))
    svc = ForecastService(cache_dir=tmp_path / "c", art_v3=art)
    assert "does not match its card" in svc.models()["B"]["error"]


@pytest.mark.skipif(not HAVE_ART, reason="Model A artifact not present")
def test_service_with_real_model_a_artifact(tmp_path):
    r = service(tmp_path, NOW, models=None).forecast("FOL")
    a = r["models"]["A"]
    assert r["status"] == "ok" and a["available"] and a["version"] == "v2-frozen"
    assert a["training_cutoff_exclusive"] == "2024-01-01" and len(a["artifact_sha256"]) == 64


# ---------------------------------------------------------------- artifact provisioning

def _fake_artifacts(root):
    from src.v3_artifacts import FILES
    for f in FILES:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"content of {f}".encode())


def test_artifact_package_round_trip_is_hash_pinned(tmp_path):
    from src.v3_artifacts import FILES, install, pack
    src_root, dest = tmp_path / "src", tmp_path / "dest"
    _fake_artifacts(src_root)
    out, h = pack(tmp_path / "dist", root=src_root, selfcheck=False)
    assert pack(tmp_path / "dist2", root=src_root, selfcheck=False)[1] == h      # byte-reproducible
    data = out.read_bytes()
    man = install(data, h, dest=dest, mode="private")
    assert set(man) == set(FILES) and all((dest / f).read_bytes() == (src_root / f).read_bytes() for f in FILES)
    with pytest.raises(ValueError, match="does not match the pinned"):
        install(data, "0" * 64, dest=tmp_path / "d2", mode="private")
    assert not (tmp_path / "d2").exists()


def test_artifact_install_rejects_unexpected_members_and_public_without_rights(tmp_path, restricted_policy):
    import hashlib
    import io
    import tarfile
    from src.v3_artifacts import install
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        ti = tarfile.TarInfo("../../etc/evil")
        ti.size = 1
        tar.addfile(ti, io.BytesIO(b"x"))
        ti = tarfile.TarInfo("data/processed/daily_joined.csv")     # private data never accepted
        ti.size = 1
        tar.addfile(ti, io.BytesIO(b"x"))
    data = buf.getvalue()
    with pytest.raises(ValueError, match="unexpected archive member"):
        install(data, hashlib.sha256(data).hexdigest(), dest=tmp_path, mode="private")
    with pytest.raises(PermissionError):
        install(data, hashlib.sha256(data).hexdigest(), dest=tmp_path, mode="public")


def test_ensure_artifacts_without_configuration_does_nothing(tmp_path, monkeypatch):
    from src.v3_artifacts import ensure_artifacts
    monkeypatch.delenv("WATERSHED_ARTIFACT_URL", raising=False)
    assert ensure_artifacts(root=tmp_path).startswith("missing") and not any(tmp_path.iterdir())



def test_current_policy_permits_public_cdec_and_every_release_complies():
    from src.v3_release import check_releases
    from src.v3_service import cdec_display_block_reason
    assert cdec_display_block_reason(mode="public") is None
    assert check_releases() == []


def test_release_check_catches_files_the_policy_does_not_allow(tmp_path):
    from tests.conftest import restricted_policy_dict
    import shutil
    from src.v3_release import check_releases
    src = sorted((ROOT / "public_data").glob("watershedshift-public-data-*.2"))
    if not src:
        pytest.skip("CDEC-inclusive release not present")
    shutil.copytree(src[-1], tmp_path / src[-1].name)
    p = tmp_path / "pol.json"
    p.write_text(json.dumps(restricted_policy_dict()))
    probs = check_releases(tmp_path, policy_path=p)
    assert any("not cleared" in x for x in probs)
    (tmp_path / src[-1].name / "power_daily.csv").write_text("tampered")
    assert any("checksum mismatch" in x for x in check_releases(tmp_path))


@pytest.mark.skipif(not (ROOT / "models/v3/model_cards.json").exists(), reason="v3 artifacts not built")
def test_selfcheck_refuses_models_that_do_not_reproduce(tmp_path):
    import shutil
    from src.v3_artifacts import INSTALLED_MANIFEST, install, pack, verify_selfcheck
    out, h = pack(tmp_path / "dist")
    dest = tmp_path / "site"
    install(out.read_bytes(), h, dest=dest, mode="private")
    assert verify_selfcheck(dest) == {}
    m = json.loads((dest / INSTALLED_MANIFEST).read_text())
    m["selfcheck"]["models/v3/model_C.joblib"][0] += 1.0              # simulate a library-version drift
    (dest / INSTALLED_MANIFEST).write_text(json.dumps(m))
    bad = verify_selfcheck(dest)
    assert list(bad) == ["models/v3/model_C.joblib"] and "differ" in bad["models/v3/model_C.joblib"]


def test_truncated_archive_is_refused_not_crashing(tmp_path, monkeypatch):
    import hashlib
    from src import v3_artifacts as va
    src_root = tmp_path / "src"
    _fake_artifacts(src_root)
    out, _ = va.pack(tmp_path / "dist", root=src_root, selfcheck=False)
    cut = out.read_bytes()[:200]
    with pytest.raises(ValueError, match="cannot be read"):
        va.install(cut, hashlib.sha256(cut).hexdigest(), dest=tmp_path / "d", mode="private")

    class R:
        content = cut

        def raise_for_status(self):
            pass
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    monkeypatch.setenv("WATERSHED_ARTIFACT_URL", "https://example.invalid/bundle.tar.gz")
    monkeypatch.setenv("WATERSHED_ARTIFACT_SHA256", hashlib.sha256(cut).hexdigest())
    status = va.ensure_artifacts(root=tmp_path / "site")
    assert status.startswith("install refused or failed") and not (tmp_path / "site/models").exists()



class FakeGHResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeGitHub:
    """GitHub contents API stand-in: a PUT without a blob SHA creates the file, or answers 422 if it exists."""
    MESSAGES = {401: "Bad credentials", 404: "Not Found", 409: "is at abc but expected def", 422: "sha wasn't supplied"}

    def __init__(self, put_codes=(), net_error=None):
        self.files, self.puts, self.put_codes, self.net_error = {}, [], list(put_codes), net_error

    def put(self, url, headers, json, timeout):
        if self.net_error:
            raise self.net_error
        self.puts.append({"url": url, "headers": headers, "json": json})
        if self.put_codes:
            code = self.put_codes.pop(0)
            return FakeGHResponse(code, {"message": self.MESSAGES.get(code, "error")})
        if url in self.files:
            return FakeGHResponse(422, {"message": self.MESSAGES[422]})
        self.files[url] = json["content"]
        path = url.split("/contents/")[1]
        return FakeGHResponse(201, {"content": {"html_url": f"https://github.com/o/records/blob/main/{path}"},
                                    "commit": {"sha": "c0ffee"}})

    def get(self, url, headers, timeout, params=None):
        return FakeGHResponse(200 if url in self.files else 404, {})


def test_github_record_store_creates_each_record_once(tmp_path):
    import base64
    rec = {"station_id": "FOL", "issued_utc": "2026-11-22T18:00:00Z", "models": {"A": 1}}
    gh = FakeGitHub()
    s = RecordStore(root=tmp_path / "local", github_repo="o/records", github_token="t0ken-secret", session=gh)
    out = s.save(rec)
    assert s.backend == "github" and s.persistent and "GitHub" in s.describe()
    assert out["location"].startswith("https://github.com/o/records/blob/main/forecast-records/2026/11/wsf-v3-FOL-")
    assert out["commit"] == "c0ffee" and out["id"] in out["location"]
    sent = gh.puts[0]
    assert "sha" not in sent["json"]                           # create-only: GitHub refuses if the path exists
    assert sent["url"].startswith("https://api.github.com/repos/o/records/contents/forecast-records/2026/11/")
    assert json.loads(base64.b64decode(sent["json"]["content"])) == rec
    assert sent["headers"]["Authorization"] == "Bearer t0ken-secret"
    with pytest.raises(FileExistsError):
        s.save(rec)
    assert len(gh.files) == 1 and not (tmp_path / "local").exists()


def test_github_record_store_retries_conflicts_and_reports_safe_errors():
    import requests
    rec = {"station_id": "FOL", "issued_utc": "2026-11-22T18:00:00Z"}
    waits = []
    gh = FakeGitHub(put_codes=[409, 409])
    out = RecordStore(github_repo="o/records", github_token="t0ken-secret", session=gh, sleep=waits.append).save(rec)
    assert out["commit"] == "c0ffee" and waits == [1, 2] and len(gh.puts) == 3
    for store, expect in [
        (RecordStore(github_repo="o/records", github_token="t0ken-secret", session=FakeGitHub(put_codes=[401])),
         "HTTP 401: Bad credentials"),
        (RecordStore(github_repo="o/records", github_token="t0ken-secret", session=FakeGitHub(put_codes=[422])),
         "HTTP 422"),                                          # 422 but the file is absent: an error, not "exists"
        (RecordStore(github_repo="o/records", github_token="t0ken-secret",
                     session=FakeGitHub(net_error=requests.ConnectionError("t0ken-secret in a URL"))),
         "could not be reached (ConnectionError)"),
        (RecordStore(github_repo="o/records", github_token=None, session=FakeGitHub()), "WATERSHED_RECORD_GITHUB_TOKEN"),
    ]:
        with pytest.raises(RecordStoreError) as exc:
            store.save(rec)
        assert expect in str(exc.value) and "t0ken-secret" not in str(exc.value)


# ---------------------------------------------------------------- reliability: model installation, NASA POWER

def test_archive_without_manifest_is_refused_cleanly(tmp_path):
    import hashlib
    import io
    import tarfile
    from src.v3_artifacts import FILES, install
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            data = f"content of {f}".encode()
            ti = tarfile.TarInfo(f)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    blob = buf.getvalue()
    with pytest.raises(ValueError, match="no MANIFEST.json"):
        install(blob, hashlib.sha256(blob).hexdigest(), dest=tmp_path / "d", mode="private")


class FakeResponse:
    def __init__(self, status, content=b"bundle"):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_bundle_download_retries_only_transient_failures(monkeypatch):
    import requests
    from src import v3_artifacts as va
    replies, waits = [FakeResponse(503), FakeResponse(502), FakeResponse(200, b"ok")], []
    monkeypatch.setattr(requests, "get", lambda *a, **k: replies.pop(0))
    assert va.download("https://example.invalid/b", sleep=waits.append) == b"ok" and waits == [2, 4]
    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(1) or FakeResponse(404))
    with pytest.raises(requests.HTTPError):
        va.download("https://example.invalid/b", sleep=waits.append)
    assert len(calls) == 1                                     # a 404 is final, not retried
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(503))
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        va.download("https://example.invalid/b", sleep=lambda s: None)


def test_failed_installation_is_retried_and_reported(tmp_path):
    clock = [1000.0]
    statuses = ["install refused or failed: download failed after 3 attempts (HTTP 503)", "installed"]
    installs = []

    def installer():
        installs.append(clock[0])
        return statuses.pop(0)
    calls = []
    counting = {k: (lambda *a, k=k: calls.append(k)) for k in ("cdec", "power", "gefs")}
    svc = service(tmp_path, NOW, models=None, **counting)
    svc.artifact_installer, svc._mono = installer, lambda: clock[0]
    svc._load_models = lambda: (mock_models() if installs and not statuses else
                                {k: {"error": "model artifact missing"} for k in ("A", "B", "B2", "C")})
    r = svc.forecast("FOL")
    assert r["status"] == "unavailable" and "artifacts are not installed" in r["reasons"][0]
    assert "HTTP 503" in r["reasons"][0] and "tries again automatically" in r["reasons"][0]
    assert "forecast_model --build" not in r["reasons"][0] and calls == []
    clock[0] += 10
    svc.forecast("FOL")
    assert len(installs) == 1                                  # not retried on every request
    clock[0] += 60
    svc.fetchers = {"cdec": lambda n: synthetic_cdec(n), "power": synthetic_power, "gefs": synthetic_gefs}
    r = svc.forecast("FOL")
    assert len(installs) == 2 and r["status"] == "ok" and r["models"]["A"]["available"]


def _models_with_b():
    models = mock_models()
    models["B"] = dict(models["A"], features=models["A"]["features"][:9] +
                       ["rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"] + models["A"]["features"][9:])
    return models


def test_slow_nasa_power_does_not_delay_model_a(tmp_path):
    release = threading.Event()

    def slow_power(n):
        release.wait(20)
        return synthetic_power(n)
    svc = service(tmp_path, NOW, models=_models_with_b(), power=slow_power)
    svc.power_wait_s = 0.3
    t0 = time.monotonic()
    r = svc.forecast("FOL")
    assert time.monotonic() - t0 < 5
    assert r["status"] == "ok" and r["models"]["A"]["available"]
    assert not r["models"]["B"]["available"] and "NASA POWER" in r["models"]["B"]["reasons"][0]
    assert "did not respond within 0.3 seconds" in r["source_status"]["power"]["error"]
    release.set()                                              # POWER answers; the background download finishes
    svc._power_jobs[next(iter(svc._power_jobs))][0].result(timeout=10)
    r = svc.forecast("FOL")
    assert r["models"]["B"]["available"] and r["source_status"]["power"] == dict(r["source_status"]["power"], ok=True, cached=True)


def test_power_failure_backs_off_and_unexpected_errors_never_block_model_a(tmp_path):
    clock, calls = [0.0], []

    def failing(n):
        calls.append(1)
        raise SourceError("POWER 503")
    svc = service(tmp_path, NOW, models=_models_with_b(), power=failing)
    svc._mono = lambda: clock[0]
    for _ in range(3):
        r = svc.forecast("FOL")
        assert r["models"]["A"]["available"] and "POWER 503" in r["source_status"]["power"]["error"]
    assert len(calls) == 1                                     # failure reused, not retried by every visitor
    clock[0] += 301
    svc.forecast("FOL")
    assert len(calls) == 2
    broken = service(tmp_path / "b", NOW, models=_models_with_b(), power=lambda n: {}["series"])
    r = broken.forecast("FOL")
    assert r["status"] == "ok" and r["models"]["A"]["available"] and "KeyError" in r["source_status"]["power"]["error"]

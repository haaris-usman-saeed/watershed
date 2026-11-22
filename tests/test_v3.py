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
from src.v3_service import (APPROVED_STATIONS, ForecastService, PolicyBlocked, RateLimited, RecordStore, SourceCache,
                            SourceError, TokenBucket, check_display_allowed, days_remaining, weather_features)

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
    p = ROOT / "forecasts/records" / f"{led[0]['forecast_id']}.json"
    if p.exists():
        import hashlib
        assert hashlib.sha256(p.read_bytes()).hexdigest() == led[0]["sha256"]


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


@pytest.mark.skipif(not (ROOT / "data/v3/gefs_basin_precip.csv").exists(), reason="GEFS history not built")
def test_gefs_history_uses_forecast_cycles_only_and_operational_published_before_issue():
    d = pd.read_csv(ROOT / "data/v3/gefs_basin_precip.csv", parse_dates=["init_date"])
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
    policy = json.loads((ROOT / "config/data_release_policy.json").read_text())
    with pytest.raises(PolicyBlocked):
        check_display_allowed(["cdec_storage_sensor15"], policy, mode="public")
    check_display_allowed(["cdec_storage_sensor15"], policy, mode="private")
    check_display_allowed(["noaa_gefs_operational", "nasa_power"], policy, mode="public")


def test_public_service_refuses_before_fetching(tmp_path, monkeypatch):
    monkeypatch.setenv("WATERSHED_DEPLOYMENT", "public")
    calls = []
    svc = service(tmp_path, NOW, cdec=lambda n: (calls.append(1), synthetic_cdec(n))[1])
    with pytest.raises(PolicyBlocked):
        svc.forecast("FOL")
    assert calls == []


def test_release_pipeline_never_packages_unresolved_sources(tmp_path):
    from src.v3_release import build_release
    man = build_release(tmp_path, version="test")
    names = [f["path"] for f in man["files"]]
    assert all(("cdec" not in n.lower()) and ("weekly" not in n) and ("storage" not in n) for n in names)
    assert man["excluded_pending_clearance"]
    with pytest.raises(FileExistsError):
        build_release(tmp_path, version="test")                 # published versions are never overwritten


def test_record_store_is_write_once(tmp_path):
    s = RecordStore(root=tmp_path)
    rec = {"station_id": "FOL", "issued_utc": "2026-11-22T18:00:00Z"}
    s.save(rec)
    with pytest.raises(FileExistsError):
        s.save(rec)

    class FakeS3:
        def __init__(self):
            self.objs = {}

        def head_object(self, Bucket, Key):
            if Key not in self.objs:
                raise KeyError(Key)

        def put_object(self, Bucket, Key, Body, ContentType):
            self.objs[Key] = Body
    s3 = RecordStore(client=FakeS3(), bucket="b")
    out = s3.save(rec)
    assert out["location"].startswith("s3://b/")
    with pytest.raises(FileExistsError):
        s3.save(rec)


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

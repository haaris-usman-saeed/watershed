"""v3 forecast service: shared source cache, request coalescing, rate limits, and Models A, B, B2, C.

Used in-process by the Streamlit page (one ForecastService per server process) and from the shell:
  python -m src.v3_service --station FOL          # one forecast, printed as JSON summary
  python -m src.v3_service --warm                 # refresh every source once (for a scheduled job)

Design
- Sources are cached once for ALL reservoirs, keyed by refresh bucket, not per visitor:
    CDEC storage + snow (all six reservoirs, core pillows):  6-hour UTC bucket
    NASA POWER (dam points + upstream-box cells):            12-hour UTC bucket
    GEFS 00Z cycle of the origin Monday:                     cycle date (immutable once complete)
- Coalescing: concurrent requests for the same cache key wait on one in-flight fetch.
- Rate limits: a global token bucket on upstream HTTP fetch jobs, and a per-key cooldown on
  visitor-forced refreshes.
- Integrity: every cache entry stores the SHA-256 of its payload; a mismatch deletes the entry and
  refetches. Raw CDEC files keep their logged SHA-256 (src.forecast_live.read_log).
- Publication gate: config/data_release_policy.json. With WATERSHED_DEPLOYMENT=public, a source
  whose public_display is false makes the service refuse to return values derived from it.
- Never fills: a model is "unavailable" (with reasons) when any input is missing, stale, or its
  upstream fetch failed. Observations after the origin are never used.
- NASA POWER never holds up Model A: it is fetched in the background alongside CDEC, and a forecast waits
  for it at most POWER_WAIT_S. If it is slower, B and B2 are skipped for that forecast while the download
  continues for later ones. After a POWER failure, new attempts wait POWER_RETRY_AFTER_FAILURE_S.
- Model artifacts: when the app passes an installer (src.v3_artifacts.ensure_artifacts), a failed
  installation is retried at most once per ARTIFACT_RETRY_S instead of lasting until a restart, and the
  installer's status is shown in the refusal.
- Model selection: ForecastService(model_keys=...) runs only the named models (default all four). With
  model_keys=("A",), as on the website's Generate Forecast page, NASA POWER and GEFS are never fetched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APPROVED_STATIONS = ("DNP", "EXC", "FOL", "NML", "ORO", "SHA")
SCHEMA = "watershedshift-forecast/2.0"
CDEC_BUCKET_H, POWER_BUCKET_H = 6, 12
HISTORY_DAYS = 220
WEATHER_LAG = 7
FORCE_COOLDOWN_S = 600
UPSTREAM_JOBS_PER_MIN = 6
POWER_WAIT_S = float(os.environ.get("WATERSHED_POWER_WAIT_S", "45"))   # longest a forecast waits for NASA POWER
POWER_RETRY_AFTER_FAILURE_S = 300        # after a failed POWER download, wait this long before trying again
POWER_FETCH_DEADLINE_S = 180             # a POWER download starts no new request after this many seconds
ARTIFACT_RETRY_S = 60                    # a failed model installation is retried at most this often
ALL_MODEL_KEYS = ("A", "B", "B2", "C")
MODEL_LABELS = {"A": "Model A: frozen v2 (storage, snow, season)",
                "B": "Model B: + recent weather at the dam (NASA POWER)",
                "B2": "Model B2: + recent weather over the upstream area (NASA POWER)",
                "C": "Model C: + GEFS precipitation forecast, days 1-16"}


def utcnow():
    return datetime.now(timezone.utc)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class SourceError(RuntimeError):
    pass


class PolicyBlocked(RuntimeError):
    pass


class RateLimited(RuntimeError):
    pass


# ------------------------------------------------------------------ cache, coalescing, rate limiting

class TokenBucket:
    def __init__(self, per_minute, clock=time.monotonic):
        self.capacity, self.rate, self.clock = per_minute, per_minute / 60.0, clock
        self.tokens, self.t = float(per_minute), clock()
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens < 1:
                return False
            self.tokens -= 1
            return True


class SourceCache:
    """JSON entries on disk with payload hashes, plus in-process coalescing of identical fetches."""

    def __init__(self, root, bucket=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.locks, self.guard = {}, threading.Lock()
        self.bucket = bucket or TokenBucket(UPSTREAM_JOBS_PER_MIN)
        self.upstream_calls = 0

    def _path(self, key):
        return self.root / (key.replace("/", "_") + ".json")

    def read(self, key):
        p = self._path(key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text())
            body = json.dumps(obj["payload"], sort_keys=True).encode()
            if _sha(body) != obj["sha256"]:
                raise ValueError("hash mismatch")
            return obj
        except (ValueError, KeyError, json.JSONDecodeError):
            p.unlink(missing_ok=True)          # corrupted entry: discard and refetch
            return None

    def write(self, key, payload, meta=None):
        body = json.dumps(payload, sort_keys=True).encode()
        obj = {"key": key, "written_utc": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256": _sha(body),
               "meta": meta or {}, "payload": payload}
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(obj))
        tmp.replace(self._path(key))           # atomic: readers never see a partial entry
        return obj

    def get_or_fetch(self, key, fetcher, force=False):
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:                               # coalescing: one fetch per key at a time
            hit = None if force else self.read(key)
            if hit is not None:
                return hit, True
            if not self.bucket.take():
                raise RateLimited("Upstream request budget exhausted; try again in a minute.")
            self.upstream_calls += 1
            payload, meta = fetcher()
            return self.write(key, payload, meta), False


# ------------------------------------------------------------------ release policy

def load_policy():
    return json.loads((ROOT / "config" / "data_release_policy.json").read_text())


def deployment_mode():
    return os.environ.get("WATERSHED_DEPLOYMENT", "private").strip().lower()


def check_display_allowed(sources, policy=None, mode=None):
    """Raise PolicyBlocked if this deployment is public and any source may not be displayed publicly."""
    policy = policy or load_policy()
    if (mode or deployment_mode()) != "public":
        return
    blocked = [s for s in sources if not policy["sources"].get(s, {}).get("public_display", False)]
    if blocked:
        raise PolicyBlocked("Public display of " + ", ".join(blocked) + " is not cleared "
                            "(docs/DATA_PUBLICATION_AUDIT.md). This public deployment cannot show forecasts that "
                            "contain those values until the owner records written permission.")


def cdec_display_block_reason(policy=None, mode=None):
    """Message explaining why CDEC-derived values may not be shown on this deployment, or None if allowed."""
    try:
        check_display_allowed(["cdec_storage_sensor15", "cdec_snow_sensor82"], policy, mode)
    except PolicyBlocked as exc:
        return str(exc)
    return None


# ------------------------------------------------------------------ feature helpers (pure)

def weather_features(daily, t, lag=WEATHER_LAG, suffix=""):
    """Same definitions as src.features.construct_features: windows end at t - lag, full windows only."""
    t = pd.Timestamp(t)
    end = t - pd.Timedelta(days=lag)

    def window(col, n, how):
        idx = pd.date_range(end - pd.Timedelta(days=n - 1), end, freq="D")
        v = daily[col].reindex(idx) if col in daily else pd.Series(np.nan, index=idx)
        if v.isna().any():
            return np.nan
        return float(v.sum() if how == "sum" else v.mean())

    return {f"rain_7d_mm{suffix}": window("rain_mm", 7, "sum"), f"rain_30d_mm{suffix}": window("rain_mm", 30, "sum"),
            f"temp_30d_c{suffix}": window("temp_c", 30, "mean"), f"humidity_30d_pct{suffix}": window("humidity_pct", 30, "mean")}


def days_remaining(target_date, issued_utc):
    """Whole days from the issue instant to the end of the target date (UTC)."""
    end = datetime.combine(pd.Timestamp(target_date).date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    return round((end - issued_utc).total_seconds() / 86400, 2)


# ------------------------------------------------------------------ the service

class ForecastService:
    def __init__(self, cache_dir=None, art_v3=ROOT / "models" / "v3", fetchers=None, clock=utcnow,
                 artifact_installer=None, power_wait_s=None, monotonic=time.monotonic, model_keys=ALL_MODEL_KEYS):
        unknown = [k for k in model_keys if k not in ALL_MODEL_KEYS]
        if unknown or "A" not in model_keys:
            raise ValueError(f"model_keys must include 'A' and be drawn from {ALL_MODEL_KEYS}; got {tuple(model_keys)}")
        self.model_keys = tuple(k for k in ALL_MODEL_KEYS if k in model_keys)   # canonical order
        self.cache = SourceCache(cache_dir or os.environ.get("WATERSHED_CACHE_DIR", ROOT / "data" / "v3" / "cache"))
        self.art_v3 = Path(art_v3)
        self.clock = clock
        self.fetchers = fetchers or {}
        self.force_times = {}
        self._models = None
        self._lock = threading.Lock()
        self.artifact_installer = artifact_installer      # e.g. src.v3_artifacts.ensure_artifacts; None = never install
        self.artifact_status = None                       # last status string returned by the installer
        self._artifact_tried = None
        self.power_wait_s = POWER_WAIT_S if power_wait_s is None else float(power_wait_s)
        self._mono = monotonic
        self._bg = None                                   # background executor for NASA POWER, created on first use
        self._power_jobs = {}                             # cache key -> [future, monotonic time it finished or None]
        self._power_lock = threading.Lock()

    # ----- models
    def models(self):
        with self._lock:
            missing = self._models is None or "error" in self._models.get("A", {})
            if missing and self.artifact_installer is not None:
                now = self._mono()
                if self._artifact_tried is None or now - self._artifact_tried >= ARTIFACT_RETRY_S:
                    self._artifact_tried = now
                    self.artifact_status = self.artifact_installer()
                    if self.artifact_status == "installed":
                        self._models = None                 # reload from the newly installed files
            if self._models is None:
                self._models = self._load_models()
            return self._models

    def _load_models(self):
        import joblib
        out = {}
        try:
            from src.forecast_model import load_artifact
            m, card = load_artifact()
            out["A"] = {"model": m, "features": card["feature_order"], "cutoff": card["training_cutoff_exclusive"],
                        "offsets": card["interval_offsets_pp"], "clim": card["snow_climatology_in"],
                        "coverage": card["historical_interval_coverage"], "sha256": card["artifact_sha256"],
                        "version": "v2-frozen", "core": card["core_snow_pillows"]}
        except (FileNotFoundError, ValueError) as exc:
            out["A"] = {"error": str(exc)}
        try:
            from src.v3_artifacts import verify_selfcheck
            bad = verify_selfcheck(ROOT)
        except Exception as exc:  # noqa: BLE001  a self-check that cannot run counts as failed
            bad = {"models/v2_forecast/model.joblib": f"self-check could not run: {exc}"}
        if "models/v2_forecast/model.joblib" in bad and "model" in out.get("A", {}):
            out["A"] = {"error": "Model A refused: " + bad["models/v2_forecast/model.joblib"]}
        self._selfcheck_failures = bad
        cards_p = self.art_v3 / "model_cards.json"
        cards = json.loads(cards_p.read_text())["models"] if cards_p.exists() else {}
        for k in ("B", "B2", "C"):
            c = cards.get(k)
            p = self.art_v3 / f"model_{k}.joblib"
            if c is None or not p.exists():
                out[k] = {"error": f"v3 artifact for Model {k} not built (python -m src.v3_experiment --final)"}
                continue
            if _sha(p.read_bytes()) != c["artifact_sha256"]:
                out[k] = {"error": f"Model {k} artifact hash does not match its card; refusing to load"}
                continue
            if f"models/v3/model_{k}.joblib" in bad:
                out[k] = {"error": f"Model {k} refused: " + bad[f"models/v3/model_{k}.joblib"]}
                continue
            m = joblib.load(p)   # trusted: our own artifact, hash-checked above
            out[k] = {"model": m, "features": c["feature_order"], "cutoff": c["training_cutoff_exclusive"],
                      "offsets": c["interval_offsets_pp"], "clim": c["snow_climatology_in"], "sha256": c["artifact_sha256"],
                      "version": "v3", "coverage": c.get("historical_interval_coverage_2026_test", {})}
        return out

    # ----- sources
    def _bucket(self, hours, now):
        return now.strftime("%Y%m%d") + f"T{(now.hour // hours) * hours:02d}"

    def _allow_force(self, key, now):
        last = self.force_times.get(key)
        if last and (now - last).total_seconds() < FORCE_COOLDOWN_S:
            return False
        self.force_times[key] = now
        return True

    def cdec(self, now, force=False):
        key = f"cdec_{self._bucket(CDEC_BUCKET_H, now)}"
        force = force and self._allow_force(key, now)
        return self.cache.get_or_fetch(key, lambda: self.fetchers.get("cdec", fetch_cdec)(now), force=force)

    def power(self, now, force=False):
        key = f"power_{self._bucket(POWER_BUCKET_H, now)}"
        force = force and self._allow_force(key, now)
        return self.cache.get_or_fetch(key, lambda: self.fetchers.get("power", fetch_power)(now), force=force)

    def _power_job(self, now, force=False):
        """The background POWER fetch for this refresh bucket, shared by concurrent forecasts. A failed fetch is
        reused (not retried) for POWER_RETRY_AFTER_FAILURE_S."""
        key = f"power_{self._bucket(POWER_BUCKET_H, now)}"
        with self._power_lock:
            t = self._mono()
            job = self._power_jobs.get(key)
            if job is not None:
                fut, done_at = job
                if not fut.done():
                    return fut
                if fut.exception() is not None and done_at is not None and t - done_at < POWER_RETRY_AFTER_FAILURE_S:
                    return fut
            if self._bg is None:
                self._bg = ThreadPoolExecutor(max_workers=2, thread_name_prefix="watershed-power")
            self._power_jobs = {k: j for k, j in self._power_jobs.items() if not j[0].done()}
            job = [None, None]
            fut = self._bg.submit(self.power, now, force)
            job[0] = fut
            fut.add_done_callback(lambda f, job=job: job.__setitem__(1, self._mono()))
            self._power_jobs[key] = job
            return fut

    def gefs(self, cycle):
        key = f"gefs_{cycle.isoformat()}"
        hit = self.cache.read(key)
        if hit is not None and hit["payload"].get("complete"):
            return hit, True
        return self.cache.get_or_fetch(key, lambda: self.fetchers.get("gefs", fetch_gefs)(cycle), force=hit is not None)

    # ----- one reservoir
    def forecast(self, station, force_refresh=False):
        if station not in APPROVED_STATIONS:
            raise ValueError(f"unknown reservoir {station!r}")
        check_display_allowed(["cdec_storage_sensor15", "cdec_snow_sensor82"])
        issued = self.clock()
        status, warnings = {}, []
        loaded = self.models()
        if "error" in loaded.get("A", {"error": ""}):
            why = "Model artifacts are not installed on this deployment, so no forecast can be made"
            if self.artifact_installer is not None and self.artifact_status not in (None, "present", "installed"):
                why += (f". Last installation attempt: {self.artifact_status}. The server tries again automatically, "
                        f"at most once every {ARTIFACT_RETRY_S} seconds.")
            else:
                why += ": " + loaded["A"]["error"]
            return self._failed(station, issued, why)
        needs_power = any(k in self.model_keys for k in ("B", "B2"))
        power_job = self._power_job(issued, force_refresh) if needs_power else None   # runs alongside the CDEC refresh
        power_deadline = self._mono() + self.power_wait_s
        try:
            cdec_entry, cdec_cached = self.cdec(issued, force_refresh)
            status["cdec"] = {"ok": True, "cached": cdec_cached, "retrieved_utc": cdec_entry["written_utc"]}
        except (SourceError, RateLimited, RuntimeError) as exc:
            return self._failed(station, issued, f"CDEC refresh failed: {exc}")
        if power_job is None:
            return self._infer(station, issued, cdec_entry, None, status, warnings)
        try:
            pw_entry, pw_cached = power_job.result(timeout=max(0.0, power_deadline - self._mono()))
            status["power"] = {"ok": True, "cached": pw_cached, "retrieved_utc": pw_entry["written_utc"]}
        except FuturesTimeout:
            pw_entry = None
            status["power"] = {"ok": False, "error": f"NASA POWER did not respond within {self.power_wait_s:g} seconds, so "
                                                     "Models B and B2 were skipped for this forecast. The download continues "
                                                     "in the background and later forecasts will use it."}
        except Exception as exc:  # noqa: BLE001  a POWER problem may disable only B and B2, never Model A
            pw_entry = None
            status["power"] = {"ok": False, "error": str(exc) if isinstance(exc, (SourceError, RateLimited))
                               else f"{type(exc).__name__}: {exc}"}
        return self._infer(station, issued, cdec_entry, pw_entry, status, warnings)

    def _failed(self, station, issued, why):
        return {"schema": SCHEMA, "record_type": "prospective_research_forecast", "is_heldout_test_prediction": False,
                "station_id": station, "status": "unavailable", "issued_utc": issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reasons": [why], "models": {}}

    def _infer(self, station, issued, cdec_entry, pw_entry, status, warnings):
        from src.forecast_live import candidate_origins, design_row, inference_features, season_of
        from src.features import load_breaks
        models = self.models()
        cd = cdec_entry["payload"]
        st = cd["stations"].get(station)
        if not st or not st["storage"]:
            return self._failed(station, issued, "No CDEC storage values were retrieved for this reservoir.")
        storage = pd.Series(st["storage"], dtype=float)
        storage.index = pd.to_datetime(storage.index)
        storage = storage.dropna()
        swe = pd.Series(cd["snow_index"].get(station, {}), dtype=float)
        swe.index = pd.to_datetime(swe.index)
        swe = swe.dropna()
        cap_hist = pd.DataFrame(cd["capacity_history"])
        from src import build_dataset as bd
        cap_at = lambda d: float(bd.capacity_series(station, pd.DatetimeIndex([pd.Timestamp(d)]), cap_hist).iloc[0])  # noqa: E731
        issue_day = pd.Timestamp(issued.date())
        breaks = load_breaks().get(station, [])
        clim_a = models.get("A", {}).get("clim", {}).get(station)
        origin, feats_a, considered = None, None, []
        for t in candidate_origins(issue_day):
            f, missing = inference_features(storage, cap_at, swe, clim_a or [np.nan] * 366, t, breaks)
            considered.append({"origin_date": str(t.date()), "eligible": not missing, "missing": missing})
            if not missing:
                origin, feats_a = t, f
                break
        rec = {"schema": SCHEMA, "record_type": "prospective_research_forecast", "is_heldout_test_prediction": False,
               "data_mode": "real", "station_id": station, "reservoir_name": st["name"],
               "issued_utc": issued.strftime("%Y-%m-%dT%H:%M:%SZ"), "origins_considered": considered,
               "storage_last_obs_date": str(storage.index.max().date()),
               "snow_last_obs_date": str(swe.index.max().date()) if len(swe) else None,
               "source_status": status, "warnings": warnings, "models": {},
               "sources": {"cdec": cdec_entry["meta"], "power": pw_entry["meta"] if pw_entry else None}}
        if origin is None:
            rec.update(status="unavailable", reasons=["No Monday origin within 14 days of the issue date has complete "
                                                      "storage and snow inputs. Older origins are never used."])
            return rec
        target = origin + pd.Timedelta(days=30)
        cap = cap_at(origin)
        s_af = float(storage[origin])
        age = (issue_day - origin).days
        if age > 7:
            warnings.append(f"Stale origin: {age} days before the issue date (a newer Monday lacked inputs).")
        if (issue_day - storage.index.max()).days > 5:
            warnings.append(f"Latest storage observation {storage.index.max().date()} is more than 5 days old.")
        rec.update(status="ok", origin_date=str(origin.date()), target_date=str(target.date()),
                   days_remaining=days_remaining(target, issued), capacity_af=cap,
                   observed={"storage_af": s_af, "storage_pct": 100 * s_af / cap, "date": str(origin.date())},
                   snow={"swe_date": str((origin - pd.Timedelta(days=1)).date()), "swe_basin_in": feats_a["swe_basin_in"],
                         "swe_change_14d_in": feats_a["swe_change_14d_in"]},
                   recent_history=[{"date": str(d.date()), "storage_af": float(v), "storage_pct": 100 * float(v) / cap_at(d),
                                    "used_by_forecast": bool(d <= origin)} for d, v in storage[storage.index >= issue_day - pd.Timedelta(days=180)].items()])
        weather = {}
        if pw_entry is not None:
            for src, suf in (("dam_point", ""), ("upstream_box_mean", "_box")):
                d = pd.DataFrame(pw_entry["payload"]["series"].get(src, {}).get(station, {}))
                if len(d):
                    d.index = pd.to_datetime(d.index)
                    weather.update(weather_features(d, origin, suffix=suf))
                    rec.setdefault("weather_history", {})[src] = ({str(k.date()): float(v) for k, v in d.rain_mm.dropna().items()}
                                                                 if "rain_mm" in d else {})
        cycle = origin.date()
        gefs = None
        if "C" in self.model_keys:
            try:
                ge, ge_cached = self.gefs(cycle)
                gefs = ge["payload"]
                status["gefs"] = {"ok": bool(gefs.get("complete")), "cached": ge_cached, "cycle": f"{cycle}T00Z",
                                  "published_utc_max": gefs.get("published_utc_max"), "note": gefs.get("note")}
            except (SourceError, RateLimited, RuntimeError) as exc:
                status["gefs"] = {"ok": False, "error": str(exc)}
        for k in self.model_keys:
            rec["models"][k] = self._score(k, models.get(k, {}), station, origin, feats_a, weather, gefs, issued, cap, s_af, season_of(origin), swe)
        return rec

    def _score(self, k, m, station, origin, feats_a, weather, gefs, issued, cap, s_af, season, swe):
        from src.forecast_live import design_row
        out = {"label": MODEL_LABELS[k], "available": False, "reasons": []}
        if "error" in m:
            out["reasons"].append(m["error"])
            return out
        t1 = origin - pd.Timedelta(days=1)
        f = dict(feats_a)
        f["swe_anom_in"] = f["swe_basin_in"] - float(m["clim"][station][t1.dayofyear - 1])
        info_cutoff = {"storage": str(origin.date()), "snow": str(t1.date())}
        if k in ("B", "B2"):
            suf = "" if k == "B" else "_box"
            need = [c for c in ("rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct")]
            vals = {c + suf: weather.get(c + suf, np.nan) for c in need}
            if any(np.isnan(v) for v in vals.values()):
                out["reasons"].append("NASA POWER weather for the 37 days ending 7 days before the origin is incomplete "
                                      "or could not be retrieved.")
                return out
            f.update(vals)
            info_cutoff["weather"] = str((origin - pd.Timedelta(days=WEATHER_LAG)).date())
        if k == "C":
            if not gefs or not gefs.get("complete"):
                out["reasons"].append(f"GEFS 00Z cycle of {origin.date()} is missing or incomplete"
                                      + (f" ({gefs.get('note')})" if gefs else "") + ".")
                return out
            pub = gefs.get("published_utc_max")
            if not pub or datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) > issued:
                out["reasons"].append("GEFS cycle was not published before the issue time.")
                return out
            f["gefs_p_d1_7_mm"] = gefs["values"][station]["gefs_p_d1_7_mm"]
            f["gefs_p_d8_16_mm"] = gefs["values"][station]["gefs_p_d8_16_mm"]
            info_cutoff["weather_forecast"] = f"GEFS {origin.date()} 00Z, published by {pub}"
        cols = [c for c in m["features"] if not c.startswith("station_")]
        row = {c: f[c] for c in cols}
        row.update({f"station_{s}": float(s == station) for s in APPROVED_STATIONS})
        x = pd.DataFrame([row])[m["features"]]
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            out["reasons"].append("A required input is missing.")
            return out
        pp = float(m["model"].predict(x)[0])
        lo_off, hi_off = m["offsets"][season]
        pct0 = 100 * s_af / cap
        out.update(available=True, version=m["version"], training_cutoff_exclusive=m["cutoff"], artifact_sha256=m["sha256"],
                   information_cutoff=info_cutoff, inputs={c: float(x[c].iloc[0]) for c in x.columns},
                   change_pp=pp, change_af=pp / 100 * cap, projected_af=s_af + pp / 100 * cap, projected_pct=pct0 + pp,
                   interval={"method": "sym_season, model-specific", "season": season, "offsets_pp": [lo_off, hi_off],
                             "lower_change_pp": pp + lo_off, "upper_change_pp": pp + hi_off,
                             "lower_projected_pct": pct0 + pp + lo_off, "upper_projected_pct": pct0 + pp + hi_off,
                             "future_coverage_established": False,
                             "historical_coverage": (m.get("coverage") or {}).get(station)},
                   plausible=bool(0 <= pct0 + pp <= 100))
        if k == "C" and gefs:
            out["gefs"] = {"cycle": f"{origin.date()}T00Z", "published_utc_max": gefs.get("published_utc_max"),
                           "source": gefs.get("source"), "horizon": "days 1-16 only; nothing represents days 17-30",
                           "values": gefs["values"][station], "bytes_sha256": gefs.get("bytes_sha256")}
        return out


# ------------------------------------------------------------------ default upstream fetchers

def fetch_cdec(now):
    """All six reservoirs and core pillows, last HISTORY_DAYS, via the v2 refresh and cleaning code."""
    import src.forecast_live as fl
    from src.forecast_model import load_artifact
    try:
        _, card = load_artifact()
        core = card["core_snow_pillows"]
    except (FileNotFoundError, ValueError) as exc:
        raise SourceError(f"model card with core pillows unavailable: {exc}")
    issue = pd.Timestamp(now.date())
    start = str((issue - pd.Timedelta(days=HISTORY_DAYS)).date())
    run_dir = ROOT / "data" / "v3" / "cache" / "cdec_raw" / now.strftime("%Y%m%dT%H%M%S")
    pillows = sorted({p for v in core.values() for p in v})
    try:
        log = fl.fetch(run_dir, start, str(issue.date()), list(APPROVED_STATIONS), pillows)
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"{type(exc).__name__}: {exc}")
    if not log or not all(r.get("sha256_ok") for r in log):
        raise SourceError("downloaded files do not match their logged hashes")
    storage, audits, snow, _, cap_hist = fl.clean(run_dir, start, str(issue.date()), list(APPROVED_STATIONS))
    idx = fl.basin_index(snow, core) if snow is not None else pd.DataFrame(columns=["station_id", "date", "swe_basin_in"])
    names = pd.read_csv(ROOT / "stations.csv").set_index("station_id").reservoir_name
    payload = {"stations": {s: {"name": names[s], "storage": {str(d.date()): (None if pd.isna(v) else float(v))
                                                             for d, v in storage[s].storage_af.items()}}
                            for s in storage},
               "snow_index": {s: {str(d.date()): float(v) for d, v in g.set_index("date").swe_basin_in.dropna().items()}
                              for s, g in idx.groupby("station_id")},
               "capacity_history": cap_hist.to_dict(orient="list")}
    meta = {"provider": "CDEC (California DWR)", "endpoint": fl.CDEC_URL, "window": [start, str(issue.date())],
            "files": log, "retrieved_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    return payload, meta


def fetch_power(now):
    """Last 60 days of NASA POWER at the dam points and upstream-box cells."""
    from src.fetch import POWER, get
    from src.v3_basins import basin_boxes
    from src.v3_power import PLAUSIBLE, grid_cells
    end = now.date()
    start = end - timedelta(days=60)
    st = pd.read_csv(ROOT / "stations.csv").set_index("station_id")
    boxes = basin_boxes()
    jobs = [(s, "dam_point", float(st.loc[s, "lat"]), float(st.loc[s, "lon"])) for s in APPROVED_STATIONS]
    jobs += [(s, "box", la, lo) for s in APPROVED_STATIONS for la, lo in grid_cells(boxes[s])]
    deadline = time.monotonic() + POWER_FETCH_DEADLINE_S

    def one(j):
        if time.monotonic() > deadline:
            raise SourceError(f"gave up after {POWER_FETCH_DEADLINE_S} s")
        params = {"parameters": "T2M,PRECTOTCORR,RH2M", "community": "AG", "longitude": j[3], "latitude": j[2],
                  "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON", "time-standard": "LST"}
        r = get(POWER, params, lambda r: "properties" in r.json(), tries=2, timeout=30)
        obj = r.json()
        par, fill = obj["properties"]["parameter"], obj["header"].get("fill_value", -999.0)
        w = pd.DataFrame({"temp_c": par["T2M"], "rain_mm": par["PRECTOTCORR"], "humidity_pct": par["RH2M"]})
        w = w.mask(w <= fill + 1e-9)
        for c, (lo, hi) in PLAUSIBLE.items():
            w.loc[(w[c] < lo) | (w[c] > hi), c] = np.nan
        w.index = pd.to_datetime(w.index, format="%Y%m%d").strftime("%Y-%m-%d")
        return w, _sha(r.content), r.url

    try:
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, jobs))
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"NASA POWER: {type(exc).__name__}: {exc}")
    series = {"dam_point": {}, "upstream_box_mean": {}}
    boxframes = {}
    for (s, kind, _, _), (w, h, u) in zip(jobs, res):
        if kind == "dam_point":
            series["dam_point"][s] = w.to_dict()
        else:
            boxframes.setdefault(s, []).append(w)
    for s, fr in boxframes.items():
        series["upstream_box_mean"][s] = pd.concat(fr).groupby(level=0).mean().to_dict()
    meta = {"provider": "NASA LaRC POWER", "endpoint": POWER, "window": [str(start), str(end)],
            "requests": len(jobs), "responses_sha256": [h for _, h, _ in res], "retrieved_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    return {"series": series}, meta


def fetch_gefs(cycle):
    from src.v3_gefs import extract_cycle
    try:
        rows = extract_cycle(cycle)
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"GEFS: {type(exc).__name__}: {exc}")
    complete = all(r.get("complete") for r in rows)
    payload = {"cycle": cycle.isoformat(), "complete": complete, "source": rows[0].get("source"),
               "note": rows[0].get("note"), "published_utc_max": rows[0].get("published_utc_max") or None,
               "bytes_sha256": rows[0].get("bytes_sha256"),
               "values": {r["station_id"]: {"gefs_p_d1_7_mm": r.get("gefs_p_d1_7_mm"), "gefs_p_d8_16_mm": r.get("gefs_p_d8_16_mm")}
                          for r in rows}}
    meta = {"provider": "NOAA NCEP GEFSv12 via NODD on AWS", "cycle": f"{cycle}T00Z", "retrieved_utc": rows[0].get("retrieved_utc")}
    return payload, meta


# ------------------------------------------------------------------ records

GITHUB_API = "https://api.github.com"


class RecordStoreError(RuntimeError):
    """A record could not be saved. The message is safe to show visitors: it never contains credentials."""


def _s3_error_code(exc):
    """The error code of a botocore ClientError (for example "404", "NoSuchKey", "PreconditionFailed"), else None."""
    resp = getattr(exc, "response", None) or {}
    code = (resp.get("Error") or {}).get("Code")
    status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return str(code) if code is not None else (str(status) if status is not None else None)


def _github_message(r):
    try:
        return str(r.json().get("message", ""))[:200]
    except ValueError:
        return ""


class RecordStore:
    """Immutable forecast records, in one of three backends (first configured wins):

    - GitHub repository (WATERSHED_RECORD_GITHUB_REPO = "owner/name" and WATERSHED_RECORD_GITHUB_TOKEN, a
      fine-grained token with Contents read/write on that repository only; optional
      WATERSHED_RECORD_GITHUB_BRANCH). Each record is created as forecast-records/YYYY/MM/<id>.json through the
      contents API without a blob SHA, which GitHub refuses when the path already exists, and every record is
      its own commit, so git history shows when it was written. Needs no extra dependency.
    - S3 or S3-compatible bucket (WATERSHED_RECORD_BUCKET, optional WATERSHED_RECORD_ENDPOINT_URL; needs boto3).
      Written with If-None-Match: * after a HEAD in which only "not found" counts as absent.
    - Local directory (default), exclusive create, mode 0444. Lost on hosts with temporary disks.

    Write-once in every backend: an existing record is never overwritten. Record IDs end with the first 8 hex
    characters of the record's SHA-256, so two different forecasts issued in the same second cannot collide."""

    def __init__(self, root=None, client=None, bucket=None, prefix="forecast-records/", endpoint_url=None,
                 github_repo=None, github_token=None, github_branch=None, session=None, sleep=time.sleep):
        self.root = Path(root or os.environ.get("WATERSHED_RECORD_DIR", ROOT / "forecasts" / "records_v3"))
        self.bucket = bucket or os.environ.get("WATERSHED_RECORD_BUCKET")
        self.endpoint_url = endpoint_url or os.environ.get("WATERSHED_RECORD_ENDPOINT_URL")
        self.github_repo = github_repo or os.environ.get("WATERSHED_RECORD_GITHUB_REPO")
        self._github_token = github_token or os.environ.get("WATERSHED_RECORD_GITHUB_TOKEN")
        self.github_branch = github_branch or os.environ.get("WATERSHED_RECORD_GITHUB_BRANCH")
        self.client, self.prefix, self.session, self._sleep = client, prefix, session, sleep

    @property
    def backend(self):
        return "github" if self.github_repo else ("s3" if self.bucket else "local")

    @property
    def persistent(self):
        """True for GitHub and bucket backends. A local directory is lost on hosts with temporary disks."""
        return self.backend != "local"

    def describe(self):
        return {"github": "persistent record store (GitHub)", "s3": "persistent record store (bucket)"}.get(
            self.backend, "local disk only: lost when the app restarts on hosts with temporary storage such as "
                          "Streamlit Community Cloud")

    def _s3(self):
        if self.client is None:
            import boto3
            self.client = boto3.client("s3", endpoint_url=self.endpoint_url) if self.endpoint_url else boto3.client("s3")
        return self.client

    @staticmethod
    def record_id(rec, body):
        return f"wsf-v3-{rec['station_id']}-{rec['issued_utc'].replace(':', '').replace('-', '')}-{_sha(body)[:8]}"

    def save(self, rec, record_id=None):
        """Write `rec` once. `record_id` names the record explicitly (official weekly records); by default the ID is
        derived from the reservoir, issue time and content hash."""
        body = json.dumps(rec, indent=1, allow_nan=False).encode()
        rid = record_id or self.record_id(rec, body)
        if self.backend == "github":
            return self._save_github(rid, rec, body)
        if self.backend == "s3":
            return self._save_s3(rid, body)
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / f"{rid}.json"
        try:
            with open(p, "xb") as fh:         # exclusive create: never overwrites, even under concurrency
                fh.write(body)
        except FileExistsError:
            raise FileExistsError(p) from None
        p.chmod(0o444)
        return {"id": rid, "location": str(p), "sha256": _sha(body)}

    def _save_s3(self, rid, body):
        key = self.prefix + rid + ".json"
        s3 = self._s3()
        try:
            s3.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001  only "not found" means the record is absent
            if _s3_error_code(exc) not in ("404", "NoSuchKey", "NotFound"):
                raise
        else:
            raise FileExistsError(key)
        try:
            s3.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json", IfNoneMatch="*")
        except Exception as exc:  # noqa: BLE001
            if _s3_error_code(exc) in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
                raise FileExistsError(key) from exc
            raise
        return {"id": rid, "location": f"s3://{self.bucket}/{key}", "sha256": _sha(body)}

    def _save_github(self, rid, rec, body):
        import base64
        if not self._github_token:
            raise RecordStoreError("WATERSHED_RECORD_GITHUB_TOKEN is not set, so the record could not be saved to "
                                   f"{self.github_repo}.")
        if self.session is None:
            import requests
            self.session = requests.Session()
        issued = rec["issued_utc"]
        path = f"{self.prefix}{issued[:4]}/{issued[5:7]}/{rid}.json"
        url = f"{GITHUB_API}/repos/{self.github_repo}/contents/{path}"
        headers = {"Authorization": f"Bearer {self._github_token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28"}
        data = {"message": f"Add forecast record {rid}", "content": base64.b64encode(body).decode()}
        if self.github_branch:
            data["branch"] = self.github_branch
        for attempt in range(3):
            try:
                r = self.session.put(url, headers=headers, json=data, timeout=30)
            except Exception as exc:  # noqa: BLE001  network errors: report the type only
                raise RecordStoreError(f"GitHub could not be reached ({type(exc).__name__}).") from None
            if r.status_code == 201:                 # created; 200 would mean an update, which needs a SHA we never send
                j = r.json()
                return {"id": rid, "location": (j.get("content") or {}).get("html_url") or f"github:{self.github_repo}/{path}",
                        "sha256": _sha(body), "commit": (j.get("commit") or {}).get("sha")}
            if r.status_code == 409 and attempt < 2:  # the branch moved under a concurrent write: try again
                self._sleep(1 + attempt)
                continue
            if r.status_code == 422:                 # GitHub refuses a create without SHA when the file exists
                g = self.session.get(url, headers=headers, timeout=30,
                                     params={"ref": self.github_branch} if self.github_branch else None)
                if g.status_code == 200:
                    raise FileExistsError(path)
            raise RecordStoreError(f"GitHub refused the record (HTTP {r.status_code}"
                                   + (f": {_github_message(r)}" if _github_message(r) else "") + ").")
        raise RecordStoreError("GitHub kept reporting a conflicting write (HTTP 409).")


def main():
    import warnings
    warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta", category=DeprecationWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", choices=APPROVED_STATIONS)
    ap.add_argument("--warm", action="store_true")
    a = ap.parse_args()
    svc = ForecastService()
    if a.warm:
        now = utcnow()
        svc.cdec(now), svc.power(now)
        print("warmed CDEC and POWER caches")
    if a.station:
        r = svc.forecast(a.station)
        brief = {k: r.get(k) for k in ("station_id", "status", "issued_utc", "origin_date", "target_date", "days_remaining",
                                       "storage_last_obs_date", "snow_last_obs_date", "warnings", "reasons")}
        brief["models"] = {k: ({"change_pp": round(v["change_pp"], 2), "interval": [round(v["interval"]["lower_change_pp"], 2),
                                                                                round(v["interval"]["upper_change_pp"], 2)]}
                               if v.get("available") else {"unavailable": v["reasons"]}) for k, v in r.get("models", {}).items()}
        brief["source_status"] = r.get("source_status")
        print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()

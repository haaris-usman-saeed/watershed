"""Prospective forecast scoring: the reference implementation of docs/prospective/PROTOCOL.md.

Run (always started by a person; there is no scheduler):
  python -m src.prospective score forecasts/records/<id>.json --stage provisional   # from target date + 2 days
  python -m src.prospective score forecasts/records/<id>.json --stage settled       # from target date + 30 days
  python -m src.prospective summary [--stage settled]

What it does
- Verifies that the protocol document and the fixed baseline table match the SHA-256 values in
  config/prospective_protocol.json (refuses otherwise), and that the forecast record matches its ledger SHA-256.
- Scores only records listed in a ledger: the two pre-protocol issued records (docs/v3/issued_forecast_ledger.jsonl)
  and official weekly records (docs/prospective/official_ledger.jsonl). Visitor forecasts are never scored.
- Retrieves CDEC daily storage (sensor 15) for the origin and target dates at scoring time, keeps the raw response
  with its SHA-256 in data/prospective/raw/ (git-ignored), and computes the realised change exactly as the training
  target: 100 x (S[target] - S[origin]) / capacity in force at the origin (taken from the record). Never interpolated.
- Writes one write-once score file per record, reservoir and stage under results/prospective/scores/. An existing
  score is never overwritten; a later CDEC revision is visible by comparing stages, not by rewriting scores.
Forecast records are only read.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "config" / "prospective_protocol.json"
DOC = ROOT / "docs" / "prospective" / "PROTOCOL.md"
BASELINE = ROOT / "config" / "prospective_doy_baseline.csv"
LEDGERS = {"pre_protocol_issued": ROOT / "docs" / "v3" / "issued_forecast_ledger.jsonl",
           "official_weekly": ROOT / "docs" / "prospective" / "official_ledger.jsonl"}
SCORES = ROOT / "results" / "prospective" / "scores"
RAW = ROOT / "data" / "prospective" / "raw"
CDEC = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
MODELS = ("A", "B", "B2", "C")


class ProtocolError(RuntimeError):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def load_protocol(protocol=PROTOCOL, doc=DOC, baseline=BASELINE):
    cfg = json.loads(Path(protocol).read_text())
    if sha_file(doc) != cfg["document_sha256"]:
        raise ProtocolError("docs/prospective/PROTOCOL.md differs from the registered protocol; amendments need a new "
                            "protocol version (see its Amendments section).")
    if sha_file(baseline) != cfg["baseline"]["table_sha256"]:
        raise ProtocolError("config/prospective_doy_baseline.csv differs from the registered baseline table.")
    cfg["_config_sha256"] = sha_file(protocol)
    return cfg


def load_baseline(path=BASELINE):
    return pd.read_csv(path).set_index(["station_id", "origin_day_of_year"]).baseline_doy_pp


def classify(rid, record_sha, ledgers=LEDGERS):
    """Ledger class of a record, verifying its SHA-256; None if it is in no ledger (e.g. a visitor forecast)."""
    for cls, path in ledgers.items():
        if not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                if e.get("forecast_id") == rid:
                    if e["sha256"] != record_sha:
                        raise ProtocolError(f"{rid}: record SHA-256 does not match its ledger entry; refusing to score")
                    return cls
    return None


def extract(record):
    """Per-reservoir entries from either record schema (v2 issued record or v3 interactive record)."""
    out = []
    if "stations" in record:                                   # v2 schema: one Model A forecast per reservoir
        for s in record["stations"]:
            e = {"station_id": s["station_id"], "status": s["status"], "reasons": s.get("reasons", []),
                 "origin_date": s.get("origin_date"), "target_date": s.get("target_date"),
                 "capacity_af": s.get("capacity_af"), "issue_storage_af": (s.get("observed") or {}).get("storage_af"),
                 "models": {}}
            f = s.get("forecast") or {}
            if s["status"] == "ok" and "change_pp" in f:
                e["models"]["A"] = {"change_pp": f["change_pp"], "lower_change_pp": f["lower_change_pp"],
                                    "upper_change_pp": f["upper_change_pp"], "projected_pct": f.get("projected_pct")}
            out.append(e)
        return out
    e = {"station_id": record["station_id"], "status": record["status"], "reasons": record.get("reasons", []),
         "origin_date": record.get("origin_date"), "target_date": record.get("target_date"),
         "capacity_af": record.get("capacity_af"), "issue_storage_af": (record.get("observed") or {}).get("storage_af"),
         "models": {}}
    for k in MODELS:
        m = (record.get("models") or {}).get(k) or {}
        if m.get("available"):
            iv = m["interval"]
            e["models"][k] = {"change_pp": m["change_pp"], "lower_change_pp": iv["lower_change_pp"],
                              "upper_change_pp": iv["upper_change_pp"], "projected_pct": m.get("projected_pct")}
    out.append(e)
    return out


def ready(stage, target_date, retrieved_utc, cfg):
    """A stage may be scored only from target_date + its waiting period (UTC calendar days)."""
    wait = cfg["scoring"]["stage_wait_days"][stage]
    return pd.Timestamp(retrieved_utc.date()) >= pd.Timestamp(target_date) + pd.Timedelta(days=wait)


def parse_cdec_storage(text):
    d = pd.read_csv(io.StringIO(text))
    if d.empty:
        return pd.Series(dtype=float)
    date = pd.to_datetime(d["DATE TIME"].astype(str), format="%Y%m%d %H%M").dt.normalize()
    return pd.Series(pd.to_numeric(d["VALUE"], errors="coerce").to_numpy(), index=date).dropna()


def breaks_between(station, origin, target, path=ROOT / "data" / "series_breaks.csv"):
    b = pd.read_csv(path, parse_dates=["break_date"])
    b = b[b.station_id == station]
    return bool(((b.break_date > pd.Timestamp(origin)) & (b.break_date <= pd.Timestamp(target))).any())


def score_entry(e, storage, baseline, cfg):
    """Score one reservoir entry against a retrieved storage series (date -> AF)."""
    out = {k: e[k] for k in ("station_id", "origin_date", "target_date", "capacity_af")}
    if e["status"] != "ok" or not e["models"]:
        return dict(out, status="no_forecast", reasons=e.get("reasons", []))
    o, t, cap = pd.Timestamp(e["origin_date"]), pd.Timestamp(e["target_date"]), float(e["capacity_af"])
    if breaks_between(e["station_id"], o, t):
        return dict(out, status="unscorable", reasons=["documented series break between origin and target"])
    s_o, s_t = storage.get(o, np.nan), storage.get(t, np.nan)
    if not (np.isfinite(s_o) and np.isfinite(s_t)):
        return dict(out, status="unscorable", reasons=["CDEC storage missing on the "
                                                       + ("origin" if not np.isfinite(s_o) else "target") + " date"])
    y = 100 * (s_t - s_o) / cap
    rev = None if e.get("issue_storage_af") is None else s_o - float(e["issue_storage_af"])
    models = {}
    for k, m in e["models"].items():
        err = m["change_pp"] - y
        models[k] = {"forecast_change_pp": m["change_pp"], "error_pp": err, "abs_error_pp": abs(err),
                     "interval_pp": [m["lower_change_pp"], m["upper_change_pp"]],
                     "interval_hit": bool(m["lower_change_pp"] <= y <= m["upper_change_pp"]),
                     "level_error_pp": None if m.get("projected_pct") is None else m["projected_pct"] - 100 * s_t / cap}
    b = float(baseline[(e["station_id"], o.dayofyear)])
    return dict(out, status="scored", realised_change_pp=y, storage_origin_af=s_o, storage_target_af=s_t,
                origin_revision_af=rev,
                origin_revised=bool(rev is not None and abs(rev) > cfg["scoring"]["revision_flag_pct_capacity"] / 100 * cap),
                models=models,
                baselines={"doy_seasonal": {"forecast_change_pp": b, "abs_error_pp": abs(b - y)},
                           "persistence": {"forecast_change_pp": 0.0, "abs_error_pp": abs(y)}})


def write_once(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as fh:                    # never overwrites an existing score
        fh.write(json.dumps(obj, indent=1, allow_nan=False, default=float))
    path.chmod(0o444)
    return path


def fetch_storage(station, start, end, get=None):
    if get is None:
        from src.fetch import get as _get

        def get(params):
            return _get(CDEC, params, lambda r: r.text.startswith("STATION_ID"), tries=3, timeout=60)
    r = get({"Stations": station, "SensorNums": 15, "dur_code": "D", "Start": str(start), "End": str(end)})
    return parse_cdec_storage(r.text), r.content, r.url


def score_record(path, stage, now=None, get=None, scores=SCORES, raw=RAW, cfg=None, ledgers=LEDGERS):
    """Score every reservoir of one ledger-listed record that is ready for `stage`. Returns written paths and
    the reservoirs skipped because they are not ready yet or were already scored."""
    cfg = cfg or load_protocol()
    if stage not in cfg["scoring"]["stage_wait_days"]:
        raise ValueError(f"unknown stage {stage!r}")
    path = Path(path)
    body = path.read_bytes()
    record = json.loads(body)
    rid = record.get("forecast_id") or path.stem
    cls = classify(rid, sha_bytes(body), ledgers)
    if cls is None:
        raise ProtocolError(f"{rid} is not in an issued or official ledger; visitor forecasts are not scored")
    now = now or datetime.now(timezone.utc)
    baseline = load_baseline()
    written, skipped = [], []
    for e in extract(record):
        st = e["station_id"]
        dest = Path(scores) / stage / rid / f"{st}.json"
        if dest.exists():
            skipped.append((st, "already scored"))
            continue
        if e["status"] == "ok" and not ready(stage, e["target_date"], now, cfg):
            skipped.append((st, f"not ready until {pd.Timestamp(e['target_date']).date()} + "
                                f"{cfg['scoring']['stage_wait_days'][stage]} days"))
            continue
        meta = None
        if e["status"] == "ok" and e["models"]:
            o, t = pd.Timestamp(e["origin_date"]).date(), pd.Timestamp(e["target_date"]).date()
            storage, content, url = fetch_storage(st, o, t, get=get)
            rp = Path(raw) / rid / f"{st}_{stage}_{now.strftime('%Y%m%dT%H%M%SZ')}.csv"
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_bytes(content)
            meta = {"url": url, "sha256": sha_bytes(content), "bytes": len(content), "raw_file": str(rp.relative_to(ROOT))
                    if rp.is_relative_to(ROOT) else str(rp)}
            res = score_entry(e, storage, baseline, cfg)
        else:
            res = score_entry(e, pd.Series(dtype=float), baseline, cfg)
        obj = {"schema": "watershedshift-prospective-score/1.0", "protocol_version": cfg["version"],
               "protocol_config_sha256": cfg["_config_sha256"], "scorer_sha256": sha_file(Path(__file__)),
               "record_id": rid, "record_class": cls, "record_sha256": sha_bytes(body), "stage": stage,
               "retrieved_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "cdec_response": meta, **res}
        written.append(write_once(dest, obj))
    return written, skipped


def load_scores(stage="settled", scores=SCORES):
    rows = []
    for p in sorted((Path(scores) / stage).glob("*/*.json")):
        s = json.loads(p.read_text())
        base = {k: s.get(k) for k in ("record_id", "record_class", "station_id", "origin_date", "target_date", "status",
                                      "realised_change_pp", "origin_revised")}
        for k in MODELS:
            m = (s.get("models") or {}).get(k)
            base[k] = m["forecast_change_pp"] if m else np.nan
            base[f"{k}_hit"] = m["interval_hit"] if m else np.nan
        for b in ("doy_seasonal", "persistence"):
            base[b] = ((s.get("baselines") or {}).get(b) or {}).get("forecast_change_pp", np.nan)
        rows.append(base)
    return pd.DataFrame(rows)


def summarize(df):
    """Descriptive summary by record class: counts, availability, MAE and 90% interval coverage per model, and the
    baselines on the same rows as each model."""
    out = {}
    for cls, g in df.groupby("record_class"):
        sc = g[g.status == "scored"]
        res = {"entries": int(len(g)), "scored": int(len(sc)), "no_forecast": int((g.status == "no_forecast").sum()),
               "unscorable": int((g.status == "unscorable").sum()), "origin_revised": int(sc.origin_revised.fillna(False).sum()),
               "models": {}}
        for k in MODELS:
            r = sc[sc[k].notna()]
            if len(r):
                y = r.realised_change_pp
                res["models"][k] = {"n": int(len(r)), "mae_pp": float(np.mean(np.abs(r[k] - y))),
                                    "coverage_90": float(r[f"{k}_hit"].astype(float).mean()) if f"{k}_hit" in r else None,
                                    "doy_seasonal_mae_pp_same_rows": float(np.mean(np.abs(r.doy_seasonal - y))),
                                    "persistence_mae_pp_same_rows": float(np.mean(np.abs(r.persistence - y)))}
        out[cls] = res
    return out


def _coverage_ci(off, block, n_boot, seed):
    """95% moving-block bootstrap CI of interval coverage, resampling origin weeks (all reservoirs together)."""
    rng = np.random.default_rng(seed)
    dates = np.sort(off.date.unique())
    hits = [off.loc[off.date == d, "A_hit"].astype(float).to_numpy() for d in dates]
    sums, counts = np.array([h.sum() for h in hits]), np.array([len(h) for h in hits])
    starts = np.arange(max(1, len(dates) - block + 1))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < len(dates):
            s0 = rng.choice(starts)
            idx.extend(range(s0, min(s0 + block, len(dates))))
        idx = np.array(idx[:len(dates)])
        reps[b] = sums[idx].sum() / counts[idx].sum()
    return [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]


def primary_analysis(df, cfg):
    """P1, P2 and S1-S3 of the protocol, on settled scores of official weekly records only. Refuses before the
    registered number of weekly origins has been scored."""
    from src.evaluation import block_bootstrap_diff
    p = cfg["primary"]
    off = df[(df.record_class == "official_weekly") & (df.status == "scored")].copy()
    n_origins = off[off.A.notna()].origin_date.nunique()
    if n_origins < p["min_weekly_origins"]:
        raise ProtocolError(f"primary analysis needs {p['min_weekly_origins']} scored official weekly origins; "
                            f"{n_origins} so far. Interim summaries are descriptive only.")
    first = sorted(off[off.A.notna()].origin_date.unique())[:p["min_weekly_origins"]]
    off = off[off.origin_date.isin(first)].sort_values(["origin_date", "station_id"])
    off = off.assign(date=pd.to_datetime(off.origin_date), target_delta_pp=off.realised_change_pp)

    def paired(a, b):
        rows = off[off[a].notna() & off[b].notna()]
        boots = {f"{bl}_weeks": block_bootstrap_diff(rows, a, b, block=bl, n_boot=p["n_boot"], seed=p["seed"])
                 for bl in p["block_weeks"]}
        return {"n_rows": int(len(rows)), "bootstrap": boots,
                "verdict": "supported" if all(v["ci95_high"] < 0 for v in boots.values()) else "not established"}

    a_rows = off[off.A.notna()]
    out = {"n_origins": len(first), "first_origin": str(first[0]), "last_origin": str(first[-1]),
           "P1": paired("A", "doy_seasonal"),
           "P2": {"coverage_90": float(a_rows.A_hit.astype(float).mean()),
                  "ci95": _coverage_ci(a_rows, p["coverage_block_weeks"], p["n_boot"], p["seed"]),
                  "by_station": {s: float(g.A_hit.astype(float).mean()) for s, g in a_rows.groupby("station_id")}}}
    for sec in cfg["secondary"]:
        a, b = [x.strip() for x in sec["comparison"].split(" vs ")]
        out[sec["id"]] = paired(a, b)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("record")
    s.add_argument("--stage", choices=["provisional", "settled"], required=True)
    m = sub.add_parser("summary")
    m.add_argument("--stage", choices=["provisional", "settled"], default="settled")
    a = ap.parse_args()
    if a.cmd == "score":
        written, skipped = score_record(a.record, a.stage)
        for p in written:
            print("wrote", p.relative_to(ROOT))
        for st, why in skipped:
            print("skipped", st, "-", why)
        return
    load_protocol()
    df = load_scores(a.stage)
    print(json.dumps(summarize(df) if len(df) else {"note": f"no {a.stage} scores yet"}, indent=1))


if __name__ == "__main__":
    main()

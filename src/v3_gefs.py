"""NOAA GEFS v12 precipitation forecasts aggregated over the v3 upstream-area boxes (Model C inputs).

Run: python -m src.v3_gefs --history [--workers 16]   # all eligible Monday 00Z cycles, resumable
     python -m src.v3_gefs --cycle 2026-11-16         # one cycle

Sources (public, NOAA Open Data Dissemination on AWS, no credentials)
- Reforecast: s3://noaa-gefs-retrospective/GEFSv12/reforecast/ - control member c00, 00Z, daily,
  2000-2019. RETROSPECTIVE: generated around 2019-2020 with the GEFSv12 model; never issued in
  real time on those dates. Labelled source="reforecast".
- Operational: s3://noaa-gefs-pds/gefs.YYYYMMDD/00/atmos/ - control member gec00, GEFSv12, from
  2020-09-24 onward. Labelled source="operational". Each file's S3 Last-Modified is recorded as
  the evidence of when it was published.
- 2020-01-01 to 2020-09-23 (GEFSv11 era, different model and layout) is not used.

Feature definition (fixed in advance)
  gefs_p_d1_7_mm  = basin-box mean of forecast precipitation accumulated over lead hours (0, 168]
  gefs_p_d8_16_mm = basin-box mean over lead hours (168, 384]
from 6-hour accumulation messages "(h-6)-h hour acc". All 64 messages must be present, else the
cycle is recorded as incomplete and no feature value is produced. No precipitation after day 16
is represented: GEFS daily 00Z reforecasts end at day 16, and nothing fills days 17-30.
Only the control member is used (ensemble mean and spread would multiply the download fivefold).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import requests

from src.v3_basins import STATIONS, basin_boxes, box_mask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v3" / "gefs_basin_precip.csv"
REFORECAST = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OPERATIONAL = "https://noaa-gefs-pds.s3.amazonaws.com"
REFORECAST_END = date(2019, 12, 31)
OPERATIONAL_V12_START = date(2020, 9, 24)        # first full 00Z cycle after the GEFSv12 implementation
LEADS = list(range(6, 385, 6))                   # 64 six-hour accumulations
D1_7 = 168
COLUMNS = ["init_date", "source", "station_id", "gefs_p_d1_7_mm", "gefs_p_d8_16_mm", "complete", "n_messages",
           "bytes_sha256", "published_utc_max", "retrieved_utc", "note"]
_session_local = threading.local()


def session():
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "WatershedShift-research-script/0.3 (python-requests)"
        _session_local.s = s
    return s


def get(url, headers=None, tries=4, timeout=60):
    last = None
    for a in range(tries):
        try:
            r = session().get(url, headers=headers or {}, timeout=timeout)
            if r.status_code == 404:
                return r
            if r.ok:
                return r
            last = r.status_code
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
        time.sleep(2 * 2 ** a)
    raise RuntimeError(f"GET failed {url}: {last!r}")


def source_for(d):
    if d <= REFORECAST_END:
        return "reforecast"
    if d >= OPERATIONAL_V12_START:
        return "operational"
    return None


def parse_idx(text):
    """[(offset, end_or_None, lead_start, lead_end, line)] for APCP accumulation messages."""
    lines = [l for l in text.splitlines() if l.strip()]
    offs = [int(l.split(":")[1]) for l in lines]
    out = []
    for i, l in enumerate(lines):
        f = l.split(":")
        if len(f) < 6 or f[3] != "APCP" or "hour acc" not in f[5]:
            continue
        a, b = map(int, f[5].split(" ")[0].split("-"))
        out.append((offs[i], offs[i + 1] - 1 if i + 1 < len(offs) else None, a, b, l))
    return out


def message_plan(d):
    """List of (url, lead_end) message requests for the cycle; byte ranges resolved from .idx files."""
    ymd = d.strftime("%Y%m%d")
    plan = []
    if source_for(d) == "reforecast":
        for folder in ("Days:1-10", "Days:10-16"):
            url = f"{REFORECAST}/{d.year}/{ymd}00/c00/{folder}/apcp_sfc_{ymd}00_c00.grib2"
            r = get(url + ".idx")
            if r.status_code == 404:
                return None, f"missing {folder} index"
            for off, end, a, b, _ in parse_idx(r.text):
                if b - a == 6 and b in LEADS:
                    plan.append((url, off, end, b))
    else:
        for h in LEADS:
            sub, res = ("pgrb2sp25", "0p25") if h <= 240 else ("pgrb2ap5", "0p50")
            pref = "pgrb2s" if sub == "pgrb2sp25" else "pgrb2a"
            url = f"{OPERATIONAL}/gefs.{ymd}/00/atmos/{sub}/gec00.t00z.{pref}.{res}.f{h:03d}"
            r = get(url + ".idx")
            if r.status_code == 404:
                return None, f"missing lead {h}"
            hit = [(off, end, a, b) for off, end, a, b, _ in parse_idx(r.text) if b == h and a == h - 6]
            if not hit:
                return None, f"no 6-hour APCP message at lead {h}"
            off, end, a, b = hit[0]
            plan.append((url, off, end, b))
    leads = sorted(p[3] for p in plan)
    if leads != LEADS:
        return None, f"incomplete leads ({len(leads)} of {len(LEADS)})"
    return plan, ""


_masks = {}


def decode_basin_means(msg, boxes):
    import eccodes
    h = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        la1, lo1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        di, dj = eccodes.codes_get(h, "iDirectionIncrementInDegrees"), eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        jpos = eccodes.codes_get(h, "jScansPositively")
        units = eccodes.codes_get(h, "units")
        vals = eccodes.codes_get_values(h).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    if units not in ("kg m**-2", "kg m-2"):
        raise ValueError(f"unexpected APCP units {units}")
    key = (ni, nj, la1, lo1, di, dj, jpos)
    if key not in _masks:
        lats = la1 + (dj if jpos else -dj) * np.arange(nj)
        lons = (lo1 + di * np.arange(ni)) % 360
        _masks[key] = {s: box_mask(lats, lons, b) for s, b in boxes.items()}
    return {s: float(vals[m].mean()) for s, m in _masks[key].items()}


def published_times(urls):
    """S3 Last-Modified of each distinct file (evidence of availability), operational only."""
    out = []
    for u in sorted(set(urls)):
        r = session().head(u, timeout=60)
        if r.ok and "Last-Modified" in r.headers:
            out.append(datetime.strptime(r.headers["Last-Modified"], "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc))
    return out


def extract_cycle(d, boxes=None):
    boxes = boxes or basin_boxes()
    src = source_for(d)
    base = {"init_date": d.isoformat(), "source": src, "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if src is None:
        return [{**base, "station_id": s, "complete": False, "note": "no GEFSv12 product for this date"} for s in STATIONS]
    plan, why = message_plan(d)
    if plan is None:
        return [{**base, "station_id": s, "complete": False, "note": why} for s in STATIONS]
    sums = {s: [0.0, 0.0] for s in STATIONS}
    hsh = hashlib.sha256()
    for url, off, end, lead in sorted(plan, key=lambda p: p[3]):
        msg = get(url, headers={"Range": f"bytes={off}-{'' if end is None else end}"}).content
        hsh.update(msg)
        for s, v in decode_basin_means(msg, boxes).items():
            sums[s][0 if lead <= D1_7 else 1] += v
    pub = published_times([p[0] for p in plan]) if src == "operational" else []
    pub_max = max(pub).strftime("%Y-%m-%dT%H:%M:%SZ") if pub else ""
    note = ("retrospective reforecast, not issued in real time" if src == "reforecast" else
            f"operational; {len(pub)} files, latest published {pub_max}")
    return [{**base, "station_id": s, "gefs_p_d1_7_mm": sums[s][0], "gefs_p_d8_16_mm": sums[s][1], "complete": True,
             "n_messages": len(plan), "bytes_sha256": hsh.hexdigest(), "published_utc_max": pub_max, "note": note}
            for s in STATIONS]


def history_dates(end=None):
    end = end or date.today()
    d, out = date(2001, 1, 1), []
    d += timedelta(days=(7 - d.weekday()) % 7)
    while d <= end:
        if source_for(d):
            out.append(d)
        d += timedelta(days=7)
    return out


def run_history(workers=16, end=None):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev.loc[prev.complete.astype(bool), "init_date"])
    todo = [d for d in history_dates(end) if d.isoformat() not in done]
    print(f"{len(todo)} cycles to fetch ({len(done)} already complete)", flush=True)
    boxes = basin_boxes()
    lock = threading.Lock()
    t0 = time.time()
    n = [0]

    def work(d):
        try:
            rows = extract_cycle(d, boxes)
        except Exception as exc:  # noqa: BLE001  recorded, never filled in
            rows = [{"init_date": d.isoformat(), "source": source_for(d), "station_id": s, "complete": False,
                     "note": f"error: {type(exc).__name__}: {exc}"[:300]} for s in STATIONS]
        with lock:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            n[0] += 1
            if n[0] % 20 == 0:
                print(f"{n[0]}/{len(todo)} cycles, {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    finalize()


def finalize():
    """Keep the last complete record per (init_date, station); sort; write deterministically."""
    df = pd.read_csv(OUT)
    df["complete"] = df.complete.astype(bool)
    df = df.sort_values(["init_date", "station_id", "complete"]).drop_duplicates(["init_date", "station_id"], keep="last")
    df.sort_values(["init_date", "station_id"]).to_csv(OUT, index=False)
    print(df.groupby(["source", "complete"]).size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--cycle", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cycle:
        print(json.dumps(extract_cycle(date.fromisoformat(a.cycle)), indent=1))
    if a.history:
        run_history(a.workers)


if __name__ == "__main__":
    main()

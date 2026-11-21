"""Download raw CDEC daily reservoir storage and NASA POWER daily weather.

Run: python -m src.fetch --start 2018-01-01 --end 2025-12-31 [--stations SHA ORO]

Network step ONLY. Responses are saved byte-for-byte (unmodified) under
data/raw/ together with a fetch log (URL, UTC time, HTTP status, size, SHA-256).
Parsing/cleaning happens offline in `python -m src.build_dataset`, so the
processed dataset can be rebuilt from raw files without re-downloading.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CDEC = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
CDEC_HEADER = "STATION_ID,DURATION,SENSOR_NUMBER,SENSOR_TYPE,DATE TIME,OBS DATE,VALUE,DATA_FLAG,UNITS"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "WatershedShift-research-script/0.2 (python-requests)"})


def _log(entry):
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "fetch_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def get(url, params, valid, tries=5, timeout=180):
    """GET with retries. `valid(response)` must be True to accept (CDEC sometimes
    returns an empty or header-only body transiently)."""
    last = None
    for attempt in range(tries):
        try:
            res = SESSION.get(url, params=params, timeout=timeout)
            last = res
            if res.ok and valid(res):
                return res
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"giving up on {url} {params}: {last!r}")


def save(path, res):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(res.content)
    _log({"file": str(path.relative_to(ROOT)), "url": res.url, "status": res.status_code,
          "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "bytes": len(res.content), "sha256": hashlib.sha256(res.content).hexdigest()})


def fetch_cdec(station, start, end):
    """One request per calendar year (smaller requests are far more reliable on CDEC)."""
    for year in range(int(start[:4]), int(end[:4]) + 1):
        s = max(start, f"{year}-01-01")
        e = min(end, f"{year}-12-31")
        path = RAW / "cdec" / f"{station}_sensor15_D_{year}.csv"
        if path.exists() and path.stat().st_size > len(CDEC_HEADER) + 10:
            print(f"  cdec {station} {year}: cached")
            continue
        params = {"Stations": station, "SensorNums": 15, "dur_code": "D", "Start": s, "End": e}
        res = get(CDEC, params, lambda r: r.text.startswith("STATION_ID") and r.text.count("\n") > 1)
        save(path, res)
        print(f"  cdec {station} {year}: {res.text.count(chr(10)) - 1} rows")


def fetch_power(station, lat, lon, start, end):
    path = RAW / "power" / f"{station}_daily_{start}_{end}.json"
    if path.exists():
        print(f"  power {station}: cached")
        return
    params = {"parameters": "T2M,PRECTOTCORR,RH2M", "community": "AG",
              "longitude": lon, "latitude": lat,
              "start": start.replace("-", ""), "end": end.replace("-", ""),
              "format": "JSON", "time-standard": "LST"}
    res = get(POWER, params, lambda r: "properties" in r.json())
    save(path, res)
    print(f"  power {station}: {len(res.json()['properties']['parameter']['T2M'])} days")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--stations", nargs="*", default=None)
    args = parser.parse_args()
    stations = pd.read_csv(ROOT / "stations.csv")
    if args.stations:
        stations = stations.loc[stations.station_id.isin(args.stations)]
    failed = []
    for st in stations.itertuples():
        print(f"Fetching {st.station_id} ...", flush=True)
        try:
            fetch_cdec(st.station_id, args.start, args.end)
            fetch_power(st.station_id, st.lat, st.lon, args.start, args.end)
        except Exception as exc:  # keep going; failures are reported, never filled in
            print(f"  FAILED: {exc}")
            failed.append(st.station_id)
    print("Done." + (f" FAILED stations: {failed}" if failed else " All stations fetched."))


if __name__ == "__main__":
    main()

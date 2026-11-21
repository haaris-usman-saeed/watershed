"""Download raw CDEC daily snow water content (sensor 82, adjusted) for candidate snow pillows.

Run: python -m src.fetch_snow --start 2001-01-01 --end 2025-12-31
Candidate stations (data/snow_stations.csv) are every CDEC station listed with daily
sensor 82 in the river basin(s) draining to each reservoir (CDEC staSearch, 2026-11-21).
One request per calendar year for all stations; responses are saved byte-for-byte to
data/raw/snow/ and logged in data/raw/fetch_log.jsonl.
"""
import argparse

import pandas as pd

from src.fetch import CDEC, RAW, ROOT, get, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2001-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()
    ids = pd.read_csv(ROOT / "data" / "snow_stations.csv").snow_station_id.tolist()
    for year in range(int(args.start[:4]), int(args.end[:4]) + 1):
        path = RAW / "snow" / f"snow_sensor82_D_{year}.csv"
        if path.exists() and path.stat().st_size > 1000:
            print(f"  snow {year}: cached")
            continue
        params = {"Stations": ",".join(ids), "SensorNums": 82, "dur_code": "D",
                  "Start": max(args.start, f"{year}-01-01"), "End": min(args.end, f"{year}-12-31")}
        res = get(CDEC, params, lambda r: r.text.startswith("STATION_ID") and r.text.count("\n") > 1)
        save(path, res)
        print(f"  snow {year}: {res.text.count(chr(10)) - 1} rows")


if __name__ == "__main__":
    main()

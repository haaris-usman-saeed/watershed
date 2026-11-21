"""Download historical reservoir storage and NASA POWER weather.

Run: python -m src.fetch --start 2018-01-01 --end 2025-12-31
Requires internet. All CDEC response headers, station metadata and capacities
MUST be verified against official source records before reporting study findings.
"""
import argparse
from io import StringIO
from pathlib import Path
import time
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CDEC = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "WatershedShiftAcademicResearch/0.1"})


def request(url, params):
    for attempt in range(4):
        try:
            res = SESSION.get(url, params=params, timeout=90)
            res.raise_for_status()
            return res
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def cdec_storage(station, start, end):
    params = {"Stations": station, "SensorNums": 15, "dur_code": "D", "Start": start, "End": end}
    r = request(CDEC, params)
    if "<html" in r.text[:300].lower():
        raise ValueError(f"CDEC returned HTML rather than CSV for {station}: {r.url}")
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"cdec_{station}.csv").write_text(r.text, encoding="utf-8")
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    df.columns = [str(c).strip().upper() for c in df.columns]
    date_col = next((c for c in ["OBS DATE", "DATE TIME", "DATE", "OBS_DATE"] if c in df), None)
    val_col = next((c for c in ["VALUE", "OBS_VALUE", "STORAGE"] if c in df), None)
    if not (date_col and val_col):
        raise ValueError(f"Unexpected CDEC fields for {station}: {df.columns.tolist()}")
    dt = pd.to_datetime(df[date_col], errors="coerce")
    output = pd.DataFrame({"date": dt.dt.normalize(), "storage_af": pd.to_numeric(df[val_col], errors="coerce")})
    output.loc[(output.storage_af < 0) | (~output.storage_af.notna()), "storage_af"] = float("nan")
    output = output.dropna(subset=["date", "storage_af"])
    return output.groupby("date", as_index=False)["storage_af"].last()


def nasa_weather(station, lat, lon, start, end):
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG", "longitude": lon, "latitude": lat,
        "start": start.replace("-", ""), "end": end.replace("-", ""),
        "format": "JSON", "time-standard": "LST"
    }
    result = request(POWER, params).json()
    (ROOT / "data" / "raw" / f"nasa_{station}.json").write_text(__import__('json').dumps(result), encoding="utf-8")
    data = result["properties"]["parameter"]
    frame = pd.DataFrame({key: pd.Series(vals) for key, vals in data.items()})
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d", errors="coerce")
    frame.index.name = "date"
    frame = frame.rename(columns={"T2M": "temp_c", "PRECTOTCORR": "rain_mm", "RH2M": "humidity_pct"})
    frame = frame.reset_index()
    for c in ["temp_c", "rain_mm", "humidity_pct"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
        frame.loc[frame[c] <= -900, c] = float("nan")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--stations", nargs="*", default=None)
    args = parser.parse_args()
    stations = pd.read_csv(ROOT / "stations.csv")
    if args.stations:
        stations = stations.loc[stations.station_id.isin(args.stations)]
    collected = []
    for station in stations.itertuples():
        print(f"Fetching {station.station_id} ...", flush=True)
        try:
            storage = cdec_storage(station.station_id, args.start, args.end)
            weather = nasa_weather(station.station_id, station.lat, station.lon, args.start, args.end)
            merged = storage.merge(weather, on="date", how="left", validate="one_to_one")
            merged["station_id"] = station.station_id
            merged["reservoir_name"] = station.reservoir_name
            merged["capacity_af"] = station.capacity_af
            merged["lat"] = station.lat
            merged["lon"] = station.lon
            collected.append(merged)
            print(f"  {len(merged)} storage days; weather missing days={merged.temp_c.isna().sum()}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
        time.sleep(1)
    if not collected:
        raise RuntimeError("No stations downloaded. Check network, station IDs and sensor availability.")
    result = pd.concat(collected, ignore_index=True).sort_values(["station_id", "date"])
    destination = ROOT / "data" / "processed" / "daily_joined.csv"
    destination.parent.mkdir(exist_ok=True, parents=True)
    result.to_csv(destination, index=False)
    print(f"Saved {len(result)} rows to {destination}")

if __name__ == "__main__":
    main()

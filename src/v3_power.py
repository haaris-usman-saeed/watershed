"""NASA POWER daily weather for v3: dam-point series (Model B) and upstream-box means (Model B2).

Run: python -m src.v3_power --end 2026-11-21 [--workers 4]

- Dam point: the same request as src.fetch.fetch_power (T2M, PRECTOTCORR, RH2M, community AG, LST),
  for 2001-01-01 to --end. Downloaded fresh so 2026 values are included.
- Upstream box (B2, separate v3 experiment): every POWER grid cell centre (MERRA-2 grid, 0.5 deg
  latitude by 0.625 deg longitude) inside the reservoir's v3 box (src/v3_basins.py); the box mean is
  the unweighted mean over cells reporting that day. Rule fixed in advance.
Raw JSON responses go to data/v3/power/ (NASA POWER: redistribution permitted with citation);
daily series to data/v3/public/power_daily.csv with a source column.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.fetch import POWER, get
from src.v3_basins import STATIONS, basin_boxes

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "v3" / "power"
PUBLIC = ROOT / "data" / "v3" / "public"
START = "2001-01-01"
LAT_STEP, LON_STEP = 0.5, 0.625
PLAUSIBLE = {"temp_c": (-40, 55), "rain_mm": (0, 500), "humidity_pct": (0, 100)}   # same as build_dataset


def grid_cells(box):
    lats = np.arange(-90, 90.01, LAT_STEP)
    lons = np.arange(-180, 180, LON_STEP)
    la = lats[(lats >= box["lat_min"]) & (lats <= box["lat_max"])]
    lo = lons[(lons >= box["lon_min"]) & (lons <= box["lon_max"])]
    return [(float(a), float(b)) for a in la for b in lo]


def fetch_point(key, lat, lon, end, log):
    path = RAW / f"{key}_{START}_{end}.json"
    if not path.exists():
        params = {"parameters": "T2M,PRECTOTCORR,RH2M", "community": "AG", "longitude": lon, "latitude": lat,
                  "start": START.replace("-", ""), "end": end.replace("-", ""), "format": "JSON", "time-standard": "LST"}
        res = get(POWER, params, lambda r: "properties" in r.json(), tries=5, timeout=300)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(res.content)
        log.append({"file": path.name, "url": res.url, "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sha256": hashlib.sha256(res.content).hexdigest(), "bytes": len(res.content)})
    obj = json.loads(path.read_text())
    par, fill = obj["properties"]["parameter"], obj["header"].get("fill_value", -999.0)
    w = pd.DataFrame({"temp_c": par["T2M"], "rain_mm": par["PRECTOTCORR"], "humidity_pct": par["RH2M"]})
    w.index = pd.to_datetime(w.index, format="%Y%m%d")
    w = w.mask(w <= fill + 1e-9)
    for c, (lo, hi) in PLAUSIBLE.items():
        w.loc[(w[c] < lo) | (w[c] > hi), c] = np.nan
    return w, obj["header"]["api"]["version"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    st = pd.read_csv(ROOT / "stations.csv").set_index("station_id")
    boxes = basin_boxes()
    jobs = [(s, "dam", f"{s}_dam", float(st.loc[s, "lat"]), float(st.loc[s, "lon"])) for s in STATIONS]
    for s in STATIONS:
        for la, lo in grid_cells(boxes[s]):
            jobs.append((s, "box", f"cell_{la:+.3f}_{lo:+.3f}", la, lo))
    log = []
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: fetch_point(j[2], j[3], j[4], a.end, log), jobs))
    frames, versions = [], set()
    for (s, kind, key, la, lo), (w, ver) in zip(jobs, res):
        versions.add(ver)
        frames.append(w.assign(station_id=s, kind=kind, cell=key))
    allw = pd.concat(frames).rename_axis("date").reset_index()
    dam = allw[allw.kind == "dam"].drop(columns=["kind", "cell"]).assign(source="dam_point")
    box = allw[allw.kind == "box"].groupby(["station_id", "date"])[["temp_c", "rain_mm", "humidity_pct"]].mean().reset_index()
    ncell = allw[allw.kind == "box"].groupby("station_id").cell.nunique()
    box = box.assign(source="upstream_box_mean")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    out = pd.concat([dam, box])[["station_id", "source", "date", "rain_mm", "temp_c", "humidity_pct"]]
    out.sort_values(["station_id", "source", "date"]).to_csv(PUBLIC / "power_daily.csv", index=False, date_format="%Y-%m-%d",
                                                            float_format="%.4f")
    meta = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "period": [START, a.end],
            "power_api_versions": sorted(versions), "cells_per_box": ncell.to_dict(), "boxes": boxes,
            "new_downloads": log}
    (RAW / "power_manifest.json").write_text(json.dumps(meta, indent=1))
    print("cells per box:", ncell.to_dict(), "rows:", len(out), "api:", versions)


if __name__ == "__main__":
    main()

"""v5: CDEC daily inflow (sensor 76) and outflow (sensor 23) for 2024-01-01 .. 2026-11-20, for retrospective testing.

Run: python -m src.v5.flows_ext --fetch      # network: 6 reservoirs x 2 sensors x 2 chunks

The 2001-2023 flows are the v4 download (data/v4/flows_daily.csv), unchanged. This module adds the later period
with the same parser and plausibility rules (src.v4.flows.parse_cdec / clean). Raw responses are stored byte-for-byte
with SHA-256 in data/v5/raw/ (git-ignored). Sensors 76 and 23 are not covered by config/data_release_policy.json, so
nothing derived from them is published; recent CDEC values are provisional and may be revised.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

from src.v4.flows import CDEC, CDEC_HEADER, SENSORS, clean, parse_cdec
from src.v5 import DATA, ROOT, STATIONS

RAW = DATA / "raw" / "cdec_flow"
MANIFEST = DATA / "raw" / "manifest.jsonl"
FLOWS_LATE = DATA / "flows_daily_2024_2026.csv"
CHUNKS = (("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-11-20"))


def fetch(get=None):
    if get is None:
        from src.fetch import get as _get

        def get(params):
            return _get(CDEC, params, lambda r: r.text.startswith(CDEC_HEADER))
    RAW.mkdir(parents=True, exist_ok=True)
    out = []
    for st in STATIONS:
        for name, sensor in SENSORS.items():
            for s, e in CHUNKS:
                p = RAW / f"{st}_s{sensor}_{s}_{e}.csv"
                if not p.exists():
                    r = get({"Stations": st, "SensorNums": sensor, "dur_code": "D", "Start": s, "End": e})
                    p.write_bytes(r.content)
                    with open(MANIFEST, "a") as fh:
                        fh.write(json.dumps({"file": str(p.relative_to(ROOT)), "url": r.url, "bytes": len(r.content),
                                             "sha256": hashlib.sha256(r.content).hexdigest(),
                                             "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                                 + "\n")
                out.append(p)
    return out


def build_late(raw_dir=RAW):
    frames = []
    for p in sorted(raw_dir.glob("*.csv")):
        st, sensor = p.stem.split("_")[:2]
        name = {f"s{v}": k for k, v in SENSORS.items()}[sensor]
        frames.append(clean(parse_cdec(p.read_text()), name).assign(station_id=st, variable=name))
    long = pd.concat(frames, ignore_index=True).drop_duplicates(["station_id", "variable", "date"], keep="last")
    wide = long.pivot_table(index=["station_id", "date"], columns="variable", values="value", aggfunc="first").reset_index()
    wide = wide[(wide.date >= "2024-01-01") & (wide.date <= CHUNKS[-1][1])]
    return wide[["station_id", "date"] + list(SENSORS)].sort_values(["station_id", "date"])


def load_all_flows():
    """2001-2023 (v4 download) plus 2024-2026 (this module), one row per station-day observed."""
    early = pd.read_csv(ROOT / "data" / "v4" / "flows_daily.csv", parse_dates=["date"])
    parts = [early]
    if FLOWS_LATE.exists():
        parts.append(pd.read_csv(FLOWS_LATE, parse_dates=["date"]))
    f = pd.concat(parts, ignore_index=True)
    assert not f.duplicated(["station_id", "date"]).any()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        paths = fetch()
        late = build_late()
        late.to_csv(FLOWS_LATE, index=False, date_format="%Y-%m-%d")
        cov = late.groupby("station_id")[list(SENSORS)].apply(lambda g: g.notna().mean().round(3))
        print(f"{len(paths)} raw files; {len(late)} station-days -> {FLOWS_LATE.relative_to(ROOT)}\n{cov}")


if __name__ == "__main__":
    main()

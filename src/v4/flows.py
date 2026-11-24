"""v4: CDEC daily reservoir inflow (sensor 76) and outflow (sensor 23), and a water-balance audit.

Run: python -m src.v4.flows --fetch     # network: 6 reservoirs x 2 sensors, 2001-01-01 to 2023-12-31
     python -m src.v4.flows --audit     # coverage and closure -> results/v4/flow_audit.json

Development period only. Raw responses are stored byte-for-byte with SHA-256 in data/v4/raw/ (git-ignored),
and logged in data/v4/raw/manifest.jsonl (not the tracked v2 manifest). Sensors 76 and 23 are not covered by
config/data_release_policy.json, so nothing derived from them is published.

Water balance: storage change S(t) - S(t-1) is compared with (inflow - outflow) x 1.983471 AF per cfs-day.
CDEC inflow at many reservoirs is itself computed by the operator from storage change and releases, so a
close daily balance does not show that inflow is independently measured (docs/v4/FLOW_DATA.md).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v4 import DEV_END, STATIONS

ROOT = Path(__file__).resolve().parents[2]
CDEC = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
CDEC_HEADER = "STATION_ID,DURATION,SENSOR_NUMBER,SENSOR_TYPE,DATE TIME,OBS DATE,VALUE,DATA_FLAG,UNITS"
CFS_DAY_AF = 1.983471                       # acre-feet in one cfs-day
SENSORS = {"inflow_cfs": 76, "outflow_cfs": 23}
FETCH_START, FETCH_END = "2001-01-01", "2023-12-31"
RAW = ROOT / "data" / "v4" / "raw" / "cdec_flow"
MANIFEST = ROOT / "data" / "v4" / "raw" / "manifest.jsonl"
FLOWS = ROOT / "data" / "v4" / "flows_daily.csv"
AUDIT = ROOT / "results" / "v4" / "flow_audit.json"
MAX_CFS = 1_000_000.0                        # far above any recorded flow at these dams


def _chunks(start, end, years=5):
    s = pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(pd.Timestamp(year=s.year + years, month=1, day=1) - pd.Timedelta(days=1), pd.Timestamp(end))
        yield s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")
        s = e + pd.Timedelta(days=1)


def fetch(stations=STATIONS, start=FETCH_START, end=FETCH_END, get=None):
    """Download every station x sensor x 5-year chunk; skip files already present. Returns paths."""
    if pd.Timestamp(end) >= DEV_END:
        raise ValueError("v4 development fetches stop before 2024-01-01")
    if get is None:
        from src.fetch import get as _get

        def get(params):
            return _get(CDEC, params, lambda r: r.text.startswith(CDEC_HEADER))
    RAW.mkdir(parents=True, exist_ok=True)
    out = []
    for st in stations:
        for name, sensor in SENSORS.items():
            for s, e in _chunks(start, end):
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


def parse_cdec(text):
    """CDEC CSVDataServlet daily file -> date, value (NaN when missing). Date = DATE TIME (end of day)."""
    d = pd.read_csv(io.StringIO(text))
    if d.empty:
        return pd.DataFrame(columns=["date", "value"])
    date = pd.to_datetime(d["DATE TIME"].astype(str), format="%Y%m%d %H%M").dt.normalize()
    value = pd.to_numeric(d["VALUE"], errors="coerce")
    return pd.DataFrame({"date": date, "value": value})


def clean(df, name):
    """Physical plausibility only: outflow cannot be negative; |flow| above MAX_CFS is an error.
    Negative computed inflow (evaporation exceeding inflow) is kept: it is what the operator reports."""
    v = df["value"].where(df["value"].abs() <= MAX_CFS)
    if name == "outflow_cfs":
        v = v.where(v >= 0)
    return df.assign(value=v)


def build_flows(raw_dir=RAW):
    frames = []
    for p in sorted(Path(raw_dir).glob("*.csv")):
        st, sensor = p.stem.split("_")[:2]
        name = {f"s{v}": k for k, v in SENSORS.items()}[sensor]
        d = clean(parse_cdec(p.read_text()), name)
        frames.append(d.assign(station_id=st, variable=name))
    if not frames:
        raise FileNotFoundError(f"no raw flow files in {raw_dir}; run python -m src.v4.flows --fetch")
    long = pd.concat(frames, ignore_index=True).drop_duplicates(["station_id", "variable", "date"], keep="last")
    wide = long.pivot_table(index=["station_id", "date"], columns="variable", values="value", aggfunc="first")
    wide = wide.reset_index()
    wide = wide[wide.date < DEV_END]
    return wide[["station_id", "date"] + [c for c in SENSORS if c in wide]].sort_values(["station_id", "date"])


def audit(panel):
    """Coverage and water-balance closure per reservoir on the daily panel (dates < DEV_END)."""
    out = {"period": [str(panel.date.min().date()), str(panel.date.max().date())], "cfs_day_af": CFS_DAY_AF,
           "stations": {}}
    for st, g in panel.groupby("station_id"):
        g = g.set_index("date").sort_index()
        cap = g.capacity_af
        net = (g.inflow_cfs - g.outflow_cfs) * CFS_DAY_AF
        ds = g.storage_af.diff()
        same = g.seg == g.seg.shift(1)
        r = (ds - net).where(same)
        ok = r.notna()
        s30 = (g.storage_af.shift(-30) - g.storage_af).where(g.seg == g.seg.shift(-30))
        net30 = net[::-1].rolling(30, min_periods=30).sum()[::-1].shift(-1)     # sum over days t+1 .. t+30
        y = 100 * s30 / cap
        f = 100 * net30 / cap
        both = y.notna() & f.notna()
        out["stations"][st] = {
            "inflow_coverage": round(float(g.inflow_cfs.notna().mean()), 4),
            "outflow_coverage": round(float(g.outflow_cfs.notna().mean()), 4),
            "negative_inflow_share": round(float((g.inflow_cfs < 0).mean()), 4),
            "daily_residual_median_abs_pct_capacity": round(float((100 * r[ok].abs() / cap[ok]).median()), 4),
            "daily_residual_p95_abs_pct_capacity": round(float((100 * r[ok].abs() / cap[ok]).quantile(0.95)), 4),
            "daily_residual_mean_pct_capacity": round(float((100 * r[ok] / cap[ok]).mean()), 5),
            "days_compared": int(ok.sum()),
            "thirty_day_mae_pp_flow_sum_vs_storage": round(float(np.abs(y[both] - f[both]).mean()), 4),
            "thirty_day_bias_pp": round(float((f[both] - y[both]).mean()), 4),
            "thirty_day_corr": round(float(np.corrcoef(y[both], f[both])[0, 1]), 5),
            "thirty_day_windows": int(both.sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        paths = fetch()
        flows = build_flows()
        FLOWS.parent.mkdir(parents=True, exist_ok=True)
        flows.to_csv(FLOWS, index=False, date_format="%Y-%m-%d")
        print(f"{len(paths)} raw files; {len(flows)} station-days -> {FLOWS.relative_to(ROOT)}")
    if a.audit:
        from src.v4.dataset import load_panel
        res = audit(load_panel())
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

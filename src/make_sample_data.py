"""Generate SYNTHETIC UI fixtures that follow the output contract exactly.

THESE ARE NOT REAL OBSERVATIONS OR PREDICTIONS. Station IDs (SYN1..SYN3),
names ("Synthetic Reservoir ..."), coordinates and every value are invented
so the frontend can be built before the real pipeline finishes.

Run: python -m src.make_sample_data   (writes app/sample_data/)
"""
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.contract import SCHEMA_VERSION, validate_dir

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "sample_data"
RNG = np.random.default_rng(20261121)
STATIONS = [
    ("SYN1", "Synthetic Reservoir A (NOT REAL)", 40.0, -122.0, 4_000_000),
    ("SYN2", "Synthetic Reservoir B (NOT REAL)", 39.0, -121.0, 3_000_000),
    ("SYN3", "Synthetic Reservoir C (NOT REAL)", 38.0, -120.5, 1_000_000),
]
START, END = "2022-01-01", "2025-12-31"
SYN_NOTE = "SYNTHETIC SAMPLE DATA for UI development only. Not real observations, predictions or accuracy."


def history():
    days = pd.date_range(START, END, freq="D")
    doy = days.dayofyear.to_numpy()
    rows = []
    for sid, _, _, _, cap in STATIONS:
        pct = 60 + 20 * np.sin(2 * np.pi * (doy - 60) / 365.25) + np.cumsum(RNG.normal(0, 0.15, len(days)))
        pct = np.clip(pct, 15, 98)
        storage = np.round(pct / 100 * cap)
        wet = np.isin(days.month, [11, 12, 1, 2, 3, 4])
        rain = np.where(RNG.random(len(days)) < np.where(wet, 0.35, 0.05), RNG.gamma(1.2, 8, len(days)), 0.0)
        temp = 15 - 9 * np.cos(2 * np.pi * (doy - 15) / 365.25) + RNG.normal(0, 2, len(days))
        hum = np.clip(60 + 15 * np.cos(2 * np.pi * (doy - 15) / 365.25) + RNG.normal(0, 6, len(days)), 5, 100)
        df = pd.DataFrame({"station_id": sid, "date": days, "storage_af": storage,
                           "storage_pct": np.round(100 * storage / cap, 3), "rain_mm": np.round(rain, 2),
                           "temp_c": np.round(temp, 2), "humidity_pct": np.round(hum, 2)})
        gap = RNG.choice(len(df), size=25, replace=False)  # missing days are blank, never interpolated
        df.loc[gap, ["storage_af", "storage_pct"]] = np.nan
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def predictions(hist):
    out = []
    for sid, name, _, _, cap in STATIONS:
        h = hist[hist.station_id == sid].set_index("date")
        for d in pd.date_range("2024-01-01", "2025-12-01", freq="W-MON"):
            t = d + pd.Timedelta(days=30)
            if t > pd.Timestamp(END):
                continue
            s0, s1, sm30 = h.storage_af.get(d), h.storage_af.get(t), h.storage_af.get(d - pd.Timedelta(days=30))
            if any(pd.isna(v) for v in (s0, s1, sm30)):
                continue
            y = 100 * (s1 - s0) / cap
            pred = y + RNG.normal(0, 2.0)
            out.append({"station_id": sid, "reservoir_name": name, "date": d, "target_date": t,
                        "target_delta_pp": round(y, 3), "pred_delta_pp": round(pred, 3),
                        "lower_90_pp": round(pred - 3.5, 3), "upper_90_pp": round(pred + 3.5, 3),
                        "baseline_persistence_pp": 0.0,
                        "baseline_trend_pp": round(100 * (s0 - sm30) / cap, 3),
                        "baseline_seasonal_pp": round(y * 0.6 + RNG.normal(0, 2.5), 3),
                        "storage_af": s0, "capacity_af": cap, "storage_pct": round(100 * s0 / cap, 3)})
    return pd.DataFrame(out)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def metrics(p):
    cols = {"ML": "pred_delta_pp", "Persistence": "baseline_persistence_pp",
            "30-day trend": "baseline_trend_pp", "Historical seasonal": "baseline_seasonal_pp"}
    cover = lambda d: float(((d.target_delta_pp >= d.lower_90_pp) & (d.target_delta_pp <= d.upper_90_pp)).mean())
    m = {"schema_version": SCHEMA_VERSION, "data_mode": "SYNTHETIC_SAMPLE",
         "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    m.update({k: mae(p.target_delta_pp, p[c]) for k, c in cols.items()})
    m.update({"interval_90_coverage": cover(p), "interval_90_halfwidth_pp": 3.5,
              "train_count": 600, "val_count": 150, "test_count": int(len(p)),
              "test_station_count": int(p.station_id.nunique()), "note": SYN_NOTE,
              "split": {"train_target_end_exclusive": "2023-01-01", "val_start": "2023-01-01",
                        "val_target_end_exclusive": "2024-01-01", "test_start": "2024-01-01",
                        "test_target_end_inclusive": "2025-12-31", "horizon_days": 30,
                        "weather_lag_days": 7, "sampling": "weekly (Mondays)"},
              "per_station": {}})
    for sid, d in p.groupby("station_id"):
        m["per_station"][sid] = {"n": int(len(d)), **{k: mae(d.target_delta_pp, d[c]) for k, c in cols.items()},
                                 "interval_90_coverage": cover(d)}
    best = min(["Persistence", "30-day trend", "Historical seasonal"], key=lambda k: m[k])
    diff = m["ML"] - m[best]
    m["bootstrap"] = {"method": "SYNTHETIC placeholder: moving-block bootstrap over weekly test dates",
                      "n_boot": 0, "block_weeks": 5, "best_baseline": best,
                      "mae_diff_ml_minus_best_baseline": {"estimate": diff, "ci95_low": diff - 0.5,
                                                          "ci95_high": diff + 0.5}}
    return m


FEATURES = [
    ("storage_pct", "Storage at forecast origin", "% of capacity", "CDEC sensor 15", 0),
    ("prev7_pct", "Storage change over previous 7 days", "pp of capacity", "CDEC sensor 15", 0),
    ("prev30_pct", "Storage change over previous 30 days", "pp of capacity", "CDEC sensor 15", 0),
    ("prev60_pct", "Storage change over previous 60 days", "pp of capacity", "CDEC sensor 15", 0),
    ("rain_7d_mm", "Precipitation sum, 7 days ending at origin-7d", "mm", "NASA POWER PRECTOTCORR", 7),
    ("rain_30d_mm", "Precipitation sum, 30 days ending at origin-7d", "mm", "NASA POWER PRECTOTCORR", 7),
    ("temp_30d_c", "Mean 2 m air temperature, 30 days ending at origin-7d", "degC", "NASA POWER T2M", 7),
    ("humidity_30d_pct", "Mean 2 m relative humidity, 30 days ending at origin-7d", "%", "NASA POWER RH2M", 7),
    ("season_sin", "sin(2*pi*day_of_year/365.25)", "unitless", "calendar", 0),
    ("season_cos", "cos(2*pi*day_of_year/365.25)", "unitless", "calendar", 0),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist = history()
    pred = predictions(hist)
    st = pd.DataFrame([{"station_id": s, "reservoir_name": n, "lat": la, "lon": lo, "capacity_af": c,
                        "capacity_source": "SYNTHETIC", "status": "included", "first_date": START,
                        "last_date": END, "n_storage_days": int(hist[hist.station_id == s].storage_af.notna().sum()),
                        "missing_storage_pct": round(100 * hist[hist.station_id == s].storage_af.isna().mean(), 3)}
                       for s, n, la, lo, c in STATIONS])
    st.loc[len(st)] = ["SYN4", "Synthetic Reservoir D (NOT REAL, excluded example)", 37.5, -120.0, 2_000_000,
                       "SYNTHETIC", "excluded", START, END, 0, 100.0]
    st.to_csv(OUT / "stations.csv", index=False)
    hist.to_csv(OUT / "history_daily.csv", index=False, date_format="%Y-%m-%d")
    pred.to_csv(OUT / "test_predictions.csv", index=False, date_format="%Y-%m-%d")
    (OUT / "metrics.json").write_text(json.dumps(metrics(pred), indent=2))
    (OUT / "feature_catalog.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "features": [
        {"name": n, "description": d, "units": u, "source": s, "min_lag_days": l} for n, d, u, s, l in FEATURES]},
        indent=2))
    (OUT / "README.md").write_text(f"# SYNTHETIC UI FIXTURES — NOT REAL DATA\n\n{SYN_NOTE}\n\n"
                                   "Regenerate with `python -m src.make_sample_data`. Schema: docs/UI_CONTRACT.md.\n")
    errs = validate_dir(OUT)
    if errs:
        raise SystemExit("\n".join(errs))
    print(f"Wrote synthetic fixtures to {OUT}: history={len(hist)} rows, predictions={len(pred)} rows")


if __name__ == "__main__":
    main()

"""Versioned public data release, restricted to sources the release policy allows.

Run: python -m src.v3_release --version 2026-11-22.2 [--out public_data]
     python -m src.v3_release --check            # every published file allowed by policy and checksum-intact

Reads config/data_release_policy.json. A candidate is packaged only if every source it derives from has
publish_derived = true. Each release directory is write-once: an existing version is never changed; a
correction is a new version.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from src.v3_basins import PAD_DEG, basin_boxes

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "data_release_policy.json"
CDEC = ["cdec_storage_sensor15", "cdec_snow_sensor82"]
LIVE = ROOT / "data" / "live" / "2026-11-22"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ builders for CDEC-derived tables

def _storage(dst):
    from src.forecast_live import clean
    h = pd.read_csv(ROOT / "data/processed/daily_joined.csv")[["station_id", "date", "storage_af", "storage_pct",
                                                                "capacity_af", "cdec_flag"]]
    stations = sorted(h.station_id.unique())
    storage, _, _, _, cap_hist = clean(LIVE, "2025-01-01", LIVE.name, stations)
    from src import build_dataset as bd
    rows = []
    for s, df in storage.items():
        d = df[df.index >= "2026-01-01"].dropna(subset=["storage_af"])
        cap = bd.capacity_series(s, d.index, cap_hist).to_numpy()
        rows.append(pd.DataFrame({"station_id": s, "date": d.index.strftime("%Y-%m-%d"), "storage_af": d.storage_af.to_numpy(),
                                  "storage_pct": 100 * d.storage_af.to_numpy() / cap, "capacity_af": cap,
                                  "cdec_flag": d.cdec_flag.to_numpy()}))
    out = pd.concat([h] + rows, ignore_index=True).sort_values(["station_id", "date"])
    out.to_csv(dst, index=False, float_format="%.4f")


def _pillows(dst):
    from src.build_dataset import load_snow
    from src.forecast_live import raw_dir
    h = pd.read_csv(ROOT / "data/processed/snow_daily.csv")
    with raw_dir(LIVE):
        live, _ = load_snow("2026-01-01", LIVE.name)
    live = live[live.date >= "2026-01-01"].assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d"))
    out = pd.concat([h, live[h.columns]], ignore_index=True).sort_values(["snow_station_id", "date"])
    out.to_csv(dst, index=False)


def _copy(src):
    def f(dst):
        shutil.copyfile(ROOT / src, dst)
    return f


CANDIDATES = [
    {"path": "gefs_basin_precip.csv", "build": _copy("data/v3/gefs_basin_precip.csv"),
     "sources": ["noaa_gefs_reforecast", "noaa_gefs_operational"],
     "description": "GEFSv12 control-member precipitation forecast, 00Z Monday cycles, averaged over each reservoir's upstream "
                    "box: lead hours (0,168] and (168,384]. source = reforecast (retrospective, not issued in real time) or "
                    "operational (published_utc_max = latest S3 Last-Modified of the 64 files used)."},
    {"path": "power_daily.csv", "build": _copy("data/v3/public/power_daily.csv"), "sources": ["nasa_power"],
     "description": "NASA POWER daily weather (PRECTOTCORR, T2M, RH2M; LST) at each dam (dam_point) and averaged over POWER "
                    "grid-cell centres inside the upstream box (upstream_box_mean). Implausible values set to missing."},
    {"path": "cdec_daily_storage.csv", "build": _storage, "sources": ["cdec_storage_sensor15"],
     "description": "Cleaned CDEC daily end-of-day reservoir storage (sensor 15), 2001-01-01 to 2026-11-20. storage_pct uses the "
                    "capacity in force on each date (data/capacity_history.csv). Out-of-range values and unreviewed one-day "
                    "spikes are blank, never interpolated. cdec_flag: e = estimated, r = revised. 2026 values are provisional."},
    {"path": "cdec_snow_pillow_daily.csv", "build": _pillows, "sources": ["cdec_snow_sensor82"],
     "description": "Cleaned CDEC daily snow water content (sensor 82, adjusted), inches. 2001-2025: all 51 candidate pillows; "
                    "2026: the core pillows used by the models. Missing stays missing; small negatives (>= -0.5 in) set to 0."},
    {"path": "basin_snow_index.csv", "build": _copy("data/v3/private/snow_index_v3.csv"), "sources": ["cdec_snow_sensor82"],
     "description": "Daily basin SWE index: mean of core pillows, only when at least 2/3 report (src.features rules)."},
    {"path": "weekly_features_v3.csv", "build": _copy("data/v3/private/weekly_v3.csv"),
     "sources": CDEC + ["nasa_power", "noaa_gefs_reforecast", "noaa_gefs_operational"],
     "description": "v3 weekly Monday-origin table: 30-day storage-change target, storage, snow, weather (dam point and upstream "
                    "box, 7-day lag) and GEFS features (usable cycles only)."},
    {"path": "audits/coverage.csv", "build": _copy("data/processed/coverage.csv"), "sources": CDEC,
     "description": "Per-station storage coverage and cleaning audit (v2)."},
    {"path": "audits/station_year_audit.csv", "build": _copy("data/processed/station_year_audit.csv"), "sources": CDEC,
     "description": "Per station-year missing days and flags (v2)."},
    {"path": "audits/discontinuity_scan.csv", "build": _copy("data/processed/discontinuity_scan.csv"), "sources": CDEC,
     "description": "Dry-day level shifts flagged for review (v2)."},
    {"path": "research/v3_test_predictions_2026.csv", "build": _copy("data/v3/private/test_predictions_v3.csv"),
     "sources": CDEC + ["nasa_power", "noaa_gefs_operational"],
     "description": "The preregistered v3 2026 test: per-row predictions and intervals of A, B, B2, C, frozen A and baselines."},
    {"path": "research/v3_development_oof.csv", "build": _copy("data/v3/private/oof_v3.csv"),
     "sources": CDEC + ["nasa_power", "noaa_gefs_reforecast", "noaa_gefs_operational"],
     "description": "v3 development out-of-fold predictions, WY2008-2025 (used for model-specific intervals)."},
    *[{"path": f"ui_outputs/{f}", "build": _copy(f"outputs/{f}"), "sources": CDEC + ["nasa_power"],
       "description": "UI contract file for the website's backtest pages (docs/UI_CONTRACT.md)."}
      for f in ("stations.csv", "history_daily.csv", "test_predictions.csv", "metrics.json", "feature_catalog.json")],
    {"path": "ui_outputs/forecast_snapshot.json",
     "build": _copy("forecasts/records/wsf-v2-2026-11-22-20261122T052508Z.json"), "sources": CDEC,
     "description": "The first issued Model A forecast (2026-11-22), shown in the website's Issued record tab. Unmodified copy."},
    {"path": "issued_forecasts/wsf-v2-2026-11-22-20261122T052508Z.json",
     "build": _copy("forecasts/records/wsf-v2-2026-11-22-20261122T052508Z.json"), "sources": CDEC,
     "description": "Issued Model A forecast record, unmodified (SHA-256 in docs/v3/issued_forecast_ledger.jsonl)."},
    {"path": "issued_forecasts/wsf-v3-ORO-20261122T064412Z.json",
     "build": _copy("forecasts/records_v3/wsf-v3-ORO-20261122T064412Z.json"),
     "sources": CDEC + ["nasa_power", "noaa_gefs_operational"],
     "description": "First interactive v3 forecast (Oroville, Models A-C), unmodified."},
]


def allowed(sources, policy):
    return all(policy["sources"].get(s, {}).get("publish_derived") for s in sources)


def build_release(out_root, version, policy_path=POLICY):
    policy = json.loads(Path(policy_path).read_text())
    out = Path(out_root) / f"watershedshift-public-data-{version}"
    if out.exists():
        raise FileExistsError(f"{out} exists; published releases are never changed. Use a new version.")
    out.mkdir(parents=True)
    files, excluded = [], []
    for c in CANDIDATES:
        if not allowed(c["sources"], policy):
            excluded.append(f"{c['path']} (sources not cleared: {', '.join(s for s in c['sources'] if not policy['sources'].get(s, {}).get('publish_derived'))})")
            continue
        dst = out / c["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            c["build"](dst)
        except FileNotFoundError as exc:
            excluded.append(f"{c['path']} (input not available: {exc})")
            continue
        entry = {"path": c["path"], "sha256": sha(dst), "description": c["description"], "sources": c["sources"],
                 "attribution": sorted({policy["sources"][s]["attribution"] for s in c["sources"]})}
        if dst.suffix == ".csv":
            df = pd.read_csv(dst)
            entry.update(rows=len(df), fields=list(df.columns),
                         missing_share_by_column={k: round(float(v), 4) for k, v in df.isna().mean().items()})
        files.append(entry)
    boxes = {"rule": f"bounding box of the reservoir and every CDEC candidate snow station of its river basin(s), padded "
                     f"{PAD_DEG} degrees", "boxes": basin_boxes()}
    (out / "upstream_boxes.json").write_text(json.dumps(boxes, indent=1))
    files.append({"path": "upstream_boxes.json", "sha256": sha(out / "upstream_boxes.json"), "description": "box definitions",
                  "sources": []})
    for extra in ("data/capacity_history.csv", "data/series_breaks.csv", "data/spike_review.csv", "data/snow_stations.csv"):
        dst = out / "metadata" / Path(extra).name
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / extra, dst)
        files.append({"path": f"metadata/{Path(extra).name}", "sha256": sha(dst), "description": f"copy of {extra}", "sources": []})
    man = {"name": "WatershedShift public data", "version": version,
           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "policy_version": policy["policy_version"], "rights_evidence": "docs/DATA_PUBLICATION_AUDIT.md",
           "files": files, "excluded_pending_clearance": excluded,
           "citation": "WatershedShift (2026). Public data for 30-day reservoir storage research forecasts, "
                       f"version {version}. https://github.com/haaris-usman-saeed/watershed"}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    (out / "SHA256SUMS.txt").write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in files))
    readme = [f"# WatershedShift public data, version {version}", "",
              "Each file's sources, fields, missingness and checksum are in MANIFEST.json. This release is immutable; "
              "corrections are published as a new version. Values from CDEC are provisional and may be revised by "
              "the data providers; no endorsement by any data provider is implied.", ""]
    if excluded:
        readme += ["Not included:", ""] + [f"- {e}" for e in excluded] + [""]
    readme += ["Attribution:", ""] + sorted({f"- {a}" for f in files for a in f.get("attribution", [])})
    (out / "README.md").write_text("\n".join(readme) + "\n")
    return man


def check_releases(root=ROOT / "public_data", policy_path=POLICY):
    """Every file in every release is listed, checksum-intact and allowed by the current policy."""
    policy = json.loads(Path(policy_path).read_text())
    problems = []
    for d in sorted(Path(root).glob("watershedshift-public-data-*")):
        man = json.loads((d / "MANIFEST.json").read_text())
        listed = {f["path"] for f in man["files"]}
        on_disk = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"MANIFEST.json", "SHA256SUMS.txt", "README.md"}
        if on_disk != listed:
            problems.append(f"{d.name}: files not in manifest {sorted(on_disk - listed)} or missing {sorted(listed - on_disk)}")
        for f in man["files"]:
            p = d / f["path"]
            if p.exists() and sha(p) != f["sha256"]:
                problems.append(f"{d.name}/{f['path']}: checksum mismatch")
            if not allowed(f.get("sources", []), policy):
                problems.append(f"{d.name}/{f['path']}: sources {f.get('sources')} not cleared by the release policy")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    ap.add_argument("--out", default=str(ROOT / "public_data"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        probs = check_releases(Path(a.out))
        print("\n".join(probs) if probs else "public releases OK (listed, checksum-intact, allowed by policy)")
        raise SystemExit(1 if probs else 0)
    man = build_release(Path(a.out), a.version)
    print(json.dumps({"version": man["version"], "files": [(f["path"], f.get("rows")) for f in man["files"]],
                      "excluded": man["excluded_pending_clearance"]}, indent=1))


if __name__ == "__main__":
    main()

"""Versioned public data release, restricted to sources whose redistribution is confirmed.

Run: python -m src.v3_release --version 2026-11-22 [--out public_data]

Reads config/data_release_policy.json. Packages only derived tables whose every source has
publish_derived = true. Each release directory is write-once: an existing version is never
overwritten or changed; a correction is a new version.
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
CANDIDATES = [
    {"path": "gefs_basin_precip.csv", "src": "data/v3/gefs_basin_precip.csv",
     "sources": ["noaa_gefs_reforecast", "noaa_gefs_operational"],
     "description": "GEFSv12 control-member precipitation forecast, 00Z Monday cycles, averaged over each reservoir's "
                    "upstream box: lead hours (0,168] and (168,384]. source = reforecast (retrospective, not issued in real "
                    "time) or operational (published_utc_max = latest S3 Last-Modified of the 64 files used).",
     "fields": {"init_date": "cycle date (00 UTC)", "source": "reforecast | operational", "station_id": "reservoir",
                "gefs_p_d1_7_mm": "mm, lead hours 0-168", "gefs_p_d8_16_mm": "mm, lead hours 168-384",
                "complete": "all 64 six-hour messages present", "n_messages": "messages used",
                "bytes_sha256": "SHA-256 of the concatenated GRIB2 messages", "published_utc_max": "operational only",
                "retrieved_utc": "download time", "note": "provenance note"}},
    {"path": "power_daily.csv", "src": "data/v3/public/power_daily.csv", "sources": ["nasa_power"],
     "description": "NASA POWER daily weather (PRECTOTCORR, T2M, RH2M; LST) at each dam location (dam_point) and averaged over "
                    "POWER grid-cell centres inside the upstream box (upstream_box_mean). Implausible values set to missing.",
     "fields": {"station_id": "reservoir", "source": "dam_point | upstream_box_mean", "date": "local standard time day",
                "rain_mm": "mm/day", "temp_c": "degC", "humidity_pct": "%"}},
]
EXCLUDED = ["CDEC daily storage (sensor 15) and any table containing it", "CDEC snow water content (sensor 82) and the "
            "basin snow index", "weekly feature tables, forecasts, residuals and model artifacts (fitted to CDEC values)"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_release(out_root, version, policy_path=ROOT / "config" / "data_release_policy.json"):
    policy = json.loads(Path(policy_path).read_text())
    out = Path(out_root) / f"watershedshift-public-data-{version}"
    if out.exists():
        raise FileExistsError(f"{out} exists; published releases are never changed. Use a new version.")
    out.mkdir(parents=True)
    files = []
    for c in CANDIDATES:
        ok = all(policy["sources"].get(s, {}).get("publish_derived") for s in c["sources"])
        src = ROOT / c["src"]
        if not ok or not src.exists():
            continue
        shutil.copyfile(src, out / c["path"])
        df = pd.read_csv(out / c["path"])
        files.append({"path": c["path"], "sha256": sha(out / c["path"]), "rows": len(df),
                      "missing_share_by_column": {k: round(float(v), 4) for k, v in df.isna().mean().items()},
                      "description": c["description"], "fields": c["fields"], "sources": c["sources"],
                      "attribution": [policy["sources"][s]["attribution"] for s in c["sources"]]})
    boxes = {"rule": f"bounding box of the reservoir and every CDEC candidate snow station of its river basin(s), padded "
                     f"{PAD_DEG} degrees", "boxes": basin_boxes()}
    (out / "upstream_boxes.json").write_text(json.dumps(boxes, indent=1))
    files.append({"path": "upstream_boxes.json", "sha256": sha(out / "upstream_boxes.json"), "description": "box definitions"})
    man = {"name": "WatershedShift public data", "version": version,
           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "policy_version": policy["policy_version"], "rights_evidence": "docs/DATA_PUBLICATION_AUDIT.md",
           "files": files, "excluded_pending_clearance": EXCLUDED,
           "citation": "WatershedShift (2026). Public weather-input data for 30-day reservoir storage research forecasts, "
                       f"version {version}. https://github.com/haaris-usman-saeed/watershed"}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    (out / "SHA256SUMS.txt").write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in files))
    readme = [f"# WatershedShift public data, version {version}", "",
              "Derived weather inputs only. Each file's sources, field definitions, missingness and checksum are in "
              "MANIFEST.json. This release is immutable; corrections are published as a new version.", "",
              "Not included until the California Department of Water Resources confirms redistribution terms:", ""]
    readme += [f"- {e}" for e in EXCLUDED] + ["", "Attribution:", ""]
    readme += sorted({f"- {a}" for f in files for a in f.get("attribution", [])})
    (out / "README.md").write_text("\n".join(readme) + "\n")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", default=str(ROOT / "public_data"))
    a = ap.parse_args()
    man = build_release(Path(a.out), a.version)
    print(json.dumps({"version": man["version"], "files": [(f["path"], f.get("rows")) for f in man["files"]]}, indent=1))


if __name__ == "__main__":
    main()

"""Validate a data directory against the frontend's view of the contract.

    python app/check_contract.py app/sample_data
    python app/check_contract.py outputs --require-real

Exit code 1 if a required file is missing or unusable, or (with --require-real)
if metrics.json does not report data_mode = "real".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui_contract as C  # noqa: E402
import ui_io  # noqa: E402
import ui_metrics as M  # noqa: E402

REQUIRED = (C.STATIONS_FILE, C.HISTORY_FILE, C.PREDICTIONS_FILE, C.METRICS_FILE, C.FEATURES_FILE)


def main():
    ap = argparse.ArgumentParser(description="Check WatershedShift outputs against the UI contract.")
    ap.add_argument("data_dir", nargs="?", default=None, help="defaults to WATERSHED_DATA_DIR or outputs/")
    ap.add_argument("--require-real", action="store_true", help="fail unless data_mode is \"real\"")
    a = ap.parse_args()
    d = Path(a.data_dir).resolve() if a.data_dir else ui_io.resolve_data_dir()[0]
    print(f"Data directory: {d}\n")
    b = ui_io.load_dir(d)
    failed = False
    for name in C.ALL_FILES:
        r = b["reports"][name]
        mark = {"ok": "OK  ", "warn": "WARN", "missing": "MISS", "invalid": "FAIL"}[r.status]
        print(f"[{mark}] {name}" + (f"  ({r.rows:,} rows)" if r.rows else ""))
        if r.error:
            print(f"        error: {r.error}")
        for w in r.warnings:
            print(f"        note: {w}")
        if r.missing_optional:
            print(f"        optional fields absent: {', '.join(r.missing_optional)}")
        if r.extra_columns:
            print(f"        columns the UI ignores: {', '.join(r.extra_columns)}")
        if name in REQUIRED and not r.usable:
            failed = True
    print(f"\ndata_mode: {b['data_mode']!r}")
    if a.require_real and b["data_mode"] != C.REAL_DATA_MODE:
        print("  not \"real\": the app will show its synthetic/unverified warning")
        failed = True
    nm = M.normalize_metrics(b["metrics"])
    comp = M.mae_table(b["preds"]) if b["preds"] is not None else None
    if comp is not None or nm.present:
        v = M.verdict(nm, comp)
        print(f"\nVerdict ({v.kind}): {v.headline}\n  {v.detail}")
    if b["preds"] is not None:
        for msg in M.cross_checks(nm, comp, M.coverage(b["preds"]), b["preds"]):
            print(f"  cross-check: {msg}")
        if nm.present and nm.bootstrap_unparsed:
            print(f"  bootstrap key '{nm.bootstrap_raw_key}' present but not parseable by ui_metrics.py")
    print("\nFAILED" if failed else "\nPASSED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
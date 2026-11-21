"""Report facts for the v2 submission, extracted mechanically from existing repository evidence.

Run: python -m src.report_facts            writes results/v2/report_facts.json and report_facts.md
     python -m src.report_facts --check    regenerates in memory and fails if the committed files differ

No model is trained and no held-out computation is repeated. Every fact records its value, units, exact
source file(s), whether that source is tracked in git or local only, and its kind:
  observed                 counted directly from processed observation files
  computed                 read from frozen evaluation outputs (results/v2/final, results/v2/development)
  documented configuration read from frozen configuration, code constants or documentation
Facts that cannot be found are written as "not found". Output is deterministic (no timestamps).
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from src.features import SNOW_LAG, WEATHER_LAG
from src.models import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "v2" / "report_facts.json"
OUT_MD = ROOT / "results" / "v2" / "report_facts.md"
NOT_FOUND = "not found"

F_METRICS = "results/v2/final/metrics.json"
F_RUN = "results/v2/final/RUN_RECORD.json"
F_CONFIG = "config/v2_frozen.json"
F_PREREG = "docs/preregistration.md"
F_DAILY = "data/processed/daily_joined.csv"
F_COVERAGE = "data/processed/coverage.csv"
F_COV_MD = "data/coverage_table.md"
F_WEEKLY = "data/processed/weekly_features.csv"
F_SNOWIDX = "data/processed/snow_basin_index.csv"
F_SNOWST = "data/snow_stations.csv"
F_CAP = "data/capacity_history.csv"
F_BREAKS = "data/series_breaks.csv"
F_OOF = "results/v2/development/oof_predictions.csv"
F_DEVSUM = "results/v2/development/summary.json"
F_PRED = "results/v2/final/test_predictions_public.csv"


def tracked(path):
    r = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True)
    return r.returncode == 0


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class Facts:
    def __init__(self):
        self.items = []

    def add(self, fid, name, value, units, sources, kind, note=""):
        assert kind in ("observed", "computed", "documented configuration")
        srcs = [sources] if isinstance(sources, str) else list(sources)
        self.items.append({"id": fid, "name": name, "value": value, "units": units, "kind": kind,
                           "sources": [{"path": s, "tracked_in_git": tracked(s)} for s in srcs], "note": note})


def r4(x):
    return round(float(x), 4)


def build():
    f = Facts()
    m = json.loads((ROOT / F_METRICS).read_text())
    run = json.loads((ROOT / F_RUN).read_text())
    cfg = json.loads((ROOT / F_CONFIG).read_text())
    prereg = (ROOT / F_PREREG).read_text()

    # 1-3. dates, reservoir-days, valid and missing days
    daily = pd.read_csv(ROOT / F_DAILY, parse_dates=["date"])
    cov = pd.read_csv(ROOT / F_COVERAGE).set_index("station_id")
    per = {}
    for sid, g in daily.groupby("station_id"):
        valid, missing = int(g.storage_af.notna().sum()), int(g.storage_af.isna().sum())
        assert valid == int(cov.loc[sid, "n_storage_days"]) and missing == int(cov.loc[sid, "missing_storage_days"])
        per[sid] = {"first_date": g.date.min().strftime("%Y-%m-%d"), "last_date": g.date.max().strftime("%Y-%m-%d"),
                    "calendar_days": int(len(g)), "valid_storage_days": valid, "missing_storage_days": missing}
    f.add("1", "Start and end date per reservoir", {k: [v["first_date"], v["last_date"]] for k, v in per.items()},
          "date", [F_DAILY, F_COVERAGE, F_COV_MD], "observed")
    f.add("2", "Total reservoir-day observations (calendar station-days in the joined daily table)",
          {"reservoirs": len(per), "station_days": int(len(daily))}, "reservoir-days", [F_DAILY], "observed",
          "One row per reservoir per calendar day 2001-01-01 to 2025-12-31; storage may be missing on a row.")
    f.add("3", "Valid and missing storage days per reservoir",
          {k: {"valid": v["valid_storage_days"], "missing": v["missing_storage_days"]} for k, v in per.items()},
          "days", [F_DAILY, F_COVERAGE, F_COV_MD], "observed",
          "Missing = no usable CDEC value or set to missing by the elevation-adjudicated spike review. Never interpolated.")

    # 4. row counts
    weekly = pd.read_csv(ROOT / F_WEEKLY, parse_dates=["date", "target_date"])
    oof = pd.read_csv(ROOT / F_OOF)
    dev = json.loads((ROOT / F_DEVSUM).read_text())
    n_train = int((weekly.target_date < "2024-01-01").sum())
    n_test = int(((weekly.date >= "2024-01-01") & (weekly.target_date <= "2025-12-31")).sum())
    assert n_train == m["train_count"] and n_test == m["test_count"] and len(oof) == m["val_count"]
    f.add("4", "Weekly rows: total, final training, rolling development forecasts, held-out test",
          {"weekly_rows_total": int(len(weekly)), "final_training_rows": n_train,
           "rolling_development_forecasts_all_folds_WY2008_2023": int(len(oof)),
           "rolling_development_tuning_folds_WY2008_2015": dev["n_rows"]["tuning"],
           "rolling_development_confirmation_folds_WY2016_2023": dev["n_rows"]["confirmation"],
           "held_out_test_rows": n_test, "held_out_test_reservoirs": m["test_station_count"]},
          "rows", [F_WEEKLY, F_OOF, F_DEVSUM, F_METRICS], "observed",
          "Training = all weekly rows with target_date < 2024-01-01 (also the development pool). "
          "Test = origin >= 2024-01-01 and target_date <= 2025-12-31.")

    # 5. snow pillows
    st = pd.read_csv(ROOT / F_SNOWST)
    idx = pd.read_csv(ROOT / F_SNOWIDX)
    cand = st.groupby("reservoir_id").size().astype(int).to_dict()
    core = idx.groupby("station_id").n_core.max().astype(int).to_dict()
    f.add("5", "Candidate and core SWE snow pillows per reservoir",
          {k: {"candidate": int(cand.get(k, 0)), "core": int(core.get(k, 0))} for k in sorted(per)},
          "pillows", [F_SNOWST, F_SNOWIDX, "src/features.py"], "observed",
          f"Candidates total {int(len(st))}. Core rule (src/features.py core_snow_stations): a pillow is core if at "
          "least 75% of Dec-May seasons in WY2002-2023 have at least 80% daily availability.")

    # 6. SWE lag
    assert SNOW_LAG == cfg["snow_lag_days"] == m["split"]["snow_lag_days"]
    f.add("6", "SWE observation lag", SNOW_LAG, "days", [F_CONFIG, "src/features.py", F_METRICS],
          "documented configuration", "SWE features at origin t use values from t-1 and earlier. "
          f"Weather lag for secondary models: {WEATHER_LAG} days (src/features.py, config).")

    # 7. primary feature list
    stations = sorted(weekly[weekly.target_date < "2024-01-01"].station_id.unique())
    cols = FEATURE_SETS["storage_snow"] + [f"station_{s}" for s in stations]
    assert FEATURE_SETS["storage_snow"] == [c for c in cfg["primary_model"]["features"] if c != "station one-hot"]
    f.add("7", "Exact primary model input columns (HGB SWE no weather)", cols, "column names",
          [F_CONFIG, "src/models.py", "src/train_v2.py"], "documented configuration",
          "FEATURE_SETS['storage_snow'] plus models.design() station one-hot columns in sorted order. "
          "No NASA POWER weather column is an input of the primary model.")

    # 8. interval
    f.add("8", "90% interval method and calibration period",
          {"method": cfg["interval"]["method"], "nominal": cfg["interval"]["nominal"],
           "calibration": cfg["interval"]["calibration"],
           "half_width_by_origin_season_pp": {k: r4(v) for k, v in m["interval_90_halfwidth_by_season_pp"].items()}},
          "pp of capacity", [F_CONFIG, F_METRICS, F_OOF], "documented configuration",
          "Symmetric interval = prediction +/- the 90th percentile of absolute out-of-fold residuals per season.")

    # 9. test MAEs
    names = ["ML", "HGB (v1 features)", "HGB + SWE", "HGB no weather", "Ridge", "Smoothed DOY seasonal",
             "Historical seasonal", "30-day trend", "Persistence"]
    f.add("9", "Held-out 2024-2025 test MAE of the primary model, secondary models and baselines",
          {n: r4(m[n]) for n in names}, "pp of capacity", [F_METRICS], "computed",
          "ML = preregistered primary (HGB SWE no weather). Historical seasonal = monthly seasonal baseline. "
          f"Strongest baseline fixed in advance: {m['bootstrap']['best_baseline']}.")

    # 10. Oroville capacity and break windows
    cap = pd.read_csv(ROOT / F_CAP)
    brk = pd.read_csv(ROOT / F_BREAKS, parse_dates=["break_date"])
    oro_cap = [{"valid_from": r.valid_from, "valid_to": r.valid_to, "capacity_af": int(r.capacity_af)}
               for r in cap[cap.station_id == "ORO"].itertuples()]
    b = brk[brk.station_id == "ORO"].break_date.iloc[0]
    mondays = pd.date_range("2001-01-01", "2025-12-31", freq="W-MON")
    span = list(mondays[(mondays + pd.Timedelta(days=30) >= b) & (mondays - pd.Timedelta(days=60) < b)])
    oro_dates = set(weekly[weekly.station_id == "ORO"].date)
    other = set(weekly[weekly.station_id == "SHA"].date)
    assert not any(d in oro_dates for d in span) and all(d in other for d in span)
    f.add("10", "Oroville capacity definitions and excluded reporting-break windows",
          {"capacity_history": oro_cap, "break_date": b.strftime("%Y-%m-%d"),
           "excluded_origin_rule": "origin t excluded if t - 60 days < break_date <= t + 30 days",
           "excluded_origins_first": span[0].strftime("%Y-%m-%d"), "excluded_origins_last": span[-1].strftime("%Y-%m-%d"),
           "excluded_origin_count": len(span),
           "excluded_in_test_period": sum(d >= pd.Timestamp("2024-01-01") for d in span),
           "oro_test_rows": m["per_station"]["ORO"]["n"]},
          "acre-feet, dates, rows", [F_CAP, F_BREAKS, F_WEEKLY, F_METRICS], "observed",
          "Capacity in force is used on each date. The excluded origins are absent from the ORO weekly rows and present for SHA.")

    # 11. CIs
    bb = m["bootstrap"]["by_comparison_and_block"]
    f.add("11", "Primary minus seasonal baselines, MAE difference with 95% moving-block bootstrap CI",
          {comp: {k: {"estimate": r4(v["estimate"]), "ci95_low": r4(v["ci95_low"]), "ci95_high": r4(v["ci95_high"])}
                  for k, v in blocks.items()} for comp, blocks in bb.items()},
          "pp of capacity", [F_METRICS], "computed",
          f"n_boot {m['bootstrap']['n_boot']}, seed 42 (config), blocks of weekly origin dates; negative = primary better.")

    # 12. coverage per reservoir
    f.add("12", "90% interval coverage and mean width per reservoir (held-out test)",
          {**{k: {"coverage": r4(v["interval_90_coverage"]), "mean_width_pp": r4(v["interval_mean_width_pp"]), "n": v["n"]}
              for k, v in sorted(m["per_station"].items())},
           "all": {"coverage": r4(m["interval_90_coverage"]), "mean_width_pp": r4(m["interval_90_mean_width_pp"]),
                   "n": m["test_count"]}},
          "fraction, pp of capacity, rows", [F_METRICS], "computed",
          "The half-width depends only on origin season, so mean widths differ only through the seasonal mix of "
          "origins (ORO lacks the 13 break-window origins in June to August 2024).")

    # 13. excluded stations
    status = cov.status.to_dict()
    f.add("13", "Excluded stations cannot enter training",
          {"station_status": status, "excluded_count": int((cov.status != "included").sum()),
           "enforced_by": "src/features.py main: filter_eligible(..., eligible_stations()) before weekly_features.csv",
           "regression_test": "tests/test_features.py::test_excluded_station_never_enters_features"},
          "", [F_COVERAGE, "src/features.py", "tests/test_features.py"], "documented configuration",
          "All six reservoirs pass the <= 10% missing rule, so no station is excluded in practice.")

    # 14. disclosure
    mt = re.search(r"\*\*(the v2 configuration was developed after observing v1 test performance[^*]+)\*\*", prereg)
    f.add("14", "v1 and v2 test-set reuse disclosure", mt.group(1) if mt else NOT_FOUND, "", [F_PREREG, F_METRICS],
          "documented configuration", "Same 2024-2025 test period (593 rows) for v1 and v2.")

    # 15. frozen configuration identity
    prereg_hashes = dict(re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", prereg))
    cur = {p: sha256(p) for p in run["frozen_hashes"]}
    log = subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", F_PREREG], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    f.add("15", "Identity and source commit of the frozen v2 configuration",
          {"config_file": F_CONFIG, "config_version": cfg["version"], "primary_model": cfg["primary_model"]["name"],
           "config_sha256": cur[F_CONFIG],
           "hashes_match_preregistration": cur == {k: prereg_hashes.get(k) for k in cur},
           "hashes_match_run_record": cur == run["frozen_hashes"],
           "preregistration_commit": log[-1] if log else NOT_FOUND,
           "final_run_commit": run["git_commit_at_run"], "final_run_utc": run["first_run_utc"]},
          "", [F_CONFIG, F_PREREG, F_RUN], "documented configuration",
          "Hashes recomputed from the working tree when this file was generated.")
    return {"title": "WatershedShift v2 report facts", "generated_by": "python -m src.report_facts",
            "kinds": {"observed": "counted from processed observation files",
                      "computed": "read from frozen evaluation outputs",
                      "documented configuration": "read from frozen configuration, code constants or documentation"},
            "facts": f.items}


def to_md(d):
    lines = [f"# {d['title']}", "",
             f"Generated by `{d['generated_by']}` from existing repository evidence; no model was retrained. "
             "The machine-readable twin is `results/v2/report_facts.json`. Kinds: "
             + "; ".join(f"**{k}** = {v}" for k, v in d["kinds"].items()) + ". "
             "Sources marked (local) are git-ignored derived data on the owner's machine (rebuild with the README steps); "
             "sources marked (tracked) are in git.", ""]
    for it in d["facts"]:
        src = ", ".join(f"`{s['path']}` ({'tracked' if s['tracked_in_git'] else 'local'})" for s in it["sources"])
        lines += [f"## {it['id']}. {it['name']}", "",
                  f"- Kind: {it['kind']}" + (f"; units: {it['units']}" if it["units"] else ""),
                  f"- Source: {src}"]
        if it["note"]:
            lines.append(f"- Note: {it['note']}")
        lines += ["", "```json", json.dumps(it["value"], indent=2), "```", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = build()
    js, md = json.dumps(d, indent=2) + "\n", to_md(d)
    if a.check:
        ok = OUT_JSON.read_text() == js and OUT_MD.read_text() == md
        print("report facts:", "OK (committed files match regenerated)" if ok else "MISMATCH")
        raise SystemExit(0 if ok else 1)
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(md)


if __name__ == "__main__":
    main()

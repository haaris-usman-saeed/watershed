"""Frozen v2 evaluation (single preregistered run on the 2024-2025 held-out period).

Run:
  python -m src.train_v2 --rehearsal   # same code path on DEVELOPMENT data only (train < 2023, test = 2023)
  python -m src.train_v2 --final       # the one preregistered held-out evaluation

Guards for --final
- docs/preregistration.md must be committed (tracked and unmodified in the working tree).
- SHA-256 of config/v2_frozen.json and of the frozen code files must equal the values listed
  in the preregistration.
- If results/v2/final/RUN_RECORD.json exists, a rerun is allowed only with the identical
  configuration and code hashes (reproducibility check). A different configuration requires
  explicit owner approval (--owner-approved-new-config) and is reported as a separate run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from src.contract import SCHEMA_VERSION, validate_dir
from src.evaluation import BLOCK_LENGTHS, block_bootstrap_diff, coverage, mae, water_year
from src.features import FEATURES, SNOW, STORAGE, SEASON, WEATHER
from src.models import (FEATURE_SETS, add_snow_anomaly, baselines, design, hgb, interval_offsets, ridge, season_of,
                        snow_climatology)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "v2_frozen.json"
PREREG = ROOT / "docs" / "preregistration.md"
FROZEN_CODE = ["src/train_v2.py", "src/models.py", "src/features.py", "src/evaluation.py", "src/build_dataset.py"]
V1_PRED = ROOT / "results" / "v1" / "test_predictions_public.csv"
ORO_BREAK = pd.Timestamp("2024-07-02")
SECONDARY = {"HGB (v1 features)": "storage_weather", "HGB + SWE": "storage_weather_snow",
             "HGB no weather": "storage"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes():
    return {"config/v2_frozen.json": sha(CONFIG), **{f: sha(ROOT / f) for f in FROZEN_CODE}}


def check_preregistration():
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PREREG)], cwd=ROOT, capture_output=True)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PREREG)], cwd=ROOT)
    if tracked.returncode != 0 or dirty.returncode != 0:
        raise SystemExit("REFUSED: docs/preregistration.md must be committed and unmodified before --final.")
    text = PREREG.read_text()
    for f, h in frozen_hashes().items():
        if not re.search(rf"`{re.escape(f)}`\s*\|\s*`{h}`", text):
            raise SystemExit(f"REFUSED: hash of {f} does not match the preregistration ({h}).")


def check_run_record(out_dir, allow_new):
    rec = out_dir / "RUN_RECORD.json"
    if rec.exists():
        prev = json.loads(rec.read_text())
        if prev["frozen_hashes"] != frozen_hashes() and not allow_new:
            raise SystemExit("REFUSED: a v2 final run with a different configuration already exists. "
                             "A further held-out evaluation requires explicit owner approval.")
        return prev
    return None


def load(train_end, test_start, test_end):
    df = pd.read_csv(ROOT / "data/processed/weekly_features.csv", parse_dates=["date", "target_date"])
    train = df[df.target_date < train_end].copy()
    test = df[(df.date >= test_start) & (df.target_date <= test_end)].copy()
    assert train.target_date.max() < test.date.min()
    snow = pd.read_csv(ROOT / "data/processed/snow_basin_index.csv", parse_dates=["date"])
    clim = snow_climatology(snow, train_end)
    return add_snow_anomaly(train, clim), add_snow_anomaly(test, clim), df


def run(mode, allow_new=False):
    cfg = json.loads(CONFIG.read_text())
    if mode == "final":
        check_preregistration()
        out_dir = ROOT / "results" / "v2" / "final"
        prev = check_run_record(out_dir, allow_new)
        train_end, test_start, test_end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
        cal_wys = range(2008, 2024)
    else:
        out_dir = ROOT / "results" / "v2" / "rehearsal"
        prev = None
        train_end, test_start, test_end = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
        cal_wys = range(2008, 2023)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test, full = load(train_end, test_start, test_end)
    stations = sorted(train.station_id.unique())
    hp = {k: v for k, v in cfg["primary_model"]["hyperparameters"].items() if k not in ("l2_regularization", "random_state")}
    assert cfg["primary_model"]["hyperparameters"]["l2_regularization"] == 1.0
    primary_cols = FEATURE_SETS["storage_snow"]

    out = test[["station_id", "reservoir_name", "date", "target_date", "target_delta_pp", "storage_af",
                "capacity_af", "storage_pct"]].copy()
    out["pred_delta_pp"] = hgb(hp).fit(design(train, primary_cols, stations), train.target_delta_pp) \
        .predict(design(test, primary_cols, stations))
    for name, fs in SECONDARY.items():
        col = "pred_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + "_pp"
        out[col] = hgb(hp).fit(design(train, FEATURE_SETS[fs], stations), train.target_delta_pp) \
            .predict(design(test, FEATURE_SETS[fs], stations))
    rc = FEATURE_SETS["storage_weather"]
    out["pred_ridge_pp"] = ridge().fit(design(train, rc, stations), train.target_delta_pp).predict(design(test, rc, stations))
    out = out.join(baselines(train, test))

    # interval: sym_season calibrated on development out-of-fold residuals of the primary model
    oof = pd.read_csv(ROOT / "results/v2/development/oof_predictions.csv", parse_dates=["date"])
    oof = oof[oof.wy.isin(list(cal_wys))]
    cal = pd.DataFrame({"r": oof.target_delta_pp - oof["pred_HGB SWE no weather"], "season": oof.season,
                        "station_id": oof.station_id})
    out["season"] = season_of(out.date)
    lo, hi = interval_offsets(cal, cfg["interval"]["method"])(out)
    out["lower_90_pp"], out["upper_90_pp"] = out.pred_delta_pp + lo, out.pred_delta_pp + hi
    out["wy"] = water_year(out.date)

    models = {"ML": "pred_delta_pp", "HGB (v1 features)": "pred_hgb_v1_features_pp", "HGB + SWE": "pred_hgb_swe_pp",
              "HGB no weather": "pred_hgb_no_weather_pp", "Ridge": "pred_ridge_pp",
              "Persistence": "baseline_persistence_pp", "30-day trend": "baseline_trend_pp",
              "Historical seasonal": "baseline_seasonal_pp", "Smoothed DOY seasonal": "baseline_doy_pp"}
    y = out.target_delta_pp
    inside = (y >= out.lower_90_pp) & (y <= out.upper_90_pp)

    def group_table(key):
        rows = {}
        for k, g in out.groupby(key):
            gi = (g.target_delta_pp >= g.lower_90_pp) & (g.target_delta_pp <= g.upper_90_pp)
            rows[str(k)] = {"n": int(len(g)), **{m: mae(g.target_delta_pp, g[c]) for m, c in models.items()},
                            "interval_90_coverage": float(gi.mean()),
                            "interval_mean_width_pp": float((g.upper_90_pp - g.lower_90_pp).mean())}
        return rows

    strongest = cfg["strongest_baseline"]
    comparisons = {f"ML minus {strongest}": ("pred_delta_pp", models[strongest]),
                   "ML minus Historical seasonal (monthly)": ("pred_delta_pp", "baseline_seasonal_pp")}
    boot = {lab: {f"{b}_weeks": block_bootstrap_diff(out, a, c, block=b) for b in BLOCK_LENGTHS}
            for lab, (a, c) in comparisons.items()}

    v1_compare = None
    if mode == "final" and V1_PRED.exists():
        v1_compare = v1_comparison(out, pd.read_csv(V1_PRED, parse_dates=["date", "target_date"]))

    halfwidth_by_season = {s: float(np.median(out.loc[out.season == s, "upper_90_pp"] - out.loc[out.season == s, "pred_delta_pp"]))
                           for s in sorted(out.season.unique())}
    metrics = {"schema_version": SCHEMA_VERSION, "data_mode": "real", "model_version": "v2",
               "evaluation_mode": mode, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **{m: mae(y, out[c]) for m, c in models.items()},
               "interval_90_coverage": float(inside.mean()),
               "interval_90_halfwidth_pp": float(((out.upper_90_pp - out.lower_90_pp) / 2).mean()),
               "interval_90_halfwidth_by_season_pp": halfwidth_by_season,
               "interval_90_mean_width_pp": float((out.upper_90_pp - out.lower_90_pp).mean()),
               "train_count": int(len(train)), "val_count": int(len(oof)), "test_count": int(len(out)),
               "test_station_count": int(out.station_id.nunique()),
               "note": ("v2 historical backtest; not an operational forecast. ML = HistGradientBoosting on storage history, "
                        "basin snow water equivalent (CDEC snow pillows, t-1) and season; NASA POWER weather is not an input "
                        "of the primary model. The v2 configuration was developed after the v1 results on this same "
                        "2024-2025 test period were seen, so this period is not fully independent of the revision. "
                        "Overlapping 30-day windows: CIs use moving-block bootstrap over dates. Season-conditional 90% "
                        "intervals calibrated on 2008-2023 rolling out-of-fold residuals; coverage is not guaranteed. "
                        "SWE and weather are retrospective (revised) series."),
               "split": {"train_target_end_exclusive": str(train_end.date()), "val_start": "rolling folds WY2008-2023",
                         "val_target_end_exclusive": str(train_end.date()), "test_start": str(test_start.date()),
                         "test_target_end_inclusive": str(out.target_date.max().date()), "horizon_days": 30,
                         "weather_lag_days": 7, "snow_lag_days": 1, "sampling": "weekly (Mondays)",
                         "data_period": cfg["dataset_period"]},
               "per_station": group_table("station_id"), "per_season": group_table("season"),
               "per_water_year": group_table("wy"),
               "bootstrap": {"method": "moving-block bootstrap over weekly test origin dates (reservoirs resampled jointly per date); "
                                       "two-sided 95% percentile CI; block lengths fixed in advance at 5, 9 and 13 weeks",
                             "n_boot": 2000, "block_weeks": 5, "best_baseline": strongest,
                             "best_baseline_selected_on": "rolling development folds WY2016-2023",
                             "mae_diff_ml_minus_best_baseline": {**{k: v for k, v in boot[f"ML minus {strongest}"]["5_weeks"].items()
                                                                   if k in ("estimate", "ci95_low", "ci95_high")},
                                                                 "share_boot_ml_better": boot[f"ML minus {strongest}"]["5_weeks"]["share_better"]},
                             "by_comparison_and_block": boot},
               "v1_vs_v2": v1_compare,
               "v1_reference": json.loads((ROOT / "results/v1/metrics.json").read_text()) if mode == "final" else None,
               "frozen_hashes": frozen_hashes()}
    pub_cols = ["station_id", "reservoir_name", "date", "target_date", "target_delta_pp", "pred_delta_pp", "lower_90_pp",
                "upper_90_pp", "baseline_persistence_pp", "baseline_trend_pp", "baseline_seasonal_pp", "baseline_doy_pp",
                "pred_hgb_v1_features_pp", "pred_hgb_swe_pp", "pred_hgb_no_weather_pp", "pred_ridge_pp", "season", "wy"]
    out[pub_cols].to_csv(out_dir / "test_predictions_public.csv", index=False, date_format="%Y-%m-%d")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if mode == "final":
        if prev is not None:
            same = all(abs(prev["headline"][k] - metrics[k]) < 1e-9 for k in prev["headline"])
            print("RERUN with identical frozen configuration; headline identical to first run:", same)
        else:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
            (out_dir / "RUN_RECORD.json").write_text(json.dumps({
                "first_run_utc": metrics["generated_utc"], "git_commit_at_run": commit, "frozen_hashes": frozen_hashes(),
                "headline": {k: metrics[k] for k in ("ML", "Smoothed DOY seasonal", "Historical seasonal", "interval_90_coverage")}},
                indent=2))
        write_ui_outputs(out, metrics, ROOT / "outputs")
    else:
        write_ui_outputs(out, metrics, out_dir / "ui_outputs_rehearsal", history_src=ROOT / "outputs" / "history_daily.csv")
    print(json.dumps({k: metrics[k] for k in models} | {"coverage": metrics["interval_90_coverage"],
                     "mean_width": metrics["interval_90_mean_width_pp"], "n": len(out)}, indent=1))
    print(json.dumps(boot, indent=1)[:3000])
    if v1_compare:
        print(json.dumps(v1_compare, indent=1))
    return metrics


def v1_comparison(out, v1):
    """Paired v2 vs v1 on rows whose target definition is identical in both versions."""
    v1 = v1[["station_id", "date", "target_delta_pp", "pred_delta_pp"]].rename(
        columns={"target_delta_pp": "v1_target", "pred_delta_pp": "pred_v1_gbm_pp"})
    both = out.merge(v1, on=["station_id", "date"], how="inner")
    both = both[~((both.station_id == "ORO") & (both.date < ORO_BREAK))]   # v1 used post-2024 ORO capacity
    if not np.allclose(both.target_delta_pp, both.v1_target, atol=1e-6):
        raise ValueError("v1 and v2 targets differ on rows expected to be identical")
    return {"n_common_rows": int(len(both)),
            "excluded": "ORO rows with origin before 2024-07-02 (v1 used the post-2024 capacity for those rows)",
            "v1_gbm_mae": mae(both.target_delta_pp, both.pred_v1_gbm_pp),
            "v2_primary_mae": mae(both.target_delta_pp, both.pred_delta_pp),
            "bootstrap_v2_minus_v1": {f"{b}_weeks": block_bootstrap_diff(both, "pred_delta_pp", "pred_v1_gbm_pp", block=b)
                                      for b in BLOCK_LENGTHS}}


def write_ui_outputs(out, metrics, o, history_src=None):
    """UI contract v1.0 files (additive extras only)."""
    o.mkdir(parents=True, exist_ok=True)
    if o.resolve() != (ROOT / "outputs").resolve():   # rehearsal: work on copies, never touch outputs/
        import shutil
        shutil.copy(ROOT / "outputs" / "stations.csv", o / "stations.csv")
        if history_src is not None:
            shutil.copy(history_src, o / "history_daily.csv")
    cols = ["station_id", "reservoir_name", "date", "target_date", "target_delta_pp", "pred_delta_pp", "lower_90_pp",
            "upper_90_pp", "baseline_persistence_pp", "baseline_trend_pp", "baseline_seasonal_pp", "baseline_doy_pp",
            "pred_hgb_v1_features_pp", "pred_hgb_swe_pp", "pred_hgb_no_weather_pp", "pred_ridge_pp", "storage_af",
            "capacity_af", "storage_pct", "season"]
    out[cols].to_csv(o / "test_predictions.csv", index=False, date_format="%Y-%m-%d")
    ui = {k: v for k, v in metrics.items() if k not in ("frozen_hashes",)}
    (o / "metrics.json").write_text(json.dumps(ui, indent=2, default=float))
    catalog = [
        ("storage_pct", "Storage at forecast origin", "% of capacity in force", "CDEC sensor 15", 0, True),
        ("prev7_pct", "Storage change over previous 7 days", "pp of capacity", "CDEC sensor 15", 0, True),
        ("prev30_pct", "Storage change over previous 30 days", "pp of capacity", "CDEC sensor 15", 0, True),
        ("prev60_pct", "Storage change over previous 60 days", "pp of capacity", "CDEC sensor 15", 0, True),
        ("swe_basin_in", "Basin snow water equivalent index (mean of core snow pillows) on t-1", "inches",
         "CDEC sensor 82 (adjusted SWE)", 1, True),
        ("swe_change_14d_in", "Change in basin SWE index from t-15 to t-1", "inches", "CDEC sensor 82", 1, True),
        ("swe_anom_in", "Basin SWE index on t-1 minus smoothed day-of-year mean of training years", "inches",
         "CDEC sensor 82", 1, True),
        ("season_sin", "sin(2*pi*day_of_year/365.25)", "unitless", "calendar", 0, True),
        ("season_cos", "cos(2*pi*day_of_year/365.25)", "unitless", "calendar", 0, True),
        ("rain_7d_mm", "Precipitation sum, 7 days ending at t-7", "mm", "NASA POWER PRECTOTCORR", 7, False),
        ("rain_30d_mm", "Precipitation sum, 30 days ending at t-7", "mm", "NASA POWER PRECTOTCORR", 7, False),
        ("temp_30d_c", "Mean 2 m air temperature, 30 days ending at t-7", "degC", "NASA POWER T2M", 7, False),
        ("humidity_30d_pct", "Mean 2 m relative humidity, 30 days ending at t-7", "%", "NASA POWER RH2M", 7, False)]
    (o / "feature_catalog.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "model_version": "v2", "features": [
        {"name": n, "description": d, "units": u, "source": s, "min_lag_days": l, "used_by_primary_model": used}
        for n, d, u, s, l, used in catalog]}, indent=2))
    hist = pd.read_csv(o / "history_daily.csv")
    snow = pd.read_csv(ROOT / "data/processed/snow_basin_index.csv")[["station_id", "date", "swe_basin_in"]]
    hist = hist.drop(columns=[c for c in ("swe_basin_in",) if c in hist]).merge(snow, on=["station_id", "date"], how="left")
    hist.to_csv(o / "history_daily.csv", index=False)
    errs = validate_dir(o)
    print("contract:", errs or "OK")
    if errs:
        raise SystemExit("UI contract violated")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearsal", action="store_true")
    g.add_argument("--final", action="store_true")
    ap.add_argument("--owner-approved-new-config", action="store_true")
    a = ap.parse_args()
    run("final" if a.final else "rehearsal", allow_new=a.owner_approved_new_config)

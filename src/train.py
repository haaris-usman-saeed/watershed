"""Chronological held-out evaluation of baselines, Ridge and gradient boosting.

Run: python -m src.train   (reads data/processed/weekly_features.csv)

Splits (by forecast origin AND target date, so every outcome is confined to its split):
  train : target_date <  2023-01-01
  valid : date >= 2023-01-01 and target_date < 2024-01-01
  test  : date >= 2024-01-01 and target_date <= last data day
Origins whose 30-day window straddles a boundary are dropped (outcome embargo).
Model selection (hyper-parameters, interval width, "best baseline") uses the
validation split only; test outcomes are touched once, for the final report.
"""
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.contract import SCHEMA_VERSION, validate_dir
from src.features import FEATURES, HORIZON, WEATHER_LAG

ROOT = Path(__file__).resolve().parents[1]
VAL_START, TEST_START = "2023-01-01", "2024-01-01"
GBM_GRID = [dict(learning_rate=lr, max_iter=n, min_samples_leaf=leaf, max_depth=d)
            for lr, n, leaf, d in [(0.05, 100, 20, 3), (0.05, 200, 20, 3), (0.03, 300, 30, 3),
                                   (0.05, 150, 40, None), (0.1, 60, 20, 2)]]
N_BOOT, BLOCK_WEEKS, SEED = 2000, 5, 42
BASELINES = {"Persistence": "baseline_persistence_pp", "30-day trend": "baseline_trend_pp",
             "Historical seasonal": "baseline_seasonal_pp"}


def design(frame, stations):
    oh = pd.get_dummies(frame.station_id).reindex(columns=stations, fill_value=0).astype(float)
    oh.columns = [f"station_{c}" for c in oh.columns]
    return pd.concat([frame[FEATURES].reset_index(drop=True), oh.reset_index(drop=True)], axis=1)


def gbm(params):
    return HistGradientBoostingRegressor(l2_regularization=1.0, random_state=SEED, **params)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def baselines(fit, target):
    out = pd.DataFrame(index=target.index)
    out["baseline_persistence_pp"] = 0.0
    out["baseline_trend_pp"] = target.prev30_pct
    clim = fit.assign(m=fit.date.dt.month).groupby(["station_id", "m"]).target_delta_pp.mean()
    out["baseline_seasonal_pp"] = [clim.get((s, d.month), np.nan) for s, d in zip(target.station_id, target.date)]
    return out


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def block_bootstrap(df, cols, n_boot=N_BOOT, block=BLOCK_WEEKS, seed=SEED):
    """Moving-block bootstrap over weekly origin dates; all stations on a date move together
    (keeps cross-reservoir and overlapping-window dependence inside blocks)."""
    rng = np.random.default_rng(seed)
    dates = np.sort(df.date.unique())
    by_date = {d: g for d, g in df.groupby("date")}
    n = len(dates)
    starts = np.arange(n - block + 1)
    reps = {c: [] for c in cols}
    for _ in range(n_boot):
        chosen = []
        while len(chosen) < n:
            s = rng.choice(starts)
            chosen.extend(dates[s:s + block])
        sample = pd.concat([by_date[d] for d in chosen[:n]])
        for c in cols:
            reps[c].append(mae(sample.target_delta_pp, sample[c]))
    return {c: np.array(v) for c, v in reps.items()}


def ci(a):
    return {"ci95_low": float(np.quantile(a, 0.025)), "ci95_high": float(np.quantile(a, 0.975))}


def run():
    df = pd.read_csv(ROOT / "data/processed/weekly_features.csv", parse_dates=["date", "target_date"])
    last_day = df.target_date.max()
    train = df[df.target_date < VAL_START].copy()
    valid = df[(df.date >= VAL_START) & (df.target_date < TEST_START)].copy()
    test = df[(df.date >= TEST_START) & (df.target_date <= last_day)].copy()
    assert train.target_date.max() < valid.date.min() and valid.target_date.max() < test.date.min()
    stations = sorted(df.station_id.unique())
    Xtr, Xva, Xte = design(train, stations), design(valid, stations), design(test, stations)
    ytr, yva, yte = train.target_delta_pp, valid.target_delta_pp, test.target_delta_pp

    # ---- model selection on validation only
    grid = [(mae(yva, gbm(p).fit(Xtr, ytr).predict(Xva)), i) for i, p in enumerate(GBM_GRID)]
    best = GBM_GRID[min(grid)[1]]
    val_model = gbm(best).fit(Xtr, ytr)
    val_pred = val_model.predict(Xva)
    val_ridge = ridge().fit(Xtr, ytr).predict(Xva)
    val_base = baselines(train, valid)
    val_scores = {"ML": mae(yva, val_pred), "Ridge": mae(yva, val_ridge),
                  **{k: mae(yva, val_base[c]) for k, c in BASELINES.items()}}
    best_baseline = min(BASELINES, key=lambda k: val_scores[k])
    q = float(np.quantile(np.abs(yva - val_pred), 0.9, method="higher"))  # split-conformal style

    # ---- refit on all pre-test outcomes, predict test once
    fit = df[df.target_date < TEST_START].copy()
    Xfit = design(fit, stations)
    final = gbm(best).fit(Xfit, fit.target_delta_pp)
    final_ridge = ridge().fit(Xfit, fit.target_delta_pp)
    out = test[["station_id", "reservoir_name", "date", "target_date", "target_delta_pp",
                "storage_af", "capacity_af", "storage_pct"]].copy()
    out["pred_delta_pp"] = final.predict(Xte)
    out["lower_90_pp"], out["upper_90_pp"] = out.pred_delta_pp - q, out.pred_delta_pp + q
    out = out.join(baselines(fit, test))
    out["pred_ridge_pp"] = final_ridge.predict(Xte)
    assert out[list(BASELINES.values())].notna().all().all()

    cols = {"ML": "pred_delta_pp", "Ridge": "pred_ridge_pp", **BASELINES}
    inside = (out.target_delta_pp >= out.lower_90_pp) & (out.target_delta_pp <= out.upper_90_pp)
    boot = block_bootstrap(out, list(cols.values()))
    diff = boot["pred_delta_pp"] - boot[BASELINES[best_baseline]]
    point_diff = mae(yte, out.pred_delta_pp) - mae(yte, out[BASELINES[best_baseline]])

    metrics = {"schema_version": SCHEMA_VERSION, "data_mode": "real",
               "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    metrics.update({k: mae(yte, out[c]) for k, c in cols.items()})
    metrics.update({
        "interval_90_coverage": float(inside.mean()), "interval_90_halfwidth_pp": q,
        "train_count": int(len(train)), "val_count": int(len(valid)), "test_count": int(len(test)),
        "test_station_count": int(test.station_id.nunique()),
        "note": ("Historical backtest on 2024-2025 held-out dates; not an operational forecast. "
                 "Weekly origins have overlapping 30-day windows, so rows are not independent; CIs use a "
                 "moving-block bootstrap over dates. 90% interval = 90th percentile of absolute 2023 validation "
                 "errors (single wet year); coverage is not guaranteed under distribution shift. Weather is NASA "
                 "POWER grid-cell data at the dam, not watershed rainfall, and is a retrospective (revised) series. "
                 "ORO windows spanning the 2024-07-02 storage-table change are excluded."),
        "split": {"train_target_end_exclusive": VAL_START, "val_start": VAL_START,
                  "val_target_end_exclusive": TEST_START, "test_start": TEST_START,
                  "test_target_end_inclusive": last_day.strftime("%Y-%m-%d"), "horizon_days": HORIZON,
                  "weather_lag_days": WEATHER_LAG, "sampling": "weekly (Mondays)"},
        "validation": {"scores_mae_pp": val_scores, "gbm_grid_mae_pp": [g for g, _ in grid],
                       "gbm_selected_params": best, "best_baseline_on_validation": best_baseline},
        "per_station": {},
        "bootstrap": {"method": "moving-block bootstrap over weekly test origin dates (stations resampled "
                                "jointly per date); percentile 95% CI",
                      "n_boot": N_BOOT, "block_weeks": BLOCK_WEEKS, "best_baseline": best_baseline,
                      "best_baseline_selected_on": "validation",
                      "mae_ci95": {k: ci(boot[c]) for k, c in cols.items()},
                      "mae_diff_ml_minus_best_baseline": {"estimate": point_diff, **ci(diff),
                                                          "share_boot_ml_better": float((diff < 0).mean())}},
    })
    for s, g in out.groupby("station_id"):
        gi = (g.target_delta_pp >= g.lower_90_pp) & (g.target_delta_pp <= g.upper_90_pp)
        metrics["per_station"][s] = {"n": int(len(g)), **{k: mae(g.target_delta_pp, g[c]) for k, c in cols.items()},
                                     "interval_90_coverage": float(gi.mean())}

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"gbm": final, "ridge": final_ridge, "stations": stations, "features": list(Xfit.columns),
                 "halfwidth_90_pp": q, "gbm_params": best}, ROOT / "models/forecast.joblib")
    OUT = ROOT / "outputs"
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "test_predictions.csv", index=False, date_format="%Y-%m-%d")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_feature_catalog(OUT)
    errs = validate_dir(OUT)
    print("contract:", errs or "OK")
    show = {k: round(v, 3) for k, v in metrics.items() if isinstance(v, (int, float))}
    print(json.dumps(show, indent=1))
    print("validation:", json.dumps(metrics["validation"], indent=1))
    print("bootstrap diff:", json.dumps(metrics["bootstrap"]["mae_diff_ml_minus_best_baseline"]))
    print(pd.DataFrame(metrics["per_station"]).T.round(2).to_string())


def write_feature_catalog(out_dir):
    from src.make_sample_data import FEATURES as CATALOG
    (out_dir / "feature_catalog.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "features": [
        {"name": n, "description": d, "units": u, "source": s, "min_lag_days": l} for n, d, u, s, l in CATALOG]},
        indent=2))


if __name__ == "__main__":
    run()

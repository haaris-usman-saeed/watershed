"""Evaluate honest baselines and ML on future, non-overlapping dates."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct",
           "rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct",
           "season_sin", "season_cos"]


def to_features(frame, stations):
    onehots = pd.get_dummies(frame.station_id, prefix="station", dtype=int)
    onehots = onehots.reindex(columns=[f"station_{s}" for s in stations], fill_value=0)
    return pd.concat([frame[NUMERIC].reset_index(drop=True), onehots.reset_index(drop=True)], axis=1)


def run():
    data_file = ROOT / "data/processed/weekly_features.csv"
    if not data_file.exists():
        raise FileNotFoundError("Run python -m src.features first")
    df = pd.read_csv(data_file, parse_dates=["date", "target_date"])
    # Training target outcomes must be observable before validation starts.
    train = df.loc[df.target_date < "2023-01-01"].copy()
    valid = df.loc[(df.date >= "2023-01-01") & (df.target_date < "2024-01-01")].copy()
    test = df.loc[(df.date >= "2024-01-01") & (df.target_date <= "2025-12-31")].copy()
    if min(len(train), len(valid), len(test)) < 30:
        raise RuntimeError(f"Not enough data: train={len(train)}, val={len(valid)}, test={len(test)}")
    stations = sorted(df.station_id.unique().tolist())
    def new_model(lr=0.05, n=180, leaf=20):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("model", HistGradientBoostingRegressor(learning_rate=lr,
                           max_iter=n, min_samples_leaf=leaf, l2_regularization=1,
                           random_state=42))])
    Xtr, Xva = to_features(train, stations), to_features(valid, stations)
    Xte = to_features(test, stations)
    ytr, yva, yte = train.target_delta_pp, valid.target_delta_pp, test.target_delta_pp
    settings = [(0.05, 100, 20), (0.05, 220, 20), (0.025, 250, 30)]
    scored = []
    for opts in settings:
        model = new_model(*opts).fit(Xtr, ytr)
        pred = model.predict(Xva)
        scored.append((mean_absolute_error(yva, pred), opts))
    best_opts = min(scored, key=lambda p: p[0])[1]
    # Train on all observations with fully known labels before 2024.
    fitting = df.loc[df.target_date < "2024-01-01"].copy()
    final = new_model(*best_opts).fit(to_features(fitting, stations), fitting.target_delta_pp)
    pred = final.predict(Xte)
    # Caution: interval calibration is based only on the pre-test validation period,
    # so we train a separate validation-era model for calibration.
    cal_model = new_model(*best_opts).fit(Xtr, ytr)
    val_errors = np.abs(yva.to_numpy() - cal_model.predict(Xva))
    q = float(np.quantile(val_errors, 0.9, method="higher"))
    test_pred = test[["station_id", "reservoir_name", "date", "target_date", "target_delta_pp", "storage_af", "capacity_af"]].copy()
    test_pred["pred_delta_pp"] = pred
    test_pred["lower_90_pp"] = pred - q
    test_pred["upper_90_pp"] = pred + q
    # Baseline 1: unchanged current level.
    test_pred["baseline_persistence_pp"] = 0.0
    # Baseline 2: extrapolate observed 30-day slope forward.
    test_pred["baseline_trend_pp"] = test.prev30_pct.to_numpy()
    # Baseline 3: average historical change for this station / month in fitting data.
    fitting = fitting.assign(month=fitting.date.dt.month)
    by_month = fitting.groupby(["station_id", "month"]).target_delta_pp.mean()
    overall = fitting.groupby("station_id").target_delta_pp.mean()
    test_pred["baseline_seasonal_pp"] = [by_month.get((s, d.month), overall.get(s, 0.0)) for s, d in zip(test.station_id, test.date)]
    predictions = {"ML": "pred_delta_pp", "Persistence": "baseline_persistence_pp",
                   "30-day trend": "baseline_trend_pp", "Historical seasonal": "baseline_seasonal_pp"}
    metrics = {name: float(mean_absolute_error(yte, test_pred[col])) for name, col in predictions.items()}
    metrics["interval_90_coverage"] = float(((test_pred.target_delta_pp >= test_pred.lower_90_pp) & (test_pred.target_delta_pp <= test_pred.upper_90_pp)).mean())
    metrics["interval_90_halfwidth_pp"] = q
    metrics["train_count"] = int(len(train)); metrics["val_count"] = int(len(valid)); metrics["test_count"] = int(len(test))
    metrics["test_station_count"] = int(test.station_id.nunique())
    metrics["note"] = "One temporal holdout; grouped uncertainty bootstrap not yet implemented. 90% interval is empirical and not guaranteed calibrated under distribution shift."
    model_dir = ROOT / "models"; model_dir.mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "stations": stations, "features": list(to_features(fitting, stations).columns), "halfwidth": q}, model_dir / "forecast.joblib")
    (ROOT / "outputs").mkdir(exist_ok=True)
    test_pred.to_csv(ROOT / "outputs/test_predictions.csv", index=False)
    (ROOT / "outputs/metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    run()

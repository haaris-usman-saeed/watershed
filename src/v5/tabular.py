"""v5 tabular models (Ridge, HGB) and the four baselines on the v2 Monday rows, per fold.

Training rows for fold start s: v2 Monday rows with target_date < s ("weekly"), or every 2nd daily origin with a valid
target and target_date < s ("daily", the networks' sample set: fitting plus early-stopping days). swe_anom_in uses the
snow climatology of days before s (src.models.snow_climatology / add_snow_anomaly, unchanged).
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models import add_snow_anomaly, baselines, design, snow_climatology
from src.v5 import CONFIRM_WYS, DEV_END, MODELS, RETRO, STATIONS, TUNE_WYS
from src.v5.registry import FROZEN_HGB, TABULAR, cols

STATION_LIST = sorted(STATIONS)
DAY0 = pd.Timestamp("2001-01-01")


def folds(which):
    """(fold_id, train_end, eval_start, eval_end_inclusive, target_end_inclusive)"""
    out = []
    if which in ("tune", "dev"):
        out += [(f"WY{w}", pd.Timestamp(f"{w - 1}-10-01"), pd.Timestamp(f"{w - 1}-10-01"),
                 pd.Timestamp(f"{w}-09-30"), DEV_END - pd.Timedelta(days=1)) for w in TUNE_WYS]
    if which in ("confirm", "dev"):
        out += [(f"WY{w}", pd.Timestamp(f"{w - 1}-10-01"), pd.Timestamp(f"{w - 1}-10-01"),
                 pd.Timestamp(f"{w}-09-30"), DEV_END - pd.Timedelta(days=1)) for w in CONFIRM_WYS]
    if which == "retro":
        out += [(k, pd.Timestamp(v["train_end"]), pd.Timestamp(v["start"]), pd.Timestamp(v["end"]),
                 pd.Timestamp(v["target_end"])) for k, v in RETRO.items()]
    return out


def fold_frames(feat, snow, train_end, ev_start, ev_end, tgt_end):
    v2 = feat[feat.v2_row]
    clim = snow_climatology(snow, train_end)
    train = add_snow_anomaly(v2[v2.target_date < train_end], clim).reset_index(drop=True)
    test = v2[(v2.date >= ev_start) & (v2.date <= ev_end) & (v2.target_date <= tgt_end)]
    test = add_snow_anomaly(test, clim).reset_index(drop=True)
    assert train.target_date.max() < test.date.min()
    daily = feat[feat.target_delta_pp.notna() & (feat.target_date < train_end)]
    daily = daily[((daily.date - DAY0).dt.days % 2) == 0]
    daily = add_snow_anomaly(daily, clim).reset_index(drop=True)
    return train, test, daily


def make_model(algo, hp=None):
    if algo == "hgb":
        p = dict(FROZEN_HGB) if hp is None else dict(hp)
        p.setdefault("l2_regularization", 1.0)
        p.setdefault("random_state", 42)
        return HistGradientBoostingRegressor(**p)
    if algo == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    if algo == "ridge_imp":
        if hp is None or hp.get("alpha") == "ridgecv":
            reg = RidgeCV(alphas=np.logspace(-2, 3, 20))
        else:
            reg = Ridge(alpha=hp["alpha"])
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), reg)
    raise ValueError(algo)


def matrix(frame, mid, daily=False):
    c = cols(TABULAR[mid]["codes"]) + (["gefs_age_days"] if daily else [])
    return design(frame, c, STATION_LIST)


def model_summary(model, algo):
    if algo == "hgb":
        leaves = [p.get_n_leaf_nodes() for it in model._predictors for p in it]
        return {"n_trees": int(model.n_iter_), "mean_leaves": float(np.mean(leaves)), "total_leaves": int(np.sum(leaves))}
    reg = model[-1]
    return {"alpha": float(reg.alpha_ if hasattr(reg, "alpha_") else reg.alpha), "n_coefficients": int(reg.coef_.size)}


def fit_predict(mid, train, test, daily, hp=None, save_to=None):
    cfg = TABULAR[mid]
    tr = daily if cfg["train"] == "daily" else train
    Xtr, Xte = matrix(tr, mid, cfg["train"] == "daily"), matrix(test, mid, cfg["train"] == "daily")
    model = make_model(cfg["algo"], hp)
    t0 = time.perf_counter()
    model.fit(Xtr, tr.target_delta_pp)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Xte)
    pred_s = time.perf_counter() - t0
    train_pred = model.predict(Xtr)
    info = {"n_train": int(len(tr)), "n_val": 0, "n_test": int(len(test)), "n_features": int(Xtr.shape[1]),
            "features": list(Xtr.columns), "fit_seconds": fit_s, "predict_seconds": pred_s,
            "train_mae": float(np.mean(np.abs(train_pred - tr.target_delta_pp))),
            "train_period": [str(tr.date.min().date()), str(tr.target_date.max().date())],
            **model_summary(model, cfg["algo"])}
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": list(Xtr.columns), "hp": hp, "config": cfg}, save_to, compress=3)
    return pred, info


def run_folds(feat, snow, which, model_ids, hp_by_model=None, stage="frozen", save=True, log=print):
    """Predictions of every model and baseline on the evaluation rows of each fold."""
    frames, infos = [], {}
    for fid, tr_end, ev_s, ev_e, tg_e in folds(which):
        train, test, daily = fold_frames(feat, snow, tr_end, ev_s, ev_e, tg_e)
        out = test[["station_id", "date", "target_date", "target_delta_pp", "storage_pct"]].copy()
        out["fold"] = fid
        b = baselines(train, test)
        out["BL-PERS"], out["BL-TREND"] = b.baseline_persistence_pp.to_numpy(), b.baseline_trend_pp.to_numpy()
        out["BL-MONTH"], out["BL-DOY"] = b.baseline_seasonal_pp.to_numpy(), b.baseline_doy_pp.to_numpy()
        infos[f"BASELINES {fid}"] = {"n_train": int(len(train)), "n_val": 0, "n_test": int(len(test))}
        for mid in model_ids:
            hp = (hp_by_model or {}).get(mid)
            name = mid if stage == "frozen" else stage
            path = MODELS / "tabular" / name / f"{fid}.joblib" if save else None
            out[name], infos[f"{name} {fid}"] = fit_predict(mid, train, test, daily, hp, path)
        frames.append(out)
        log(f"  {fid}: train {len(train)} weekly / {len(daily)} daily rows, eval {len(test)}")
    return pd.concat(frames, ignore_index=True), infos

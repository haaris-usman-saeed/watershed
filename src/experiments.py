"""Development experiments: expanding-window, rolling-origin evaluation by water year (v2).

Run: python -m src.experiments

Uses ONLY observations whose origin AND 30-day target date are before 2024-01-01; the
2024-2025 held-out period is never loaded into this module (asserted).

Folds: for evaluation water year W (Oct 1 of W-1 to Sep 30 of W), train on every row with
target_date < Oct 1 of W-1 (expanding window, outcomes fully observed before the fold starts),
evaluate on origins inside W. Origins at the end of W whose targets fall in W+1 are evaluated
in W and are not used for training until fold W+2, so no outcome is used before it is observed.

Predetermined protocol
- Hyper-parameters of each HGB variant: chosen on tuning folds WY2008-2015 (pooled MAE).
- Comparison and selection: confirmation folds WY2016-2023 (not used for hyper-parameters).
- Primary v2 model = HGB variant with the lowest pooled MAE on confirmation folds.
- Strongest baseline = baseline with the lowest pooled MAE on confirmation folds.
- Interval method: evaluated on folds WY2012-2023, each calibrated only on residuals from
  earlier folds; choose the narrowest method whose pooled coverage is within [0.87, 0.93],
  else the method whose coverage is closest to 0.90.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.evaluation import BLOCK_LENGTHS, block_bootstrap_diff, coverage, mae, water_year
from src.features import SNOW
from src.models import (BASELINES, FEATURE_SETS, HGB_GRID, HGB_VARIANTS, INTERVAL_METHODS, add_snow_anomaly,
                        baselines, design, hgb, interval_offsets, ridge, season_of, snow_climatology)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v2" / "development"
HOLDOUT_START = pd.Timestamp("2024-01-01")
TUNE_WYS = list(range(2008, 2016))
CONFIRM_WYS = list(range(2016, 2024))
ALL_WYS = TUNE_WYS + CONFIRM_WYS
INTERVAL_WYS = list(range(2012, 2024))
COVERAGE_BAND = (0.87, 0.93)


def load_dev():
    df = pd.read_csv(ROOT / "data/processed/weekly_features.csv", parse_dates=["date", "target_date"])
    dev = df[(df.date < HOLDOUT_START) & (df.target_date < HOLDOUT_START)].copy()
    assert dev.target_date.max() < HOLDOUT_START and dev.date.max() < HOLDOUT_START
    dev["wy"] = water_year(dev.date)
    dev["season"] = season_of(dev.date)
    snow = pd.read_csv(ROOT / "data/processed/snow_basin_index.csv", parse_dates=["date"])
    snow = snow[snow.date < HOLDOUT_START]
    return dev.reset_index(drop=True), snow


def fold(dev, snow, wy):
    start = pd.Timestamp(f"{wy - 1}-10-01")
    end = pd.Timestamp(f"{wy}-10-01")
    train = dev[dev.target_date < start]
    test = dev[(dev.date >= start) & (dev.date < end)]
    assert train.target_date.max() < test.date.min()
    clim = snow_climatology(snow, start)
    return add_snow_anomaly(train, clim), add_snow_anomaly(test, clim)


def fit_predict(train, test, cols, stations, params):
    m = hgb(params).fit(design(train, cols, stations), train.target_delta_pp)
    return m.predict(design(test, cols, stations))


def tune(dev, snow, stations):
    rows = []
    folds = {w: fold(dev, snow, w) for w in TUNE_WYS}
    for name, fs in HGB_VARIANTS.items():
        for i, params in enumerate(HGB_GRID):
            err, n = 0.0, 0
            for w, (tr, te) in folds.items():
                p = fit_predict(tr, te, FEATURE_SETS[fs], stations, params)
                err += np.abs(te.target_delta_pp.to_numpy() - p).sum()
                n += len(te)
            rows.append({"variant": name, "grid_index": i, **params, "pooled_mae_tuning": err / n, "n": n})
            print(f"  tune {name:20s} {i} {err / n:.3f}", flush=True)
    t = pd.DataFrame(rows)
    best = {v: HGB_GRID[int(g.loc[g.pooled_mae_tuning.idxmin(), "grid_index"])] for v, g in t.groupby("variant")}
    return t, best


def evaluate(dev, snow, stations, best):
    preds = []
    for w in ALL_WYS:
        tr, te = fold(dev, snow, w)
        out = te[["station_id", "date", "target_date", "wy", "season", "target_delta_pp", "storage_pct"]].copy()
        out = out.join(baselines(tr, te))
        for name, fs in HGB_VARIANTS.items():
            out[f"pred_{name}"] = fit_predict(tr, te, FEATURE_SETS[fs], stations, best[name])
        rcols = FEATURE_SETS["storage_weather"]
        out["pred_Ridge"] = ridge().fit(design(tr, rcols, stations), tr.target_delta_pp).predict(design(te, rcols, stations))
        out["n_train"] = len(tr)
        preds.append(out)
        print(f"  fold WY{w}: train={len(tr)} eval={len(te)}", flush=True)
    return pd.concat(preds, ignore_index=True)


def model_cols():
    cols = {name: f"pred_{name}" for name in HGB_VARIANTS}
    cols["Ridge"] = "pred_Ridge"
    cols.update(BASELINES)
    return cols


def pooled(p, cols):
    return {k: mae(p.target_delta_pp, p[c]) for k, c in cols.items()}


def by(p, key, cols):
    return pd.DataFrame([{key: k, "n": len(g), **{m: mae(g.target_delta_pp, g[c]) for m, c in cols.items()}}
                         for k, g in p.groupby(key)])


def interval_study(p, col):
    p = p.assign(r=p.target_delta_pp - p[col])
    rows, per = [], []
    for method in INTERVAL_METHODS:
        lo_all, hi_all, sub_all = [], [], []
        for w in INTERVAL_WYS:
            cal = p[p.wy < w]
            sub = p[p.wy == w]
            lo, hi = interval_offsets(cal, method)(sub)
            lo_all.append(sub[col].to_numpy() + lo)
            hi_all.append(sub[col].to_numpy() + hi)
            sub_all.append(sub)
        s = pd.concat(sub_all)
        lo, hi = np.concatenate(lo_all), np.concatenate(hi_all)
        rows.append({"method": method, "n": len(s), "coverage": coverage(s.target_delta_pp, lo, hi),
                     "mean_width_pp": float(np.mean(hi - lo))})
        s = s.assign(lo=lo, hi=hi)
        for key in ("season", "station_id", "wy"):
            for k, g in s.groupby(key):
                per.append({"method": method, "group_type": key, "group": k, "n": len(g),
                            "coverage": coverage(g.target_delta_pp, g.lo, g.hi), "mean_width_pp": float((g.hi - g.lo).mean())})
    t = pd.DataFrame(rows)
    ok = t[t.coverage.between(*COVERAGE_BAND)]
    chosen = (ok.sort_values("mean_width_pp").iloc[0] if len(ok)
              else t.iloc[(t.coverage - 0.90).abs().argsort()].iloc[0]).method
    return t, pd.DataFrame(per), chosen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev, snow = load_dev()
    stations = sorted(dev.station_id.unique())
    print(f"dev rows={len(dev)} ({dev.date.min().date()} to {dev.target_date.max().date()})")
    tuning, best = tune(dev, snow, stations)
    tuning.to_csv(OUT / "hgb_tuning.csv", index=False)
    p = evaluate(dev, snow, stations, best)
    cols = model_cols()
    wyt = pd.read_csv(ROOT / "data" / "water_year_types.csv").set_index("water_year")
    sac = {"SHA", "ORO", "FOL"}
    p["wy_type"] = [wyt.loc[w, "sac_type"] if s in sac else wyt.loc[w, "sjv_type"] for s, w in zip(p.station_id, p.wy)]
    p["wy_class"] = p.wy_type.map({"W": "wet (W/AN)", "AN": "wet (W/AN)", "BN": "below normal",
                                   "D": "dry (D/C)", "C": "dry (D/C)"})
    conf, tunep = p[p.wy.isin(CONFIRM_WYS)], p[p.wy.isin(TUNE_WYS)]
    summary = {"pooled_mae_all_folds": pooled(p, cols), "pooled_mae_tuning_folds": pooled(tunep, cols),
               "pooled_mae_confirmation_folds": pooled(conf, cols)}
    hgb_names = list(HGB_VARIANTS)
    base_names = list(BASELINES)
    primary = min(hgb_names, key=lambda k: summary["pooled_mae_confirmation_folds"][k])
    strongest = min(base_names, key=lambda k: summary["pooled_mae_confirmation_folds"][k])
    per_wy = by(p, "wy", cols)
    wins = {m: {"folds_beating_strongest_baseline": int((per_wy[m] < per_wy[strongest]).sum()), "of": len(per_wy)}
            for m in hgb_names + ["Ridge"]}
    ablations = {}
    pairs = {"weather (HGB vs HGB no weather)": ("HGB", "HGB no weather"),
             "SWE given weather (HGB + SWE vs HGB)": ("HGB + SWE", "HGB"),
             "SWE without weather (HGB SWE no weather vs HGB no weather)": ("HGB SWE no weather", "HGB no weather"),
             "primary vs strongest baseline": (primary, strongest)}
    for label, (a, b) in pairs.items():
        ca, cb = cols[a], cols[b]
        ablations[label] = {scope: {f"{bl}_weeks": block_bootstrap_diff(d, ca, cb, block=bl) for bl in BLOCK_LENGTHS}
                            for scope, d in (("confirmation_WY2016_2023", conf), ("all_WY2008_2023", p))}
    istudy, iper, imethod = interval_study(p, cols[primary])
    summary.update({"primary_model": primary, "primary_feature_set": HGB_VARIANTS[primary],
                    "primary_hyperparameters": best[primary], "all_hgb_hyperparameters": best,
                    "strongest_baseline": strongest, "interval_method": imethod,
                    "wins_by_water_year_vs_strongest_baseline": wins, "ablations_mae_difference": ablations,
                    "n_rows": {"all": len(p), "tuning": len(tunep), "confirmation": len(conf)},
                    "folds": {"tuning": TUNE_WYS, "confirmation": CONFIRM_WYS, "interval": INTERVAL_WYS}})
    keep = ["station_id", "date", "target_date", "wy", "season", "wy_type", "target_delta_pp"] + list(cols.values())
    p[keep].to_csv(OUT / "oof_predictions.csv", index=False, date_format="%Y-%m-%d")
    for key in ("station_id", "wy", "season", "wy_class"):
        by(p, key, cols).to_csv(OUT / f"mae_by_{key}_all_folds.csv", index=False)
        by(conf, key, cols).to_csv(OUT / f"mae_by_{key}_confirmation.csv", index=False)
    istudy.to_csv(OUT / "interval_methods.csv", index=False)
    iper.to_csv(OUT / "interval_methods_by_group.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("pooled_mae_all_folds", "pooled_mae_tuning_folds",
                                              "pooled_mae_confirmation_folds", "primary_model", "strongest_baseline",
                                              "interval_method", "wins_by_water_year_vs_strongest_baseline",
                                              "all_hgb_hyperparameters")}, indent=1, default=float))
    print(istudy.to_string(index=False))
    for label, r in ablations.items():
        c = r["confirmation_WY2016_2023"]
        print(label, {k: (round(v["estimate"], 3), round(v["ci95_low"], 3), round(v["ci95_high"], 3)) for k, v in c.items()})
    print(by(p, "wy", cols).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

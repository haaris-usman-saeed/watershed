"""v5 report: consolidated tables, statistics, inventories and figures from the saved runs (BENCHMARK_PLAN.md §7-8).

Run: python -m src.v5.report       # after src.v5.run has finished every stage

Writes aggregates only (no CDEC value) to results/v5/: metrics/*.csv, stats/*.json|csv, model_inventory.csv,
feature_inventory.csv, hyperparameter_registry.json, row_accounting.csv, figures/*.png|svg, summary.json.
Per-row merged predictions go to data/v5/preds/all_predictions.parquet (private).
"""
import json

import numpy as np
import pandas as pd

from src.v5 import CONFIRM_WYS, DATA, RESULTS, RETRO, ROOT, SEEDS, TUNE_WYS
from src.v5.data import FLOW, GROUPS, load_features
from src.v5.evaluate import (QUANTILES, conformal90, family, point_metrics, quantile_intervals, unadjusted)
from src.v5.neural import load_runs
from src.v5.registry import (BASELINES, FROZEN_HGB, N4_CANDIDATES, NEURAL, NN_CODES, NN_DEFAULT, TABULAR, TUNED, code,
                             cols, search_space)

PREDS = DATA / "preds"
OUT = RESULTS
REF = "B-HGB-SNCR"
DEV_FOLDS = [f"WY{w}" for w in TUNE_WYS + CONFIRM_WYS]
CONF_FOLDS = [f"WY{w}" for w in CONFIRM_WYS]
TUNE_FOLDS = [f"WY{w}" for w in TUNE_WYS]
TAB_INPUTS = [c for g in ("S", "N", "Wd", "Wu", "G", "Q", "C") for c in GROUPS[g] if c != "swe_anom_in"]
ARTIFACT = "REF-V2-ARTIFACT"


# ------------------------------------------------------------------ model metadata

def describe(mid):
    """Descriptive name, information code, algorithm, class, hyperparameter regime and training regime."""
    if mid in BASELINES:
        return {"name": BASELINES[mid], "code": f"{mid}", "algorithm": "baseline", "class": "src.models.baselines",
                "library": "numpy/pandas", "experiment": "both", "hp": "none (nothing is fitted beyond means)",
                "regime": "per-fold refit, v2 Monday training rows", "inputs": "target history (+ prev30_pct for trend)"}
    if mid == ARTIFACT:
        return {"name": "frozen v2 artifact 'HGB SWE no weather'", "code": code("HGB", ["S", "N", "C", "R"], " · artifact ≤2023"),
                "algorithm": "HistGradientBoostingRegressor", "class": "sklearn.ensemble.HistGradientBoostingRegressor",
                "library": "scikit-learn", "experiment": "reference (recorded predictions; not retrained)",
                "hp": "frozen v2", "regime": "trained once on targets before 2024-01-01 (v2 final)", "inputs": "15 columns"}
    tuned = mid in TUNED
    base = TUNED.get(mid, mid)
    if base in TABULAR:
        t = TABULAR[base]
        algo = {"hgb": "HistGradientBoostingRegressor", "ridge": "Ridge (RidgeCV)", "ridge_imp": "Ridge (median imputation + indicators)"}[t["algo"]]
        cls = {"hgb": "sklearn.ensemble.HistGradientBoostingRegressor",
               "ridge": "Pipeline(StandardScaler, RidgeCV(logspace(-2,3,20)))",
               "ridge_imp": "Pipeline(SimpleImputer(median, indicators), StandardScaler, Ridge/RidgeCV)"}[t["algo"]]
        regime = "per-fold refit, " + ("v2 Monday rows" if t["train"] == "weekly" else "every 2nd daily origin")
        c = code("HGB" if t["algo"] == "hgb" else "Ridge", t["codes"], " · daily" if t["train"] == "daily" else "")
        return {"name": ("tuned " if tuned else "") + t["origin"], "code": c + (" · tuned" if tuned else ""),
                "algorithm": algo, "class": cls, "library": "scikit-learn 1.9.1", "experiment": "A (tuned)" if tuned else t["exp"],
                "hp": "equal-budget tuned (20 configurations)" if tuned else ("frozen v2" if t["algo"] == "hgb" else "RidgeCV (efficient LOO)"),
                "regime": regime,
                "inputs": f"{len(cols(t['codes'])) + 6 + (1 if t['train'] == 'daily' else 0)} columns: "
                          + " ".join(f"{g}{len(GROUPS[g])}" for g in t["codes"] if g in GROUPS) + " R6"
                          + (" + gefs_age" if t["train"] == "daily" else "") + (" (+ missing indicators)" if t["algo"] == "ridge_imp" else ""),
                "input_columns": ", ".join(cols(t["codes"])) + ", 6 station one-hots" + (", gefs_age_days" if t["train"] == "daily" else "")}
    n = NEURAL[base]
    cand = n.get("cand")
    if tuned and n["kind"] == "n4" and (OUT / "selection.json").exists():
        cand = json.loads((OUT / "selection.json").read_text())[mid]["hp"].get("cand", cand)
    algo = {"lstm": "N1 shared LSTM", "gru": "N2 shared GRU", "tcn": "N3 TCN", "n4": f"N4 encoder-decoder LSTM ({cand})"}[n["kind"]]
    short = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "n4": f"N4-{cand}"}[n["kind"]]
    codes_ = [c for c in NN_CODES[n["gefs"]] if not ("flow" in n["exclude"] and c == "Q")]
    return {"name": ("tuned " if tuned else "") + n["origin"], "code": code(short, codes_, "-seq · tuned" if tuned else "-seq30"),
            "algorithm": algo, "class": {"lstm": "src.v4.nets.SharedRNN(cell='lstm')", "gru": "src.v4.nets.SharedRNN(cell='gru')",
                                         "tcn": "src.v4.nets.TCN", "n4": "src.v4.nets.WaterBalanceED"}[n["kind"]],
            "library": "PyTorch 2.15.0", "experiment": "A (tuned)" if tuned else n["exp"],
            "hp": "equal-budget tuned (20 configurations)" if tuned else "v4 defaults",
            "regime": "per-fold refit, every 2nd daily origin, early stopping on the last 365 days before the fold, 3 seeds",
            "inputs": f"daily sequences L={L_of(mid)}: " + ",".join(c for c in codes_ if c not in ("G", "R")) + " + doy"
                      + "; static: log cap" + (" + GEFS(4)" if n["gefs"] else "") + "; reservoir embedding",
            "input_columns": "per day: storage %, arcsinh inflow/outflow pp, SWE/20, dam and box rain/temp/RH (values x masks), doy sin/cos; "
                             "static log capacity" + ("; GEFS d1-7, d8-16 (arcsinh/3), availability, age/7" if n["gefs"] else "")
                             + ("; flow channels withheld (mask 0)" if n["exclude"] else "")}


def L_of(mid):
    if mid in TUNED and (OUT / "selection.json").exists():
        return int(json.loads((OUT / "selection.json").read_text())[mid]["hp"]["length"])
    return int(NN_DEFAULT["length"])


# ------------------------------------------------------------------ loading

def _merge(left, right, cols_):
    m = left.merge(right[["station_id", "date"] + cols_], on=["station_id", "date"], how="left", validate="one_to_one")
    assert m[cols_].notna().all().all(), f"missing predictions for {cols_[:3]}"
    return m


def load_all():
    """One wide frame per population group: development (tuning + confirmation folds) and retrospective."""
    dev = pd.read_parquet(PREDS / "tabular_dev.parquet")
    tuned_tab = pd.concat([pd.read_parquet(PREDS / f"tabular_tuned_{w}.parquet") for w in ("tune", "confirm")], ignore_index=True)
    dev = dev.merge(tuned_tab[["station_id", "date", "T-RIDGE", "T-HGB-V5"]], on=["station_id", "date"], how="left")
    infos = {"tabular": json.loads((OUT / "runs" / "tabular_dev_info.json").read_text())}
    for w in ("tune", "confirm"):
        infos["tabular"].update(json.loads((OUT / "runs" / f"tabular_tuned_{w}_info.json").read_text()))
    nn_info = {}
    for m in NEURAL:
        f, inf = load_runs("frozen", m, DEV_FOLDS, SEEDS)
        dev = _merge(dev, f, [c for c in f.columns if c.startswith(m + "_") or c == m])
        nn_info.update(inf)
    for tid, base in TUNED.items():
        if base in NEURAL:
            f, inf = load_runs("tuned", tid, DEV_FOLDS, SEEDS)
            dev = _merge(dev, f, [c for c in f.columns if c.startswith(tid)])
            nn_info.update(inf)
    retro = pd.read_parquet(PREDS / "tabular_retro.parquet")
    infos["tabular_retro"] = json.loads((OUT / "runs" / "tabular_retro_info.json").read_text())
    for m in NEURAL:
        f, inf = load_runs("frozen", m, list(RETRO), SEEDS)
        retro = _merge(retro, f, [c for c in f.columns if c.startswith(m + "_") or c == m])
        nn_info.update({f"retro {k}": v for k, v in inf.items()})
    for tid, base in TUNED.items():
        if base in NEURAL:
            f, inf = load_runs("tuned", tid, list(RETRO), SEEDS)
            retro = _merge(retro, f, [c for c in f.columns if c.startswith(tid)])
            nn_info.update({f"retro {k}": v for k, v in inf.items()})
    # frozen v2 artifact: recorded predictions (HELD: v2 final; T26: v3 final, pred_A_frozen)
    h = pd.read_csv(ROOT / "results/v2/final/test_predictions_public.csv", parse_dates=["date"])[["station_id", "date", "pred_delta_pp"]]
    t = pd.read_csv(ROOT / "data/v3/private/test_predictions_v3.csv", parse_dates=["date"])[["station_id", "date", "pred_A_frozen"]]
    art = pd.concat([h.rename(columns={"pred_delta_pp": ARTIFACT}), t.rename(columns={"pred_A_frozen": ARTIFACT})])
    retro = _merge(retro, art, [ARTIFACT])
    infos["nn"] = nn_info
    feat = load_features(None)
    flags = feat[feat.v2_row][["station_id", "date"] + TAB_INPUTS].copy()
    flags["gefs_available"] = flags.gefs_p_d1_7_mm.notna()
    flags["complete"] = flags[TAB_INPUTS].notna().all(axis=1)
    for df in (dev, retro):
        df.drop(columns=[c for c in ("gefs_available", "complete") if c in df], inplace=True)
    dev = dev.merge(flags[["station_id", "date", "gefs_available", "complete"]], on=["station_id", "date"], how="left")
    retro = retro.merge(flags[["station_id", "date", "gefs_available", "complete"]], on=["station_id", "date"], how="left")
    for df in (dev, retro):
        wyt = pd.read_csv(ROOT / "data" / "water_year_types.csv").set_index("water_year")
        wy = df.date.dt.year + (df.date.dt.month >= 10).astype(int)
        sac = df.station_id.isin(["SHA", "ORO", "FOL"])
        typ = np.where(sac, wy.map(wyt.sac_type), wy.map(wyt.sjv_type))
        df["wy"] = wy
        df["wy_type"] = typ
        df["wy_class"] = pd.Series(typ).map({"W": "wet (W/AN)", "AN": "wet (W/AN)", "BN": "dry (BN/D/C)", "D": "dry (BN/D/C)",
                                             "C": "dry (BN/D/C)"}).fillna("unclassified").to_numpy()
        df["regime"] = np.select([df.target_delta_pp >= 10, df.target_delta_pp <= -10],
                                 ["rapid fill (>= +10 pp)", "rapid drawdown (<= -10 pp)"], "other")
    return dev, retro, infos


# ------------------------------------------------------------------ tables

def model_columns(frame):
    base = list(BASELINES) + list(TABULAR) + list(NEURAL)
    extra = [m for m in TUNED if m in frame] + ([ARTIFACT] if ARTIFACT in frame else [])
    return [m for m in base if m in frame] + extra


def compute_info(mid, infos, folds_):
    """Training rows, validation rows, training MAE, training and inference time, parameters, over `folds_`."""
    out = {}
    base = TUNED.get(mid, mid)
    if mid in BASELINES or mid == ARTIFACT:
        src = infos["tabular"] if not set(folds_) & set(RETRO) else infos["tabular_retro"]
        recs = [src.get(f"BASELINES {f}") for f in folds_ if f"BASELINES {f}" in src]
        if recs and mid in BASELINES:
            out.update(train_rows=[r["n_train"] for r in recs], val_rows=0)
        return out
    if base in TABULAR:
        src = infos["tabular_retro"] if set(folds_) & set(RETRO) else infos["tabular"]
        recs = [src[f"{mid} {f}"] for f in folds_ if f"{mid} {f}" in src]
        if not recs:
            return out
        out.update(train_rows=[r["n_train"] for r in recs], val_rows=0, train_mae=float(np.mean([r["train_mae"] for r in recs])),
                   train_seconds_total=float(sum(r["fit_seconds"] for r in recs)),
                   infer_ms_per_row=float(1000 * sum(r["predict_seconds"] for r in recs) / sum(r["n_test"] for r in recs)),
                   n_features=recs[0]["n_features"])
        if "n_trees" in recs[0]:
            out.update(n_trees=float(np.mean([r["n_trees"] for r in recs])), mean_leaves=float(np.mean([r["mean_leaves"] for r in recs])),
                       params=f"{int(np.mean([r['n_trees'] for r in recs]))} trees, {int(np.mean([r['total_leaves'] for r in recs]))} leaves (fold mean)")
        else:
            out.update(alpha=[r["alpha"] for r in recs], params=f"{recs[0]['n_coefficients']} coefficients")
        return out
    src = infos["nn"]
    pre = "retro " if set(folds_) & set(RETRO) else ""
    recs = [src[f"{pre}{mid} {f} s{s}"] for f in folds_ for s in SEEDS if f"{pre}{mid} {f} s{s}" in src]
    if not recs:
        return out
    by_fold = {f: [src[f"{pre}{mid} {f} s{s}"] for s in SEEDS] for f in folds_}
    out.update(train_rows=[v[0]["n_fit"] for v in by_fold.values()], val_rows=[v[0]["n_val"] for v in by_fold.values()],
               train_mae=float(np.mean([r["train_mae"] for r in recs])),
               val_mae=float(np.mean([r["val_mae_best"] for r in recs])),
               train_seconds_total=float(sum(r["train_seconds"] for r in recs)),
               infer_ms_per_row=float(1000 * sum(r["infer_seconds"] for r in recs) / sum(r["n_test"] for r in recs)),
               params=f"{recs[0]['n_params']} trainable", n_params=int(recs[0]["n_params"]),
               epochs_best_mean=float(np.mean([r["best_epoch"] for r in recs])))
    return out


def metrics_table(frame, models, infos, folds_, conformal_order=None, label=""):
    rows = []
    doy = point_metrics(frame.target_delta_pp, frame["BL-DOY"])["mae"]
    ref = point_metrics(frame.target_delta_pp, frame[REF])["mae"] if REF in frame else np.nan
    for m in models:
        if m not in frame or frame[m].isna().any():
            continue
        r = {"model": m, **describe(m), **point_metrics(frame.target_delta_pp, frame[m])}
        r["skill_vs_doy"] = 1 - r["mae"] / doy
        r["delta_mae_vs_hgb_ref"] = r["mae"] - ref
        seeds = [f"{m}_s{s}" for s in SEEDS]
        if all(c in frame for c in seeds):
            v = [point_metrics(frame.target_delta_pp, frame[c])["mae"] for c in seeds]
            r["seed_mae_mean"], r["seed_mae_sd"] = float(np.mean(v)), float(np.std(v, ddof=1))
        qc = [f"{m}_q{t}" for t in QUANTILES]
        if all(c in frame for c in qc):
            r.update({f"native_{k}": v for k, v in quantile_intervals(frame.target_delta_pp.to_numpy(), frame[qc].to_numpy()).items()})
        if conformal_order is not None and f"cf90_{m}" in frame:
            hw = frame[f"cf90_{m}"]
            ok = hw.notna()
            r["conformal90_coverage"] = float(np.mean((frame.target_delta_pp[ok] - frame[m][ok]).abs() <= hw[ok]))
            r["conformal90_width"] = float(np.mean(2 * hw[ok]))
        inf = compute_info(m, infos, folds_)
        tr = inf.pop("train_rows", None)
        va = inf.pop("val_rows", None)
        r["train_rows"] = (f"{min(tr)}–{max(tr)} per fold (sum {sum(tr)})" if isinstance(tr, list) and len(tr) > 1 else
                           (str(tr[0]) if isinstance(tr, list) else ""))
        r["validation_rows"] = (f"{min(va)}–{max(va)} per fold" if isinstance(va, list) and len(va) > 1 else
                                (str(va[0]) if isinstance(va, list) else ("0" if va == 0 else "")))
        r["test_rows"] = int(len(frame))
        r.update({k: v for k, v in inf.items() if not isinstance(v, list)})
        r["population"] = label
        rows.append(r)
    return pd.DataFrame(rows)


def breakdown(frame, models, by):
    out = []
    for k, g in frame.groupby(by):
        for m in models:
            if m in g and not g[m].isna().any():
                out.append({by: k, "model": m, "n": len(g), "mae": float(np.mean(np.abs(g[m] - g.target_delta_pp))),
                            "bias": float(np.mean(g[m] - g.target_delta_pp))})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ statistics (BENCHMARK_PLAN.md §7)

FAMILY_F = {
    "P1": ("HGB v5 vs HGB[S,N,C,R] (existing HGB primary configuration, per-fold refit)", {"HGB-V5": 1, REF: -1}),
    "P2": ("Ridge competitiveness under identical information: [Ridge[all] - HGB v5] - [Ridge[S,C,R] - HGB[S,C,R]]",
           {"A-RIDGE": 1, "HGB-V5": -1, "B-RIDGE-SCR": -1, "B-HGB-SCR": 1}),
    "P3": ("LSTM [all] vs HGB v5", {"A-N1": 1, "HGB-V5": -1}),
    "P4": ("GRU [all] vs HGB v5", {"A-N2": 1, "HGB-V5": -1}),
    "P5": ("TCN [all] vs HGB v5", {"A-N3": 1, "HGB-V5": -1}),
    "P6": ("N4 wb_noaux [all] vs HGB v5", {"A-N4": 1, "HGB-V5": -1}),
    "P7": ("HGB v5 (all variables) vs HGB[S,C,R] (storage, calendar, reservoir only)", {"HGB-V5": 1, "B-HGB-SCR": -1}),
    "P8": ("Water-balance structure: N4 wb_noaux [all] vs N4 direct [all]", {"A-N4": 1, "A-N4-DIRECT": -1}),
}
FAMILY_T = {
    "T1": ("tuned HGB v5 vs HGB[S,N,C,R] (frozen reference)", {"T-HGB-V5": 1, REF: -1}),
    "T2": ("tuned Ridge[all] vs tuned HGB v5", {"T-RIDGE": 1, "T-HGB-V5": -1}),
    "T3": ("tuned LSTM [all] vs tuned HGB v5", {"T-N1": 1, "T-HGB-V5": -1}),
    "T4": ("tuned GRU [all] vs tuned HGB v5", {"T-N2": 1, "T-HGB-V5": -1}),
    "T5": ("tuned TCN [all] vs tuned HGB v5", {"T-N3": 1, "T-HGB-V5": -1}),
    "T6": ("tuned N4 [all] vs tuned HGB v5", {"T-N4": 1, "T-HGB-V5": -1}),
}


def secondary_contrasts():
    s = {}
    for m in [k for k in TABULAR if k.startswith("B-HGB") and k != REF] + ["HGB-V5"]:
        s[f"{m} vs {REF}"] = (f"{describe(m)['code']} vs HGB[S,N,C,R]", {m: 1, REF: -1})
    for a, b in (("A-RIDGE", "B-RIDGE-SCR"), ("A-N1", "B-N1"), ("A-N2", "B-N2"), ("A-N3", "B-N3"), ("A-N4", "B-N4-NOAUX"),
                 ("A-N4-DIRECT", "B-N4-DIRECT"), ("A-HGB-DAILY", "HGB-V5"), ("B-N1", "B-N1-NOFLOW"), ("A-RIDGE", "B-RIDGE-SWdCR")):
        s[f"{a} vs {b}"] = (f"{describe(a)['code']} vs {describe(b)['code']}", {a: 1, b: -1})
    for t, b in TUNED.items():
        s[f"{t} vs {b}"] = (f"tuned vs frozen: {describe(b)['code']}", {t: 1, b: -1})
    for v in ("B-N4-NOAUX", "B-N4-BAL", "B-N4-PRIM", "B-N4-STRICT"):
        s[f"{v} vs B-N4-DIRECT"] = (f"N4 {v} vs N4 direct (original inputs)", {v: 1, "B-N4-DIRECT": -1})
        s[f"{v} vs B-N1"] = (f"N4 {v} vs N1 (original inputs)", {v: 1, "B-N1": -1})
    for m in ("HGB-V5", REF, "A-N1", "A-N2", "A-N3", "A-N4", "A-RIDGE"):
        s[f"{m} vs BL-DOY"] = (f"{describe(m)['code']} vs DOY climatology", {m: 1, "BL-DOY": -1})
    return s


def _flat(res, family_name, population):
    rows = []
    for cid, r in res.items():
        row = {"family": family_name, "population": population, "id": cid, "description": r["description"],
               "contrast": " ".join(f"{'+' if w > 0 else '-'}{abs(w)}*{c}" for c, w in r["weights"].items()),
               "estimate_pp": r["estimate"], "n_rows": r["n_rows"], "n_dates": r["n_dates"], "p_max": r["p_max"]}
        for b, v in r["blocks"].items():
            row[f"ci95_{b}"] = f"[{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}]"
            row[f"p_{b}"] = v["p_two_sided"]
        row["p_holm"] = r.get("p_holm", np.nan)
        row["verdict"] = r.get("verdict", r.get("verdict_unadjusted"))
        rows.append(row)
    return rows


def statistics(conf, retro):
    out = {"family_F": family(conf, FAMILY_F), "family_T": family(conf, FAMILY_T),
           "secondary": unadjusted(conf, {k: v for k, v in secondary_contrasts().items() if all(c in conf for c in v[1])})}
    for w in RETRO:
        g = retro[retro.fold == w]
        out[f"posthoc_{w}_F"] = family(g, FAMILY_F)
        out[f"posthoc_{w}_T"] = family(g, FAMILY_T)
        art = {f"{m} vs {ARTIFACT}": (f"{describe(m)['code']} vs frozen v2 artifact", {m: 1, ARTIFACT: -1})
               for m in ("HGB-V5", "T-HGB-V5", REF, "A-N1", "A-N4", "T-N4", "A-HGB-DAILY")}
        out[f"posthoc_{w}_vs_artifact"] = unadjusted(g, art)
    rows = _flat(out["family_F"], "F (frozen, Holm)", "confirmation WY2016-2023")
    rows += _flat(out["family_T"], "T (tuned, Holm)", "confirmation WY2016-2023")
    rows += _flat(out["secondary"], "secondary (unadjusted, exploratory)", "confirmation WY2016-2023")
    for w in RETRO:
        for k, lab in (("F", "F (frozen, Holm) - POST-HOC"), ("T", "T (tuned, Holm) - POST-HOC"),
                       ("vs_artifact", "vs frozen artifact (unadjusted) - POST-HOC")):
            rows += _flat(out[f"posthoc_{w}_{k}"], lab, f"{w} (post-hoc retrospective)")
    return out, pd.DataFrame(rows)


# ------------------------------------------------------------------ inventories

def feature_inventory():
    meta = {
        "storage_pct": ("S", "100 x S(t) / C(t)", "% of capacity", "day t", "current fill; bounds future change (flood space, dead pool)"),
        "prev7_pct": ("S", "100 x (S(t) - S(t-7)) / C(t), same series segment", "pp", "t-7..t", "recent momentum (operations persist)"),
        "prev30_pct": ("S", "as prev7 with 30 days", "pp", "t-30..t", "monthly momentum"),
        "prev60_pct": ("S", "as prev7 with 60 days", "pp", "t-60..t", "seasonal phase of fill/drawdown"),
        "swe_basin_in": ("N", "basin snow index (core pillows, sensor 82) at t-1", "inches", "t-1", "stored snow = future melt inflow"),
        "swe_change_14d_in": ("N", "SWE(t-1) - SWE(t-15)", "inches", "t-15..t-1", "melt onset / accumulation"),
        "swe_anom_in": ("N", "SWE(t-1) minus smoothed DOY climatology from days before the fold", "inches", "t-1", "snowpack relative to normal"),
        "rain_7d_mm": ("Wd", "NASA POWER precipitation at the dam point, sum over t-13..t-7", "mm", "t-13..t-7", "recent rain (7-day lag)"),
        "rain_30d_mm": ("Wd", "sum over t-36..t-7", "mm", "t-36..t-7", "antecedent wetness"),
        "temp_30d_c": ("Wd", "mean 2-m temperature over t-36..t-7", "deg C", "t-36..t-7", "melt energy, evaporation"),
        "humidity_30d_pct": ("Wd", "mean relative humidity over t-36..t-7", "%", "t-36..t-7", "evaporative demand"),
        "gefs_p_d1_7_mm": ("G", "GEFSv12 control member 00Z on the origin Monday, upstream-box mean precipitation, days 1-7", "mm",
                           "forecast for t+1..t+7", "known-at-origin forecast of near-term rain"),
        "gefs_p_d8_16_mm": ("G", "as above, days 8-16", "mm", "forecast for t+8..t+16", "medium-range rain"),
        "inflow_7d_pp": ("Q", "mean over t-6..t of 100 x inflow(cfs) x 1.983471 / C(day); missing unless all 7 days observed",
                         "% of capacity per day", "t-6..t", "current hydrologic inflow (rising limb, melt, baseflow memory)"),
        "inflow_30d_pp": ("Q", "as above over t-29..t (30 days)", "% of capacity per day", "t-29..t",
                          "horizon-length inflow regime; recession memory"),
        "outflow_7d_pp": ("Q", "mean over t-6..t of 100 x outflow(cfs) x 1.983471 / C(day); missing unless all 7 days observed",
                          "% of capacity per day", "t-6..t", "current release schedule (operations persist)"),
        "outflow_30d_pp": ("Q", "as above over t-29..t", "% of capacity per day", "t-29..t", "monthly release regime (irrigation, flood control)"),
        "season_sin": ("C", "sin(2 pi doy / 365.25)", "-", "day t", "annual cycle"),
        "season_cos": ("C", "cos(2 pi doy / 365.25)", "-", "day t", "annual cycle"),
    }
    for suf in ("_box",):
        for c in ("rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"):
            g, d, u, w, j = meta[c]
            meta[c + suf] = ("Wu", d.replace("at the dam point", "averaged over the upstream box").replace("mean ", "upstream-box mean "),
                             u, w, j + " (upstream basin)")
    rows = [{"feature": k, "group": v[0], "definition": v[1], "units": v[2], "window": v[3], "justification": v[4],
             "in_HGB_v5": True} for k, v in meta.items()]
    rows += [{"feature": f"station_{s}", "group": "R", "definition": "one-hot reservoir indicator", "units": "0/1", "window": "-",
              "justification": "reservoir-specific operations and basin", "in_HGB_v5": True} for s in ("DNP", "EXC", "FOL", "NML", "ORO", "SHA")]
    rows.append({"feature": "gefs_age_days", "group": "G", "definition": "days since the GEFS cycle used (0 on Mondays)", "units": "days",
                 "window": "-", "justification": "daily-origin training only (A-HGB-DAILY); 0 on every evaluation row", "in_HGB_v5": False})
    return pd.DataFrame(rows)


def hp_registry():
    sel = json.loads((OUT / "selection.json").read_text())
    reg = {"frozen": {"HGB (all B-HGB-*, HGB-V5, A-HGB-DAILY)": dict(FROZEN_HGB, max_leaf_nodes=31, max_bins=255, loss="squared_error",
                                                                       early_stopping="auto (inactive: n < 10000)"),
                      "Ridge (B-RIDGE-*, A-RIDGE)": {"alphas": "logspace(-2, 3, 20)", "selection": "RidgeCV efficient LOO within each fold"},
                      "networks (B-N*, A-N*)": dict(NN_DEFAULT, stride=2, early_stopping_days=365, clip=1.0, optimizer="AdamW",
                                                     loss="pinball over 9 quantiles (N4: composite)"),
                      "N4 candidates": N4_CANDIDATES},
           "tuned": {k: v for k, v in sel.items() if not k.startswith("_")}, "rule": sel.get("_rule"),
           "search_spaces": {t: search_space(t) for t in TUNED}}
    return reg


# ------------------------------------------------------------------ main

def main():
    import matplotlib
    matplotlib.use("Agg")
    from src.v5.figures import make_figures
    dev, retro, infos = load_all()
    models_dev = model_columns(dev)
    models_retro = model_columns(retro)
    # conformal intervals: development folds in order; retrospective windows calibrated on all development folds (+HELD for T26)
    allp = pd.concat([dev.assign(group="dev"), retro.assign(group="retro")], ignore_index=True)
    order = DEV_FOLDS + list(RETRO)
    for m in models_retro:
        if m in allp and m != ARTIFACT:
            avail = allp[allp[m].notna()]
            hw = conformal90(avail, m, [f for f in order if f in set(avail.fold)])
            allp.loc[avail.index, f"cf90_{m}"] = hw
    dev = allp[allp.group == "dev"].reset_index(drop=True)
    retro = allp[allp.group == "retro"].reset_index(drop=True)
    PREDS.mkdir(parents=True, exist_ok=True)
    allp.to_parquet(PREDS / "all_predictions.parquet", index=False)
    conf, tune = dev[dev.fold.isin(CONF_FOLDS)], dev[dev.fold.isin(TUNE_FOLDS)]
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    pops = {"tuning_WY2012-2015": (tune, TUNE_FOLDS, None),
            "confirmation_WY2016-2023": (conf, CONF_FOLDS, True),
            "confirmation_gefs_available": (conf[conf.gefs_available], CONF_FOLDS, True),
            "confirmation_complete_information": (conf[conf.complete], CONF_FOLDS, True),
            "HELD_posthoc": (retro[retro.fold == "HELD"], ["HELD"], True),
            "T26_posthoc": (retro[retro.fold == "T26"], ["T26"], True)}
    tables = {}
    for name, (frame, folds_, cf) in pops.items():
        ms = models_dev if name.startswith(("tuning", "confirmation")) else models_retro
        t = metrics_table(frame, ms, infos, folds_, cf, name)
        if name.startswith("tuning"):
            t["selected_on_these_rows"] = t.model.isin(list(TUNED))
        t.to_csv(OUT / "metrics" / f"{name}.csv", index=False)
        tables[name] = t
        for by in ("station_id", "wy_class", "regime", "fold"):
            breakdown(frame, ms, by).to_csv(OUT / "metrics" / f"{name}_by_{by}.csv", index=False)
    stats, stats_flat = statistics(conf, retro)
    (OUT / "stats").mkdir(parents=True, exist_ok=True)
    (OUT / "stats" / "statistics.json").write_text(json.dumps(stats, indent=1, default=float))
    stats_flat.to_csv(OUT / "stats" / "comparisons.csv", index=False)
    inv = pd.concat([pd.DataFrame([{"model": m, **describe(m)} for m in models_retro])])
    inv.to_csv(OUT / "model_inventory.csv", index=False)
    feature_inventory().to_csv(OUT / "feature_inventory.csv", index=False)
    (OUT / "hyperparameter_registry.json").write_text(json.dumps(hp_registry(), indent=1, default=str))
    row_acc = []
    for m in models_retro:
        for f in DEV_FOLDS + list(RETRO):
            i = compute_info(m, infos, [f])
            tr, va = i.get("train_rows"), i.get("val_rows")
            if tr is None:
                continue
            n_test = int(((allp.fold == f) & allp[m].notna()).sum()) if m in allp else 0
            row_acc.append({"model": m, "code": describe(m)["code"], "fold": f, "train_rows": tr[0],
                            "validation_rows": va[0] if isinstance(va, list) else va, "test_rows": n_test,
                            "train_unit": "labelled daily origins (every 2nd day)" if (m in NEURAL or TUNED.get(m) in NEURAL or m == "A-HGB-DAILY")
                            else "v2 Monday rows"})
    pd.DataFrame(row_acc).to_csv(OUT / "row_accounting.csv", index=False)
    summary = {"populations": {k: v[["model", "mae", "rmse", "bias", "r2"]].set_index("model").round(4).to_dict("index")
                               for k, v in tables.items()},
               "family_F": {k: {x: v[x] for x in ("estimate", "p_max", "p_holm", "verdict")} for k, v in stats["family_F"].items()},
               "family_T": {k: {x: v[x] for x in ("estimate", "p_max", "p_holm", "verdict")} for k, v in stats["family_T"].items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    tables_markdown(tables, stats_flat, OUT / "RESULTS_TABLES.md")
    make_figures(dev, retro, tables, stats, OUT / "figures")
    print(json.dumps(summary["family_F"], indent=1, default=float))
    print(json.dumps(summary["family_T"], indent=1, default=float))



# ------------------------------------------------------------------ Markdown tables (generated; never hand-typed)

def _f(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{d}f}" if isinstance(x, (float, np.floating)) else str(x))


def tables_markdown(tables, stats_flat, path):
    lines = ["# v5 results tables (generated by `python -m src.v5.report`; do not edit by hand)", "",
             "MAE, RMSE, bias (prediction − observation) and max/median absolute error are in percentage points of capacity. "
             "Networks: point forecast = mean of the 3 seed medians. Training time: total over the population's folds "
             "(networks: 3 seeds, 9 concurrent single-thread workers; tabular: one process, all cores). Each table is one "
             "population; numbers are never pooled across tables.", ""]
    for name, t in tables.items():
        lines += [f"## {name}", "", "| Model | Code | Input features | Training rows | Validation rows | Test rows | MAE | RMSE | Bias | R² | "
                  "Training time (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.sort_values("mae").iterrows():
            inputs = r["inputs"]
            lines.append(f"| {r['model']} | {r['code']} | {inputs} | {r.get('train_rows', '')} | {r.get('validation_rows', '')} | "
                         f"{r['test_rows']} | {_f(r['mae'])} | {_f(r['rmse'])} | {_f(r['bias'])} | {_f(r['r2'])} | "
                         f"{_f(r.get('train_seconds_total'), 1)} |")
        lines += ["", f"### {name}: additional metrics", "",
                  "| Model | Median AE | Max AE | Skill vs DOY | ΔMAE vs HGB[S,N,C,R] | Seed MAE mean ± SD | Training MAE | "
                  "Parameters | Native 90% cov / width | Conformal 90% cov / width | Inference (ms/row) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.sort_values("mae").iterrows():
            seed = f"{r['seed_mae_mean']:.3f} ± {r['seed_mae_sd']:.3f}" if pd.notna(r.get("seed_mae_mean", np.nan)) else ""
            nat = f"{r['native_cov90']:.3f} / {r['native_width90']:.2f}" if pd.notna(r.get("native_cov90", np.nan)) else ""
            cf = f"{r['conformal90_coverage']:.3f} / {r['conformal90_width']:.2f}" if pd.notna(r.get("conformal90_coverage", np.nan)) else ""
            lines.append(f"| {r['model']} | {_f(r['median_ae'])} | {_f(r['max_ae'], 2)} | {_f(r['skill_vs_doy'])} | "
                         f"{_f(r['delta_mae_vs_hgb_ref'], 3)} | {seed} | {_f(r.get('train_mae'))} | {_f(r.get('params'))} | {nat} | {cf} | "
                         f"{_f(r.get('infer_ms_per_row'), 4)} |")
        lines.append("")
    lines += ["## Statistical comparisons", "", "| Family | Population | ID | Description | Estimate (pp) | 95% CI 5w | 9w | 13w | p_max | "
              "p_Holm | Verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in stats_flat.iterrows():
        lines.append(f"| {r['family']} | {r['population']} | {r['id']} | {r['description']} | {r['estimate_pp']:+.3f} | "
                     f"{r['ci95_5w']} | {r['ci95_9w']} | {r['ci95_13w']} | {r['p_max']:.4f} | {_f(r['p_holm'], 4)} | {r['verdict']} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""v5 model registry: every configuration in docs/v5/BENCHMARK_PLAN.md, with its information codes and provenance.

Codes: S storage, N snow, Wd dam weather, Wu upstream weather, G GEFS, Q flows, C calendar, R reservoir.
"""
import json

import numpy as np

from src.v5 import ROOT
from src.v5.data import GROUPS

FROZEN_HGB = {k: v for k, v in json.loads((ROOT / "config/v2_frozen.json").read_text())["primary_model"]["hyperparameters"].items()}
NN_DEFAULT = {"hidden": 64, "layers": 1, "emb_dim": 4, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "batch": 256,
              "length": 30, "max_epochs": 30, "patience": 4}
N4_CANDIDATES = {
    "wb_balanced": {"mode": "water_balance", "r_max": 1.0,
                    "weights": {"quantile": 1.0, "storage": 1.0, "inflow": 0.5, "outflow": 0.5, "feasible": 1.0, "residual": 0.5}},
    "wb_primary": {"mode": "water_balance", "r_max": 1.0,
                   "weights": {"quantile": 1.0, "storage": 0.25, "inflow": 0.1, "outflow": 0.1, "feasible": 1.0, "residual": 0.5}},
    "wb_strict": {"mode": "water_balance", "r_max": 0.0,
                  "weights": {"quantile": 1.0, "storage": 1.0, "inflow": 0.5, "outflow": 0.5, "feasible": 1.0, "residual": 0.5}},
    "wb_noaux": {"mode": "water_balance", "r_max": 1.0,
                 "weights": {"quantile": 1.0, "storage": 1.0, "inflow": 0.0, "outflow": 0.0, "feasible": 1.0, "residual": 0.5}},
    "direct": {"mode": "direct", "r_max": 0.0, "weights": {"quantile": 1.0, "storage": 1.0, "feasible": 1.0}},
}


def cols(codes):
    return [c for g in codes if g in GROUPS for c in GROUPS[g]]


def code(algo, codes, suffix=""):
    return f"{algo}[{','.join(codes)}]{suffix}"


ALL = ["S", "N", "Wd", "Wu", "G", "Q", "C", "R"]
# Tabular configurations: algo hgb | ridge | ridge_imp ; train "weekly" (v2 Monday rows) or "daily" (every 2nd day)
TABULAR = {
    "B-HGB-SWdCR": dict(algo="hgb", codes=["S", "Wd", "C", "R"], train="weekly", exp="B",
                        origin="v1/v2 'HGB' (storage and weather)"),
    "B-HGB-SCR": dict(algo="hgb", codes=["S", "C", "R"], train="weekly", exp="B", origin="v2 'HGB no weather'"),
    "B-HGB-SNCR": dict(algo="hgb", codes=["S", "N", "C", "R"], train="weekly", exp="B",
                       origin="v2 primary 'HGB SWE no weather' = v3 Model A (HGB reference)"),
    "B-HGB-SNWdCR": dict(algo="hgb", codes=["S", "N", "Wd", "C", "R"], train="weekly", exp="B",
                         origin="v2 'HGB + SWE' = v3 Model B"),
    "B-HGB-SNWuCR": dict(algo="hgb", codes=["S", "N", "Wu", "C", "R"], train="weekly", exp="B", origin="v3 Model B2"),
    "B-HGB-SNGCR": dict(algo="hgb", codes=["S", "N", "G", "C", "R"], train="weekly", exp="B",
                        origin="v3 Model C (here trained on all rows, GEFS missing where unavailable)"),
    "B-HGB-SNQCR": dict(algo="hgb", codes=["S", "N", "Q", "C", "R"], train="weekly", exp="B",
                        origin="v4 post-hoc flow ablation"),
    "HGB-V5": dict(algo="hgb", codes=ALL, train="weekly", exp="A+B", origin="new: HGB v5, full information"),
    "B-RIDGE-SWdCR": dict(algo="ridge", codes=["S", "Wd", "C", "R"], train="weekly", exp="B", origin="v1 Ridge"),
    "B-RIDGE-SCR": dict(algo="ridge", codes=["S", "C", "R"], train="weekly", exp="B", origin="v4 ridge autoregression baseline"),
    "A-RIDGE": dict(algo="ridge_imp", codes=ALL, train="weekly", exp="A", origin="new: Ridge with full information"),
    "A-HGB-DAILY": dict(algo="hgb", codes=ALL, train="daily", exp="A",
                        origin="new: HGB v5 inputs + gefs_age_days, trained on the networks' daily-origin sample set"),
}
BASELINES = {"BL-PERS": "persistence", "BL-TREND": "30-day trend", "BL-MONTH": "monthly climatology",
             "BL-DOY": "DOY kernel climatology (sigma 15 d)"}
NEURAL = {
    "B-N1": dict(kind="lstm", gefs=False, exclude=(), exp="B", origin="v4 N1 LSTM"),
    "B-N1-NOFLOW": dict(kind="lstm", gefs=False, exclude=("flow",), exp="B", origin="v4 N1 without flows (post-hoc ablation)"),
    "B-N2": dict(kind="gru", gefs=False, exclude=(), exp="B", origin="v4 N2 GRU"),
    "B-N3": dict(kind="tcn", gefs=False, exclude=(), exp="B", origin="v4 N3 TCN"),
    "B-N4-NOAUX": dict(kind="n4", cand="wb_noaux", gefs=False, exclude=(), exp="B", origin="v4 N4 wb_noaux (selected N4)"),
    "B-N4-BAL": dict(kind="n4", cand="wb_balanced", gefs=False, exclude=(), exp="B", origin="v4 N4 wb_balanced"),
    "B-N4-PRIM": dict(kind="n4", cand="wb_primary", gefs=False, exclude=(), exp="B", origin="v4 N4 wb_primary"),
    "B-N4-STRICT": dict(kind="n4", cand="wb_strict", gefs=False, exclude=(), exp="B", origin="v4 N4 wb_strict"),
    "B-N4-DIRECT": dict(kind="n4", cand="direct", gefs=False, exclude=(), exp="B", origin="v4 N4 direct-head ablation"),
    "A-N1": dict(kind="lstm", gefs=True, exclude=(), exp="A", origin="N1 + GEFS static inputs"),
    "A-N2": dict(kind="gru", gefs=True, exclude=(), exp="A", origin="N2 + GEFS static inputs"),
    "A-N3": dict(kind="tcn", gefs=True, exclude=(), exp="A", origin="N3 + GEFS static inputs"),
    "A-N4": dict(kind="n4", cand="wb_noaux", gefs=True, exclude=(), exp="A", origin="N4 wb_noaux + GEFS static inputs"),
    "A-N4-DIRECT": dict(kind="n4", cand="direct", gefs=True, exclude=(), exp="A", origin="N4 direct + GEFS static inputs"),
}
NN_CODES = {False: ["S", "Q", "N", "Wd", "Wu", "C", "R"], True: ["S", "Q", "N", "Wd", "Wu", "G", "C", "R"]}

# ------------------------------------------------------------------ equal-budget search (BENCHMARK_PLAN.md section 6)
N_CONFIGS = 20
TUNED = {"T-RIDGE": "A-RIDGE", "T-HGB-V5": "HGB-V5", "T-N1": "A-N1", "T-N2": "A-N2", "T-N3": "A-N3", "T-N4": "A-N4"}
ALGO_INDEX = {"T-RIDGE": 0, "T-HGB-V5": 1, "T-N1": 2, "T-N2": 3, "T-N3": 4, "T-N4": 5}


def search_space(tuned_id):
    """Configuration 0 is the frozen one; 1..19 are a seeded random sample (numpy seed 42 + algorithm index)."""
    rng = np.random.default_rng(42 + ALGO_INDEX[tuned_id])
    if tuned_id == "T-RIDGE":
        return [{"alpha": "ridgecv"}] + [{"alpha": float(a)} for a in np.logspace(-3, 4, N_CONFIGS - 1)]
    if tuned_id == "T-HGB-V5":
        out = [dict(FROZEN_HGB, max_leaf_nodes=31, max_features=1.0)]
        for _ in range(N_CONFIGS - 1):
            out.append({"learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
                        "max_iter": int(rng.choice([100, 200, 300, 500, 800])),
                        "max_leaf_nodes": int(rng.choice([7, 15, 31, 63])),
                        "max_depth": [None, 4, 6, 8][int(rng.integers(4))],
                        "min_samples_leaf": int(rng.choice([10, 20, 30, 50, 100])),
                        "l2_regularization": float(rng.choice([0.0, 0.1, 1.0, 3.0, 10.0])),
                        "max_features": float(rng.choice([0.5, 0.75, 1.0])), "random_state": 42})
        return out
    base = dict(NN_DEFAULT, cand="wb_noaux") if tuned_id == "T-N4" else dict(NN_DEFAULT)
    out = [base]
    for _ in range(N_CONFIGS - 1):
        c = dict(NN_DEFAULT)
        c.update(hidden=int(rng.choice([32, 64, 128])), dropout=float(rng.choice([0.0, 0.1, 0.2, 0.3])),
                 lr=float(rng.choice([3e-4, 1e-3, 3e-3])), weight_decay=float(rng.choice([1e-5, 1e-4, 1e-3])),
                 length=int(rng.choice([30, 60, 90])), batch=int(rng.choice([128, 256, 512])))
        if tuned_id == "T-N3":
            c["kernel"] = int(rng.choice([2, 3, 5]))
        else:
            c["layers"] = int(rng.choice([1, 2]))
        if tuned_id == "T-N4":
            c["cand"] = str(rng.choice(["wb_noaux", "wb_balanced", "wb_primary", "wb_strict"]))
        out.append(c)
    return out

"""v4 development experiments (run in the v4 environment, requirements-v4.txt).

  python -m src.v4.experiment context              # history length 30/90/180/365, LSTM, tuning folds WY2012-2015
  python -m src.v4.experiment compare --length L   # LSTM and GRU x seeds vs baselines, confirmation folds WY2016-2023
  python -m src.v4.experiment ablation --length L  # HGB + flow features; LSTM without flow inputs (after compare)

Protocol (fixed before any v4 result was seen; docs/v4/PROTOCOL.md)
- Data: dates before 2024-01-01 only. 2024-2025 (v2 held-out) and 2026 (v3 test) are never loaded.
- Folds: the v2 expanding-window rule by water year (src.v4.sequences.fold_split).
- Selection: the history length is chosen on tuning folds WY2012-2015 only. Network sizes and training settings
  are fixed defaults (no search yet). Comparisons are reported on confirmation folds WY2016-2023.
- Evaluation rows: the v2 Monday rows (data/processed/weekly_features.csv); every model and baseline is scored
  on exactly the same rows. The neural models train on daily origins.
- Baselines, refit per fold on the same weekly rows as v2: persistence, 30-day trend, monthly seasonal,
  smoothed day-of-year seasonal, ridge autoregression, and HGB with the frozen v2 features and hyperparameters.
- Uncertainty: moving-block bootstrap over origin dates (5, 9, 13 weeks; src.evaluation), seed means and SDs.

Aggregates go to results/v4/development/<stage>/ (tracked); per-row predictions to data/v4/preds/ (private).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.v4 import DEV_END, STATIONS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v4" / "development"
PREDS = ROOT / "data" / "v4" / "preds"
TUNE_WYS = [2012, 2013, 2014, 2015]
CONFIRM_WYS = list(range(2016, 2024))
LENGTHS = [30, 90, 180, 365]
DEFAULTS = {"hidden": 64, "layers": 1, "emb_dim": 4, "dropout": 0.2}
TRAIN = {"max_epochs": 30, "patience": 4, "batch": 256, "lr": 1e-3, "weight_decay": 1e-4}
STRIDE = 2                        # every 2nd daily origin in the fitting set (about 3.5x the weekly sample count)
V2_FEATURES = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct", "swe_basin_in", "swe_change_14d_in",
               "swe_anom_in", "season_sin", "season_cos"]
AR_FEATURES = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct", "season_sin", "season_cos"]


def load(lmax=365):
    from src.experiments import load_dev
    from src.v4.dataset import load_panel
    from src.v4.sequences import SequenceData
    dev, snow = load_dev()                         # v2 weekly rows with origin and target before 2024-01-01
    assert dev.target_date.max() < DEV_END
    return SequenceData(load_panel(), lmax=lmax), dev, snow


def baselines_for_fold(dev, snow, wy):
    """Baselines on the v2 Monday rows of `wy`, refit on weekly rows with target_date before the fold."""
    from src.experiments import fold
    from src.models import baselines, design, hgb, ridge
    tr, te = fold(dev, snow, wy)
    stations = sorted(dev.station_id.unique())
    out = baselines(tr, te).rename(columns={"baseline_persistence_pp": "persistence", "baseline_trend_pp": "trend_30d",
                                            "baseline_seasonal_pp": "monthly_seasonal", "baseline_doy_pp": "doy_seasonal"})
    hp = json.loads((ROOT / "config/v2_frozen.json").read_text())["primary_model"]["hyperparameters"]
    params = {k: v for k, v in hp.items() if k not in ("l2_regularization", "random_state")}
    out["hgb_v2"] = hgb(params).fit(design(tr, V2_FEATURES, stations), tr.target_delta_pp).predict(
        design(te, V2_FEATURES, stations))
    out["ridge_ar"] = ridge().fit(design(tr, AR_FEATURES, stations), tr.target_delta_pp).predict(
        design(te, AR_FEATURES, stations))
    te = te.reset_index(drop=True)
    return pd.concat([te[["station_id", "date", "target_date", "target_delta_pp", "season"]], out.reset_index(drop=True)],
                     axis=1)


def run_nn(data, wy, kind, length, seed, weekly, log=print, exclude=()):
    import torch
    from src.v4.nets import QUANTILES, build
    from src.v4.sequences import N_STATIC, fold_split, monday_rows
    from src.v4.train import fit, predict, set_seed
    torch.set_num_threads(8)
    fit_idx, val_idx = fold_split(data.samples, wy, stride=STRIDE)
    eval_idx, rows = monday_rows(data.samples, weekly, wy)
    set_seed(seed)                                  # weight initialisation is part of the seeded run
    model = build(kind, data.n_inputs, N_STATIC, len(STATIONS), length, **DEFAULTS)
    info = fit(model, data, fit_idx, val_idx, length, seed=seed, **TRAIN, drop_sources=exclude, log=None)
    q = predict(model, data, eval_idx, length, drop_sources=exclude)
    robust = {name: predict(model, data, eval_idx, length, drop_sources=tuple(groups) + tuple(exclude))[:, QUANTILES.index(0.5)]
              for name, groups in (("no_flow", ("flow",)), ("no_snow", ("snow",)),
                                   ("no_weather", ("weather_dam", "weather_box")))}
    log(f"  WY{wy} {kind}{' without ' + '+'.join(exclude) if exclude else ''} L={length} seed={seed}: epochs {info['epochs_run']} (best {info['best_epoch']}), "
        f"{info['train_seconds']:.0f} s, eval MAE {np.mean(np.abs(q[:, QUANTILES.index(0.5)] - rows.target_delta_pp)):.3f}")
    return q, robust, rows, info


FLOW_FEATURES = ["inflow_7d_pp", "inflow_30d_pp", "outflow_7d_pp", "outflow_30d_pp"]


def flow_features(panel, rows):
    """Mean daily inflow and outflow (pp of capacity per day) over the 7 and 30 days ending at the origin, for
    weekly rows (station_id, date). Same information rule as the sequences: flows up to and including t.
    NaN unless every day of the window is observed (HGB handles NaN natively)."""
    from src.v4.sequences import CFS_DAY_AF
    out = []
    for st, g in panel.groupby("station_id"):
        g = g.set_index("date").sort_index()
        f = pd.DataFrame(index=g.index)
        for c, name in (("inflow_cfs", "inflow"), ("outflow_cfs", "outflow")):
            pp = 100 * g[c] * CFS_DAY_AF / g.capacity_af
            f[f"{name}_7d_pp"] = pp.rolling(7, min_periods=7).mean()
            f[f"{name}_30d_pp"] = pp.rolling(30, min_periods=30).mean()
        out.append(f.assign(station_id=st).reset_index())
    f = pd.concat(out)
    return rows[["station_id", "date"]].merge(f, on=["station_id", "date"], how="left")[FLOW_FEATURES].to_numpy()


def stage_ablation(args):
    """Is any neural gain architecture or information? (a) HGB with the frozen v2 configuration plus recent-flow
    features; (b) N1 trained and evaluated without flow inputs. Same confirmation folds and rows as `compare`."""
    from src.evaluation import BLOCK_LENGTHS, block_bootstrap_diff
    from src.experiments import fold
    from src.models import design, hgb
    from src.v4.dataset import load_panel
    from src.v4.nets import QUANTILES
    comp = pd.read_csv(PREDS / "compare_predictions.csv", parse_dates=["date", "target_date"])
    data, dev, snow = load()
    panel = load_panel()
    hp = json.loads((ROOT / "config/v2_frozen.json").read_text())["primary_model"]["hyperparameters"]
    params = {k: v for k, v in hp.items() if k not in ("l2_regularization", "random_state")}
    stations = sorted(dev.station_id.unique())
    frames, info_all = [], {}
    for wy in CONFIRM_WYS:
        tr, te = fold(dev, snow, wy)
        te = te.reset_index(drop=True)
        cols = V2_FEATURES + FLOW_FEATURES
        Xtr = pd.concat([design(tr, V2_FEATURES, stations), pd.DataFrame(flow_features(panel, tr), columns=FLOW_FEATURES)], axis=1)
        Xte = pd.concat([design(te, V2_FEATURES, stations), pd.DataFrame(flow_features(panel, te), columns=FLOW_FEATURES)], axis=1)
        f = te[["station_id", "date", "target_delta_pp"]].copy()
        f["hgb_v2_flows"] = hgb(params).fit(Xtr, tr.target_delta_pp).predict(Xte)
        meds = []
        for seed in args.seeds:
            q, _, rows, info = run_nn(data, wy, "lstm", args.length, seed, dev, exclude=("flow",))
            assert (rows.date.to_numpy() == f.date.to_numpy()).all()
            f[f"lstm_noflow_s{seed}"] = q[:, QUANTILES.index(0.5)]
            meds.append(q[:, QUANTILES.index(0.5)])
            info_all[f"lstm_noflow WY{wy} seed{seed}"] = {k: info[k] for k in ("best_epoch", "epochs_run", "train_seconds")}
        f["lstm_noflow_ens"] = np.mean(meds, axis=0)
        frames.append(f)
    p = pd.concat(frames, ignore_index=True)
    c = comp.set_index(["station_id", "date"])
    for col in ("hgb_v2", "doy_seasonal", "lstm_ens", "gru_ens"):
        if col in c:
            p[col] = c[col].reindex(pd.MultiIndex.from_frame(p[["station_id", "date"]])).to_numpy()
    assert p[["hgb_v2", "lstm_ens"]].notna().all().all()
    p.to_csv(PREDS / "ablation_predictions.csv", index=False)
    cols = [k for k in ("hgb_v2", "hgb_v2_flows", "lstm_noflow_ens", "lstm_ens", "gru_ens") if k in p]
    pairs = [("lstm_ens", "hgb_v2_flows"), ("lstm_noflow_ens", "hgb_v2"), ("hgb_v2_flows", "hgb_v2"),
             ("lstm_ens", "lstm_noflow_ens")]
    summary = {"stage": "ablation", "folds": CONFIRM_WYS, "history_length": args.length, "seeds": args.seeds,
               "n_rows": int(len(p)), "flow_features": FLOW_FEATURES, "pooled_mae_pp": maes(p, cols),
               "mae_by_station": maes(p, cols, "station_id"),
               "seed_mae_pp_lstm_noflow": [float(np.mean(np.abs(p.target_delta_pp - p[f"lstm_noflow_s{s}"]))) for s in args.seeds],
               "block_bootstrap": {f"{a} vs {b}": {f"{bl}_weeks": block_bootstrap_diff(p, a, b, block=bl) for bl in BLOCK_LENGTHS}
                                   for a, b in pairs},
               "compute": info_all}
    out_dir = OUT / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["pooled_mae_pp"], indent=1))


def interval_metrics(y, q):
    from src.v4.nets import INTERVALS, QUANTILES
    out = {}
    for nom, (lo, hi) in INTERVALS.items():
        a, b = q[:, QUANTILES.index(lo)], q[:, QUANTILES.index(hi)]
        out[str(nom)] = {"coverage": float(np.mean((y >= a) & (y <= b))), "mean_width_pp": float(np.mean(b - a))}
    tau = np.array(QUANTILES)
    e = y[:, None] - q
    pin = np.maximum(tau * e, (tau - 1) * e)
    out["mean_pinball_pp"] = float(pin.mean())
    out["approx_crps_pp"] = float(2 * pin.mean())      # 2 x mean pinball over the 9 quantile levels
    return out


def maes(df, cols, by=None):
    f = lambda g: {c: round(float(np.mean(np.abs(g.target_delta_pp - g[c]))), 4) for c in cols}  # noqa: E731
    return f(df) if by is None else {str(k): f(g) for k, g in df.groupby(by)}


def plot_curves(curves, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(curves)
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8), squeeze=False)
    for ax, k in zip(axes[0], keys):
        c = curves[k]
        ax.plot(range(1, len(c["train_pinball"]) + 1), c["train_pinball"], label="train")
        ax.plot(range(1, len(c["val_pinball"]) + 1), c["val_pinball"], label="early-stopping set")
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("pinball loss (10 pp units)", fontsize=8)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stage_context(args):
    from src.v4.nets import QUANTILES
    data, dev, snow = load()
    out_dir = OUT / "context"
    out_dir.mkdir(parents=True, exist_ok=True)
    res, curves, rows_all = {}, {}, []
    for length in LENGTHS:
        preds = []
        for wy in TUNE_WYS:
            q, _, rows, info = run_nn(data, wy, "lstm", length, 0, dev)
            preds.append(rows.assign(pred=q[:, QUANTILES.index(0.5)], wy=wy))
            curves[f"L{length} WY{wy}"] = info["curves"]
            res.setdefault(str(length), {"fold_info": {}})["fold_info"][str(wy)] = {
                k: info[k] for k in ("best_epoch", "epochs_run", "train_seconds", "n_params", "n_fit", "n_val")}
        p = pd.concat(preds, ignore_index=True)
        res[str(length)]["pooled_mae_pp"] = float(np.mean(np.abs(p.pred - p.target_delta_pp)))
        rows_all.append(p.assign(length=length))
    best = min(LENGTHS, key=lambda L: res[str(L)]["pooled_mae_pp"])
    summary = {"stage": "context", "folds": TUNE_WYS, "model": "lstm", "seed": 0, "defaults": DEFAULTS, "train": TRAIN,
               "stride": STRIDE, "results": res, "selected_length": best,
               "rule": "lowest pooled MAE on the Monday rows of tuning folds WY2012-2015"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "curves.json").write_text(json.dumps(curves))
    plot_curves({k: v for k, v in curves.items() if k.endswith("WY2015")}, out_dir / "loss_curves_WY2015.png",
                "LSTM loss curves by history length (tuning fold WY2015, seed 0)")
    PREDS.mkdir(parents=True, exist_ok=True)
    pd.concat(rows_all).to_csv(PREDS / "context_predictions.csv", index=False)
    print(json.dumps({L: res[str(L)]["pooled_mae_pp"] for L in LENGTHS}, indent=1), "selected", best)


def stage_compare(args):
    from src.evaluation import BLOCK_LENGTHS, block_bootstrap_diff
    from src.v4.nets import QUANTILES
    data, dev, snow = load()
    out_dir = OUT / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds, seeds, L = args.models, args.seeds, args.length
    frames, info_all, curves = [], {}, {k: {} for k in kinds}
    t_start = time.perf_counter()
    for wy in CONFIRM_WYS:
        base = baselines_for_fold(dev, snow, wy)
        for kind in kinds:
            meds, qs = [], []
            for seed in seeds:
                q, robust, rows, info = run_nn(data, wy, kind, L, seed, dev)
                assert (rows.station_id.to_numpy() == base.station_id.to_numpy()).all()
                assert (rows.date.to_numpy() == base.date.to_numpy()).all()
                base[f"{kind}_s{seed}"] = q[:, QUANTILES.index(0.5)]
                for name, med in robust.items():
                    base[f"{kind}_s{seed}_{name}"] = med
                meds.append(q[:, QUANTILES.index(0.5)])
                qs.append(q)
                info_all[f"{kind} WY{wy} seed{seed}"] = {k: info[k] for k in ("best_epoch", "epochs_run", "train_seconds",
                                                                                "n_params", "n_fit", "n_val")}
                if seed == seeds[0]:
                    curves[kind][f"WY{wy}"] = info["curves"]
            base[f"{kind}_ens"] = np.mean(meds, axis=0)
            qe = np.mean(qs, axis=0)
            for i, tau in enumerate(QUANTILES):
                base[f"{kind}_ens_q{tau}"] = qe[:, i]
            for name in ("no_flow", "no_snow", "no_weather"):
                base[f"{kind}_ens_{name}"] = np.mean([base[f"{kind}_s{s}_{name}"] for s in seeds], axis=0)
        frames.append(base.assign(wy=wy))
    p = pd.concat(frames, ignore_index=True)
    PREDS.mkdir(parents=True, exist_ok=True)
    p.to_csv(PREDS / "compare_predictions.csv", index=False)
    base_cols = ["persistence", "trend_30d", "monthly_seasonal", "doy_seasonal", "ridge_ar", "hgb_v2"]
    nn_cols = [f"{k}_ens" for k in kinds]
    cols = base_cols + nn_cols
    seed_mae = {k: [float(np.mean(np.abs(p.target_delta_pp - p[f"{k}_s{s}"]))) for s in seeds] for k in kinds}
    wyt = pd.read_csv(ROOT / "data" / "water_year_types.csv").set_index("water_year")
    sac = p.station_id.isin(["SHA", "ORO", "FOL"])                 # Sacramento Valley index; others San Joaquin
    p["wy_type"] = np.where(sac, p.wy.map(wyt.sac_type), p.wy.map(wyt.sjv_type))
    p["regime"] = np.select([p.target_delta_pp >= 10, p.target_delta_pp <= -10], ["rapid fill (>= +10 pp)",
                                                                                  "rapid drawdown (<= -10 pp)"], "other")
    boots = {}
    for k in nn_cols:
        for b in ("hgb_v2", "doy_seasonal"):
            boots[f"{k} vs {b}"] = {f"{bl}_weeks": block_bootstrap_diff(p, k, b, block=bl) for bl in BLOCK_LENGTHS}
    q_cols = {k: [f"{k}_ens_q{tau}" for tau in QUANTILES] for k in kinds}
    intervals = {k: {"all": interval_metrics(p.target_delta_pp.to_numpy(), p[q_cols[k]].to_numpy()),
                     "by_station": {s: interval_metrics(g.target_delta_pp.to_numpy(), g[q_cols[k]].to_numpy())
                                    for s, g in p.groupby("station_id")}} for k in kinds}
    robustness = {k: {name: round(float(np.mean(np.abs(p.target_delta_pp - p[f"{k}_ens_{name}"]))), 4)
                      for name in ("no_flow", "no_snow", "no_weather")} for k in kinds}
    summary = {
        "stage": "compare", "folds": CONFIRM_WYS, "history_length": L, "seeds": seeds, "defaults": DEFAULTS,
        "train": TRAIN, "stride": STRIDE, "n_rows": int(len(p)),
        "pooled_mae_pp": maes(p, cols),
        "seed_mae_pp": {k: {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                        for k, v in seed_mae.items()},
        "mae_by_station": maes(p, cols, "station_id"), "mae_by_water_year": maes(p, cols, "wy"),
        "mae_by_season": maes(p, cols, "season"), "mae_by_water_year_type": maes(p, cols, "wy_type"),
        "mae_by_regime": maes(p, cols, "regime"), "regime_counts": p.regime.value_counts().to_dict(),
        "block_bootstrap": boots, "intervals": intervals, "robustness_missing_source_at_inference_mae_pp": robustness,
        "compute": {"fold_seed_runs": info_all, "total_seconds": time.perf_counter() - t_start},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_dir / "curves.json").write_text(json.dumps(curves))
    for k in kinds:
        plot_curves(curves[k], out_dir / f"loss_curves_{k}.png",
                    f"{k.upper()} loss curves, confirmation folds (L={L}, seed {seeds[0]})")
    print(json.dumps(summary["pooled_mae_pp"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("context")
    c = sub.add_parser("compare")
    c.add_argument("--length", type=int, required=True)
    c.add_argument("--models", nargs="+", default=["lstm", "gru"])
    c.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    b = sub.add_parser("ablation")
    b.add_argument("--length", type=int, required=True)
    b.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    {"context": stage_context, "compare": stage_compare, "ablation": stage_ablation}[a.stage](a)


if __name__ == "__main__":
    main()

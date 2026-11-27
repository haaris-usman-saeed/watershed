"""v5 evaluation: point metrics, paired moving-block bootstrap contrasts with Holm correction, and intervals.

The bootstrap draws its block indices exactly as src.evaluation.block_bootstrap does (same generator, seed and order),
so for a two-model contrast it reproduces src.evaluation.block_bootstrap_diff (tested). A contrast is any weighted sum
of per-model MAEs on the same rows, e.g. {A: 1, B: -1} for MAE(A) - MAE(B), or a difference-in-differences.
"""
import numpy as np
import pandas as pd

from src.evaluation import BLOCK_LENGTHS, N_BOOT, SEED, _block_indices

QUANTILES = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
INTERVALS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975)}
ALPHA = 0.05


def point_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    sst = np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)), "r2": float(1 - np.sum(e ** 2) / sst) if sst > 0 else float("nan"),
            "median_ae": float(np.median(np.abs(e))), "max_ae": float(np.max(np.abs(e)))}


def contrast_bootstrap(df, weights, block=5, n_boot=N_BOOT, seed=SEED):
    """Bootstrap distribution of sum_c w_c * MAE_c over origin dates (all reservoirs on a date move together)."""
    dates = np.sort(df.date.unique())
    g = df.groupby("date")
    counts = g.size().reindex(dates).to_numpy()
    sums = {c: g.apply(lambda x, c=c: np.abs(x.target_delta_pp - x[c]).sum(), include_groups=False).reindex(dates).to_numpy()
            for c in weights}
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_indices(len(dates), block, rng)
        n = counts[idx].sum()
        reps[b] = sum(w * sums[c][idx].sum() / n for c, w in weights.items())
    est = sum(w * np.mean(np.abs(df.target_delta_pp - df[c])) for c, w in weights.items())
    p = min(1.0, 2 * min((np.sum(reps >= 0) + 1) / (n_boot + 1), (np.sum(reps <= 0) + 1) / (n_boot + 1)))
    return {"estimate": float(est), "ci95_low": float(np.quantile(reps, 0.025)), "ci95_high": float(np.quantile(reps, 0.975)),
            "share_below_zero": float(np.mean(reps < 0)), "p_two_sided": float(p), "block_weeks": block, "n_boot": n_boot}


def test_contrast(df, weights, blocks=BLOCK_LENGTHS):
    res = {f"{b}w": contrast_bootstrap(df, weights, block=b) for b in blocks}
    p_max = max(r["p_two_sided"] for r in res.values())
    all_exclude = all(r["ci95_high"] < 0 or r["ci95_low"] > 0 for r in res.values())
    return {"estimate": res[f"{blocks[0]}w"]["estimate"], "p_max": p_max, "all_ci_exclude_zero": all_exclude, "blocks": res,
            "n_rows": int(len(df)), "n_dates": int(df.date.nunique())}


def holm(pvals):
    """Holm step-down adjusted p-values, same order as given."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def family(df, contrasts):
    """contrasts: {id: (description, weights)}. Holm over the family on p_max; supported iff adjusted p_max < 0.05."""
    out = {cid: dict(description=d, weights=w, **test_contrast(df, w)) for cid, (d, w) in contrasts.items()}
    adj = holm([out[c]["p_max"] for c in out])
    for c, a in zip(out, adj):
        out[c]["p_holm"] = float(a)
        est = out[c]["estimate"]
        out[c]["verdict"] = ("supported: first better" if a < ALPHA and est < 0 else
                             "supported: second better" if a < ALPHA and est > 0 else "not established")
    return out


def unadjusted(df, contrasts):
    out = {cid: dict(description=d, weights=w, **test_contrast(df, w)) for cid, (d, w) in contrasts.items()}
    for c in out:
        est = out[c]["estimate"]
        ok = out[c]["all_ci_exclude_zero"]
        out[c]["verdict_unadjusted"] = ("first better (all 3 CIs exclude 0)" if ok and est < 0 else
                                        "second better (all 3 CIs exclude 0)" if ok and est > 0 else "not established")
    return out


def quantile_intervals(y, q):
    """q: [n, 9] quantiles (QUANTILES order)."""
    y = np.asarray(y, float)
    out = {}
    for nom, (lo, hi) in INTERVALS.items():
        a, b = q[:, QUANTILES.index(lo)], q[:, QUANTILES.index(hi)]
        out[f"cov{nom}"] = float(np.mean((y >= a) & (y <= b)))
        out[f"width{nom}"] = float(np.mean(b - a))
    tau = np.array(QUANTILES)
    e = y[:, None] - q
    out["mean_pinball"] = float(np.maximum(tau * e, (tau - 1) * e).mean())
    return out


def conformal90(frame, col, fold_order):
    """Chronological split-conformal 90 % interval: rows of fold k use the 90th percentile (method 'higher') of the
    model's |residuals| on every earlier fold in `fold_order`. Returns per-row half-widths (NaN where no calibration)."""
    hw = pd.Series(np.nan, index=frame.index)
    r = (frame.target_delta_pp - frame[col]).abs()
    for k, f in enumerate(fold_order):
        cal = frame.fold.isin(fold_order[:k])
        if k == 0 or not cal.any():
            continue
        hw[frame.fold == f] = np.quantile(r[cal].to_numpy(), 0.90, method="higher")
    return hw

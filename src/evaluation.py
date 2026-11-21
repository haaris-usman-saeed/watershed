"""Shared evaluation utilities: MAE, moving-block bootstrap, water years, intervals."""
import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 42
BLOCK_LENGTHS = (5, 9, 13)


def mae(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def water_year(dates):
    """California water year: Oct 1 (year N-1) to Sep 30 (year N) is water year N."""
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.year + (d.dt.month >= 10).astype(int)).to_numpy()


def _block_indices(n_dates, block, rng):
    starts = np.arange(max(1, n_dates - block + 1))
    out = []
    while len(out) < n_dates:
        s = rng.choice(starts)
        out.extend(range(s, min(s + block, n_dates)))
    return np.array(out[:n_dates])


def block_bootstrap(df, cols, block=5, n_boot=N_BOOT, seed=SEED):
    """Moving-block bootstrap over sorted unique origin dates. All rows (stations) that share
    an origin date are resampled together, which keeps cross-reservoir dependence, and blocks
    of consecutive weeks keep the dependence created by overlapping 30-day windows.
    Returns {col: array of bootstrap MAEs}."""
    rng = np.random.default_rng(seed)
    dates = np.sort(df.date.unique())
    groups = [df[df.date == d] for d in dates]
    y = [g.target_delta_pp.to_numpy(float) for g in groups]
    preds = {c: [g[c].to_numpy(float) for g in groups] for c in cols}
    abs_err = {c: [np.abs(yy - pp) for yy, pp in zip(y, preds[c])] for c in cols}
    sums = {c: np.array([e.sum() for e in abs_err[c]]) for c in cols}
    counts = np.array([len(yy) for yy in y])
    reps = {c: np.empty(n_boot) for c in cols}
    for b in range(n_boot):
        idx = _block_indices(len(dates), block, rng)
        n = counts[idx].sum()
        for c in cols:
            reps[c][b] = sums[c][idx].sum() / n
    return reps


def block_bootstrap_diff(df, model_col, base_col, block=5, n_boot=N_BOOT, seed=SEED):
    reps = block_bootstrap(df, [model_col, base_col], block=block, n_boot=n_boot, seed=seed)
    diff = reps[model_col] - reps[base_col]
    return {"estimate": mae(df.target_delta_pp, df[model_col]) - mae(df.target_delta_pp, df[base_col]),
            "ci95_low": float(np.quantile(diff, 0.025)), "ci95_high": float(np.quantile(diff, 0.975)),
            "share_better": float((diff < 0).mean()), "block_weeks": block, "n_boot": n_boot}


def coverage(y, lo, hi):
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))

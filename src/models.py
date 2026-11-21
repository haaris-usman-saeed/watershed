"""Model, baseline and interval definitions shared by development experiments and the frozen v2 run.

Everything here is fit ONLY on the training rows passed in (outcomes with target_date before
the fold cutoff), so it can be used inside an expanding-window evaluation without leakage.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import SEASON, SNOW, STORAGE, WEATHER

SEED = 42
DOY_SIGMA_DAYS = 15.0      # smoothed day-of-year climatology kernel (predetermined)
SNOW_CLIM_SIGMA_DAYS = 10.0
MIN_CLIM_YEARS = 3

FEATURE_SETS = {
    "storage": STORAGE + SEASON,
    "storage_weather": STORAGE + WEATHER + SEASON,                 # v1 feature set
    "storage_snow": STORAGE + SNOW + ["swe_anom_in"] + SEASON,
    "storage_weather_snow": STORAGE + WEATHER + SNOW + ["swe_anom_in"] + SEASON,
}
HGB_VARIANTS = {"HGB no weather": "storage", "HGB": "storage_weather",
                "HGB + SWE": "storage_weather_snow", "HGB SWE no weather": "storage_snow"}
HGB_GRID = [dict(learning_rate=lr, max_iter=it, max_depth=d, min_samples_leaf=30)
            for lr in (0.03, 0.06) for it in (150, 300) for d in (3, None)]
BASELINES = {"Persistence": "baseline_persistence_pp", "30-day trend": "baseline_trend_pp",
             "Monthly seasonal": "baseline_seasonal_pp", "Smoothed DOY seasonal": "baseline_doy_pp"}


def design(frame, cols, stations):
    oh = pd.get_dummies(frame.station_id).reindex(columns=stations, fill_value=0).astype(float)
    oh.columns = [f"station_{c}" for c in oh.columns]
    return pd.concat([frame[cols].reset_index(drop=True).astype(float), oh.reset_index(drop=True)], axis=1)


def hgb(params):
    return HistGradientBoostingRegressor(l2_regularization=1.0, random_state=SEED, **params)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def _circ_dist(a, b, period=365.25):
    d = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    return np.minimum(d, period - d)


def baselines(train, target):
    """All seasonal baselines use exactly the same training outcomes as the ML models."""
    out = pd.DataFrame(index=target.index)
    out["baseline_persistence_pp"] = 0.0
    out["baseline_trend_pp"] = target.prev30_pct.to_numpy()
    clim = train.assign(m=train.date.dt.month).groupby(["station_id", "m"]).target_delta_pp.mean()
    overall = train.groupby("station_id").target_delta_pp.mean()
    out["baseline_seasonal_pp"] = [clim.get((s, d.month), overall.get(s, 0.0))
                                   for s, d in zip(target.station_id, target.date)]
    doy = np.empty(len(target))
    for s in target.station_id.unique():
        tr = train[train.station_id == s]
        te_mask = (target.station_id == s).to_numpy()
        w = np.exp(-0.5 * (_circ_dist(target.date[te_mask].dt.dayofyear, tr.date.dt.dayofyear) / DOY_SIGMA_DAYS) ** 2)
        doy[te_mask] = (w @ tr.target_delta_pp.to_numpy()) / w.sum(axis=1)
    out["baseline_doy_pp"] = doy
    return out


def snow_climatology(snow_index, cutoff):
    """Smoothed day-of-year mean basin SWE per station from days strictly before `cutoff`."""
    s = snow_index[(snow_index.date < cutoff)].dropna(subset=["swe_basin_in"])
    out = {}
    grid = np.arange(1, 367)
    for st, g in s.groupby("station_id"):
        if g.date.dt.year.nunique() < MIN_CLIM_YEARS:
            continue
        w = np.exp(-0.5 * (_circ_dist(grid, g.date.dt.dayofyear) / SNOW_CLIM_SIGMA_DAYS) ** 2)
        out[st] = pd.Series((w @ g.swe_basin_in.to_numpy()) / w.sum(axis=1), index=grid)
    return out


def add_snow_anomaly(rows, clim):
    """swe_anom_in = swe_basin_in (t-1) minus climatology for day-of-year of t-1 (training years only)."""
    rows = rows.copy()
    doy = (rows.date - pd.Timedelta(days=1)).dt.dayofyear
    ref = [clim[s][d] if s in clim else np.nan for s, d in zip(rows.station_id, doy)]
    rows["swe_anom_in"] = rows.swe_basin_in.to_numpy() - np.asarray(ref, dtype=float)
    return rows


def season_of(dates):
    m = pd.to_datetime(pd.Series(dates)).dt.month
    return m.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}).to_numpy()


# ---------- prediction intervals (calibrated on out-of-sample residuals from earlier periods)
MIN_GROUP_N = 150
INTERVAL_METHODS = ["sym_global", "sym_season", "sym_station", "asym_global", "asym_season"]


def interval_offsets(cal, method, alpha=0.10):
    """cal: DataFrame of calibration residuals r = y - pred with station_id and season.
    Returns a function rows -> (lower_offset, upper_offset)."""
    def sym(x):
        return -np.quantile(np.abs(x), 1 - alpha, method="higher"), np.quantile(np.abs(x), 1 - alpha, method="higher")

    def asym(x):
        return np.quantile(x, alpha / 2, method="lower"), np.quantile(x, 1 - alpha / 2, method="higher")

    fn = sym if method.startswith("sym") else asym
    glob = fn(cal.r.to_numpy())
    key = {"sym_season": "season", "asym_season": "season", "sym_station": "station_id"}.get(method)
    groups = {}
    if key:
        for k, g in cal.groupby(key):
            if len(g) >= MIN_GROUP_N:
                groups[k] = fn(g.r.to_numpy())

    def apply(rows):
        keys = rows[key].to_numpy() if key else [None] * len(rows)
        lo = np.array([groups.get(k, glob)[0] for k in keys])
        hi = np.array([groups.get(k, glob)[1] for k in keys])
        return lo, hi
    return apply

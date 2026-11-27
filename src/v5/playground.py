"""Model playground data: retrospective results of the ten study models, for the website.

The website's forecasts come only from HGB[S,N,C,R] · frozen v2 artifact. The playground lets visitors compare the
other nine study models with it on past windows. Nothing here fits, refits or re-scores a model, and no new statistic
is computed:
- everything a public deployment shows comes from tracked aggregate files under results/v5/ (metrics and paired
  comparisons written by python -m src.v5.report);
- MAE differences are differences of tracked MAEs on the same rows;
- per-row predictions (data/v5/preds/all_predictions.parquet) are private and git-ignored. They are loaded only where
  they exist, and on a public deployment only for models whose every input source config/data_release_policy.json
  clears for public display. Five models use CDEC inflow/outflow (sensors 76 and 23), which the policy does not list,
  so their per-row predictions are never shown publicly.

Windows (never pooled):
- confirmation WY2016-2023: development data used for the v5 comparisons (exploratory);
- 2024-2025 and 2026: v2's and v3's test windows, already viewed, so every v5 number there is post-hoc.
"""
from __future__ import annotations

import pandas as pd

from src.v5 import ROOT

METRICS = ROOT / "results" / "v5" / "metrics"
STATS = ROOT / "results" / "v5" / "stats" / "comparisons.csv"
PREDS = ROOT / "data" / "v5" / "preds" / "all_predictions.parquet"

STORAGE, SNOW = "cdec_storage_sensor15", "cdec_snow_sensor82"
POWER, GEFS_OP, GEFS_RE = "nasa_power", "noaa_gefs_operational", "noaa_gefs_reforecast"
FLOWS = "cdec_flow_sensors76_23"            # not in config/data_release_policy.json: never cleared for public display
ALL_SOURCES = (STORAGE, SNOW, POWER, GEFS_OP, GEFS_RE, FLOWS)
ALL_INPUTS = ("Storage and its 7, 30 and 60-day changes; basin snowpack, its 14-day change and anomaly; NASA POWER "
              "weather at the dam and over the upstream area; GEFS precipitation forecast days 1-7 and 8-16 (blank "
              "where none); CDEC inflow and outflow over 7 and 30 days; season; reservoir")
SEQ_INPUTS = ("Daily sequences of the same inputs as Ridge[all] over the preceding days, plus GEFS as static "
              "inputs; trained with pinball loss (the median is scored), mean of 3 seeds")

WINDOWS = {
    "confirmation": dict(label="Confirmation WY2016-2023", file="confirmation_WY2016-2023",
                         population="confirmation WY2016-2023",
                         folds=tuple(f"WY{y}" for y in range(2016, 2024)), main_ref="B-HGB-SNCR",
                         status="Development data: the folds the v5 comparisons were planned on. Exploratory, but the "
                                "fair basis for comparing models."),
    "held": dict(label="2024-2025", file="HELD_posthoc", population="HELD (post-hoc retrospective)", folds=("HELD",),
                 main_ref="REF-V2-ARTIFACT",
                 status="Post-hoc: v2's preregistered test window, already viewed. Only the four baselines and the "
                        "main HGB reproduce numbers from that one-shot test."),
    "t26": dict(label="2026", file="T26_posthoc", population="T26 (post-hoc retrospective)", folds=("T26",),
                main_ref="REF-V2-ARTIFACT",
                status="Post-hoc: v3's preregistered test window, already viewed. No model may be chosen from it."),
}

# The ten study models in report numbering. Model 6 is the site's HGB; its run id depends on the window.
MAIN_NUMBER = 6
TEN = [
    dict(number=1, run="BL-PERS", name="Persistence", spec="Persistence baseline",
         inputs="None: predicts no change", sources=(STORAGE,)),
    dict(number=2, run="BL-TREND", name="30-day trend", spec="Trend extrapolation baseline",
         inputs="Repeats the storage change of the previous 30 days", sources=(STORAGE,)),
    dict(number=3, run="BL-MONTH", name="Monthly seasonal mean", spec="Monthly seasonal baseline",
         inputs="Mean training-period change by reservoir and origin month", sources=(STORAGE,)),
    dict(number=4, run="BL-DOY", name="Day-of-year seasonal", spec="Smoothed day-of-year seasonal baseline",
         inputs="Training-period changes by reservoir, Gaussian-weighted by day of year (15-day bandwidth)",
         sources=(STORAGE,)),
    dict(number=5, run="A-RIDGE", name="Ridge", spec="Ridge[all] · frozen hyperparameters, per-fold refit",
         inputs=ALL_INPUTS + ". Missing values median-imputed with indicators", sources=ALL_SOURCES),
    dict(number=6, run=None, name="HGB (main model)", spec="HGB[S,N,C,R] · frozen v2 artifact",
         inputs="Storage and its 7, 30 and 60-day changes; basin snowpack, its 14-day change and anomaly; season; "
                "reservoir", sources=(STORAGE, SNOW)),
    dict(number=7, run="A-N1", name="LSTM", spec="LSTM[all] · frozen configuration, per-fold refit",
         inputs=SEQ_INPUTS, sources=ALL_SOURCES),
    dict(number=8, run="A-N2", name="GRU", spec="GRU[all] · frozen configuration, per-fold refit",
         inputs=SEQ_INPUTS, sources=ALL_SOURCES),
    dict(number=9, run="A-N3", name="TCN", spec="TCN[all] · frozen configuration, per-fold refit",
         inputs=SEQ_INPUTS, sources=ALL_SOURCES),
    dict(number=10, run="A-N4", name="Water-balance LSTM (N4)", spec="N4 wb_noaux[all] · frozen configuration, per-fold refit",
         inputs=SEQ_INPUTS + ". Predicts daily inflow, release and losses and sums them into the 30-day change",
         sources=ALL_SOURCES),
]
OTHERS = [m for m in TEN if m["number"] != MAIN_NUMBER]


def run_id(model, window):
    """The v5 run id that holds this model's results in this window."""
    if model["number"] == MAIN_NUMBER:
        return WINDOWS[window]["main_ref"]
    return model["run"]


def spec(model, window):
    """The precise specification of this model's results in this window (model 6 differs by window)."""
    if model["number"] == MAIN_NUMBER and WINDOWS[window]["main_ref"] != "REF-V2-ARTIFACT":
        return "HGB[S,N,C,R] · frozen hyperparameters, per-fold refit"
    return model["spec"]


def main_note(window):
    if WINDOWS[window]["main_ref"] == "REF-V2-ARTIFACT":
        return "Model 6 is the saved frozen v2 artifact that makes the site's forecasts (trained on targets before 2024)."
    return ("Model 6 is the per-fold refit of the same configuration, HGB[S,N,C,R] · frozen hyperparameters: the saved "
            "artifact was trained on these years, so it cannot be scored on them.")


def by_number(number):
    return next(m for m in TEN if m["number"] == number)


def _read(name):
    p = METRICS / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p.relative_to(ROOT)} is missing (python -m src.v5.report)")
    return pd.read_csv(p)


def summary(window):
    """One row per study model with its tracked metrics in this window, plus its MAE difference from model 6."""
    w = WINDOWS[window]
    t = _read(w["file"]).set_index("model")
    ref = run_id(by_number(MAIN_NUMBER), window)
    rows = []
    for m in TEN:
        rid = run_id(m, window)
        if rid not in t.index:
            continue
        r = t.loc[rid]
        rows.append({"number": m["number"], "name": m["name"], "spec": spec(m, window), "run": rid, "n": int(r["n"]),
                     "mae": r["mae"], "rmse": r["rmse"], "bias": r["bias"], "r2": r["r2"],
                     "median_ae": r["median_ae"], "max_ae": r["max_ae"], "skill_vs_doy": r["skill_vs_doy"],
                     "coverage90": r.get("conformal90_coverage"), "width90": r.get("conformal90_width"),
                     "delta_mae_vs_main": r["mae"] - t.loc[ref, "mae"] if ref in t.index else float("nan")})
    out = pd.DataFrame(rows)
    if len(out) and out.n.nunique() != 1:
        raise ValueError(f"{w['file']}: the ten models are not scored on the same rows ({sorted(out.n.unique())})")
    return out


def breakdown(window, by):
    """Tracked MAE and bias by reservoir ('station_id'), regime ('regime') or water-year class ('wy_class')."""
    t = _read(f"{WINDOWS[window]['file']}_by_{by}")
    ids = {run_id(m, window): m for m in TEN}
    t = t[t.model.isin(ids)].copy()
    t["number"] = t.model.map(lambda r: ids[r]["number"])
    t["name"] = t.model.map(lambda r: ids[r]["name"])
    return t.rename(columns={"model": "run"}).reset_index(drop=True)


def comparisons(window, model):
    """Tracked paired block-bootstrap comparisons with this model first, in this window (never computed here)."""
    if not STATS.exists():
        return pd.DataFrame()
    c = pd.read_csv(STATS)
    rid = run_id(model, window)
    c = c[(c.population == WINDOWS[window]["population"]) & c.contrast.str.startswith(f"+1*{rid} ")]
    return c.reset_index(drop=True)


def display_block(model, policy=None, mode=None):
    """Why this model's per-row predictions may not be shown on this deployment, or None."""
    from src.v3_service import PolicyBlocked, check_display_allowed
    try:
        check_display_allowed(list(model["sources"]), policy=policy, mode=mode)
    except PolicyBlocked as exc:
        return str(exc)
    return None


def rows(window, policy=None, mode=None, path=None):
    """Per-row predictions of the ten models in this window, or (None, reason).

    Returns (frame, hidden): frame has station_id, date, target_date, target_delta_pp, storage_pct, regime and one
    column per run id; `hidden` maps the number of each model withheld on this deployment to the reason.
    """
    path = path or PREDS
    if not path.exists():
        return None, {"all": "Per-row predictions are private (CDEC-derived) and are not on this deployment."}
    keep = ["station_id", "date", "target_date", "target_delta_pp", "storage_pct", "regime"]
    try:
        import pyarrow.parquet as pq
        have = set(pq.read_schema(path).names)
        p = pd.read_parquet(path, columns=[c for c in keep + ["fold"] + [run_id(m, window) for m in TEN] if c in have])
    except ImportError as exc:                       # pyarrow missing
        return None, {"all": f"Per-row predictions could not be read here ({exc})."}
    p = p[p.fold.isin(WINDOWS[window]["folds"])]
    hidden = {}
    for m in TEN:
        rid = run_id(m, window)
        why = display_block(m, policy=policy, mode=mode)
        if why:
            hidden[m["number"]] = why
        elif rid in p:
            keep.append(rid)
    out = p[[c for c in keep if c in p]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["target_date"] = pd.to_datetime(out["target_date"])
    return out.sort_values(["station_id", "date"]).reset_index(drop=True), hidden


def replay(frame, window, station, origin):
    """Every shown model's 30-day forecast from one past origin, against what happened (all in % of capacity)."""
    r = frame[(frame.station_id == station) & (frame.date == pd.Timestamp(origin))]
    if r.empty:
        return pd.DataFrame()
    r = r.iloc[0]
    s0, obs = float(r.storage_pct), float(r.target_delta_pp)
    out = []
    for m in TEN:
        rid = run_id(m, window)
        if rid not in frame or pd.isna(r[rid]):
            continue
        pred = float(r[rid])
        out.append({"number": m["number"], "name": m["name"], "run": rid, "storage_now_pct": s0,
                    "pred_change_pp": pred, "pred_day30_pct": s0 + pred, "obs_change_pp": obs,
                    "obs_day30_pct": s0 + obs, "error_pp": pred - obs})
    return pd.DataFrame(out)

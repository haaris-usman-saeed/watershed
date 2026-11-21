"""Evaluation logic shown in the UI. Pure functions, no Streamlit.

Two sources of numbers, kept distinct on screen:
  * metrics.json: written by the engineering pipeline; authoritative for headline results.
  * computed here from test_predictions.csv: used for per-reservoir and filtered views,
    and to cross-check metrics.json. Never used to replace a pipeline value silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import ui_contract as C

TIE_TOL_PP = 0.005


def available_models(preds):
    if preds is None:
        return []
    return [m for m in C.MODELS if m.pred_col in preds.columns and preds[m.pred_col].notna().any()]


def common_rows(preds, models):
    cols = ["target_delta_pp"] + [m.pred_col for m in models]
    return preds.dropna(subset=cols)


def mae_table(preds, models=None):
    """MAE per model on rows where every compared model has a prediction."""
    models = models or available_models(preds)
    if preds is None or not models:
        return pd.DataFrame(columns=["key", "label", "mae", "n"])
    rows = common_rows(preds, models)
    out = []
    for m in models:
        err = (rows[m.pred_col] - rows["target_delta_pp"]).abs()
        out.append(dict(key=m.key, label=m.label, mae=float(err.mean()) if len(err) else np.nan, n=int(len(err))))
    return pd.DataFrame(out)


def coverage(preds):
    if preds is None:
        return dict(frac=np.nan, inside=0, n=0, mean_width=np.nan)
    rows = preds.dropna(subset=["target_delta_pp", "lower_90_pp", "upper_90_pp"])
    inside = rows["target_delta_pp"].between(rows["lower_90_pp"], rows["upper_90_pp"], inclusive="both")
    n = int(len(rows))
    return dict(frac=float(inside.mean()) if n else np.nan, inside=int(inside.sum()), n=n,
                mean_width=float((rows["upper_90_pp"] - rows["lower_90_pp"]).mean()) if n else np.nan)


def per_station(preds, stations=None):
    """Long table: station_id, reservoir_name, key, label, mae, n; plus coverage per station."""
    if preds is None or preds.empty:
        return pd.DataFrame(), pd.DataFrame()
    models = available_models(preds)
    long_rows, cov_rows = [], []
    for sid, g in preds.groupby("station_id", sort=True):
        name = g["reservoir_name"].dropna().iloc[0] if "reservoir_name" in g and g["reservoir_name"].notna().any() else sid
        t = mae_table(g, models)
        for r in t.itertuples(index=False):
            long_rows.append(dict(station_id=sid, reservoir_name=name, key=r.key, label=r.label, mae=r.mae, n=r.n))
        cv = coverage(g)
        cov_rows.append(dict(station_id=sid, reservoir_name=name, **cv))
    return pd.DataFrame(long_rows), pd.DataFrame(cov_rows)


# ---------------------------------------------------------------- metrics.json

@dataclass
class BootRow:
    model_key: str
    diff: float            # baseline MAE minus ML MAE; positive means ML has lower error
    lo: float
    hi: float
    level: float = 0.95
    n_boot: object = None
    method: str = ""
    sign_suspect: bool = False


@dataclass
class NormMetrics:
    present: bool = False
    data_mode: object = None
    mae: dict = field(default_factory=dict)          # model key -> float
    coverage: float = np.nan
    halfwidth: float = np.nan
    counts: dict = field(default_factory=dict)
    note: str = ""
    bootstrap: list = field(default_factory=list)
    bootstrap_raw_key: str = ""
    bootstrap_unparsed: bool = False
    limitations: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    weather_lag_days: object = None
    selected_baseline_key: str = ""
    extra: dict = field(default_factory=dict)


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [f"{k}: {x}" for k, x in v.items()]
    if isinstance(v, (list, tuple)):
        return [x if isinstance(x, (str, dict)) else str(x) for x in v]
    return [str(v)]


def _match_model(name):
    n = str(name).strip().lower()
    for m in C.MODELS:
        if n in (m.key, m.label.lower(), m.metric_key.lower(), m.pred_col.lower()):
            return m.key
    for token, key in (("persist", "persistence"), ("trend", "trend"), ("season", "seasonal")):
        if token in n:
            return key
    return None


_DIFF_POS = ("mae_diff_baseline_minus_ml", "baseline_minus_ml", "improvement_pp", "mae_improvement_pp",
             "improvement", "diff", "mae_diff", "delta", "point", "estimate", "mean")
_DIFF_NEG = ("mae_diff_ml_minus_baseline", "ml_minus_baseline")
_LO = ("ci95_low", "ci_low", "low", "lower", "ci_lower", "lo", "ci_lo")
_HI = ("ci95_high", "ci_high", "high", "upper", "ci_upper", "hi", "ci_hi")


def _parse_boot_entry(key, d):
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    mk = _match_model(key) if key is not None else None
    if mk is None:
        for k in ("baseline", "model", "comparison", "name"):
            if k in low:
                mk = _match_model(low[k])
                break
    if mk is None or mk == "ml":
        return None
    sign, diff = 1.0, np.nan
    for k in _DIFF_NEG:
        if k in low:
            sign, diff = -1.0, _num(low[k])
            break
    if np.isnan(diff):
        for k in _DIFF_POS:
            if k in low:
                diff = _num(low[k])
                break
    lo = next((_num(low[k]) for k in _LO if k in low), np.nan)
    hi = next((_num(low[k]) for k in _HI if k in low), np.nan)
    for k in ("ci", "ci95", "interval"):
        if k in low and isinstance(low[k], (list, tuple)) and len(low[k]) == 2:
            lo, hi = _num(low[k][0]), _num(low[k][1])
    if sign < 0:
        diff, lo, hi = -diff, -hi, -lo
    level = _num(low.get("level", low.get("ci_level", 0.95)))
    level = level / 100 if level > 1 else level
    return BootRow(mk, diff, lo, hi, level if np.isfinite(level) else 0.95,
                   low.get("n_boot", low.get("n_bootstrap", low.get("reps"))),
                   str(low.get("method", low.get("block", low.get("block_weeks", ""))) or ""))


def normalize_metrics(m):
    nm = NormMetrics()
    if not isinstance(m, dict):
        return nm
    nm.present = True
    K = C.METRIC_KEYS
    nm.data_mode = m.get(K["data_mode"])
    nested = m.get("mae") if isinstance(m.get("mae"), dict) else m.get("test_mae") if isinstance(m.get("test_mae"), dict) else {}
    for spec in C.MODELS:
        v = _num(m.get(spec.metric_key, nested.get(spec.metric_key, np.nan)))
        if np.isfinite(v):
            nm.mae[spec.key] = v
    nm.coverage = _num(m.get(K["coverage"]))
    if np.isfinite(nm.coverage) and nm.coverage > 1:
        nm.coverage /= 100.0
    nm.halfwidth = _num(m.get(K["halfwidth"]))
    bootmeta = m.get("bootstrap", {})
    if isinstance(bootmeta, dict):
        nm.selected_baseline_key = _match_model(bootmeta.get("best_baseline")) or ""
    for k in ("train_count", "val_count", "test_count", "test_station_count"):
        if K[k] in m:
            nm.counts[k] = m[K[k]]
    nm.note = str(m.get(K["note"]) or "")
    for bk in C.BOOTSTRAP_KEYS:
        if bk in m:
            nm.bootstrap_raw_key = bk
            raw = m[bk]
            if isinstance(raw, dict) and isinstance(raw.get("by_comparison_and_block"), dict):
                for comparison, blocks in raw["by_comparison_and_block"].items():
                    if " minus " not in comparison:
                        continue
                    label = comparison.split(" minus ", 1)[1].replace(" (monthly)", "")
                    mk = _match_model(label)
                    if not mk or mk == "ml" or not isinstance(blocks, dict):
                        continue
                    for block, vals in sorted(blocks.items(), key=lambda p: int(p[0].split("_")[0])):
                        if not isinstance(vals, dict):
                            continue
                        estimate = _num(vals.get("estimate"))
                        lo, hi = _num(vals.get("ci95_low")), _num(vals.get("ci95_high"))
                        if np.isfinite(estimate) and np.isfinite(lo) and np.isfinite(hi):
                            nm.bootstrap.append(BootRow(mk, -estimate, -hi, -lo, 0.95,
                                                       vals.get("n_boot", raw.get("n_boot")),
                                                       f"{block.replace('_', ' ')} moving-block bootstrap"))
            else:
                entries = raw.items() if isinstance(raw, dict) else [(None, e) for e in raw] if isinstance(raw, list) else []
                for key, entry in entries:
                    row = _parse_boot_entry(key, entry)
                    if row:
                        nm.bootstrap.append(row)
            nm.bootstrap_unparsed = not nm.bootstrap
            break
    for keys, attr in ((C.LIMITATIONS_KEYS, "limitations"), (C.ASSUMPTIONS_KEYS, "assumptions"), (C.SOURCES_KEYS, "sources")):
        for k in keys:
            if k in m:
                setattr(nm, attr, _as_list(m[k]))
                break
    for k in C.WEATHER_LAG_KEYS:
        if k in m:
            nm.weather_lag_days = _num(m[k])
            break
    # Flag bootstrap entries whose sign disagrees with the point MAEs.
    ml = nm.mae.get("ml")
    for b in nm.bootstrap:
        base = nm.mae.get(b.model_key)
        if ml is not None and base is not None and np.isfinite(b.diff):
            point = base - ml
            if abs(point) > 0.01 and np.sign(point) != np.sign(b.diff):
                b.sign_suspect = True
    known = set(K.values()) | {s.metric_key for s in C.MODELS} | set(C.BOOTSTRAP_KEYS) | set(C.LIMITATIONS_KEYS) \
        | set(C.ASSUMPTIONS_KEYS) | set(C.SOURCES_KEYS) | set(C.WEATHER_LAG_KEYS) | {"mae", "test_mae"}
    nm.extra = {k: v for k, v in m.items() if k not in known}
    return nm


# ---------------------------------------------------------------- verdict

@dataclass
class Verdict:
    kind: str          # "worse" | "unclear" | "better" | "point_only" | "unknown"
    headline: str
    detail: str
    best_key: str = ""
    source: str = ""   # "metrics.json" or "computed"


def best_baseline(mae: dict, preferred=None):
    cands = {k: v for k, v in mae.items() if k != "ml" and v is not None and np.isfinite(v)}
    if preferred in cands:
        return preferred
    if not cands:
        return None
    return min(cands, key=cands.get)


def verdict(nm: NormMetrics, computed: pd.DataFrame | None = None):
    mae, source = dict(nm.mae), "metrics.json"
    if ("ml" not in mae or best_baseline(mae) is None) and computed is not None and not computed.empty:
        mae, source = {r.key: r.mae for r in computed.itertuples(index=False)}, "computed"
    ml = mae.get("ml")
    bk = best_baseline(mae, nm.selected_baseline_key)
    if ml is None or bk is None:
        return Verdict("unknown", "Model comparison unavailable",
                       "Test MAE for the ML model and at least one baseline is needed to compare them.")
    base = mae[bk]
    bl = C.MODEL_BY_KEY[bk].label
    gap = base - ml
    src_note = "" if source == "metrics.json" else " Values computed in the app from test_predictions.csv because metrics.json lacks them."
    if gap <= TIE_TOL_PP:
        return Verdict("worse", f"The ML model does not beat the {bl.lower()} baseline",
                       f"On held-out test dates the ML model's mean absolute error is {ml:.2f} pp, against "
                       f"{base:.2f} pp for {bl.lower()}. On this evidence the model adds no predictive value over "
                       f"the simplest competitive rule.{src_note}", bk, source)
    boot = next((b for b in nm.bootstrap if b.model_key == bk and np.isfinite(b.lo) and np.isfinite(b.hi)), None)
    lvl = f"{boot.level:.0%}" if boot else ""
    if boot is None:
        return Verdict("point_only", f"Lower error than {bl.lower()}, uncertainty not reported",
                       f"ML MAE is {ml:.2f} pp against {base:.2f} pp for {bl.lower()}, a {gap:.2f} pp reduction. "
                       f"No bootstrap interval for this comparison was provided, so whether the gap exceeds "
                       f"sampling noise is unknown.{src_note}", bk, source)
    if boot.sign_suspect:
        return Verdict("unclear", "Improvement cannot be assessed: bootstrap sign looks inverted",
                       f"Point MAEs favour the ML model by {gap:.2f} pp, but the bootstrap estimate for "
                       f"{bl.lower()} has the opposite sign. Check the sign convention in metrics.json.", bk, source)
    if boot.lo > 0:
        return Verdict("better", f"ML beats {bl.lower()} by {gap:.2f} pp",
                       f"ML MAE {ml:.2f} pp against {base:.2f} pp. The {lvl} bootstrap interval for the reduction "
                       f"is {boot.lo:.2f} to {boot.hi:.2f} pp and excludes zero. This is a historical backtest; "
                       f"it does not establish future performance.{src_note}", bk, source)
    return Verdict("unclear", f"Improvement over {bl.lower()} is not statistically clear",
                   f"ML MAE is {ml:.2f} pp against {base:.2f} pp for {bl.lower()}, but the {lvl} bootstrap "
                   f"interval for the reduction runs from {boot.lo:.2f} to {boot.hi:.2f} pp and includes zero. "
                   f"The data do not distinguish the model from this baseline.{src_note}", bk, source)


def cross_checks(nm: NormMetrics, computed: pd.DataFrame | None, cov: dict, preds=None, tol=0.02):
    """Discrepancies between metrics.json and values recomputed from test_predictions.csv."""
    out = []
    if not nm.present or computed is None or computed.empty:
        return out
    for r in computed.itertuples(index=False):
        if r.key in nm.mae and np.isfinite(r.mae) and abs(nm.mae[r.key] - r.mae) > tol:
            out.append(f"{r.label}: metrics.json MAE {nm.mae[r.key]:.3f} pp, recomputed {r.mae:.3f} pp "
                       f"on {r.n} common rows. The pipeline may score each model on different rows.")
    if np.isfinite(nm.coverage) and np.isfinite(cov.get("frac", np.nan)) and abs(nm.coverage - cov["frac"]) > 0.005:
        out.append(f"Interval coverage: metrics.json {nm.coverage:.1%}, recomputed {cov['frac']:.1%} over {cov['n']} rows.")
    tc = nm.counts.get("test_count")
    if tc is not None and preds is not None:
        try:
            if int(tc) != len(preds):
                out.append(f"test_count is {int(tc)} in metrics.json but test_predictions.csv has {len(preds)} rows.")
        except (TypeError, ValueError):
            pass
    if np.isfinite(nm.halfwidth) and preds is not None and not preds.empty:
        hw = ((preds["upper_90_pp"] - preds["lower_90_pp"]) / 2).mean()
        if abs(hw - nm.halfwidth) > 0.02:
            out.append(f"Interval half-width: metrics.json {nm.halfwidth:.2f} pp, mean in predictions {hw:.2f} pp.")
    return out
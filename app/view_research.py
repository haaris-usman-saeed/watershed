"""D. Model and research: headline evaluation, uncertainty, features, sources and limits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import ui_charts as charts
import ui_components as ui
import ui_contract as C
import ui_metrics as M

VERDICT_STYLE = {"worse": "caution verdict", "unclear": "caution verdict", "point_only": "caution verdict",
                 "better": "good verdict", "unknown": "verdict"}

SOURCES = [
    ("California Data Exchange Center (CDEC)",
     "Daily reservoir storage, sensor 15, in acre-feet.",
     "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"),
    ("CDEC snow-water equivalent, sensor 82",
     "Adjusted historical snow-pillow series from upstream basins. Used by the selected v2 model.",
     "https://cdec.water.ca.gov/"),
    ("NASA POWER daily point API",
     "Daily T2M (2 m air temperature), PRECTOTCORR (bias-corrected precipitation) and RH2M (2 m relative "
     "humidity) for the grid cell at each reservoir's coordinates.",
     "https://power.larc.nasa.gov/docs/services/api/temporal/daily/"),
]
README_ASSUMPTIONS = [
    "Each observation is one reservoir on one Monday forecast origin; the target is the observed storage change "
    "over the next 30 days divided by documented capacity.",
    "The selected model uses CDEC snowpack through origin minus one day. Weather, available in the dataset and secondary models, ends at least seven days before the origin.",
    "Daily storage is never interpolated for inputs or targets.",
    "The current dataset covers 2001-2025. Rolling evaluation uses WY2008-2023; the preregistered v2 test covers 2024-2025. The same period was previously examined for v1.",
]
README_LIMITATIONS = [
    "NASA POWER is a coarse grid-cell estimate, not rainfall measured over the contributing watershed.",
    "POWER and CDEC values are retrospective and may have been revised since the original dates, so this is not "
    "a true as-of-time backtest.",
    "CDEC station coordinates and capacities were checked against official metadata. Oroville changed storage reporting basis in July 2024.",
    "Interval coverage on the test period is measured, not guaranteed for future data.",
]
SCOPE = ("This app shows a tabular forecasting experiment on historical data. It does not provide live forecasts, "
         "satellite imagery or CNN models, water-safety or water-quality assessments, or recommendations about "
         "where to site reservoirs.")


def _bullets(items):
    st.markdown("\n".join(f"- {ui.esc(str(i))}" for i in items))


def _scores(b, nm, computed):
    st.markdown("## Test-set scores")
    k = nm.counts
    c = st.columns(4)
    c[0].metric("Test forecasts", f"{int(k['test_count']):,}" if "test_count" in k else f"{len(b.preds):,}" if b.preds is not None else "n/a")
    c[1].metric("Reservoirs in test", str(k.get("test_station_count", b.preds["station_id"].nunique() if b.preds is not None else "n/a")))
    c[2].metric("Training rows", f"{int(k['train_count']):,}" if "train_count" in k else "n/a")
    c[3].metric("Rolling development forecasts" if (b.metrics or {}).get("model_version") == "v2" else "Validation rows",
                f"{int(k['val_count']):,}" if "val_count" in k else "n/a")

    comp = {r.key: (r.mae, r.n) for r in computed.itertuples(index=False)} if computed is not None else {}
    rows = []
    for spec in C.MODELS:
        if spec.key not in nm.mae and spec.key not in comp:
            continue
        rows.append({"Method": spec.label,
                     "Test MAE, metrics.json (pp)": nm.mae.get(spec.key, np.nan),
                     "Recomputed in app (pp)": comp.get(spec.key, (np.nan, 0))[0],
                     "What it does": spec.blurb})
    if rows:
        left, right = st.columns([3, 2], gap="large")
        with left:
            cfg = {"Test MAE, metrics.json (pp)": st.column_config.NumberColumn(format="%.3f"),
                   "Recomputed in app (pp)": st.column_config.NumberColumn(format="%.3f"),
                   "What it does": st.column_config.TextColumn(width="large")}
            ui.show_table(pd.DataFrame(rows), column_config=cfg)
            st.caption("Mean absolute error of the 30-day change, in percentage points of documented capacity. "
                       "metrics.json values come from the pipeline and drive the headline. Recomputed values use "
                       "rows where every method has a prediction and serve as a consistency check.")
        with right:
            t = pd.DataFrame([{"key": s.key, "label": s.label, "mae": nm.mae[s.key], "n": k.get("test_count", "")}
                              for s in C.MODELS if s.key in nm.mae])
            if t.empty and computed is not None:
                t = computed
            if not t.empty:
                ui.show_chart(charts.mae_bars(t, b.synthetic), key="rs_mae")


def _per_station(b):
    st.markdown("## Performance by reservoir")
    long, cov = M.per_station(b.preds)
    if long.empty:
        st.caption("No per-reservoir scores available.")
        return cov
    ui.show_chart(charts.per_station_dots(long, b.synthetic), key="rs_dots")
    losers = []
    for name, g in long.groupby("reservoir_name"):
        mae = dict(zip(g["key"], g["mae"]))
        nm = M.normalize_metrics(b.metrics)
        bk = M.best_baseline(mae, nm.selected_baseline_key)
        if "ml" in mae and bk and mae["ml"] >= mae[bk] - M.TIE_TOL_PP:
            losers.append(f"{name} ({C.MODEL_BY_KEY[bk].label.lower()} {mae[bk]:.2f} pp vs ML {mae['ml']:.2f} pp)")
    if losers:
        ui.callout(f"The ML model does not beat the best baseline at {len(losers)} of {long['reservoir_name'].nunique()} reservoirs",
                   ui.esc("; ".join(losers)) + ". Per-reservoir samples are small and weekly forecasts overlap, so "
                   "these differences are noisy.", "caution")
    st.caption("Computed in the app from test_predictions.csv on rows where every method has a prediction. "
               "Reservoirs are ordered by ML error.")
    return cov


def _coverage(b, nm, cov_rows):
    st.markdown("## Do the 90% intervals hold?")
    overall = M.coverage(b.preds)
    c = st.columns(4)
    c[0].metric("Nominal coverage", f"{C.INTERVAL_LEVEL:.0%}")
    emp = nm.coverage if np.isfinite(nm.coverage) else overall["frac"]
    c[1].metric("Empirical coverage, test period", ui.fmt_pct(emp * 100, 1),
                help="From metrics.json" if np.isfinite(nm.coverage) else "Computed in the app; metrics.json has no value.")
    c[2].metric("Interval half-width", ui.fmt_pp(nm.halfwidth, sign=False) if np.isfinite(nm.halfwidth) else "n/a",
                help="interval_90_halfwidth_pp from metrics.json.")
    c[3].metric("Mean interval width", ui.fmt_pp(overall["mean_width"], sign=False))
    if np.isfinite(emp) and emp < C.INTERVAL_LEVEL - 0.05:
        ui.callout("Intervals are too narrow",
                   f"Only {emp:.1%} of observed changes fell inside intervals meant to hold 90%. Treat the stated "
                   f"uncertainty as overconfident.", "caution")
    elif np.isfinite(emp) and emp > C.INTERVAL_LEVEL + 0.05:
        ui.callout("Intervals are wider than needed",
                   f"{emp:.1%} of observed changes fell inside the 90% intervals. They are conservative on this test "
                   f"period, which is safer than the reverse but less informative.", "note")
    if cov_rows is not None and not cov_rows.empty:
        ui.show_chart(charts.coverage_bars(cov_rows, overall, b.synthetic), key="rs_cov")
        fol = cov_rows[cov_rows["station_id"] == "FOL"]
        if not fol.empty and float(fol.iloc[0]["frac"]) < 0.85:
            ui.callout("Folsom uncertainty under-coverage",
                       f"Only {float(fol.iloc[0]["frac"]):.1%} of held-out Folsom observations fell inside nominal 90% intervals. "
                       "These bands underestimated historical uncertainty and should not be treated as reliable guidance.",
                       "caution")
    st.caption("Coverage counts how often the observed change landed inside its interval. Weekly forecasts with a "
               "30-day horizon overlap, so these counts are not independent and per-reservoir figures rest on about "
               "a hundred correlated forecasts each.")


def _bootstrap(b, nm):
    st.markdown("## Is the difference larger than sampling noise?")
    if nm.bootstrap:
        rows = sorted(nm.bootstrap, key=lambda r: [s.key for s in C.MODELS].index(r.model_key))
        plot_rows = [r for r in rows if r.method.startswith("5 weeks")] or rows
        ui.show_chart(charts.bootstrap_forest(plot_rows, b.synthetic), key="rs_boot")
        t = pd.DataFrame([{
            "Compared with": C.MODEL_BY_KEY[r.model_key].label + (f" ({r.method.split()[0]} weeks)" if r.method.split()[0].isdigit() else ""),
            "MAE reduction (pp)": r.diff, "Interval low": r.lo, "Interval high": r.hi,
            "Level": f"{r.level:.0%}",
            "Reading": ("ML clearly lower error" if r.lo > 0 else "baseline clearly lower error" if r.hi < 0
                        else "not distinguishable"),
            "Resamples": r.n_boot if r.n_boot is not None else "", "Method": r.method} for r in rows])
        cfg = {c: st.column_config.NumberColumn(format="%+.3f") for c in ("MAE reduction (pp)", "Interval low", "Interval high")}
        ui.show_table(t, column_config=cfg)
        for r in rows:
            if r.sign_suspect:
                ui.callout(f"Sign convention check: {C.MODEL_BY_KEY[r.model_key].label}",
                           "This bootstrap estimate has the opposite sign to the difference between the point MAEs in "
                           "metrics.json. The app reads positive values as ML having lower error. Confirm the "
                           "convention in UI_CONTRACT.md.", "caution")
        st.caption("Intervals come from the pipeline's bootstrap in metrics.json. An interval that crosses zero "
                   "means the test data cannot tell the two methods apart.")
    elif nm.bootstrap_unparsed:
        ui.callout("Bootstrap results present but unreadable",
                   f"metrics.json has a \"{ui.esc(nm.bootstrap_raw_key)}\" entry the app could not parse. Update "
                   f"the parser in app/ui_metrics.py to match the contract.", "caution")
    else:
        ui.callout("No bootstrap comparison provided",
                   "metrics.json has no bootstrap results, so the MAE differences above are point estimates whose "
                   "uncertainty is unknown. Do not read small differences as real improvements.", "caution")


def _features(b):
    st.markdown("## Model inputs and historical context")
    if not ui.require(b, C.FEATURES_FILE, purpose="The feature catalog"):
        return
    rows = []
    for f in b.features:
        rows.append({"Feature": f.get("name"), "Group": f.get("group", ""), "Description": f.get("description", ""),
                     "Source": f.get("source", ""), "Window (days)": f.get("window_days", ""),
                     "Lag before origin (days)": f.get("lag_days", ""), "Unit": f.get("unit", ""),
                     "Primary model input": ("Yes" if f.get("used_by_primary_model") is True else
                                             "No, context or secondary" if f.get("used_by_primary_model") is False else "Unspecified"),
                     **({"Other": f["other"]} if "other" in f else {})})
    t = pd.DataFrame(rows)
    t = t[[c for c in t.columns if t[c].astype(str).str.len().gt(0).any()]]
    for c in ("Window (days)", "Lag before origin (days)"):
        if c in t:
            t[c] = t[c].apply(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    ui.show_table(t, column_config={"Description": st.column_config.TextColumn(width="large")})
    lag = ui.weather_lag(b)
    if lag is not None:
        st.caption(f"Secondary-model weather inputs end at least {lag:.0f} days before the forecast origin. "
                   "The selected model uses storage, adjusted SWE at t-1, seasonality and reservoir identity.")


def _context(b, nm):
    st.markdown("## Data sources")
    for name, desc, url in SOURCES:
        st.markdown(f"**{name}.** {desc} [{url}]({url})")
    if nm.sources:
        _bullets(nm.sources)
    st.caption("The dataset is a newly assembled join of public sources, not new physical measurements.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("## Prediction-time assumptions")
        if nm.assumptions:
            _bullets(nm.assumptions)
        else:
            _bullets(README_ASSUMPTIONS)
            st.caption("From the project README; metrics.json lists no assumptions.")
    with right:
        st.markdown("## Limitations")
        if nm.limitations:
            _bullets(nm.limitations)
        else:
            _bullets(README_LIMITATIONS)
            st.caption("From the project README; metrics.json lists no limitations.")
    ui.callout("What this app does not do", SCOPE, "note")


def _diagnostics(b, nm, computed):
    with st.expander("Data contract diagnostics"):
        st.caption(f"Reading from {b.data_dir}. Use this to reconcile the frontend with UI_CONTRACT.md.")
        rows = []
        for name in C.ALL_FILES:
            r = b.reports.get(name)
            rows.append({"File": name, "Status": r.status, "Rows": r.rows or "",
                         "Problems": "; ".join([r.error] + r.warnings if r.error else r.warnings),
                         "Optional fields missing": ", ".join(r.missing_optional),
                         "Unused columns": ", ".join(r.extra_columns)})
        ui.show_table(pd.DataFrame(rows))
        checks = M.cross_checks(nm, computed, M.coverage(b.preds), b.preds) if b.preds is not None else []
        if checks:
            st.markdown("**metrics.json against test_predictions.csv**")
            _bullets(checks)
        elif nm.present and b.preds is not None:
            st.caption("metrics.json agrees with values recomputed from test_predictions.csv.")
        if nm.extra:
            st.markdown("**metrics.json keys the app does not display**")
            st.json(nm.extra, expanded=False)


def render():
    b = ui.load_bundle()
    ui.page_intro("Model and research",
                  "What the held-out evaluation shows, how much to trust it, and what the data can and cannot "
                  "support.")
    if not (b.reports[C.METRICS_FILE].usable or b.reports[C.PREDICTIONS_FILE].usable):
        ui.require(b, C.METRICS_FILE, C.PREDICTIONS_FILE, purpose="Model evaluation")
        _context(b, M.NormMetrics())
        return
    nm = M.normalize_metrics(b.metrics)
    computed = M.mae_table(b.preds) if b.preds is not None else None
    v = M.verdict(nm, computed)
    ui.callout(ui.esc(v.headline), ui.esc(v.detail), VERDICT_STYLE[v.kind])
    if nm.note:
        st.caption(f"Pipeline note: {nm.note}")
    if not b.reports[C.METRICS_FILE].usable:
        ui.require(b, C.METRICS_FILE, purpose="Headline metrics")

    _scores(b, nm, computed)
    cov_rows = _per_station(b) if b.preds is not None else None
    if b.preds is not None:
        _coverage(b, nm, cov_rows)
    _bootstrap(b, nm)
    _features(b)
    _context(b, nm)
    _diagnostics(b, nm, computed)
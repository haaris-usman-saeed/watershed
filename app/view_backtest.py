"""C. ML backtesting: held-out historical predictions scored against what happened."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import ui_charts as charts
import ui_components as ui
import ui_contract as C
import ui_io
import ui_metrics as M
from ui_theme import PALETTE as P

SCOPES = {"one": "This reservoir", "all": "All reservoirs"}


def _kpis(p, scope_label, preferred=None):
    table = M.mae_table(p)
    cov = M.coverage(p)
    mae = {r.key: r.mae for r in table.itertuples(index=False)}
    ml = mae.get("ml")
    bk = M.best_baseline(mae, preferred)
    c = st.columns(4)
    if ml is not None and bk:
        diff = ml - mae[bk]
        c[0].metric(f"ML mean absolute error, {scope_label}", ui.fmt_pp(ml, sign=False),
                    delta=f"{diff:+.2f} pp vs {C.MODEL_BY_KEY[bk].label.lower()}", delta_color="inverse",
                    help="Average size of the miss, in percentage points of capacity. Lower is better.")
        c[1].metric(f"Best baseline: {C.MODEL_BY_KEY[bk].label}", ui.fmt_pp(mae[bk], sign=False),
                    help="Lowest-error simple rule over the same forecasts.")
    else:
        c[0].metric("ML mean absolute error", ui.fmt_pp(ml, sign=False) if ml is not None else "n/a")
        c[1].metric("Best baseline", "n/a", help="No baseline columns in test_predictions.csv.")
    c[2].metric("Inside the 90% interval", ui.fmt_pct(cov["frac"] * 100 if cov["n"] else np.nan),
                help=f"{cov['inside']} of {cov['n']} observed changes fell inside the stated interval.")
    n = table["n"].iloc[0] if len(table) else len(p)
    c[3].metric("Forecasts scored", f"{n:,}",
                help=f"Origins {ui.fmt_date(p['date'].min(), True)} to {ui.fmt_date(p['date'].max(), True)}.")
    return table


def _inspector(b, p, sid, baselines_all):
    st.markdown("## Inspect a single forecast")
    st.caption("Pick a forecast origin to see what the model knew, what it predicted, and what happened.")
    origins = p["date"].tolist()
    key = f"bt_origin_{sid}"
    origin = st.select_slider("Forecast origin", options=origins, value=origins[len(origins) // 2], key=key,
                              format_func=lambda d: pd.Timestamp(d).strftime("%b %d, %Y"))
    row = p.loc[p["date"] == origin].iloc[0]
    cap = float(row["capacity_af"]) if "capacity_af" in row.index and pd.notna(row["capacity_af"]) else b.capacity(sid)
    left, right = st.columns([3, 2], gap="large")
    daily = ui_io.daily_series(b.history, sid, cap) if b.history is not None else None
    with left:
        if daily is not None and not daily.empty:
            fig, s0 = charts.inspector_figure(daily, row, None if (b.metrics or {}).get("model_version") == "v2" else ui.weather_lag(b), b.synthetic, baselines_all)
            ui.show_chart(fig, key=f"bt_inspect_{sid}")
            if not np.isfinite(s0):
                st.caption("No storage value on the origin date in history_daily.csv, so predicted levels "
                           "cannot be placed on the storage axis. The change values on the right still apply.")
            else:
                tgt = daily.loc[daily["date"] == row["target_date"], "storage_pct"]
                if len(tgt) and pd.notna(tgt.iloc[0]):
                    gap = (tgt.iloc[0] - s0) - row["target_delta_pp"]
                    if abs(gap) > 0.05:
                        ui.callout("Files disagree on this forecast's outcome",
                                   f"history_daily.csv implies an observed change of {tgt.iloc[0] - s0:+.2f} pp, "
                                   f"test_predictions.csv records {row['target_delta_pp']:+.2f} pp. Capacity, dates "
                                   f"or storage values may differ between the two files.", "caution")
        else:
            st.caption("history_daily.csv is unavailable, so the storage trajectory cannot be drawn.")
    with right:
        obs = row["target_delta_pp"]
        af = (lambda v: f"{v / 100 * cap:+,.0f} AF") if np.isfinite(cap) else (lambda v: "n/a")
        inside = row["lower_90_pp"] <= obs <= row["upper_90_pp"]
        trs = [f'<tr><th>Forecast</th><th>30-day change</th><th>Error</th><th>Acre-feet</th></tr>',
               f'<tr><td><span class="ws-swatch" style="background:{P["bone"]}"></span>Observed</td>'
               f'<td>{ui.fmt_pp(obs)}</td><td></td><td>{af(obs)}</td></tr>',
               f'<tr><td><span class="ws-swatch" style="background:{P["model"]}"></span>ML model</td>'
               f'<td>{ui.fmt_pp(row["pred_delta_pp"])}</td><td>{ui.fmt_pp(row["pred_delta_pp"] - obs)}</td>'
               f'<td>{af(row["pred_delta_pp"])}</td></tr>',
               f'<tr><td style="padding-left:1.9rem">90% interval</td><td colspan="2">'
               f'{row["lower_90_pp"]:+.2f} to {row["upper_90_pp"]:+.2f} pp</td>'
               f'<td class="{"ws-yes" if inside else "ws-no"}">{"contains observed" if inside else "misses observed"}</td></tr>']
        for bl in baselines_all:
            v = row.get(bl.pred_col)
            if pd.notna(v):
                trs.append(f'<tr><td><span class="ws-swatch" style="background:{P[bl.color]}"></span>{bl.label}</td>'
                           f'<td>{ui.fmt_pp(v)}</td><td>{ui.fmt_pp(v - obs)}</td><td>{af(v)}</td></tr>')
        st.markdown(f'<table class="ws-ftable">{"".join(trs)}</table>', unsafe_allow_html=True)
        st.caption(f"Made on {ui.fmt_date(row['date'], True)} for {ui.fmt_date(row['target_date'], True)}. "
                   "Error is predicted minus observed. Acre-feet convert percentage points using the capacity in "
                   "the origin-date capacity.")


def render():
    b = ui.load_bundle()
    if not ui.require(b, C.PREDICTIONS_FILE, purpose="Backtesting"):
        return
    preds = b.preds
    t0, t1 = preds["date"].min(), preds["target_date"].max()
    ui.page_intro("Backtest on held-out dates",
                  "How the model's 30-day storage predictions compare with what actually happened, and with "
                  "historical baselines selected using pre-2024 development data.")
    ui.callout("Historical predictions, not forecasts",
               f"Every value here is a prediction for a past date in the test period, {ui.fmt_date(t0, True)} to "
               f"{ui.fmt_date(t1, True)}, scored against the change that was later observed. The pipeline holds "
               f"these dates out of training and model selection; Model and research lists the split rules and "
               f"their limits. Nothing on this page predicts future storage.", "model")

    ids = [s for s in b.station_ids if s in set(preds["station_id"])] or sorted(preds["station_id"].unique())
    sid = ui.station_picker(b, "bt", ids=ids)
    p = preds[preds["station_id"] == sid].sort_values("date").reset_index(drop=True)
    baselines_all = [m for m in M.available_models(preds) if m.key != "ml"]

    nm = M.normalize_metrics(b.metrics)
    best = M.best_baseline(nm.mae, nm.selected_baseline_key) or M.best_baseline(
        {r.key: r.mae for r in M.mae_table(preds).itertuples(index=False)})
    c1, c2 = st.columns([2, 3])
    with c1:
        scope = ui.segmented("Scores for", ["one", "all"], "one", key="bt_scope", format_func=SCOPES.get)
    with c2:
        keys = [m.key for m in baselines_all]
        shown_keys = ui.multi_pills("Baselines on the chart", keys, [best] if best in keys else [], key="bt_baselines",
                                    format_func=lambda k: C.MODEL_BY_KEY[k].label)
    shown = [C.MODEL_BY_KEY[k] for k in shown_keys]

    scope_df = p if scope == "one" else preds
    table = _kpis(scope_df, b.name(sid) if scope == "one" else "all reservoirs", preferred=best)
    if (b.metrics or {}).get("model_version") == "v2" and sid == "FOL":
        fol = (b.metrics or {}).get("per_station", {}).get("FOL", {})
        cov_fol = fol.get("interval_90_coverage")
        if cov_fol is not None and cov_fol < 0.85:
            ui.callout("Folsom interval warning",
                       f"Only {cov_fol:.1%} of historical Folsom targets fell inside nominal 90% intervals. "
                       "Do not interpret these bands as calibrated future forecasts.", "caution")

    st.markdown(f"## Predicted and observed change at {ui.esc(b.name(sid))}")
    if p.empty:
        st.caption("No test forecasts for this reservoir.")
        return
    ui.show_chart(charts.backtest_timeline(p, shown, b.synthetic), key=f"bt_timeline_{sid}")
    st.caption("Each point is a separate 30-day forecast, placed at its origin date; lines join neighbouring "
               "weekly forecasts for readability and break where weeks are missing. The shaded band is the model's "
               "90% prediction interval. Units are percentage points of documented capacity.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Predicted against observed, all reservoirs")
        ui.show_chart(charts.pred_vs_actual(preds, sid, b.synthetic), key=f"bt_scatter_{sid}")
        st.caption("Points on the dotted line are perfect forecasts. The selected reservoir is highlighted.")
    with right:
        st.markdown(f"### Error by method, {SCOPES[scope].lower()}")
        if table.empty:
            st.caption("No model columns available to compare.")
        else:
            ui.show_chart(charts.mae_bars(table, b.synthetic), key=f"bt_mae_{sid}_{scope}")
            st.caption(f"Computed in the app from test_predictions.csv over {int(table['n'].iloc[0]):,} forecasts "
                       f"where every method has a value. Headline scores and their uncertainty are on Model and "
                       f"research.")

    _inspector(b, p, sid, baselines_all)

    st.markdown("## Download")
    out = (p if scope == "one" else preds).copy()
    for c in ("date", "target_date"):
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    st.download_button("Download held-out predictions as CSV", out.to_csv(index=False).encode(),
                       file_name=f"heldout_predictions_{sid if scope == 'one' else 'all'}.csv", mime="text/csv",
                       key="bt_download")
"""Model playground: the other nine study models against the site's HGB, on past windows only.

Research comparison, not forecasting. The site's forecasts come only from HGB[S,N,C,R] · frozen v2 artifact
(30-Day Forecast page). Everything here is read from tracked v5 result files through src.v5.playground; nothing is
fitted, re-scored or tested on this page. Per-row replays need the private prediction file and, on a public
deployment, are limited to models whose input sources are all cleared by config/data_release_policy.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui_components as ui
import ui_nav
from ui_theme import PALETTE as P, axis, finish, rgba

MAIN_COLOR, PICK_COLOR, OBS_COLOR = P["model"], P["temp"], P["bone"]
REGIMES = ["rapid fill (>= +10 pp)", "rapid drawdown (<= -10 pp)", "other"]


def _pg():
    from src.v5 import playground
    return playground


def _f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    x = round(float(x), d) + 0.0
    return (f"{x:+.{d}f}" if sign and x != 0 else f"{x:.{d}f}").replace("-", "−")


@st.cache_data(show_spinner=False)
def _summary(window):
    return _pg().summary(window)


@st.cache_data(show_spinner=False)
def _breakdown(window, by):
    return _pg().breakdown(window, by)


@st.cache_data(show_spinner=False)
def _comparisons(window, number):
    pg = _pg()
    return pg.comparisons(window, pg.by_number(number))


@st.cache_data(show_spinner="Reading per-row predictions")
def _rows(window, mode):
    return _pg().rows(window, mode=mode)


def _label(m):
    return f"{m['number']}. {m['name']}"


# ---------------------------------------------------------------- figures

def reservoir_figure(bd, pick, main_name, height=330):
    stations = sorted(bd.station_id.unique())
    fig = go.Figure()
    for name, color in ((main_name, MAIN_COLOR), (pick["name"], PICK_COLOR)):
        s = bd[bd.name == name].set_index("station_id").reindex(stations)
        fig.add_trace(go.Bar(x=stations, y=s["mae"], name=name, marker=dict(color=color, line=dict(width=0)),
                             customdata=np.stack([s["n"], s["bias"]], axis=1),
                             hovertemplate="%{x}<br>" + name + "<br>MAE %{y:.2f} pp<br>bias %{customdata[1]:+.2f} pp"
                                           "<br>%{customdata[0]} rows<extra></extra>"))
    fig = finish(fig, False, height, barmode="group", bargap=0.35, bargroupgap=0.08, hovermode="closest",
                 legend=dict(orientation="h", y=1.12, x=0), margin=dict(l=60, r=12, t=48, b=36))
    fig.update_yaxes(**axis(title=dict(text="MAE (pp of capacity)"), rangemode="tozero"))
    fig.update_xaxes(**axis())
    return fig


def series_figure(fr, station, pick_run, pick_name, main_run, main_name, height=380):
    d = fr[fr.station_id == station].sort_values("date")
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=rgba(P["silt"], 0.35), width=1))
    dense = len(d) > 120                         # eight water years: markers would hide the lines
    fig.add_trace(go.Scatter(x=d.date, y=d.target_delta_pp, mode="lines" if dense else "lines+markers",
                             name="Observed 30-day change", line=dict(color=OBS_COLOR, width=2),
                             marker=dict(size=8, symbol="circle"),
                             hovertemplate="%{y:+.2f} pp<extra>observed</extra>"))
    if main_run in d:
        fig.add_trace(go.Scatter(x=d.date, y=d[main_run], mode="lines", name=main_name,
                                 line=dict(color=MAIN_COLOR, width=2), hovertemplate="%{y:+.2f} pp<extra>" + main_name + "</extra>"))
    if pick_run in d:
        fig.add_trace(go.Scatter(x=d.date, y=d[pick_run], mode="lines", name=pick_name,
                                 line=dict(color=PICK_COLOR, width=2, dash="dash"),
                                 hovertemplate="%{y:+.2f} pp<extra>" + pick_name + "</extra>"))
    fig = finish(fig, False, height, hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0),
                 margin=dict(l=64, r=12, t=48, b=36))
    fig.update_yaxes(**axis(title=dict(text="30-day change (pp of capacity)"), ticksuffix=" pp"))
    fig.update_xaxes(**axis(title=dict(text="Forecast origin (Monday)")))
    return fig


def replay_figure(rp, pick_number, main_number, height=None):
    rp = rp.sort_values("number", ascending=False)
    names = [f"{n}. {nm}" for n, nm in zip(rp.number, rp.name)]
    colors = [MAIN_COLOR if n == main_number else PICK_COLOR if n == pick_number else rgba(P["silt"], 0.8)
              for n in rp.number]
    sizes = [13 if n in (main_number, pick_number) else 9 for n in rp.number]
    fig = go.Figure()
    s0, obs = float(rp.storage_now_pct.iloc[0]), float(rp.obs_day30_pct.iloc[0])
    for x, label, dash in ((s0, "Storage at origin", "dot"), (obs, "Observed on day 30", "solid")):
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color=OBS_COLOR if dash == "solid" else P["water"], width=1.5, dash=dash))
        fig.add_annotation(x=x, y=1, xref="x", yref="paper", text=label, showarrow=False, yanchor="bottom",
                           font=dict(size=11, color=P["silt"]))
    fig.add_trace(go.Scatter(x=rp.pred_day30_pct, y=names, mode="markers", name="Predicted storage on day 30",
                             marker=dict(size=sizes, color=colors, symbol="diamond", line=dict(width=2, color=P["ink"])),
                             customdata=np.stack([rp.pred_change_pp, rp.error_pp], axis=1),
                             hovertemplate="%{y}<br>day 30: %{x:.1f}%<br>change %{customdata[0]:+.2f} pp"
                                           "<br>error %{customdata[1]:+.2f} pp<extra></extra>"))
    fig = finish(fig, False, height or 90 + 34 * len(rp), showlegend=False, hovermode="closest",
                 margin=dict(l=190, r=16, t=40, b=40))
    fig.update_xaxes(**axis(title=dict(text="Storage on day 30 (% of capacity at the origin)"), ticksuffix="%"))
    fig.update_yaxes(**axis())
    return fig


# ---------------------------------------------------------------- tables

def summary_table(s, pick_number):
    out = pd.DataFrame({
        "No.": s.number, "Model": [("→ " if n == pick_number else "") + nm for n, nm in zip(s.number, s.name)],
        "Specification": s.spec, "Rows": s.n,
        "MAE (pp)": s.mae.map(_f), "MAE vs HGB (pp)": s.delta_mae_vs_main.map(lambda v: _f(v, sign=True)),
        "RMSE (pp)": s.rmse.map(_f), "Bias (pp)": s.bias.map(lambda v: _f(v, sign=True)), "R²": s.r2.map(_f),
        "Skill vs DOY": s.skill_vs_doy.map(_f),
        "90% coverage": s.coverage90.map(_f), "90% width (pp)": s.width90.map(lambda v: _f(v, 2))})
    return out


def comparison_table(c):
    return pd.DataFrame({
        "Family": c.family, "Comparison": c.description, "MAE difference (pp)": c.estimate_pp.map(lambda v: _f(v, sign=True)),
        "p (largest of 5, 9, 13-week blocks)": c.p_max.map(_f), "Holm p": c.p_holm.map(_f), "Verdict": c.verdict})


def regime_table(bd_regime, bd_wy, names):
    rows = []
    for nm in names:
        r = {"Model": nm}
        for reg in REGIMES:
            x = bd_regime[(bd_regime.name == nm) & (bd_regime.regime == reg)]
            r[reg.capitalize()] = f"{_f(x.mae.iloc[0], 2)} ({int(x.n.iloc[0])})" if len(x) else "n/a"
        for cls in sorted(bd_wy.wy_class.unique()):
            x = bd_wy[(bd_wy.name == nm) & (bd_wy.wy_class == cls)]
            r[f"{cls} years"] = f"{_f(x.mae.iloc[0], 2)} ({int(x.n.iloc[0])})" if len(x) else "n/a"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- page

def render():
    pg = _pg()
    ui.page_intro("Model playground",
                  "Compare the other nine study models with the HGB that makes this site's forecasts, on past windows. "
                  "Nothing on this page is a forecast, and none of these models is used for forecasting.")
    ui.callout("Research comparison only",
               "The site's 30-day forecasts come only from HGB[S,N,C,R] &middot; frozen v2 artifact. The other models "
               "here are v5 per-fold refits; none is frozen. Scores on 2024-2025 and 2026 are post-hoc, because those "
               "windows were already used as one-shot tests, so no model may be picked as better from them. The "
               "confirmation folds are the fair basis for comparison. Differences count only where the paired "
               "comparison table says so.", "caution")
    c1, c2 = st.columns([2, 3])
    with c1:
        window = ui.segmented("Window", list(pg.WINDOWS), "confirmation", key="pg_window",
                              format_func=lambda w: pg.WINDOWS[w]["label"])
    with c2:
        nums = [m["number"] for m in pg.OTHERS]
        pick_number = st.selectbox("Model to compare with the HGB", nums, index=nums.index(10), key="pg_model",
                                   format_func=lambda n: _label(pg.by_number(n)))
    pick, main = pg.by_number(pick_number), pg.by_number(pg.MAIN_NUMBER)
    st.caption(pg.WINDOWS[window]["status"] + " " + pg.main_note(window))
    try:
        s = _summary(window)
    except (FileNotFoundError, ValueError) as exc:
        st.markdown(f'<div class="ws-empty"><h4>Playground results are not on this deployment</h4><p>{ui.esc(str(exc))}'
                    f'</p></div>', unsafe_allow_html=True)
        return
    srow = s.set_index("number")
    if pick_number not in srow.index or pg.MAIN_NUMBER not in srow.index:
        ui.callout("Not scored in this window", ui.esc(f"{_label(pick)} or the HGB has no result in this window."), "caution")
        return
    a, m = srow.loc[pick_number], srow.loc[pg.MAIN_NUMBER]

    st.markdown(f"## {ui.esc(_label(pick))} against the HGB")
    cols = st.columns(4)
    # ASCII sign so Streamlit reads the direction; grey ("off") because a difference alone is not a verdict.
    cols[0].metric(f"MAE, {pick['name']}", f"{_f(a.mae)} pp", f"{a.mae - m.mae:+.3f} pp vs HGB", delta_color="off")
    cols[1].metric("MAE, HGB (main model)", f"{_f(m.mae)} pp")
    cols[2].metric(f"Bias, {pick['name']}", f"{_f(a.bias, sign=True)} pp", help="Mean of prediction minus observed.")
    cols[3].metric(f"Skill vs day-of-year baseline", _f(a.skill_vs_doy),
                   help="1 - MAE / MAE of the smoothed day-of-year seasonal baseline. Above 0 means fewer errors.")
    st.caption(f"{int(a.n):,} rows: every Monday origin in the window for all six reservoirs, the same rows for every "
               f"model. MAE = mean absolute error of the 30-day storage change, in percentage points of capacity.")
    ui.facts([("Specification", ui.esc(pick["spec"])), ("Inputs", ui.esc(pick["inputs"])),
              ("v5 run", f"<code>{ui.esc(pg.run_id(pick, window))}</code>"),
              ("Compared with", ui.esc(f"{pg.spec(main, window)} (run {pg.run_id(main, window)})"))])

    st.markdown("### Paired tests")
    c = _comparisons(window, pick_number)
    if c.empty:
        st.caption("No paired test with this model first was run in this window. An MAE difference without a test is "
                   "not evidence that one model is better.")
    else:
        ui.show_table(comparison_table(c))
        st.caption("Paired moving-block bootstrap from results/v5/stats/comparisons.csv (2,000 replicates; 5, 9 and "
                   "13-week blocks; the verdict uses the largest p). Holm-corrected only within the planned families. "
                   "Post-hoc rows are descriptive. Nothing is recomputed on this page.")
        ref = pg.run_id(main, window)
        if not c.contrast.str.endswith(f"*{ref}").any():
            st.caption(f"No test of this model against the HGB itself ({ref}) was run in this window.")

    st.markdown("## All ten study models")
    ui.show_table(summary_table(s, pick_number))
    st.caption("Report numbering. MAE vs HGB is a difference of the two MAEs on the same rows, not a test. 90% coverage "
               "and width are split-conformal intervals calibrated on the tuning folds; the HGB artifact has its own "
               "seasonal ranges, so it shows n/a. Networks are scored on their median prediction and the tabular models "
               "on their mean, so MAE may favour the networks and RMSE the tabular models.")

    st.markdown("## Where the errors are")
    try:
        bd = _breakdown(window, "station_id")
        bd = bd[bd.number.isin([pick_number, pg.MAIN_NUMBER])]
        ui.show_chart(reservoir_figure(bd, pick, main["name"]), key=f"pg_res_{window}_{pick_number}")
        st.caption("MAE by reservoir. Hover a bar for its bias and row count.")
        reg, wy = _breakdown(window, "regime"), _breakdown(window, "wy_class")
        ui.show_table(regime_table(reg, wy, [main["name"], pick["name"]]))
        st.caption("MAE in pp (rows in brackets) by what actually happened: rapid fill (rose by 10 pp or more), rapid "
                   "drawdown (fell by 10 pp or more), other; and by water-year class.")
    except FileNotFoundError as exc:
        st.caption(f"Breakdowns are not available: {exc}")

    st.markdown("## Replay a past 30-day forecast")
    from src.v3_service import deployment_mode
    fr, hidden = _rows(window, deployment_mode())
    if fr is None:
        ui.callout("Replays are not available here", ui.esc(hidden.get("all", "")) + " The tables and charts above come "
                   "from tracked aggregate files and are complete without them.", "note")
        return
    if hidden:
        ui.callout("Some models are withheld here",
                   ui.esc("Per-row predictions of models " + ", ".join(str(n) for n in sorted(hidden)) + " are not shown "
                          "on this public deployment: their inputs include CDEC inflow and outflow, which the data "
                          "release policy does not clear for publication."), "caution")
    stations = sorted(fr.station_id.unique())
    r1, r2 = st.columns([3, 2])
    with r1:
        station = ui.segmented("Reservoir", stations, "SHA" if "SHA" in stations else stations[0], key="pg_station")
    dates = sorted(fr[fr.station_id == station].date.unique(), reverse=True)
    with r2:
        origin = st.selectbox("Forecast origin (Monday)", dates, key=f"pg_origin_{window}_{station}",
                              format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    rp = pg.replay(fr, window, station, origin)
    if rp.empty:
        st.caption("No row for this reservoir and origin.")
        return
    tgt = pd.Timestamp(origin) + pd.Timedelta(days=30)
    st.caption(f"From the Monday {pd.Timestamp(origin):%Y-%m-%d} to {tgt:%Y-%m-%d}. Storage on day 30 is in % of the "
               f"capacity in force at the origin. Each model's prediction was made by the fold model that never saw "
               f"this window.")
    ui.show_chart(replay_figure(rp, pick_number, pg.MAIN_NUMBER), key=f"pg_replay_{window}_{station}")
    ui.show_table(pd.DataFrame({
        "Model": [f"{n}. {nm}" for n, nm in zip(rp.number, rp.name)],
        "Predicted change (pp)": rp.pred_change_pp.map(lambda v: _f(v, 2, True)),
        "Predicted day 30": rp.pred_day30_pct.map(lambda v: f"{v:.1f}%"),
        "Observed day 30": rp.obs_day30_pct.map(lambda v: f"{v:.1f}%"),
        "Error (pp)": rp.error_pp.map(lambda v: _f(v, 2, True))}))
    pick_run, main_run = pg.run_id(pick, window), pg.run_id(main, window)
    st.markdown(f"### {ui.esc(station)}: every origin in the window")
    ui.show_chart(series_figure(fr, station, pick_run, pick["name"], main_run, main["name"]),
                  key=f"pg_series_{window}_{station}_{pick_number}")
    st.caption("Observed 30-day change against the two models' predictions, one point per Monday origin.")
    if st.button("Back to the 30-Day Forecast", key="pg_to_forecast"):
        ui_nav.go("forecast")

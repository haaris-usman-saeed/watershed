"""Interactive part of the 30-Day Forecast page: Generate Forecast with Models A, B, B2 and C.

Calls src.v3_service.ForecastService in-process (one shared instance per server). The service
caches upstream data for all visitors, coalesces identical requests, rate-limits upstream fetches,
and refuses to show CDEC-derived values on a public deployment until rights are cleared.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui_components as ui
import ui_io
from ui_theme import PALETTE as P, axis, finish, rgba

MODEL_COLORS = {"A": P["model"], "B": P["temp"], "B2": P["trend"], "C": P["rain"]}
MODEL_INPUTS = {
    "A": "Storage now and its 7, 30 and 60-day changes; basin snowpack (t-1), its 14-day change and anomaly; season; reservoir.",
    "B": "Model A inputs + NASA POWER rain (7 and 30 days), temperature and humidity (30 days) at the dam, ending 7 days before the origin.",
    "B2": "Model A inputs + the same four weather measures averaged over the upstream box instead of the dam point.",
    "C": "Model A inputs + GEFS forecast precipitation over the upstream box for days 1-7 and 8-16 (00Z cycle of the origin Monday). Nothing represents days 17-30.",
}


@st.cache_resource(show_spinner=False)
def service():
    from src.v3_artifacts import ensure_artifacts
    from src.v3_service import ForecastService
    ensure_artifacts()          # no-op when present; installs hash-pinned artifacts when configured
    return ForecastService()


@st.cache_resource(show_spinner=False)
def record_store():
    from src.v3_service import RecordStore
    return RecordStore()


def _perf():
    """Historical performance from aggregate result files (tracked, publishable)."""
    out = {}
    for name, p in (("dev", "results/v3/development/summary.json"), ("final", "results/v3/final/metrics.json")):
        f = ui_io.REPO_ROOT / p
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


def _d(s, long=True):
    return ui.fmt_date(pd.Timestamp(s), long) if s else "n/a"


def endpoint_figure(r, synthetic=False, height=430):
    h = pd.DataFrame(r.get("recent_history", []))
    o, t = pd.Timestamp(r["origin_date"]), pd.Timestamp(r["target_date"])
    fig = go.Figure()
    fig.add_shape(type="rect", x0=o, x1=t, y0=0, y1=1, xref="x", yref="paper", fillcolor=rgba(P["model"], 0.05),
                  line=dict(width=0), layer="below")
    fig.add_annotation(x=o + (t - o) / 2, y=0.02, xref="x", yref="paper", showarrow=False,
                       text="30-day horizon: only the day-30 value is predicted", font=dict(size=11, color=P["silt"]))
    if not h.empty:
        h["date"] = pd.to_datetime(h["date"])
        used, after = h[h.used_by_forecast], h[~h.used_by_forecast]
        fig.add_trace(go.Scatter(x=used.date, y=used.storage_pct, mode="lines", name="Observed storage used (CDEC)",
                                 line=dict(color=P["water"], width=2), hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}%<extra></extra>"))
        if not after.empty:
            fig.add_trace(go.Scatter(x=after.date, y=after.storage_pct, mode="lines+markers", name="Observed after the origin (not used)",
                                     line=dict(color=P["bone"], width=1.2, dash="dot"), marker=dict(size=4),
                                     hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}%<extra></extra>"))
    avail = [k for k, m in r["models"].items() if m.get("available")]
    for i, k in enumerate(avail):
        m = r["models"][k]
        x = t + pd.Timedelta(hours=(i - (len(avail) - 1) / 2) * 40)
        iv = m["interval"]
        fig.add_trace(go.Scatter(
            x=[x], y=[m["projected_pct"]], mode="markers", name=f"Model {k}: day-30 estimate and 90% range",
            marker=dict(size=11, symbol="diamond", color=MODEL_COLORS[k], line=dict(width=1, color=P["ink"])),
            error_y=dict(type="data", symmetric=False, array=[iv["upper_projected_pct"] - m["projected_pct"]],
                         arrayminus=[m["projected_pct"] - iv["lower_projected_pct"]], color=MODEL_COLORS[k], thickness=2, width=6),
            hovertemplate=f"Model {k}<br>%{{x|%b %d, %Y}}<br>%{{y:.1f}}% (90%: {iv['lower_projected_pct']:.1f} to "
                          f"{iv['upper_projected_pct']:.1f})<extra></extra>"))
    fig.add_shape(type="line", x0=o, x1=o, y0=0, y1=1, xref="x", yref="paper", line=dict(color=P["model"], width=1, dash="dot"))
    fig.add_annotation(x=o, y=1, xref="x", yref="paper", text="Origin", showarrow=False, xanchor="left", yanchor="top",
                       xshift=4, font=dict(size=11, color=P["model"]))
    fig = finish(fig, synthetic, height, hovermode="closest", margin=dict(l=70, r=18, t=70, b=40))
    fig.update_yaxes(**axis(title=dict(text="% of capacity"), ticksuffix="%"))
    fig.update_xaxes(**axis(), range=[o - pd.Timedelta(days=60), t + pd.Timedelta(days=7)])
    return fig


def weather_figures(r):
    figs = []
    wh = r.get("weather_history", {})
    o = pd.Timestamp(r["origin_date"])
    if wh.get("dam_point"):
        s = pd.Series(wh["dam_point"], dtype=float)
        s.index = pd.to_datetime(s.index)
        used = (s.index <= o - pd.Timedelta(days=7)) & (s.index >= o - pd.Timedelta(days=36))
        fig = go.Figure(go.Bar(x=s.index, y=s.values, marker_color=[P["rain"] if u else rgba(P["silt"], 0.35) for u in used],
                               hovertemplate="%{x|%b %d}<br>%{y:.1f} mm<extra></extra>", name="Observed rain"))
        fig.add_annotation(xref="paper", yref="paper", x=0, y=1.08, showarrow=False, xanchor="left",
                           text="Observed precipitation at the dam (NASA POWER). Blue: days in Model B's window (ending 7 days before the origin)",
                           font=dict(size=11, color=P["silt"]))
        fig = finish(fig, False, 230, margin=dict(l=56, r=12, t=40, b=36), showlegend=False)
        fig.update_yaxes(**axis(title=dict(text="mm/day")))
        figs.append(fig)
    c = r["models"].get("C", {})
    if c.get("available"):
        v = c["gefs"]["values"]
        fig = go.Figure(go.Bar(x=["Days 1-7", "Days 8-16", "Days 17-30"], y=[v["gefs_p_d1_7_mm"], v["gefs_p_d8_16_mm"], 0],
                               marker_color=[P["rain"], rgba(P["rain"], 0.6), rgba(P["silt"], 0.15)],
                               text=[f"{v['gefs_p_d1_7_mm']:.0f} mm", f"{v['gefs_p_d8_16_mm']:.0f} mm", "not forecast"],
                               textposition="outside", hoverinfo="skip"))
        fig.add_annotation(xref="paper", yref="paper", x=0, y=1.12, showarrow=False, xanchor="left",
                           text=f"Forecast precipitation over the upstream area: GEFS {c['gefs']['cycle']} control run "
                                f"(published by {c['gefs']['published_utc_max']})", font=dict(size=11, color=P["silt"]))
        fig = finish(fig, False, 230, margin=dict(l=56, r=12, t=44, b=36), showlegend=False)
        fig.update_yaxes(**axis(title=dict(text="mm, basin mean")))
        figs.append(fig)
    return figs


def comparison_table(r, perf):
    """Paired comparisons only: each model against Model A (v3 refit) on the same rows."""
    rows = []
    dev = perf.get("dev", {}).get("all_folds", {}).get("paired", {})
    fin = perf.get("final", {}).get("paired", {})

    def pair(src, k):
        p = src.get(f"{k} vs A")
        if not p:
            return "n/a" if k != "A" else "reference"
        return f"{p[f'mae_{k}']:.2f} vs {p['mae_A']:.2f} ({p['n']:,} rows)"

    for k in ("A", "B", "B2", "C"):
        m = r["models"].get(k, {})
        rows.append({
            "Model": k, "Inputs": MODEL_INPUTS[k],
            "Available now": "yes" if m.get("available") else "no: " + "; ".join(m.get("reasons", []))[:140],
            "30-day change": ui.fmt_pp(m["change_pp"], digits=1) if m.get("available") else "",
            "90% range": f"{m['interval']['lower_change_pp']:+.1f} to {m['interval']['upper_change_pp']:+.1f} pp" if m.get("available") else "",
            "Trained on targets before": m.get("training_cutoff_exclusive", ""),
            "MAE vs Model A, development WY2008-2025": pair(dev, k) if dev else "not run",
            "MAE vs Model A, 2026 test": pair(fin, k) if fin else "not run",
        })
    return pd.DataFrame(rows)


def verdict_text(perf):
    h = perf.get("final", {}).get("hypotheses")
    if not h:
        return "The preregistered 2026 comparison has not been run on this deployment."
    names = {"H1": "C better than A", "H2": "B better than A", "H3": "B2 better than B"}
    parts = [f"{names.get(k, k)}: {v['verdict']}" for k, v in h.items()]
    sup = [k for k, v in h.items() if v["verdict"] == "supported"]
    tail = ("No model is shown to be more accurate than another; the differences above are within sampling noise."
            if not sup else "Only the supported comparisons above may be read as differences in accuracy.")
    return "Preregistered 2026 test (95% block-bootstrap intervals, 5, 9 and 13 weeks): " + "; ".join(parts) + ". " + tail


def render_interactive(b):
    from src.v3_service import APPROVED_STATIONS, PolicyBlocked, RateLimited, deployment_mode
    names = {s: b.name(s) for s in APPROVED_STATIONS} if b.stations is not None else {s: s for s in APPROVED_STATIONS}
    c1, c2, c3 = st.columns([3, 1.2, 1])
    with c1:
        cur = st.session_state.get(ui.STATION_KEY)
        sid = ui.segmented("Reservoir", list(APPROVED_STATIONS), cur if cur in APPROVED_STATIONS else "SHA",
                           key="v3_station", format_func=lambda s: names.get(s, s))
    with c2:
        force = st.checkbox("Check for newer data", key="v3_force",
                            help="Re-download sources even if a recent copy is cached (at most once per 10 minutes).")
    with c3:
        go_btn = st.button("Generate Forecast", type="primary", key="v3_go", use_container_width=True)
    key = f"v3_result_{sid}"
    if go_btn:
        with st.spinner("Retrieving CDEC observations, NASA POWER weather and the GEFS forecast, then running the models"):
            try:
                st.session_state[key] = service().forecast(sid, force_refresh=force)
                st.session_state.pop(f"{key}_err", None)
            except PolicyBlocked as exc:
                st.session_state[f"{key}_err"] = ("Not available on this public deployment", str(exc))
            except RateLimited as exc:
                st.session_state[f"{key}_err"] = ("Busy: upstream request limit reached", str(exc))
            except Exception as exc:  # noqa: BLE001  never show a forecast after an unexpected failure
                st.session_state[f"{key}_err"] = ("Forecast failed", f"{type(exc).__name__}: {exc}")
    err = st.session_state.get(f"{key}_err")
    if err:
        ui.callout(ui.esc(err[0]), ui.esc(err[1]), "caution")
        return
    r = st.session_state.get(key)
    if r is None:
        st.markdown(f'<div class="ws-empty"><h4>Select a reservoir and press Generate Forecast</h4><p>The server retrieves '
                    f'the latest eligible CDEC storage and snowpack, recent NASA POWER weather and the latest usable NOAA GEFS '
                    f'forecast, then runs Models A, B, B2 and C. Sources are shared between visitors and cached until newer '
                    f'data can exist. Deployment mode: <code>{ui.esc(deployment_mode())}</code>.</p></div>',
                    unsafe_allow_html=True)
        return
    issued = r["issued_utc"]
    st.caption(f"Last updated {issued} UTC. Sources: " + "; ".join(
        f"{k.upper()} {'ok' if v.get('ok') else 'FAILED'}{' (cached)' if v.get('cached') else ''}"
        for k, v in r.get("source_status", {}).items()))
    if r["status"] != "ok":
        items = "".join(f"<li>{ui.esc(x)}</li>" for x in r.get("reasons", []))
        st.markdown(f'<div class="ws-empty"><h4>Forecast unavailable</h4><p>No forecast is shown and no older one is '
                    f'substituted.</p><ul>{items}</ul></div>', unsafe_allow_html=True)
        return
    for w in r.get("warnings", []):
        ui.callout("Data freshness", ui.esc(w), "caution")
    avail = {k: m for k, m in r["models"].items() if m.get("available")}
    ob = r["observed"]
    cols = st.columns(1 + len(avail))
    cols[0].metric(f"Storage on {_d(ob['date'], False)}", ui.fmt_pct(ob["storage_pct"], 1))
    cols[0].caption(f"{ui.fmt_af(ob['storage_af'])}; target {_d(r['target_date'], False)}, "
                    f"{r['days_remaining']:.1f} days after issue")
    for c, (k, m) in zip(cols[1:], avail.items()):
        c.metric(f"Model {k}: projected", ui.fmt_pct(m["projected_pct"], 1), ui.fmt_pp(m["change_pp"], digits=1))
        c.caption(f"{m['change_af']:+,.0f} AF; 90%: {m['interval']['lower_projected_pct']:.1f}% to "
                  f"{m['interval']['upper_projected_pct']:.1f}%")
    if sid == "FOL":
        ui.callout("Interval caution: Folsom", "Nominal 90% ranges covered only about 75% of Folsom outcomes in the "
                   "2024-2025 backtest. Treat Folsom ranges as too narrow.", "caution")
    ui.show_chart(endpoint_figure(r), key=f"v3_chart_{sid}")
    st.caption("Each diamond is one model's estimate for the target date, with its own retrospectively calibrated 90% "
               "range. No model predicts the path between the origin and day 30, so none is drawn.")

    st.markdown("## Model comparison")
    ui.show_table(comparison_table(r, _perf()))
    perf = _perf()
    ui.callout("What the comparison shows", ui.esc(verdict_text(perf)), "model")
    st.caption("Mean absolute error in percentage points, each model against Model A (v3 refit, same features as the "
               "frozen model) on exactly the same rows, under docs/v3/PREREGISTRATION_V3.md. Development folds include "
               "2024-2025, which were seen earlier and are not an independent test. Extra inputs do not make a model "
               "better. The Model A forecast shown above is the unchanged frozen v2 model (trained on targets before "
               "2024); B, B2 and C are v3 models trained on targets before 2026.")
    cut = {k: m.get("information_cutoff") for k, m in avail.items()}
    if len({json.dumps(v, sort_keys=True) for v in cut.values()}) > 1:
        ui.callout("Different information cutoffs", "These models use different information. A uses storage to the origin "
                   "and snow to the day before. B and B2 also use weather ending 7 days before the origin. C also uses a "
                   "weather forecast issued at 00 UTC on the origin day. They are not equivalent as-of-origin forecasts.", "model")

    st.markdown("## Weather inputs")
    figs = weather_figures(r)
    for i, f in enumerate(figs):
        ui.show_chart(f, key=f"v3_wx_{sid}_{i}")
    if not figs:
        st.caption("No weather data available for this forecast.")

    st.markdown("## Data freshness")
    gs = r.get("source_status", {}).get("gefs", {})
    ui.facts([("Forecast issued", f"{ui.esc(issued)} UTC"),
              ("Latest storage observation", _d(r["storage_last_obs_date"])),
              ("Storage used (origin)", _d(r["origin_date"])),
              ("Latest snowpack index", _d(r.get("snow_last_obs_date"))),
              ("Snowpack used", _d(r["snow"]["swe_date"])),
              ("Weather window used (B, B2)", f"ends {_d(str((pd.Timestamp(r['origin_date']) - pd.Timedelta(days=7)).date()))}"),
              ("Weather forecast (C)", ui.esc(f"{gs.get('cycle', 'n/a')}, published {gs.get('published_utc_max') or 'n/a'}")),
              ("Target date", f"{_d(r['target_date'])} ({r['days_remaining']:.1f} days after issue)")])
    with st.expander("Full inputs and provenance"):
        for k, m in avail.items():
            st.markdown(f"**Model {k}** ({ui.esc(m['version'])}, artifact `{m['artifact_sha256'][:12]}`)")
            ui.show_table(pd.DataFrame({"input": list(m["inputs"]), "value": [round(v, 4) for v in m["inputs"].values()]}))
        st.json({"sources": r.get("sources"), "source_status": r.get("source_status")}, expanded=False)
    body = json.dumps(r, indent=1, default=str).encode()
    st.download_button("Download this forecast (JSON with provenance)", body,
                       file_name=f"watershedshift_{sid}_{issued.replace(':', '')}.json", mime="application/json",
                       key=f"v3_dl_{sid}")
    try:
        record_store().save(r)
    except FileExistsError:
        pass
    except Exception:  # noqa: BLE001  the forecast is shown; persistence failure is reported
        st.caption("This forecast could not be saved to the record store.")

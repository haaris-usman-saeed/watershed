"""Interactive part of the 30-Day Forecast page: Generate Forecast with the frozen v2 HGB only.

The site's forecasts come from one model: HGB[S,N,C,R] · frozen v2 artifact (the "HGB SWE no weather" model,
scikit-learn HistGradientBoostingRegressor trained on targets before 2024-01-01). It is the only model with a
preregistered, one-shot test result. The other study models are compared, retrospectively, on the Model playground
page; none of them makes forecasts here.

Calls src.v3_service.ForecastService in-process (one shared instance per server) with model_keys=("A",), so only
CDEC storage and snowpack are fetched: no NASA POWER, no GEFS. The service caches upstream data for all visitors,
coalesces identical requests, rate-limits upstream fetches, and refuses to show CDEC-derived values on a public
deployment until rights are cleared.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui_charts
import ui_components as ui
import ui_io
import ui_nav
from ui_theme import PALETTE as P, axis, finish, rgba

MAIN_KEY = "A"                                      # the service's key for the frozen v2 artifact
MAIN_NAME = "HGB (frozen v2)"
MAIN_SPEC = "HGB[S,N,C,R] · frozen v2 artifact"
MAIN_FACTS = [
    ("Model", "HGB[S,N,C,R] · frozen v2 artifact (the &ldquo;HGB SWE no weather&rdquo; model)"),
    ("Estimator", "scikit-learn HistGradientBoostingRegressor"),
    ("Inputs", "Storage now and its 7, 30 and 60-day changes; basin snowpack on the day before the origin, its 14-day "
               "change and its anomaly; season; reservoir. No weather and no weather forecast."),
    ("Hyperparameters", "Frozen: learning rate 0.03, 150 iterations, no depth limit, at least 30 samples per leaf, "
                        "L2 1.0, random state 42"),
    ("Training data", "Targets before 2024-01-01. This is the saved artifact, not a refit; it is never retrained."),
    ("90% range", "Symmetric seasonal offsets from the model's own historical errors; future coverage is not "
                  "established"),
]


@st.cache_resource(show_spinner=False)
def service():
    from src.v3_artifacts import ensure_artifacts
    from src.v3_service import ForecastService
    # The service installs hash-pinned artifacts when configured, retries a failed installation (at most once a
    # minute) instead of keeping the failure until a restart, and reports the installation status to visitors.
    # Only the frozen v2 HGB runs, so NASA POWER and GEFS are never fetched.
    return ForecastService(artifact_installer=ensure_artifacts, model_keys=(MAIN_KEY,))


@st.cache_resource(show_spinner=False)
def record_store():
    from src.v3_service import RecordStore
    return RecordStore()


def track_record(root=None):
    """The main model's frozen test results, read from tracked aggregate files (never recomputed here)."""
    root = root or ui_io.REPO_ROOT
    rows = []
    p2 = root / "results/v2/final/metrics.json"
    if p2.exists():
        m = json.loads(p2.read_text())
        rows.append({"Test window": "2024-2025 (v2 test, preregistered, run once)", "Rows": f"{m['test_count']:,}",
                     "MAE, this model": f"{m['ML']:.3f} pp",
                     "MAE, smoothed seasonal baseline": f"{m['Smoothed DOY seasonal']:.3f} pp",
                     "90% range coverage": f"{m['interval_90_coverage']:.3f}"})
    p3 = root / "results/v3/final/metrics.json"
    if p3.exists():
        m = json.loads(p3.read_text())
        a, doy = m["mae"].get("pred_A_frozen"), m["mae"].get("base_doy")
        if a and doy:
            rows.append({"Test window": "2026 (v3 test, preregistered, run once)", "Rows": f"{a['n']:,}",
                         "MAE, this model": f"{a['mae']:.3f} pp",
                         "MAE, smoothed seasonal baseline": f"{doy['mae']:.3f} pp", "90% range coverage": "n/a"})
    return pd.DataFrame(rows)


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
    m = r["models"].get(MAIN_KEY, {})
    if m.get("available"):
        iv = m["interval"]
        ui_charts.add_trajectory_guide(fig, o, t, r["observed"]["storage_pct"], m["projected_pct"],
                                       iv["lower_projected_pct"], iv["upper_projected_pct"], P["model"])
        fig.add_trace(go.Scatter(
            x=[t], y=[m["projected_pct"]], mode="markers", name=f"{MAIN_NAME}: day-30 estimate and 90% range",
            marker=dict(size=11, symbol="diamond", color=P["model"], line=dict(width=1, color=P["ink"])),
            error_y=dict(type="data", symmetric=False, array=[iv["upper_projected_pct"] - m["projected_pct"]],
                         arrayminus=[m["projected_pct"] - iv["lower_projected_pct"]], color=P["model"], thickness=2, width=6),
            hovertemplate=f"{MAIN_NAME}<br>%{{x|%b %d, %Y}}<br>%{{y:.1f}}% (90%: {iv['lower_projected_pct']:.1f} to "
                          f"{iv['upper_projected_pct']:.1f})<extra></extra>"))
    fig.add_shape(type="line", x0=o, x1=o, y0=0, y1=1, xref="x", yref="paper", line=dict(color=P["model"], width=1, dash="dot"))
    fig.add_annotation(x=o, y=1, xref="x", yref="paper", text="Origin", showarrow=False, xanchor="left", yanchor="top",
                       xshift=4, font=dict(size=11, color=P["model"]))
    fig = finish(fig, synthetic, height, hovermode="closest", margin=dict(l=70, r=18, t=70, b=40))
    fig.update_yaxes(**axis(title=dict(text="% of capacity"), ticksuffix="%"))
    fig.update_xaxes(**axis(), range=[o - pd.Timedelta(days=60), t + pd.Timedelta(days=7)])
    return fig


def detail_table(r):
    """Every field of the forecast, or why the model could not run."""
    ob, m = r["observed"], r["models"].get(MAIN_KEY, {})
    row = {"Model": MAIN_SPEC, "Status": "ok" if m.get("available") else "unavailable: " + "; ".join(m.get("reasons", [])),
           "Observed storage": f"{ob['storage_pct']:.1f}% ({ob['storage_af']:,.0f} AF)",
           "30-day change": "", "Projected": "", "90% range": "",
           "Origin": r["origin_date"], "Issued (UTC)": r["issued_utc"], "Target": r["target_date"],
           "Days remaining": f"{r['days_remaining']:.1f}", "Storage obs. used": r["origin_date"],
           "Snow obs. used": r["snow"]["swe_date"], "Trained on targets before": m.get("training_cutoff_exclusive", "")}
    if m.get("available"):
        iv = m["interval"]
        row.update({"30-day change": f"{m['change_pp']:+.2f} pp ({m['change_af']:+,.0f} AF)",
                    "Projected": f"{m['projected_pct']:.1f}% ({m['projected_af']:,.0f} AF)",
                    "90% range": f"{iv['lower_change_pp']:+.1f} to {iv['upper_change_pp']:+.1f} pp; "
                                 f"{iv['lower_projected_pct']:.1f}% to {iv['upper_projected_pct']:.1f}%"})
    return pd.DataFrame([row])


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
        with st.spinner("Retrieving the latest CDEC storage and snowpack, then running the model"):
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
                    f'the latest eligible CDEC storage and snowpack and runs {ui.esc(MAIN_SPEC)}. Sources are shared '
                    f'between visitors and cached until newer data can exist. Deployment mode: '
                    f'<code>{ui.esc(deployment_mode())}</code>.</p></div>', unsafe_allow_html=True)
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
    m = r["models"].get(MAIN_KEY, {})
    ob = r["observed"]
    cols = st.columns(2)
    cols[0].metric(f"Storage on {_d(ob['date'], False)}", ui.fmt_pct(ob["storage_pct"], 1))
    cols[0].caption(f"{ui.fmt_af(ob['storage_af'])}; target {_d(r['target_date'], False)}, "
                    f"{r['days_remaining']:.1f} days after issue")
    if m.get("available"):
        cols[1].metric(f"{MAIN_NAME}: projected on {_d(r['target_date'], False)}", ui.fmt_pct(m["projected_pct"], 1),
                       ui.fmt_pp(m["change_pp"], digits=1))
        cols[1].caption(f"{m['change_af']:+,.0f} AF; 90%: {m['interval']['lower_projected_pct']:.1f}% to "
                        f"{m['interval']['upper_projected_pct']:.1f}%")
    else:
        ui.callout("The model could not run", ui.esc(" ".join(m.get("reasons", [])) or "No reason was recorded."), "caution")
    if sid == "FOL":
        ui.callout("Interval caution: Folsom", "Nominal 90% ranges covered only about 75% of Folsom outcomes in the "
                   "2024-2025 backtest. Treat Folsom ranges as too narrow.", "caution")
    ui.show_chart(endpoint_figure(r), key=f"v3_chart_{sid}")
    st.caption("The diamond is the model's estimate for the target date, with its retrospectively calibrated 90% "
               "range.")
    ui.callout("The dashed line is a guide, not a forecast path", ui.esc(ui_charts.TRAJECTORY_NOTE), "caution")

    st.markdown("## Forecast details")
    ui.show_table(detail_table(r))
    st.caption("Origin = the Monday whose end-of-day storage the forecast starts from. Issued = when this forecast was "
               "computed. Days remaining are counted from the issue time to the end of the target date (UTC). The 90% "
               "range is calibrated on the model's own historical errors; its future coverage is not established.")

    st.markdown("## About this model")
    ui.facts(MAIN_FACTS)
    tr = track_record()
    if not tr.empty:
        ui.show_table(tr)
        st.caption("Mean absolute error of the 30-day storage change, in percentage points of capacity, on test windows "
                   "that were fixed in advance and scored once (docs/preregistration.md, docs/v3/PREREGISTRATION_V3.md). "
                   "Past accuracy does not establish future accuracy; no forecast has yet been scored prospectively.")
    ui.callout("Other models", "The other nine study models (persistence, 30-day trend, monthly and day-of-year seasonal "
               "baselines, Ridge, LSTM, GRU, TCN and the water-balance-structured LSTM) do not make forecasts on this "
               "site. Their past test results are on the Model playground page, for research comparison only.", "model")
    if st.button("Open the Model playground", key="v3_to_playground"):
        ui_nav.go("playground")

    st.markdown("## Data freshness")
    ui.facts([("Forecast issued", f"{ui.esc(issued)} UTC"),
              ("Latest storage observation", _d(r["storage_last_obs_date"])),
              ("Storage used (origin)", _d(r["origin_date"])),
              ("Latest snowpack index", _d(r.get("snow_last_obs_date"))),
              ("Snowpack used", _d(r["snow"]["swe_date"])),
              ("Target date", f"{_d(r['target_date'])} ({r['days_remaining']:.1f} days after issue)")])
    with st.expander("Full inputs and provenance"):
        if m.get("available"):
            st.markdown(f"**{ui.esc(MAIN_SPEC)}** ({ui.esc(m['version'])}, artifact `{m['artifact_sha256'][:12]}`)")
            ui.show_table(pd.DataFrame({"input": list(m["inputs"]), "value": [round(v, 4) for v in m["inputs"].values()]}))
        st.json({"sources": r.get("sources"), "source_status": r.get("source_status")}, expanded=False)
    body = json.dumps(r, indent=1, default=str).encode()
    st.download_button("Download this forecast (JSON with provenance)", body,
                       file_name=f"watershedshift_{sid}_{issued.replace(':', '')}.json", mime="application/json",
                       key=f"v3_dl_{sid}")
    saved = st.session_state.get(f"{key}_saved")
    if saved is None or saved[0] != issued:          # save once per forecast, not on every rerun
        store = record_store()
        try:
            note = f"Saved as record {store.save(r)['id']} ({store.describe()})."
        except FileExistsError:
            note = f"Already saved ({store.describe()})."
        except Exception as exc:  # noqa: BLE001  the forecast is shown; persistence failure is reported
            from src.v3_service import RecordStoreError
            note = ("This forecast could not be saved to the record store "
                    f"({exc if isinstance(exc, RecordStoreError) else type(exc).__name__}).")
        saved = (issued, note)
        st.session_state[f"{key}_saved"] = saved
    st.caption(saved[1])

"""E. 30-Day Forecast: research storage forecasts from the latest eligible observations.

Read-only. This page never fits a model, never loads model weights and never downloads data.
It displays a snapshot written by the trusted local pipeline (python -m src.forecast_live)
after validating it with src.forecast_schema. Lookup order for the snapshot:
  1. WATERSHED_FORECAST_FILE (environment variable)
  2. forecasts/latest.json in the repository (private, git-ignored)
  3. forecast_snapshot.json in the UI data directory (the synthetic fixture in app/sample_data)
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui_components as ui
import ui_io
from ui_theme import PALETTE as P, axis, finish, rgba
from src.forecast_schema import expired, next_origin_after, validate

FILE_ENV = "WATERSHED_FORECAST_FILE"
UNITS = {"pct": "% of capacity", "af": "Acre-feet"}
PICK_KEY = "fc_station"
CHART_DAYS_BEFORE = 90     # observed window shown before the origin (the snapshot carries 180 days)


# ---------------------------------------------------------------- snapshot loading

def resolve_snapshot(b):
    env = os.environ.get(FILE_ENV)
    if env:
        return Path(env).expanduser(), "env"
    repo = ui_io.REPO_ROOT / "forecasts" / "latest.json"
    if repo.exists():
        return repo, "repo"
    fixture = Path(b.data_dir) / "forecast_snapshot.json"
    if fixture.exists():
        return fixture, "data_dir"
    return None, None


@st.cache_data(show_spinner="Reading forecast snapshot")
def _read(path_str, mtime, size):
    return json.loads(Path(path_str).read_text())


def load_snapshot(path):
    p = Path(path)
    stat = p.stat()
    return _read(str(p), stat.st_mtime, stat.st_size)


# ---------------------------------------------------------------- formatting

def _d(s, long=True):
    return ui.fmt_date(pd.Timestamp(s), long) if s else "n/a"


def _signed_af(v):
    return f"{v:+,.0f} AF" if v is not None and np.isfinite(v) else "n/a"


def _empty(title, body_html):
    st.markdown(f'<div class="ws-empty"><h4>{ui.esc(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- chart

def forecast_figure(r, units, synthetic, height=420):
    pct = units == "pct"
    h = pd.DataFrame(r.get("recent_history", []))
    fig = go.Figure()
    f, ob = r["forecast"], r["observed"]
    o, t = pd.Timestamp(r["origin_date"]), pd.Timestamp(r["target_date"])
    y0 = ob["storage_pct"] if pct else ob["storage_af"]
    lo = f["lower_projected_pct"] if pct else f["lower_projected_af"]
    hi = f["upper_projected_pct"] if pct else f["upper_projected_af"]
    mid = f["projected_pct"] if pct else f["projected_af"]
    fig.add_trace(go.Scatter(x=[o, t, t, o], y=[y0, hi, lo, y0], fill="toself", mode="lines", line=dict(width=0),
                             fillcolor=rgba(P["model"], 0.16), hoverinfo="skip",
                             name="90% interval for the day-30 value (retrospectively calibrated)"))
    col = "storage_pct" if pct else "storage_af"
    if not h.empty:
        h["date"] = pd.to_datetime(h["date"])
        h[col] = pd.to_numeric(h[col], errors="coerce")
        used, after = h[h["used_by_forecast"]], h[~h["used_by_forecast"]]
        fig.add_trace(go.Scatter(x=used["date"], y=used[col], mode="lines", name="Observed storage (CDEC), used by the forecast",
                                 line=dict(color=P["water"], width=2), connectgaps=False,
                                 hovertemplate="%{x|%b %d, %Y}<br>%{y:,.1f}" + ("%" if pct else " AF") + "<extra>observed</extra>"))
        if not after.empty:
            joined = pd.concat([used.tail(1), after])
            fig.add_trace(go.Scatter(x=joined["date"], y=joined[col], mode="lines+markers",
                                     name="Observed after the origin (not used by the forecast)",
                                     line=dict(color=P["bone"], width=1.4, dash="dot"), marker=dict(size=4),
                                     connectgaps=False,
                                     hovertemplate="%{x|%b %d, %Y}<br>%{y:,.1f}" + ("%" if pct else " AF") + "<extra>observed later</extra>"))
    fig.add_trace(go.Scatter(x=[o, t], y=[y0, mid], mode="lines+markers", name="Model forecast (dashed, day 30 only)",
                             line=dict(color=P["model"], width=2.4, dash="dash"),
                             marker=dict(size=[7, 10], symbol=["circle", "diamond"], color=P["model"]),
                             hovertemplate="%{x|%b %d, %Y}<br>%{y:,.1f}" + ("%" if pct else " AF") + "<extra>forecast</extra>"))
    fig.add_trace(go.Scatter(x=[t], y=[mid], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=1, color=P["model"]),
                             error_y=dict(type="data", symmetric=False, array=[hi - mid], arrayminus=[mid - lo],
                                          color=P["model"], thickness=1.6, width=8)))
    cap = 100.0 if pct else r["capacity_af"]
    top = max([hi, y0] + ([float(np.nanmax(h[col]))] if not h.empty and h[col].notna().any() else []))
    if top >= 0.8 * cap:   # draw the capacity line only when it is near the data, so the forecast stays readable
        fig.add_hline(y=cap, line=dict(color=rgba(P["bone"], 0.45), width=1, dash="dash"))
        fig.add_annotation(xref="paper", x=0, y=cap, yref="y", text="Documented capacity", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=P["silt"]))
    for x, label, color, ypos in ((o, "Forecast origin", P["model"], 1.0),
                                  (pd.Timestamp(r["issue_date"]), "Issued", P["silt"], 0.93)):
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper", line=dict(color=color, width=1, dash="dot"))
        fig.add_annotation(x=x, y=ypos, xref="x", yref="paper", text=label, showarrow=False, xanchor="left",
                           yanchor="top", xshift=4, font=dict(size=11, color=color))
    fig = finish(fig, synthetic, height, hovermode="closest", margin=dict(l=84, r=18, t=64, b=40))
    fig.update_xaxes(range=[o - pd.Timedelta(days=CHART_DAYS_BEFORE), t + pd.Timedelta(days=4)])
    fig.update_yaxes(**axis(title=dict(text="% of capacity" if pct else "Storage, acre-feet"),
                            ticksuffix="%" if pct else "", tickformat=".0f" if pct else ",.0f"))
    fig.update_xaxes(**axis(hoverformat="%b %d, %Y"))
    return fig


def swe_figure(r, synthetic, height=230):
    h = pd.DataFrame(r.get("recent_history", []))
    fig = go.Figure()
    if not h.empty:
        h["date"] = pd.to_datetime(h["date"])
        fig.add_trace(go.Scatter(x=h["date"], y=pd.to_numeric(h["swe_basin_in"], errors="coerce"), mode="lines",
                                 name="Basin SWE index", connectgaps=False, line=dict(color=P["water"], width=2),
                                 fill="tozeroy", fillcolor=rgba(P["water"], 0.12),
                                 hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f} in<extra></extra>"))
    s = r["snow"]
    fig.add_trace(go.Scatter(x=[pd.Timestamp(s["swe_date"])], y=[s["swe_basin_in"]], mode="markers",
                             name="Value used (origin minus 1 day)", marker=dict(color=P["model"], size=9, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[pd.Timestamp(s["swe_date"])], y=[s["climatology_swe_in"]], mode="markers",
                             name="Pre-2024 average for that day of year", marker=dict(color=P["bone"], size=8, symbol="line-ew-open",
                                                                                     line=dict(width=2))))
    fig = finish(fig, synthetic, height, margin=dict(l=64, r=20, t=40, b=40))
    fig.update_yaxes(**axis(title=dict(text="SWE index (inches)"), rangemode="tozero"))
    o = pd.Timestamp(r["origin_date"])
    fig.update_xaxes(range=[o - pd.Timedelta(days=CHART_DAYS_BEFORE), pd.Timestamp(r["issue_date"]) + pd.Timedelta(days=2)])
    return fig


# ---------------------------------------------------------------- sections

def interpretation(r):
    f = r["forecast"]
    pp = f["change_pp"]
    verb = "rise" if pp > 0.05 else "fall" if pp < -0.05 else "change little"
    amount = f" by {abs(pp):.1f} percentage points ({_signed_af(f['change_af'])})" if verb != "change little" else \
        f" ({pp:+.1f} percentage points)"
    return (f"The model estimates that storage at {ui.esc(r['reservoir_name'])} will {verb}{amount} over the 30 days "
            f"ending {_d(r['target_date'])}, from {r['observed']['storage_pct']:.1f}% to {f['projected_pct']:.1f}% of "
            f"documented capacity. The retrospectively calibrated 90% range for the change is "
            f"{f['lower_change_pp']:+.1f} to {f['upper_change_pp']:+.1f} pp.")



def who():
    left, right = st.columns(2, gap="large")
    with left:
        ui.callout("What the model takes into account",
                   "How full the reservoir is now and how fast it has been filling or emptying over the last 7, 30 "
                   "and 60 days; the mountain snowpack upstream and how it compares with normal; and the time of year. "
                   "From 2001 to 2023 it learned how storage typically moves at each reservoir in each season, which "
                   "includes the usual timing of winter rain, spring snowmelt and summer releases.", "good")
    with right:
        ui.callout("What it cannot see",
                   "Storms that have not happened yet (it uses no precipitation forecast), planned releases, "
                   "flood-control decisions, water allocations or demands. A large winter storm can therefore move "
                   "storage far outside the forecast, which is why winter ranges are the widest (about plus or minus "
                   "10 points).", "caution")
    ui.callout("Who it is for, and who it is not for",
               "It is <strong>not</strong> a tool for reservoir operators. They rely on official forecasts that include "
               "precipitation and on their own release plans, for example DWR's Bulletin 120 water supply forecasts "
               "and the National Weather Service California Nevada River Forecast Center."
               "</p><p style='margin-top:0.5rem'>Where it is useful:</p><ul style='margin:0.3rem 0 0;color:inherit'>"
               "<li><strong>Research:</strong> it shows that a transparent model built only from free public data "
               "predicts 30-day storage change with about 21% less error than the best seasonal-average rule "
               "(2.33 against 2.97 points in the 2024 to 2025 test), with a published, reproducible method.</li>"
               "<li><strong>A benchmark:</strong> a clear standard that any better model, such as one adding "
               "precipitation forecasts, has to beat.</li>"
               "<li><strong>Public understanding:</strong> students, journalists and local communities get a quick, "
               "honest view of where a reservoir is heading in the next month, with its uncertainty shown.</li></ul><p>",
               "model")

def unavailable(r, snap):
    reasons = "".join(f"<li>{ui.esc(x)}</li>" for x in r.get("reasons", []))
    tried = "".join(f"<li>{ui.esc(o['origin_date'])}: {ui.esc('; '.join(o['missing']) or 'eligible')}</li>"
                    for o in r.get("origins_considered", []))
    body = (f"<p>No forecast is shown for {ui.esc(r['reservoir_name'])}. An older forecast is never substituted.</p>"
            f"<ul>{reasons}</ul>" + (f"<p>Monday origins checked:</p><ul>{tried}</ul>" if tried else "") +
            f"<p>Latest storage observation: {ui.esc(_d(r.get('storage_last_obs_date')))}. "
            f"Latest snowpack index: {ui.esc(_d(r.get('snow_last_obs_date')))}.</p>")
    _empty("Forecast unavailable", body)


def freshness(r, snap, path, source):
    m = snap.get("model", {})
    src = snap.get("sources", {})
    files = src.get("files", [])
    retrieved = max((x.get("retrieved_utc", "") for x in files), default="")
    status = ("refresh failed: " + src["refresh_error"]) if src.get("refresh_error") else \
        (f"{len(files)} CDEC responses, all matching their logged SHA-256" if src.get("all_files_match_logged_sha256")
         else "no verified agency files (synthetic fixture)" if snap.get("data_mode") != "real" else "file hashes not verified")
    rev = src.get("revision_check_vs_archive", {})
    rev_txt = "not available"
    if rev.get("available"):
        ch = sum(v["days_changed"] for v in rev["by_station"].values())
        n = sum(v["overlap_days"] for v in rev["by_station"].values())
        rev_txt = f"{ch} of {n} station-days changed vs the archived {rev['period'][0][:4]} data"
    sc = m.get("source_commit") or {}
    ui.facts([
        ("Storage observation used", f"{_d(r['observed']['date'])} (end of day, CDEC sensor 15)"),
        ("Latest storage available", _d(r.get("storage_last_obs_date"))),
        ("Snowpack value used", f"{_d(r['snow']['swe_date'])} (origin minus 1 day, CDEC sensor 82)"),
        ("Latest snowpack index available", _d(r.get("snow_last_obs_date"))),
        ("Forecast issued", f"{_d(snap['issue_date'])} ({ui.esc(snap.get('issued_utc', ''))})"),
        ("Forecast origin and target", f"{_d(r['origin_date'])} to {_d(r['target_date'])} (exactly 30 days)"),
        ("Next possible origin", f"Monday {_d(str(next_origin_after(snap['issue_date'])))}, once CDEC reports it"),
        ("Data retrieved", ui.esc(retrieved or "n/a")),
        ("Source status", ui.esc(status)),
        ("Revision check", ui.esc(rev_txt)),
        ("Model", f"{ui.esc(m.get('model_name', ''))} ({ui.esc(m.get('model_version', ''))})"),
        ("Training cutoff", f"targets before {ui.esc(m.get('training_cutoff_exclusive', 'n/a'))}"
                            + (f", {m['training_rows']:,} weekly rows" if m.get("training_rows") else "")),
        ("Model artifact SHA-256", f"<code>{ui.esc(str(m.get('artifact_sha256', 'n/a'))[:16])}</code>"),
        ("Model built from commit", f"<code>{ui.esc(str(sc.get('commit') or 'n/a')[:10])}</code>"),
        ("Forecast record", f"<code>{ui.esc(snap.get('forecast_id', ''))}</code>"),
        ("Snapshot file", f"<code>{ui.esc(str(path))}</code> ({'WATERSHED_FORECAST_FILE' if source == 'env' else source})"),
    ])


def render():
    b = ui.load_bundle()
    ui.page_intro("30-Day Forecast",
                  "Experimental research forecasts of reservoir storage 30 days ahead, made by the frozen v2 model from "
                  "the latest eligible CDEC observations. Storage, not water level: +5 pp means storage rises by 5% of "
                  "documented capacity.")
    path, source = resolve_snapshot(b)
    if path is None or not path.exists():
        _empty("No forecast snapshot on this deployment",
               "<p>Forecasts are produced locally by the trusted pipeline and are not published, because the CDEC "
               "redistribution terms are unresolved. To generate one on an authorised machine:</p>"
               "<p><code>python -m src.forecast_model --build</code><br><code>python -m src.forecast_live</code></p>"
               f"<p>The page then reads <code>forecasts/latest.json</code>, or the file named by "
               f"<code>{FILE_ENV}</code>.</p>" + (f"<p>Looked for <code>{ui.esc(str(path))}</code>.</p>" if path else ""))
        return
    try:
        snap = load_snapshot(path)
    except (OSError, ValueError) as exc:
        _empty("Forecast snapshot could not be read", f"<p><code>{ui.esc(str(path))}</code>: {ui.esc(str(exc))}</p>")
        return
    today = datetime.now(timezone.utc).date()
    errs, warns = validate(snap, today=today)
    synthetic = snap.get("data_mode") != "real"
    if not synthetic and expired(snap, today):
        _empty("Forecast expired",
               f"<p>The newest snapshot was issued {ui.esc(_d(snap.get('issue_date')))}; every target date has passed. "
               "Expired forecasts are not shown as current conditions. Run <code>python -m src.forecast_live</code> "
               "to issue a new one.</p>")
        return
    if errs:
        items = "".join(f"<li>{ui.esc(e)}</li>" for e in errs[:12])
        _empty("Forecast snapshot failed validation", f"<p>Nothing from it is displayed.</p><ul>{items}</ul>")
        return
    if synthetic:
        st.markdown('<div class="ws-banner" role="alert"><strong>Synthetic forecast fixture. Not a forecast.</strong>'
                    '<span>Invented reservoirs, invented observations and a placeholder formula in place of the v2 '
                    'model. Shown only to demonstrate the page.</span></div>', unsafe_allow_html=True)
    for w in [w for w in warns if "issued" in w and "ago" in w]:
        ui.callout("Stale snapshot", ui.esc(w), "caution")

    recs = {r["station_id"]: r for r in snap["stations"]}
    ids = list(recs)
    cur = ui.current_station(b) if b.stations is not None else st.session_state.get(ui.STATION_KEY)
    default = cur if cur in ids else ids[0]
    c1, c2 = st.columns([3, 1])
    with c1:
        sid = ui.segmented("Reservoir", ids, default, key=PICK_KEY, format_func=lambda s: recs[s]["reservoir_name"])
        if sid not in recs:
            sid = default
    with c2:
        units = ui.segmented("Units", ["pct", "af"], "pct", key="fc_units", format_func=UNITS.get)
    if sid in (b.station_ids or []):
        st.session_state[ui.STATION_KEY] = sid
    r = recs[sid]
    if r["status"] != "ok":
        unavailable(r, snap)
        return

    f, ob = r["forecast"], r["observed"]
    m = st.columns(4)
    m[0].metric(f"Storage on {_d(ob['date'], False)}", ui.fmt_pct(ob["storage_pct"], 1),
                help=f"{ui.fmt_af(ob['storage_af'])} of {ui.fmt_af(r['capacity_af'])} documented capacity")
    m[0].caption(ui.fmt_af(ob["storage_af"]))
    m[1].metric("Expected 30-day change", ui.fmt_pp(f["change_pp"], digits=1),
                help="Percentage points of documented capacity, not feet of water level")
    m[1].caption(f"{_signed_af(f['change_af'])}; 90%: {f['lower_change_pp']:+.1f} to {f['upper_change_pp']:+.1f} pp")
    m[2].metric("Projected storage", ui.fmt_pct(f["projected_pct"], 1))
    m[2].caption(f"{ui.fmt_af(f['projected_af'])}; 90%: {f['lower_projected_pct']:.1f}% to {f['upper_projected_pct']:.1f}%")
    m[3].metric("Target date", _d(r["target_date"], False))
    m[3].caption(f"Origin {_d(r['origin_date'], False)} + 30 days")

    ui.callout("Forecast interpretation", interpretation(r), "model")
    for w in r.get("warnings", []):
        ui.callout("Interval caution" if w.startswith("Interval caution") else
                   "Implausible projection" if "outside 0-100%" in w else "Data freshness", ui.esc(w), "caution")

    ui.show_chart(forecast_figure(r, units, synthetic), key=f"fc_chart_{sid}")
    st.caption("Solid: observed storage up to the origin, the only storage the model saw. Dotted: storage reported after "
               "the origin, shown for context and not used. Dashed violet: the forecast. The model predicts only the "
               "day-30 value; the dashed line and shaded wedge join the origin to that value and say nothing about the "
               "path in between. Percent-full uses the capacity in force on the origin date.")

    st.markdown("## Snowpack inputs")
    s = r["snow"]
    k = st.columns(3)
    k[0].metric(f"Basin SWE index, {_d(s['swe_date'], False)}", f"{s['swe_basin_in']:.2f} in")
    k[1].metric("14-day change", f"{s['swe_change_14d_in']:+.2f} in")
    k[2].metric("Anomaly vs pre-2024 average", f"{s['swe_anom_in']:+.2f} in")
    ui.show_chart(swe_figure(r, synthetic), key=f"fc_swe_{sid}")
    rep = (f"{s['n_core_reporting']} of {s['n_core']} core pillows reported on {_d(s['swe_date'])}"
           if s.get("n_core") else "")
    revised = (f"; {s['share_revised_flag']:.0%} of those values already carry CDEC's revised flag"
               if s.get("share_revised_flag") is not None else "")
    st.caption("Three of the model's 15 inputs come from snowpack: the basin snow-water-equivalent index on the day "
               "before the origin (mean of the basin's core CDEC snow pillows), its change over the previous 14 days, "
               "and its difference from the smoothed day-of-year average of the pre-2024 training years. "
               f"{rep}{revised}. Recent sensor 82 values may still be revised; the model was trained mostly on revised "
               "values. Values after the marked day were not available to the model. Snowpack feeds storage through melt, "
               "mainly in spring.")

    st.markdown("## Data freshness and provenance")
    freshness(r, snap, path, source)
    with st.expander("All 15 model inputs for this forecast"):
        ui.show_table(pd.DataFrame({"input": list(r["features"]), "value": [round(v, 4) for v in r["features"].values()]}))

    st.markdown("## Who this forecast helps")
    who()

    st.markdown("## What this forecast is and is not")
    iv = r["interval"]
    ui.callout("Experimental research forecast, not operational advice",
               "This is a research output from a historically evaluated model. It is not water-management advice, "
               "a water-supply outlook, or a flood or dam-safety product, and must not inform release decisions. "
               "The model has not been prospectively validated. It knows nothing about planned releases, operating "
               "rules, downstream demands or precipitation forecasts, only recent storage, snowpack and season. "
               f"The 90% range uses the frozen {ui.esc(iv['method'])} method: offsets calibrated on historical "
               f"out-of-fold errors for {ui.esc(iv['season'])} origins. Its coverage of future outcomes is not "
               f"established; in the 2024-2025 backtest it covered {iv['historical_coverage']:.0%} of outcomes at "
               f"{ui.esc(r['reservoir_name'])}, and only about 75% at Folsom.", "caution")
    st.caption("These forecast records are new predictions. They are not held-out test predictions and are not part "
               "of the frozen v2 evaluation shown on the Backtesting page.")

    rec_bytes = json.dumps({k2: v for k2, v in snap.items() if k2 != "stations"} | {"stations": [r]}, indent=2).encode()
    st.download_button("Download this forecast record (JSON, with provenance and assumptions)", rec_bytes,
                       file_name=f"{snap['forecast_id']}_{sid}.json", mime="application/json", key="fc_download")
    if not synthetic:
        st.caption("The record contains CDEC-derived values. Keep it private while CDEC redistribution terms are unresolved.")

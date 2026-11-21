"""B. Reservoir explorer: daily storage as recorded, with weather context."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import ui_charts as charts
import ui_components as ui
import ui_contract as C
import ui_io

UNITS = {"af": "Acre-feet", "pct": "% of capacity"}
WINDOWS = {"all": "Full record", "3y": "Last 3 years", "1y": "Last 12 months", "test": "Test period"}


def render():
    b = ui.load_bundle()
    if not ui.require(b, C.STATIONS_FILE, C.HISTORY_FILE, purpose="The reservoir explorer"):
        return
    ui.page_intro("Reservoir explorer",
                  "Historical observed storage with CDEC snowpack and NASA weather context. The selected v2 model uses "
                  "snowpack, not NASA weather. Missing storage is never interpolated.")
    sid = ui.station_picker(b, "ex")
    cap = b.capacity(sid)
    series = ui_io.daily_series(b.history, sid, cap)
    if series.empty or series["storage_af"].notna().sum() == 0:
        st.markdown(f'<div class="ws-empty"><h4>No storage values for {ui.esc(b.name(sid))}</h4>'
                    f'<p>history_daily.csv has no non-blank storage_af rows for station {ui.esc(sid)}.</p></div>',
                    unsafe_allow_html=True)
        return

    test_range = None
    if b.preds is not None:
        p = b.preds[b.preds["station_id"] == sid]
        if not p.empty:
            test_range = (p["date"].min(), p["target_date"].max())

    c1, c2 = st.columns(2)
    with c1:
        unit_opts = ["af", "pct"] if np.isfinite(cap) and cap > 0 else ["af"]
        units = ui.segmented("Units", unit_opts, "af", key="ex_units", format_func=UNITS.get)
    with c2:
        win_opts = ["all", "3y", "1y"] + (["test"] if test_range else [])
        win = ui.segmented("Window", win_opts, "all", key="ex_window", format_func=WINDOWS.get)

    end = series["date"].max()
    start = {"all": series["date"].min(), "3y": end - pd.DateOffset(years=3), "1y": end - pd.DateOffset(years=1),
             "test": test_range[0] if test_range else series["date"].min()}[win]
    if win == "test" and test_range:
        end = min(end, test_range[1])
    w = series[(series["date"] >= start) & (series["date"] <= end)]
    obs = w.dropna(subset=["storage_af"])

    col = "storage_pct" if units == "pct" else "storage_af"
    fmt = (lambda v: ui.fmt_pct(v, 1)) if units == "pct" else ui.fmt_af
    m = st.columns(4)
    last = obs.iloc[-1]
    m[0].metric(f"Last value in window ({ui.fmt_date(last['date'], True)})", fmt(last[col]))
    lo, hi = obs.loc[obs[col].idxmin()], obs.loc[obs[col].idxmax()]
    m[1].metric("Window minimum", fmt(lo[col]), help=f"On {ui.fmt_date(lo['date'], True)}")
    m[2].metric("Window maximum", fmt(hi[col]), help=f"On {ui.fmt_date(hi['date'], True)}")
    m[3].metric("Days with a value", f"{len(obs):,} of {len(w):,}",
                help="Calendar days in the window with a recorded storage value.")

    ui.show_chart(charts.storage_figure(w, cap, units, b.synthetic, test_range=test_range), key=f"ex_storage_{sid}")
    missing = len(w) - len(obs)
    note = (f"{missing:,} days in this window have no storage value and appear as breaks in the line. "
            if missing else "Every day in this window has a storage value. ")
    st.caption(note + f"Current documented capacity: {ui.fmt_af(cap)}. Historical percent-full values use "
               "the capacity in force on each date, including Oroville before July 2, 2024.")
    dup_warn = [x for x in b.reports[C.HISTORY_FILE].warnings if "duplicate" in x]
    if dup_warn:
        ui.callout("Duplicate dates in history_daily.csv", ui.esc("; ".join(dup_warn)), "caution")

    st.markdown("## Weather context")
    env = charts.env_figure(w, b.synthetic, test_range=test_range)
    if env is None:
        st.markdown('<div class="ws-empty"><h4>No weather columns</h4><p>history_daily.csv has no usable '
                    'precipitation, temperature or humidity values for this reservoir.</p></div>',
                    unsafe_allow_html=True)
    else:
        ui.show_chart(env, key=f"ex_env_{sid}")
        lag = ui.weather_lag(b)
        lag_note = (f" Secondary weather-model features end at least {lag:.0f} days before each origin."
                    if lag is not None else "")
        st.caption("NASA POWER daily estimates for the grid cell at the reservoir's listed coordinates. They "
                   "approximate local conditions and are not rainfall measured over the contributing watershed."
                   + lag_note + " The primary v2 model does not use NASA weather.")

    if "swe_basin_in" in w and w["swe_basin_in"].notna().any():
        st.markdown("## Snowpack context")
        ui.show_chart(charts.swe_figure(w, b.synthetic), key=f"ex_swe_{sid}")
        st.caption("CDEC sensor 82 adjusted snow-water equivalent, averaged across qualified upstream snow pillows. "
                   "The selected v2 model uses the basin index through t-1. The series is retrospective and revised.")

    h = b.history
    out = h[(h["station_id"] == sid) & (h["date"] >= start) & (h["date"] <= end)].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    st.download_button("Download this window as CSV", out.to_csv(index=False).encode(),
                       file_name=f"{sid}_{win}_history.csv", mime="text/csv", key="ex_download")
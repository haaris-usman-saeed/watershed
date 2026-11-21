"""A. Overview: where every reservoir stood at the end of the record, on a map."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import ui_charts as charts
import ui_components as ui
import ui_contract as C
import ui_nav

PICKER = "ov"


def _map_pick(event, ids):
    try:
        points = event.selection.points if hasattr(event, "selection") else event["selection"]["points"]
    except (AttributeError, KeyError, TypeError):
        return None
    for p in points or []:
        cd = p.get("customdata") if isinstance(p, dict) else getattr(p, "customdata", None)
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd in ids:
            return cd
    return None


def _panel(b, sid):
    r = b.summary.loc[b.summary["station_id"] == sid].iloc[0]
    st.markdown(f'<div class="ws-name">{ui.esc(str(r.reservoir_name))}</div>'
                f'<div class="ws-id">Station {ui.esc(str(sid))}</div>', unsafe_allow_html=True)
    rows = [("Documented capacity", ui.fmt_af(r.capacity_af))]
    if pd.notna(r.latest_af):
        rows += [("Latest storage in dataset", f"{ui.fmt_af(r.latest_af)} ({ui.fmt_pct(r.latest_pct)} of capacity)"),
                 ("Observed on", ui.fmt_date(r.last_date, long=True))]
        band = r.band
        if band:
            rows.append(("Same season, other years", f"{band['lo']:.0f}% to {band['hi']:.0f}%, median "
                                                     f"{band['median']:.0f}% ({band['years']} years)"))
        rows += [("Storage record", f"{ui.fmt_date(r.first_date, True)} to {ui.fmt_date(r.last_date, True)}"),
                 ("Days with a storage value", f"{r.obs_days:,} of {r.span_days:,} ({r.coverage_pct:.1f}%)"),
                 ("Longest gap", f"{r.longest_gap_days} days" if r.longest_gap_days else "none")]
    else:
        rows.append(("Storage record", "no storage values in history_daily.csv"))
    if b.preds is not None:
        rows.append(("Held-out test forecasts", f"{r.n_forecasts:,}"))
    ui.facts(rows)
    c1, c2 = st.columns(2)
    with c1:
        if b.history is not None and ui.wide_button("Open storage history", key="ov_go_explorer", kind="primary"):
            ui_nav.go("explorer")
    with c2:
        if b.preds is not None and r.n_forecasts and ui.wide_button("Open backtest", key="ov_go_backtest"):
            ui_nav.go("backtest")


def _table(b):
    s = b.summary
    t = pd.DataFrame({
        "Reservoir": s["reservoir_name"].astype(str),
        "ID": s["station_id"].astype(str),
        "Capacity (AF)": s["capacity_af"],
        "Latest storage (AF)": s["latest_af"],
        "% of capacity": s["latest_pct"],
        "Observed on": s["last_date"].apply(lambda d: ui.fmt_date(d)),
        "Record starts": s["first_date"].apply(lambda d: ui.fmt_date(d)),
        "Days observed": s["coverage_pct"],
        "Longest gap (days)": s["longest_gap_days"],
    })
    if b.preds is not None:
        t["Test forecasts"] = s["n_forecasts"]
    cfg = {
        "Capacity (AF)": st.column_config.NumberColumn(format="localized"),
        "Latest storage (AF)": st.column_config.NumberColumn(format="localized"),
        "% of capacity": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        "Days observed": st.column_config.NumberColumn(format="%.1f%%"),
    }
    ui.show_table(t, column_config=cfg)


def render():
    b = ui.load_bundle()
    if not ui.require(b, C.STATIONS_FILE, purpose="The overview"):
        return
    s = b.summary
    n = len(s)
    span = ""
    if s["first_date"].notna().any():
        span = f", {s['first_date'].min():%Y} to {s['last_date'].max():%Y}"
    ui.page_intro(f"{n} California reservoirs{span}",
                  "Daily California reservoir storage joined with CDEC snowpack and NASA POWER weather. "
                  "The selected v2 model uses storage history and snowpack to predict 30-day change.")

    sid = ui.station_picker(b, PICKER, label="Reservoir")

    st.markdown("## Where each reservoir stood at the end of the record")
    if b.history is None:
        ui.require(b, C.HISTORY_FILE, purpose="Storage levels")
    else:
        ui.gauge_row(b, sid)
        last = s["last_date"].max()
        st.caption(f"Latest values in the dataset, not live conditions. The most recent observation is "
                   f"{ui.fmt_date(last, long=True)}. The same-season range uses only the years in this dataset, "
                   f"so it is a short record, not a long-term historical average.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        event = ui.show_chart(charts.map_figure(s, sid, b.synthetic), key=f"ov_map_{sid}", select=True)
        picked = _map_pick(event, b.station_ids) if event is not None else None
        if picked and picked != sid:
            ui.set_station(picked, picker_keys=[f"{PICKER}_station"])
            st.rerun()
        st.caption("Marker area scales with documented capacity; color shows the latest storage as a share of "
                   "capacity. Select a reservoir on the map or with the control above.")
    with right:
        _panel(b, sid)

    st.markdown("## All reservoirs")
    _table(b)
    if b.reports[C.STATIONS_FILE].warnings:
        st.caption("stations.csv notes: " + "; ".join(b.reports[C.STATIONS_FILE].warnings))
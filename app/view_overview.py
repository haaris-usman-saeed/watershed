"""A. Overview: a guide to the site, the latest observed storage (when a forecast refresh exists) and
where each reservoir stood on the last day of the historical dataset, on a map."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import ui_charts as charts
import ui_components as ui
import ui_contract as C
import ui_nav

from datetime import datetime, timezone

PICKER = "ov"


GUIDE = [
    ("Overview", "this page: the six reservoirs, how full they are, and how much data we have."),
    ("Reservoir explorer", "the measured daily storage history of one reservoir, 2001 to 2025, with snowpack and weather "
                           "context. Data only, no predictions."),
    ("Backtesting", "the exam: forecasts made for 2024 and 2025 using only what was known at the time, compared with "
                    "what actually happened."),
    ("Model and research", "the evidence and the fine print: scores, whether the advantage could be luck, whether "
                           "the uncertainty ranges hold, and the limitations."),
    ("30-Day Forecast", "a new forecast from the latest CDEC observations, made with the same frozen model."),
]


def _guide():
    items = "".join(f"<li><strong>{ui.esc(n)}</strong>: {ui.esc(d)}</li>" for n, d in GUIDE)
    ui.callout("How to read this site",
               "WatershedShift predicts how much water each reservoir will gain or lose over the next 30 days, as a "
               "share of its capacity. It was trained on 2001 to 2023 and tested on 2024 to 2025."
               f"</p><ul style='margin:0.4rem 0 0;color:inherit'>{items}</ul><p>", "note")


def _latest_live(b):
    """Most recent observed storage from a valid, unexpired forecast refresh, or None."""
    try:
        import view_forecast
        from src.forecast_schema import expired, validate
        path, _ = view_forecast.resolve_snapshot(b)
        if path is None or not path.exists():
            return None
        snap = view_forecast.load_snapshot(path)
    except Exception:  # noqa: BLE001  the overview never fails because of the forecast snapshot
        return None
    today = datetime.now(timezone.utc).date()
    errs, _ = validate(snap, today=today)
    if errs or snap.get("data_mode") != ("real" if b.is_real else "synthetic") or expired(snap, today):
        return None
    rows = []
    for r in snap["stations"]:
        obs = [h for h in r.get("recent_history", []) if h.get("storage_af") is not None]
        if obs:
            rows.append((r["reservoir_name"], obs[-1]["storage_pct"], obs[-1]["date"]))
    return (snap, rows) if rows else None


def _live_strip(b):
    live = _latest_live(b)
    if live is None:
        ui.callout("The gauges below are not today's levels",
                   "They show the last day of the historical dataset. No recent CDEC refresh is available on this "
                   "deployment; see the 30-Day Forecast page for how one is produced.", "caution")
        return
    snap, rows = live
    st.markdown("## Latest observed storage")
    cols = st.columns(len(rows))
    for c, (name, pct, d) in zip(cols, rows):
        c.metric(name, ui.fmt_pct(pct, 1), help=f"CDEC daily storage on {d}")
        c.caption(f"on {ui.fmt_date(d, long=True)}")
    st.caption(f"From the CDEC refresh behind the 30-Day Forecast, issued {ui.fmt_date(snap['issue_date'], long=True)}. "
               "Percent of the capacity in force on that date. These recent values are not part of the historical "
               "dataset used for training and testing.")
    if ui.wide_button("Open the 30-Day Forecast", key="ov_go_forecast", kind="primary"):
        ui_nav.go("forecast")


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
        rows += [("Storage on last day of dataset", f"{ui.fmt_af(r.latest_af)} ({ui.fmt_pct(r.latest_pct)} of capacity)"),
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
        "Storage, last day of dataset (AF)": s["latest_af"],
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
        "Storage, last day of dataset (AF)": st.column_config.NumberColumn(format="localized"),
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
    _guide()
    _live_strip(b)

    sid = ui.station_picker(b, PICKER, label="Reservoir")

    last_all = s["last_date"].max()
    end_txt = f" ({ui.fmt_date(last_all, long=True)})" if pd.notna(last_all) else ""
    st.markdown(f"## Storage on the last day of the historical dataset{end_txt}")
    if b.history is None:
        ui.require(b, C.HISTORY_FILE, purpose="Storage levels")
    else:
        ui.gauge_row(b, sid)
        last = s["last_date"].max()
        st.caption(f"Values in the historical dataset used to train and test the model, not today's conditions. "
                   f"Its last day is {ui.fmt_date(last, long=True)}. The pale bar is the range on the same dates in "
                   f"other years of the dataset, a short record rather than a long-term average.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        event = ui.show_chart(charts.map_figure(s, sid, b.synthetic), key=f"ov_map_{sid}", select=True)
        picked = _map_pick(event, b.station_ids) if event is not None else None
        if picked and picked != sid:
            ui.set_station(picked, picker_keys=[f"{PICKER}_station"])
            st.rerun()
        st.caption("Marker area scales with documented capacity; color shows storage on the last day of the "
                   "historical dataset as a share of capacity. Select a reservoir on the map or with the control above.")
    with right:
        _panel(b, sid)

    st.markdown("## All reservoirs")
    _table(b)
    if b.reports[C.STATIONS_FILE].warnings:
        st.caption("stations.csv notes: " + "; ".join(b.reports[C.STATIONS_FILE].warnings))
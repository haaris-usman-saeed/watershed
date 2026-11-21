"""Streamlit building blocks shared by all views."""
from __future__ import annotations

import html
import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

import ui_contract as C
import ui_io
from ui_theme import PALETTE as P, PLOTLY_CONFIG, css

STATION_KEY = "ws_station"
esc = html.escape


# ---------------------------------------------------------------- data bundle

@dataclass
class Bundle:
    stations: object
    history: object
    preds: object
    metrics: object
    features: object
    reports: dict
    data_mode: object
    summary: object
    data_dir: object
    dir_source: str

    @property
    def is_real(self):
        return self.data_mode == C.REAL_DATA_MODE

    @property
    def synthetic(self):
        """True whenever data cannot be confirmed as real: drives warnings and watermarks."""
        return not self.is_real

    @property
    def station_ids(self):
        return [] if self.stations is None else self.stations["station_id"].tolist()

    def name(self, sid):
        if self.stations is None:
            return str(sid)
        hit = self.stations.loc[self.stations["station_id"] == sid, "reservoir_name"]
        return str(hit.iloc[0]) if len(hit) else str(sid)

    def capacity(self, sid):
        hit = self.stations.loc[self.stations["station_id"] == sid, "capacity_af"] if self.stations is not None else []
        return float(hit.iloc[0]) if len(hit) and pd.notna(hit.iloc[0]) else np.nan

    def test_range(self):
        if self.preds is None or self.preds.empty:
            return None
        return self.preds["date"].min(), self.preds["target_date"].max()


@st.cache_data(show_spinner="Reading data files")
def _load_cached(dir_str, fingerprint):
    """fingerprint (file mtimes and sizes) is part of the cache key, so new outputs reload."""
    return ui_io.load_dir(dir_str)


def load_bundle() -> Bundle:
    d, source = ui_io.resolve_data_dir()
    raw = _load_cached(str(d), ui_io.fingerprint(d))
    return Bundle(**raw, data_dir=d, dir_source=source)


# ---------------------------------------------------------------- page chrome

def inject_css():
    st.markdown(css(), unsafe_allow_html=True)


def masthead(b: Bundle):
    try:
        shown = b.data_dir.relative_to(ui_io.REPO_ROOT)
    except ValueError:
        shown = b.data_dir
    if b.is_real:
        mode = '<span class="ws-mode real">Real data</span>'
    elif b.data_mode is None:
        mode = '<span class="ws-mode synthetic">Provenance unconfirmed</span>'
    else:
        mode = f'<span class="ws-mode synthetic">data_mode: {esc(str(b.data_mode))}</span>'
    src = "from WATERSHED_DATA_DIR" if b.dir_source == "env" else "default"
    st.markdown(
        f'<div class="ws-mast"><div class="ws-mark">WatershedShift'
        f'<small>California reservoir storage, observed and backtested</small></div>'
        f'<div class="ws-source">{mode}<span>Data</span><code>{esc(str(shown))}</code><span>{src}</span></div></div>',
        unsafe_allow_html=True)


def data_mode_warning(b: Bundle):
    """Persistent warning on every view when data is not confirmed real."""
    if b.is_real:
        return
    if b.data_mode is None:
        head = "Data provenance unconfirmed"
        body = ("metrics.json is missing or has no data_mode, so this app cannot confirm the files hold real "
                "measurements. Treat every value as unverified.")
        ribbon = "Unverified data: provenance not confirmed by metrics.json"
    else:
        head = "Synthetic data. These numbers are not results."
        body = (f"metrics.json reports data_mode = \"{esc(str(b.data_mode))}\". Storage, weather, forecasts and "
                "metrics shown here are generated for development and describe no real reservoir or model.")
        ribbon = "Synthetic data: generated for development, not measured, not model results"
    st.markdown(f'<div class="ws-banner" role="alert"><strong>{head}</strong><span>{body}</span></div>'
                f'<div class="ws-ribbon" aria-hidden="true">{ribbon}</div>', unsafe_allow_html=True)


def page_intro(title, lede):
    st.markdown(f"# {title}")
    if lede:
        st.markdown(f'<p class="ws-lede">{lede}</p>', unsafe_allow_html=True)


def callout(title, body, kind="note"):
    st.markdown(f'<div class="ws-callout {kind}"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def footer(b: Bundle):
    st.markdown(
        '<div class="ws-foot">Storage: California Data Exchange Center (CDEC), daily reservoir storage. '
        'Snow: CDEC adjusted snow-water equivalent. Weather context: NASA POWER. WatershedShift evaluates '
        'historical held-out forecasts. It is not an operational forecast, a water-supply outlook or a '
        'water-safety tool.</div>', unsafe_allow_html=True)


def require(b: Bundle, *files, purpose=""):
    """Render an empty state and return False if any required file is unusable."""
    bad = [f for f in files if b.reports.get(f) is None or not b.reports[f].usable]
    if not bad:
        return True
    lines = []
    for f in bad:
        r = b.reports.get(f)
        if r is None or not r.found:
            lines.append(f"<p><code>{esc(f)}</code> was not found in <code>{esc(str(b.data_dir))}</code>.</p>")
        else:
            lines.append(f"<p><code>{esc(f)}</code> could not be used: {esc(r.error)}.</p>")
    if C.STATIONS_FILE in bad and (ui_io.REPO_ROOT / C.STATIONS_FILE).exists():
        lines.append("<p>A stations.csv exists at the repository root, but the contract reads it from the data "
                     "directory. Copy it there or point WATERSHED_DATA_DIR at a complete output folder.</p>")
    hint = ("<p>Run the pipeline (see README.md) to produce outputs, or develop against the fixtures with "
            "<code>WATERSHED_DATA_DIR=app/sample_data streamlit run app.py</code>.</p>")
    title = f"{purpose} needs data that is not available" if purpose else "Required data is not available"
    st.markdown(f'<div class="ws-empty"><h4>{esc(title)}</h4>{"".join(lines)}{hint}</div>', unsafe_allow_html=True)
    return False


# ---------------------------------------------------------------- widgets with version fallbacks

def _supports(fn, param):
    try:
        return param in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def segmented(label, options, default, key, format_func=str, label_visibility="visible"):
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, options, default=default, key=key, format_func=format_func,
                                 label_visibility=label_visibility)
        return v if v is not None else default
    return st.radio(label, options, index=options.index(default), key=key, format_func=format_func,
                    horizontal=True, label_visibility=label_visibility)


def multi_pills(label, options, default, key, format_func=str):
    if hasattr(st, "pills"):
        v = st.pills(label, options, default=default, key=key, format_func=format_func, selection_mode="multi")
        return list(v or [])
    return st.multiselect(label, options, default=default, key=key, format_func=format_func)


def show_chart(fig, key=None, select=False):
    kw = dict(theme=None, config=PLOTLY_CONFIG, key=key)
    if _supports(st.plotly_chart, "width"):
        kw["width"] = "stretch"
    else:
        kw["use_container_width"] = True
    if select:
        kw.update(on_select="rerun", selection_mode="points")
    return st.plotly_chart(fig, **kw)


def show_table(df, column_config=None, height=None):
    kw = dict(hide_index=True, column_config=column_config)
    if height:
        kw["height"] = height
    if _supports(st.dataframe, "width"):
        kw["width"] = "stretch"
    else:
        kw["use_container_width"] = True
    st.dataframe(df, **kw)


def wide_button(label, key, kind="secondary"):
    kw = dict(key=key, type=kind)
    if _supports(st.button, "width"):
        kw["width"] = "stretch"
    else:
        kw["use_container_width"] = True
    return st.button(label, **kw)


# ---------------------------------------------------------------- station selection

def current_station(b: Bundle):
    ids = b.station_ids
    s = st.session_state.get(STATION_KEY)
    if s not in ids:
        q = st.query_params.get("station")
        s = q if q in ids else (ids[0] if ids else None)
        st.session_state[STATION_KEY] = s
    return s


def set_station(sid, picker_keys=()):
    st.session_state[STATION_KEY] = sid
    for k in picker_keys:
        st.session_state.pop(k, None)
    try:
        st.query_params["station"] = sid
    except Exception:  # noqa: BLE001  query params are a convenience only
        pass


def station_picker(b: Bundle, key, label="Reservoir", ids=None):
    ids = ids or b.station_ids
    cur = current_station(b)
    if cur not in ids:
        cur = ids[0]
    wkey = f"{key}_station"

    def sync():
        v = st.session_state.get(wkey)
        if v is not None:
            set_station(v)

    fmt = b.name
    if len(ids) <= 7 and hasattr(st, "segmented_control"):
        if wkey not in st.session_state:
            st.session_state[wkey] = cur
        v = st.segmented_control(label, ids, key=wkey, format_func=fmt, on_change=sync)
        if v is None:  # deselecting a segment would leave nothing chosen
            v = cur
    else:
        v = st.selectbox(label, ids, index=ids.index(cur), key=wkey, format_func=fmt, on_change=sync)
    if v != st.session_state.get(STATION_KEY):
        set_station(v)
    return v


# ---------------------------------------------------------------- gauges (signature element)

def _gauge_svg(pct, band, on, labels, w=44, h=156):
    """Reservoir as a tube: water fill to the latest level; the bracket on the left and the pale
    overlay mark the range in other years at the same time of year; the dashed line is their median."""
    def y(v):
        return h - h * float(np.clip(v, 0, 100)) / 100

    x0 = 12
    right = 34 if labels else x0          # keep the tube centred under its label
    width = x0 + w + right
    shift = ' style="margin-left:22px"' if labels else ""
    parts = [f'<svg width="{width}" height="{h + 2}" viewBox="0 -1 {width} {h + 2}" aria-hidden="true"{shift}>']
    stroke = P["bone"] if on else P["ridge"]
    parts.append(f'<rect x="{x0}" y="0" width="{w}" height="{h}" rx="3" fill="{P["basin"]}"/>')
    parts.append(f'<line x1="{x0 + 1}" x2="{x0 + w - 1}" y1="{y(50):.1f}" y2="{y(50):.1f}" stroke="{P["ridge"]}" '
                 f'stroke-width="1" stroke-dasharray="1 3"/>')
    if pct is not None and np.isfinite(pct):
        top = y(pct)
        parts.append(f'<rect class="fill" x="{x0 + 1}" y="{top:.1f}" width="{w - 2}" height="{h - top - 1:.1f}" '
                     f'rx="2" fill="{P["water"]}" fill-opacity="0.82"/>')
        parts.append(f'<line x1="{x0 + 1}" x2="{x0 + w - 1}" y1="{top:.1f}" y2="{top:.1f}" stroke="{P["bone"]}" '
                     f'stroke-width="1.2"/>')
    if band:
        y_hi, y_lo, ym = y(band["hi"]), y(band["lo"]), y(band["median"])
        hgt = max(y_lo - y_hi, 1.5)
        parts.append(f'<rect x="{x0 + 1}" y="{y_hi:.1f}" width="{w - 2}" height="{hgt:.1f}" fill="{P["bone"]}" '
                     f'fill-opacity="0.13"/>')
        parts.append(f'<rect x="{x0 - 7}" y="{y_hi:.1f}" width="3.5" height="{hgt:.1f}" rx="1.5" fill="{P["bone"]}" '
                     f'fill-opacity="0.75"/>')
        parts.append(f'<line x1="{x0 - 9}" x2="{x0 + w + 3}" y1="{ym:.1f}" y2="{ym:.1f}" stroke="{P["bone"]}" '
                     f'stroke-width="1.5" stroke-dasharray="3 2"/>')
    parts.append(f'<rect x="{x0}" y="0" width="{w}" height="{h}" rx="3" fill="none" stroke="{stroke}" '
                 f'stroke-width="{1.6 if on else 1}"/>')
    if labels:
        for tick in (0, 50, 100):
            parts.append(f'<text x="{x0 + w + 6}" y="{y(tick) + 4:.1f}" font-size="10" fill="{P["silt_dim"]}">'
                         f'{tick}%</text>')
    parts.append("</svg>")
    return "".join(parts)


def gauge_row(b: Bundle, selected):
    s = b.summary
    if s is None or s.empty:
        return
    first = not st.session_state.get("ws_gauges_drawn")
    st.session_state["ws_gauges_drawn"] = True
    cells = []
    for i, r in enumerate(s.itertuples(index=False)):
        on = r.station_id == selected
        pct = r.latest_pct if pd.notna(r.latest_pct) else None
        label = f"{r.latest_pct:.0f}%" if pct is not None else "n/a"
        aria = (f"{esc(str(r.reservoir_name))}: {label} of capacity on {r.last_date:%b %d, %Y}"
                if pct is not None else f"{esc(str(r.reservoir_name))}: no storage data")
        cells.append(f'<div class="ws-gauge{" on" if on else ""}" role="img" aria-label="{aria}" title="{aria}">'
                     f'{_gauge_svg(pct, r.band, on, labels=i == 0)}<div class="pc">{label}</div>'
                     f'<div class="nm">{esc(str(r.reservoir_name))}</div></div>')
    legend = (f'<div class="ws-legend"><span><i style="background:{P["water"]}"></i>Storage on the last observed day</span>'
              f'<span><i style="background:{P["bone"]};opacity:.75;width:4px"></i>Range in other years within 7 days of that date</span>'
              f'<span><i style="border-top:2px dashed {P["bone"]};height:0;width:14px"></i>Median of those years</span></div>')
    st.markdown(f'<div class="ws-gauges{" animate" if first else ""}">{"".join(cells)}</div>{legend}',
                unsafe_allow_html=True)


def weather_lag(b: Bundle):
    """Minimum lag (days) of weather features, from metrics.json or feature_catalog.json; None if unknown."""
    import ui_metrics
    nm = ui_metrics.normalize_metrics(b.metrics)
    if nm.weather_lag_days is not None and np.isfinite(nm.weather_lag_days):
        return float(nm.weather_lag_days)
    lags = []
    for f in b.features or []:
        tag = f"{f.get('group', '')} {f.get('source', '')}".lower()
        if any(t in tag for t in ("weather", "power", "precip", "t2m", "rh2m")):
            try:
                lags.append(float(f.get("lag_days")))
            except (TypeError, ValueError):
                pass
    return min(lags) if lags else None


# ---------------------------------------------------------------- formatting

def fmt_af(v):
    return f"{v:,.0f} AF" if v is not None and np.isfinite(v) else "n/a"


def fmt_pp(v, sign=True, digits=2):
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:+.{digits}f} pp" if sign else f"{v:.{digits}f} pp"


def fmt_pct(v, digits=0):
    return f"{v:.{digits}f}%" if v is not None and np.isfinite(v) else "n/a"


def fmt_date(d, long=False):
    if d is None or pd.isna(d):
        return "n/a"
    return pd.Timestamp(d).strftime("%b %d, %Y" if long else "%Y-%m-%d")


def facts(pairs):
    body = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in pairs)
    st.markdown(f'<dl class="ws-facts">{body}</dl>', unsafe_allow_html=True)
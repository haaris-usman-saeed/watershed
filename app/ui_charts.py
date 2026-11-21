"""Plotly figure builders. Each takes prepared data and returns a figure; no Streamlit calls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ui_contract as C
import ui_io
from ui_theme import PALETTE as P, STORAGE_SCALE, FONT_SANS, axis, finish, rgba

HAS_MAPLIBRE = hasattr(go, "Scattermap")


def _date_str(d):
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _vline(fig, x, text, color, dash="dot", yanchor="top", row=None):
    fig.add_shape(type="line", x0=_date_str(x), x1=_date_str(x), y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color=color, width=1.2, dash=dash))
    if text:
        fig.add_annotation(x=_date_str(x), y=1, xref="x", yref="paper", text=text, showarrow=False,
                           xanchor="left", yanchor=yanchor, xshift=4, font=dict(size=11, color=color))


def _vrect(fig, x0, x1, fill, text=None, text_color=None):
    fig.add_shape(type="rect", x0=_date_str(x0), x1=_date_str(x1), y0=0, y1=1, xref="x", yref="paper",
                  fillcolor=fill, line=dict(width=0), layer="below")
    if text:
        fig.add_annotation(x=_date_str(x0), y=1, xref="x", yref="paper", text=text, showarrow=False,
                           xanchor="left", yanchor="bottom", xshift=4, font=dict(size=11, color=text_color or P["silt"]))


# ---------------------------------------------------------------- map

def _zoom_for(lats, lons, height_px=470):
    lat_span = max(float(np.ptp(lats)), 0.3) * 1.7 + 0.6
    lon_span = max(float(np.ptp(lons)), 0.3) * 1.7 + 0.6
    lat0 = float(np.mean(lats))
    z_lat = np.log2(height_px * 360 * np.cos(np.radians(lat0)) / (256 * lat_span))
    z_lon = np.log2(620 * 360 / (256 * lon_span))
    return float(np.clip(min(z_lat, z_lon) - 0.25, 4.0, 9.0))


def map_figure(summary, selected, synthetic, height=470):
    s = summary.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    Trace = go.Scattermap if HAS_MAPLIBRE else go.Scattermapbox
    fig = go.Figure()
    if s.empty:
        return finish(fig, synthetic, height)
    cap = s["capacity_af"].fillna(s["capacity_af"].median())
    size = (12 + 30 * np.sqrt(cap / cap.max())).round(1)
    pct = s["latest_pct"].clip(0, 100)
    as_of = s["last_date"].apply(lambda d: d.strftime("%b %d, %Y") if pd.notna(d) else "no storage data")
    pct_txt = s["latest_pct"].apply(lambda v: f"{v:.0f}% of capacity" if pd.notna(v) else "storage unavailable")
    hover = [f"<b>{n}</b>  {i}<br>{p}<br>latest in dataset, {a}<br>capacity {c:,.0f} AF"
             for n, i, p, a, c in zip(s["reservoir_name"], s["station_id"], pct_txt, as_of, cap)]

    sel = s[s["station_id"] == selected]
    if not sel.empty:
        fig.add_trace(Trace(lat=sel["lat"], lon=sel["lon"], mode="markers", hoverinfo="skip", showlegend=False,
                            customdata=sel["station_id"].tolist(),
                            marker=dict(size=float(size[sel.index[0]]) + 18, color=P["bone"], opacity=0.28)))
    fig.add_trace(Trace(
        lat=s["lat"], lon=s["lon"], mode="markers", customdata=s["station_id"].tolist(),
        text=hover, hovertemplate="%{text}<extra></extra>", showlegend=False,
        marker=dict(size=size.tolist(), color=pct.fillna(0).tolist(), colorscale=STORAGE_SCALE, cmin=0, cmax=100,
                    opacity=0.95,
                    colorbar=dict(title=dict(text="% full", font=dict(color=P["silt"], size=11)), thickness=8, len=0.55,
                                  x=0.985, xanchor="right", y=0.5, outlinewidth=0, ticksuffix="%",
                                  tickfont=dict(color=P["silt"], size=10), bgcolor=rgba(P["ink"], 0.6)))))
    view = dict(style="carto-darkmatter", center=dict(lat=float(s["lat"].mean()), lon=float(s["lon"].mean())),
                zoom=_zoom_for(s["lat"], s["lon"], height))
    fig = finish(fig, synthetic, height, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select")
    fig.update_layout(**({"map": view} if HAS_MAPLIBRE else {"mapbox": view}))
    return fig


# ---------------------------------------------------------------- explorer

def storage_figure(series, capacity_af, units, synthetic, test_range=None, height=380):
    pct = units == "pct"
    y = series["storage_pct"] if pct else series["storage_af"]
    fig = go.Figure()
    if test_range:
        _vrect(fig, test_range[0], test_range[1], rgba(P["model"], 0.07), "Held-out test period", P["model"])
    custom = np.column_stack([series["storage_af"].to_numpy(), series["storage_pct"].to_numpy()])
    fig.add_trace(go.Scatter(
        x=series["date"], y=y, mode="lines", name="Observed storage", connectgaps=False,
        line=dict(color=P["water"], width=1.6), fill="tozeroy", fillcolor=rgba(P["water"], 0.13),
        customdata=custom, hovertemplate="%{x|%b %d, %Y}<br>%{customdata[0]:,.0f} AF<br>%{customdata[1]:.1f}% of capacity<extra></extra>"))
    cap_y = 100 if pct else capacity_af
    if cap_y and np.isfinite(cap_y):
        fig.add_hline(y=cap_y, line=dict(color=rgba(P["bone"], 0.55), width=1, dash="dash"))
        fig.add_annotation(xref="paper", x=1, y=cap_y, yref="y", text="Documented capacity", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=11, color=P["silt"]))
    ymax = max(float(np.nanmax(y)) if y.notna().any() else 0, cap_y or 0) * 1.06
    fig = finish(fig, synthetic, height, showlegend=False, hovermode="x", margin=dict(l=84, r=18, t=30, b=40))
    fig.update_yaxes(**axis(title=dict(text="% of capacity" if pct else "Storage, acre-feet"),
                            ticksuffix="%" if pct else "", tickformat=".0f" if pct else ",.0f", range=[0, ymax]))
    fig.update_xaxes(**axis(hoverformat="%b %d, %Y"))
    return fig


ENV_SERIES = (
    ("precip_mm", "Precipitation, mm/day", "rain", "bar", "%{y:.1f} mm"),
    ("t2m_c", "Air temperature at 2 m, °C", "temp", "line", "%{y:.1f} °C"),
    ("rh2m_pct", "Relative humidity at 2 m, %", "humidity", "line", "%{y:.0f}%"),
)


def env_figure(series, synthetic, test_range=None):
    avail = [e for e in ENV_SERIES if e[0] in series.columns and series[e[0]].notna().any()]
    if not avail:
        return None
    fig = make_subplots(rows=len(avail), cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[e[1] for e in avail])
    for i, (col, title, color, kind, fmt) in enumerate(avail, start=1):
        if kind == "bar":
            tr = go.Bar(x=series["date"], y=series[col], marker=dict(color=P[color], line=dict(width=0)),
                        name=title, hovertemplate=f"%{{x|%b %d, %Y}}<br>{fmt}<extra></extra>")
        else:
            tr = go.Scatter(x=series["date"], y=series[col], mode="lines", connectgaps=False,
                            line=dict(color=P[color], width=1.1), name=title,
                            hovertemplate=f"%{{x|%b %d, %Y}}<br>{fmt}<extra></extra>")
        fig.add_trace(tr, row=i, col=1)
    fig = finish(fig, synthetic, 150 * len(avail) + 60, showlegend=False, bargap=0,
                 margin=dict(l=84, r=18, t=34, b=36), hovermode="x")
    fig.update_xaxes(**axis())
    fig.update_yaxes(**axis())
    for ann in fig.layout.annotations or []:
        if ann.text in {e[1] for e in avail}:
            ann.update(font=dict(size=12, color=P["bone"], family=FONT_SANS), x=0, xanchor="left")
    if test_range:
        fig.add_shape(type="rect", x0=_date_str(test_range[0]), x1=_date_str(test_range[1]), y0=0, y1=1,
                      xref="x", yref="paper", fillcolor=rgba(P["model"], 0.06), line=dict(width=0), layer="below")
    return fig


# ---------------------------------------------------------------- backtest

def backtest_timeline(p, baselines, synthetic, height=420):
    p = p.sort_values("date")
    fig = go.Figure()
    for i, seg in enumerate(ui_io.contiguous_segments(p)):
        fig.add_trace(go.Scatter(x=seg["date"], y=seg["upper_90_pp"], mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False, legendgroup="band"))
        fig.add_trace(go.Scatter(x=seg["date"], y=seg["lower_90_pp"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=rgba(P["model"], 0.17), name="90% prediction interval",
                                 legendgroup="band", showlegend=i == 0, hoverinfo="skip"))
    g = ui_io.break_gaps(p[["date", "target_date", "target_delta_pp", "pred_delta_pp", "lower_90_pp", "upper_90_pp"]
                           + [b.pred_col for b in baselines]])
    for b in baselines:
        fig.add_trace(go.Scatter(x=g["date"], y=g[b.pred_col], mode="lines", name=b.label, connectgaps=False,
                                 line=dict(color=P[b.color], width=1.4, dash=b.dash),
                                 hovertemplate=f"{b.label} %{{y:+.2f}} pp<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["pred_delta_pp"], mode="lines+markers", name="ML prediction", connectgaps=False,
        line=dict(color=P["model"], width=2), marker=dict(size=5, color=P["model"]),
        customdata=np.column_stack([g["lower_90_pp"], g["upper_90_pp"]]),
        hovertemplate="ML %{y:+.2f} pp (90%: %{customdata[0]:+.2f} to %{customdata[1]:+.2f})<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["target_delta_pp"], mode="lines+markers", name="Observed change", connectgaps=False,
        line=dict(color=rgba(P["bone"], 0.45), width=1), marker=dict(size=6, color=P["bone"], symbol="circle"),
        customdata=g["target_date"].dt.strftime("%b %d, %Y"),
        hovertemplate="Observed %{y:+.2f} pp, measured to %{customdata}<extra></extra>"))
    fig = finish(fig, synthetic, height, hovermode="x unified", margin=dict(l=64, r=18, t=46, b=44))
    fig.update_yaxes(**axis(title=dict(text="30-day change, pp of capacity"), zeroline=True, zerolinewidth=1.4,
                            ticksuffix=" pp"))
    fig.update_xaxes(**axis(title=dict(text="Forecast origin date"), hoverformat="Origin %b %d, %Y"))
    return fig


def pred_vs_actual(preds, selected, synthetic, height=430):
    fig = go.Figure()
    lo = float(np.nanmin([preds["target_delta_pp"].min(), preds["pred_delta_pp"].min()]))
    hi = float(np.nanmax([preds["target_delta_pp"].max(), preds["pred_delta_pp"].max()]))
    pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=rgba(P["silt"], 0.6), width=1, dash="dot"))
    fig.add_annotation(x=hi, y=hi, text="Perfect forecast", showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=11, color=P["silt"]))

    def trace(df, name, color, size, opacity):
        return go.Scatter(
            x=df["target_delta_pp"], y=df["pred_delta_pp"], mode="markers", name=name,
            marker=dict(size=size, color=color, opacity=opacity, line=dict(width=0)),
            customdata=np.column_stack([df["reservoir_name"].astype(str), df["date"].dt.strftime("%Y-%m-%d")]),
            hovertemplate="<b>%{customdata[0]}</b>, origin %{customdata[1]}<br>observed %{x:+.2f} pp<br>predicted %{y:+.2f} pp<extra></extra>")

    others = preds[preds["station_id"] != selected]
    mine = preds[preds["station_id"] == selected]
    if not others.empty:
        fig.add_trace(trace(others, "Other reservoirs", P["silt_dim"], 6, 0.45))
    if not mine.empty:
        name = str(mine["reservoir_name"].iloc[0]) if mine["reservoir_name"].notna().any() else str(selected)
        fig.add_trace(trace(mine, name, P["model"], 8, 0.95))
    fig = finish(fig, synthetic, height, margin=dict(l=64, r=18, t=40, b=52))
    fig.update_xaxes(**axis(title=dict(text="Observed 30-day change, pp"), range=[lo, hi], zeroline=True))
    fig.update_yaxes(**axis(title=dict(text="ML predicted change, pp"), range=[lo, hi], zeroline=True,
                            scaleanchor="x", scaleratio=1))
    return fig


def mae_bars(table, synthetic, height=None):
    t = table.dropna(subset=["mae"]).sort_values("mae", ascending=False)
    colors = [P[C.MODEL_BY_KEY[k].color] for k in t["key"]]
    fig = go.Figure(go.Bar(
        x=t["mae"], y=t["label"], orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f} pp" for v in t["mae"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=P["bone"], size=12), customdata=t["n"],
        hovertemplate="%{y}: %{x:.3f} pp over %{customdata} forecasts<extra></extra>"))
    fig = finish(fig, synthetic, height or 70 + 46 * len(t), showlegend=False, margin=dict(l=128, r=64, t=10, b=46),
                 bargap=0.38)
    fig.update_xaxes(**axis(title=dict(text="Mean absolute error, pp (lower is better)"),
                            range=[0, float(t["mae"].max()) * 1.22 if len(t) else 1]))
    fig.update_yaxes(**axis(ticks="", tickfont=dict(color=P["bone"], size=12), gridcolor="rgba(0,0,0,0)"))
    return fig


def inspector_figure(daily, row, lag_days, synthetic, baselines, height=380):
    origin, target = pd.Timestamp(row["date"]), pd.Timestamp(row["target_date"])
    w = daily[(daily["date"] >= origin - pd.Timedelta(days=75)) & (daily["date"] <= target + pd.Timedelta(days=14))]
    before, after = w[w["date"] <= origin], w[w["date"] >= origin]
    s0 = daily.loc[daily["date"] == origin, "storage_pct"]
    s0 = float(s0.iloc[0]) if len(s0) and pd.notna(s0.iloc[0]) else np.nan
    fig = go.Figure()
    _vrect(fig, origin, target, rgba(P["model"], 0.06))
    if lag_days and np.isfinite(lag_days):
        _vline(fig, origin - pd.Timedelta(days=int(lag_days)), "Weather inputs end", P["silt"], "dot", "bottom")
    _vline(fig, origin, "Forecast made", P["bone"], "solid")
    fig.add_trace(go.Scatter(x=before["date"], y=before["storage_pct"], mode="lines", name="Observed up to origin",
                             connectgaps=False, line=dict(color=P["water"], width=2),
                             hovertemplate="%{x|%b %d}: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=after["date"], y=after["storage_pct"], mode="lines",
                             name="Observed afterwards (unknown to model)", connectgaps=False,
                             line=dict(color=rgba(P["water"], 0.55), width=1.5, dash="dash"),
                             hovertemplate="%{x|%b %d}: %{y:.1f}%<extra></extra>"))
    if np.isfinite(s0):
        tx = _date_str(target)
        for b in baselines:
            v = row.get(b.pred_col)
            if pd.notna(v):
                fig.add_trace(go.Scatter(x=[tx], y=[s0 + v], mode="markers", name=b.label,
                                         marker=dict(size=10, symbol="line-ew-open", color=P[b.color], line=dict(width=2.5)),
                                         hovertemplate=f"{b.label}: %{{y:.1f}}%<extra></extra>"))
        pred = s0 + row["pred_delta_pp"]
        fig.add_trace(go.Scatter(
            x=[tx], y=[pred], mode="markers", name="ML prediction, 90% interval",
            marker=dict(size=12, symbol="diamond", color=P["model"]),
            error_y=dict(type="data", symmetric=False, array=[row["upper_90_pp"] - row["pred_delta_pp"]],
                         arrayminus=[row["pred_delta_pp"] - row["lower_90_pp"]], color=P["model"], thickness=2, width=10),
            hovertemplate="ML: %{y:.1f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=[tx], y=[s0 + row["target_delta_pp"]], mode="markers", name="Observed at target",
                                 marker=dict(size=10, color=P["bone"], line=dict(color=P["ink"], width=1.5)),
                                 hovertemplate="Observed: %{y:.1f}%<extra></extra>"))
    fig = finish(fig, synthetic, height, hovermode="closest", margin=dict(l=60, r=18, t=58, b=40),
                 legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, font=dict(size=11, color=P["silt"])))
    fig.update_yaxes(**axis(title=dict(text="Storage, % of capacity"), ticksuffix="%"))
    fig.update_xaxes(**axis(tickformat="%b %d"))
    return fig, s0


# ---------------------------------------------------------------- research

SYMBOLS = {"ml": "diamond", "persistence": "circle-open", "trend": "square-open", "seasonal": "triangle-up-open"}


def per_station_dots(long, synthetic):
    order = (long[long["key"] == "ml"].sort_values("mae")["reservoir_name"].tolist()
             or sorted(long["reservoir_name"].unique()))
    fig = go.Figure()
    for spec in C.MODELS:
        d = long[long["key"] == spec.key]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["mae"], y=d["reservoir_name"], mode="markers", name=spec.label,
            marker=dict(symbol=SYMBOLS[spec.key], size=13 if spec.key == "ml" else 10, color=P[spec.color],
                        line=dict(width=2, color=P[spec.color])),
            customdata=d["n"], hovertemplate=f"{spec.label}: %{{x:.2f}} pp, n = %{{customdata}}<extra>%{{y}}</extra>"))
    fig = finish(fig, synthetic, 110 + 44 * len(order), margin=dict(l=150, r=18, t=40, b=46))
    fig.update_yaxes(**axis(categoryorder="array", categoryarray=order[::-1], ticks="",
                            tickfont=dict(color=P["bone"], size=12)))
    fig.update_xaxes(**axis(title=dict(text="Mean absolute error, pp (lower is better)"), rangemode="tozero"))
    return fig


def coverage_bars(cov, overall, synthetic, nominal=C.INTERVAL_LEVEL):
    rows = pd.concat([pd.DataFrame([dict(reservoir_name="All reservoirs", **overall)]), cov], ignore_index=True)
    rows = rows.iloc[::-1]
    def col(f):
        if not np.isfinite(f):
            return P["silt_dim"]
        return P["caution"] if f < nominal - 0.05 else P["water"]
    fig = go.Figure(go.Bar(
        x=rows["frac"] * 100, y=rows["reservoir_name"], orientation="h",
        marker=dict(color=[col(f) for f in rows["frac"]], line=dict(width=0)),
        text=[f"{f:.0%} ({i}/{n})" if np.isfinite(f) else "n/a" for f, i, n in zip(rows["frac"], rows["inside"], rows["n"])],
        textposition="outside", cliponaxis=False, textfont=dict(color=P["bone"], size=11),
        hovertemplate="%{y}: %{x:.1f}% of observed changes fell inside the interval<extra></extra>"))
    fig.add_shape(type="line", x0=nominal * 100, x1=nominal * 100, y0=0, y1=1, yref="paper",
                  line=dict(color=P["bone"], width=1.2, dash="dash"))
    fig.add_annotation(x=nominal * 100, y=1, yref="paper", text=f"Nominal {nominal:.0%}", showarrow=False,
                       yanchor="bottom", font=dict(size=11, color=P["bone"]))
    fig = finish(fig, synthetic, 100 + 38 * len(rows), showlegend=False, bargap=0.35,
                 margin=dict(l=150, r=80, t=30, b=46))
    fig.update_xaxes(**axis(title=dict(text="Share of test forecasts inside the 90% interval"), range=[0, 108],
                            ticksuffix="%"))
    fig.update_yaxes(**axis(ticks="", tickfont=dict(color=P["bone"], size=12), gridcolor="rgba(0,0,0,0)"))
    return fig


def bootstrap_forest(rows, synthetic):
    fig = go.Figure()
    labels = []
    for b in rows:
        spec = C.MODEL_BY_KEY[b.model_key]
        labels.append(spec.label)
        clear_good, clear_bad = b.lo > 0, b.hi < 0
        color = P["good"] if clear_good else P["caution"] if clear_bad else P["silt"]
        fig.add_trace(go.Scatter(
            x=[b.diff], y=[spec.label], mode="markers", showlegend=False,
            marker=dict(size=12, color=color, symbol="diamond"),
            error_x=dict(type="data", symmetric=False, array=[b.hi - b.diff], arrayminus=[b.diff - b.lo],
                         color=color, thickness=2.2, width=8),
            hovertemplate=(f"vs {spec.label}: %{{x:+.2f}} pp<br>{b.level:.0%} interval {b.lo:+.2f} to {b.hi:+.2f} pp"
                           "<extra></extra>")))
    span = max([abs(v) for b in rows for v in (b.lo, b.hi, b.diff) if np.isfinite(v)] + [0.5]) * 1.25
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, yref="paper", line=dict(color=P["bone"], width=1.2))
    fig.add_annotation(x=span, y=1, yref="paper", text="ML lower error", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(size=11, color=P["silt"]))
    fig.add_annotation(x=-span, y=1, yref="paper", text="Baseline lower error", showarrow=False, xanchor="left",
                       yanchor="bottom", font=dict(size=11, color=P["silt"]))
    fig = finish(fig, synthetic, 110 + 52 * len(rows), margin=dict(l=150, r=24, t=34, b=50))
    fig.update_xaxes(**axis(title=dict(text="Baseline MAE minus ML MAE, pp"), range=[-span, span], zeroline=False))
    fig.update_yaxes(**axis(ticks="", tickfont=dict(color=P["bone"], size=12), categoryorder="array",
                            categoryarray=labels[::-1], gridcolor="rgba(0,0,0,0)"))
    return fig


# ---------------------------------------------------------------- snowpack context

def swe_figure(series, synthetic, height=245):
    """CDEC basin snow-pillow index used by the selected v2 model."""
    fig = go.Figure(go.Scatter(x=series["date"], y=series["swe_basin_in"], mode="lines",
                               name="Basin SWE index", connectgaps=False,
                               line=dict(color=P["water"], width=2),
                               fill="tozeroy", fillcolor=rgba(P["water"], 0.12),
                               hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f} inches SWE<extra></extra>"))
    fig = finish(fig, synthetic, height, showlegend=False, margin=dict(l=64, r=20, t=18, b=40))
    fig.update_yaxes(**axis(title=dict(text="SWE index (inches)"), rangemode="tozero"))
    return fig
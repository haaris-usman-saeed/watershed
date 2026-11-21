"""Design tokens, global CSS and Plotly styling.

Palette idea: a reservoir in drought. Deep water, the bleached "bathtub ring" of exposed
shoreline (bone), dry silt for low storage. The ML model gets a violet that appears nowhere
in nature so it is never confused with observations. Caution amber is reserved for the
synthetic-data warning and for negative results.
"""
from __future__ import annotations

PALETTE = {
    "ink": "#0C1A22",        # page
    "basin": "#12242E",      # raised surfaces
    "ridge": "#1F3A47",      # rules, borders, gridlines
    "bone": "#ECE5D6",       # primary text; observed values
    "silt": "#93A6AE",       # secondary text
    "silt_dim": "#6C8089",   # tertiary text, axis ticks
    "water": "#48B6CE",      # storage
    "model": "#B9A2F2",      # ML predictions and intervals
    "caution": "#F2C14E",    # synthetic data, negative findings
    "good": "#8BC4A0",       # positive findings (used sparingly)
    "persistence": "#8697A0",
    "trend": "#D9A07A",
    "seasonal": "#7C9BE0",
    "doy": "#7ED7B4",
    "rain": "#5E9BE0",
    "temp": "#E3866F",
    "humidity": "#7FC2A8",
    "dry": "#9C7E55",
}
# Low storage reads as dry ground, high as deep water.
STORAGE_SCALE = [[0.0, "#9C7E55"], [0.45, "#8E9C86"], [1.0, "#3FA9C9"]]

FONT_SANS = "'Public Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
FONT_SERIF = "'Newsreader', Georgia, 'Times New Roman', serif"


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def model_color(spec) -> str:
    return PALETTE[spec.color]


# ---------------------------------------------------------------- plotly

def axis(**kw):
    base = dict(gridcolor=rgba(PALETTE["ridge"], 0.7), zerolinecolor=rgba(PALETTE["silt"], 0.35),
                linecolor=PALETTE["ridge"], tickcolor=PALETTE["ridge"], ticks="outside", ticklen=4,
                tickfont=dict(color=PALETTE["silt_dim"], size=11), title=dict(font=dict(color=PALETTE["silt"], size=12)),
                showspikes=False)
    base.update(kw)
    return base


def layout(height=360, **kw):
    base = dict(
        height=height, margin=dict(l=56, r=18, t=28, b=44),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, color=PALETTE["silt"], size=12),
        hoverlabel=dict(bgcolor=PALETTE["basin"], bordercolor=PALETTE["ridge"],
                        font=dict(family=FONT_SANS, color=PALETTE["bone"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=PALETTE["silt"], size=12), bgcolor="rgba(0,0,0,0)"),
        xaxis=axis(), yaxis=axis(), separators=".,",
    )
    base.update(kw)
    return base


def finish(fig, synthetic: bool, height=None, **kw):
    """Apply house layout and, for non-real data, a watermark that survives screenshots."""
    fig.update_layout(**layout(height or 360, **kw))
    if synthetic:
        fig.add_annotation(text="Synthetic data", xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, textangle=-14,
                           font=dict(family=FONT_SERIF, size=54, color=rgba(PALETTE["caution"], 0.09)))
    return fig


PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


# ---------------------------------------------------------------- css

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
  --ink: %(ink)s; --basin: %(basin)s; --ridge: %(ridge)s; --bone: %(bone)s; --silt: %(silt)s;
  --silt-dim: %(silt_dim)s; --water: %(water)s; --model: %(model)s; --caution: %(caution)s;
  --good: %(good)s; --dry: %(dry)s;
  --sans: %(sans)s; --serif: %(serif)s;
}

.stApp { background: var(--ink); color: var(--bone); font-family: var(--sans); }
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp input, .stApp textarea,
.stApp button, .stApp summary, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-baseweb="select"] div {
  font-family: var(--sans);
}
.stApp [data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }

.block-container { max-width: 1320px; padding-top: 4.25rem; padding-bottom: 5.5rem; }

.stApp h1, .stApp h2 { font-family: var(--serif); font-weight: 500; color: var(--bone); letter-spacing: -0.012em; }
.stApp h1 { font-size: 2.35rem; line-height: 1.12; margin: 0.2rem 0 0.4rem; }
.stApp h2 { font-size: 1.55rem; line-height: 1.2; margin-top: 1.6rem; }
.stApp h3 { font-family: var(--sans); font-weight: 600; font-size: 1.02rem; color: var(--bone); letter-spacing: 0; margin-top: 1.1rem; }
.stApp [data-testid="stHeadingWithActionElements"] a { display: none; }

.stApp p, .stApp li { line-height: 1.55; }
.stApp [data-testid="stCaptionContainer"] p { color: var(--silt); font-size: 0.86rem; line-height: 1.5; }
.ws-lede { color: var(--silt); font-size: 1.05rem; max-width: 64ch; margin: 0 0 1.4rem; line-height: 1.55; }
.ws-measure { max-width: 72ch; }

/* Metrics: separated by a left rule rather than boxed. */
.stApp [data-testid="stMetric"] { border-left: 1px solid var(--ridge); padding: 0.1rem 0 0.1rem 0.95rem; }
.stApp [data-testid="stMetricLabel"] p { color: var(--silt); font-size: 0.84rem; }
.stApp [data-testid="stMetricValue"] { font-family: var(--sans); font-weight: 600; font-size: 1.62rem;
  color: var(--bone); font-variant-numeric: tabular-nums; }
.stApp [data-testid="stMetricDelta"] { font-variant-numeric: tabular-nums; }

.stApp [data-testid="stExpander"] details { border: 1px solid var(--ridge); border-radius: 8px; background: transparent; }
.stApp hr { border-color: var(--ridge); }

/* Masthead */
.ws-mast { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
  gap: 0.4rem 1.5rem; padding: 0 0 0.9rem; border-bottom: 1px solid var(--ridge); margin-bottom: 1.4rem; }
.ws-mark { font-family: var(--serif); font-size: 1.5rem; font-weight: 600; color: var(--bone); letter-spacing: -0.01em; }
.ws-mark small { font-family: var(--sans); font-size: 0.86rem; font-weight: 400; color: var(--silt); margin-left: 0.75rem; letter-spacing: 0; }
.ws-source { font-size: 0.82rem; color: var(--silt); display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.ws-source code { font-family: var(--sans); background: var(--basin); border: 1px solid var(--ridge); color: var(--bone);
  padding: 0.1rem 0.45rem; border-radius: 4px; font-size: 0.8rem; }
.ws-mode { padding: 0.12rem 0.5rem; border-radius: 999px; font-weight: 600; font-size: 0.76rem; border: 1px solid; }
.ws-mode.real { color: var(--good); border-color: var(--good); }
.ws-mode.synthetic { color: var(--caution); border-color: var(--caution); }

/* Synthetic data banner and fixed ribbon */
.ws-banner { border: 1px solid var(--caution); border-radius: 8px; padding: 0.85rem 1.1rem; margin: 0 0 1.2rem;
  background: repeating-linear-gradient(135deg, rgba(242,193,78,0.10) 0 14px, rgba(242,193,78,0.03) 14px 28px); }
.ws-banner strong { color: var(--caution); font-weight: 700; display: block; font-size: 1rem; margin-bottom: 0.2rem; }
.ws-banner span { color: var(--bone); font-size: 0.9rem; }
.ws-ribbon { position: fixed; left: 0; right: 0; bottom: 0; z-index: 999990; text-align: center;
  background: var(--caution); color: #1A1404; font-weight: 700; font-size: 0.8rem; padding: 0.32rem 1rem; }

/* Callouts */
.ws-callout { border-left: 3px solid var(--water); background: var(--basin); padding: 0.9rem 1.15rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0 1.1rem; }
.ws-callout h4 { font-family: var(--sans); font-size: 1rem; font-weight: 650; color: var(--bone); margin: 0 0 0.3rem; padding: 0; }
.ws-callout p { margin: 0; color: var(--silt); font-size: 0.92rem; max-width: 80ch; }
.ws-callout.caution { border-left-color: var(--caution); }
.ws-callout.caution h4 { color: var(--caution); }
.ws-callout.good { border-left-color: var(--good); }
.ws-callout.model { border-left-color: var(--model); }
.ws-callout.verdict h4 { font-family: var(--serif); font-weight: 500; font-size: 1.45rem; line-height: 1.25; margin-bottom: 0.45rem; }
.ws-callout.verdict p { font-size: 0.98rem; color: var(--bone); opacity: 0.88; }

.ws-empty { border: 1px dashed var(--ridge); border-radius: 8px; padding: 1.4rem 1.3rem; margin: 0.5rem 0 1rem; }
.ws-empty h4 { font-family: var(--sans); font-size: 1rem; color: var(--bone); margin: 0 0 0.35rem; padding: 0; }
.ws-empty p { color: var(--silt); margin: 0.15rem 0; font-size: 0.92rem; }
.ws-empty code { color: var(--bone); background: var(--basin); padding: 0.05rem 0.35rem; border-radius: 4px; }

/* Reservoir gauges: the signature element */
.ws-gauges { display: grid; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: 1.4rem 0.8rem;
  align-items: start; margin: 0.4rem 0 0.4rem; max-width: 980px; }
.ws-gauge { display: flex; flex-direction: column; align-items: center; }
.ws-gauge svg { overflow: visible; }
.ws-gauge .nm { margin-top: 0.45rem; font-size: 0.8rem; color: var(--silt); text-align: center; line-height: 1.25; min-height: 2.1em; }
.ws-gauge .pc { font-size: 1.12rem; font-weight: 600; color: var(--bone); font-variant-numeric: tabular-nums; margin-top: 0.1rem; }
.ws-gauge.on .nm { color: var(--bone); font-weight: 600; }
.ws-gauge .fill { transform-box: fill-box; transform-origin: bottom; }
.ws-gauges.animate .fill { animation: ws-rise 1.1s cubic-bezier(.2,.7,.2,1) both; }
@keyframes ws-rise { from { transform: scaleY(0); } to { transform: scaleY(1); } }
@media (prefers-reduced-motion: reduce) { .ws-gauges.animate .fill { animation: none; } }
.ws-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.3rem; font-size: 0.8rem; color: var(--silt); margin: 0.6rem 0 0; }
.ws-legend i { display: inline-block; width: 12px; height: 10px; margin-right: 0.4rem; vertical-align: -1px; border-radius: 2px; }

/* Definition list for reservoir facts */
.ws-facts { display: grid; grid-template-columns: max-content 1fr; gap: 0.42rem 1.1rem; margin: 0.6rem 0 1rem; font-size: 0.92rem; }
.ws-facts dt { color: var(--silt); }
.ws-facts dd { margin: 0; color: var(--bone); font-variant-numeric: tabular-nums; }
.ws-name { font-family: var(--serif); font-size: 1.7rem; color: var(--bone); line-height: 1.15; margin: 0.1rem 0 0; }
.ws-id { color: var(--silt); font-size: 0.85rem; }

/* Forecast inspector facts */
.ws-ftable { width: 100%%; border-collapse: collapse; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
.ws-ftable td, .ws-ftable th { padding: 0.42rem 0.4rem; border-bottom: 1px solid var(--ridge); text-align: right; }
.ws-ftable th { color: var(--silt); font-weight: 500; }
.ws-ftable td:first-child, .ws-ftable th:first-child { text-align: left; }
.ws-swatch { display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin-right: 0.5rem; border-radius: 2px; }
.ws-yes { color: var(--good); } .ws-no { color: var(--caution); }

.ws-foot { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--ridge); color: var(--silt-dim); font-size: 0.8rem; max-width: 90ch; }

@media (max-width: 640px) {
  .block-container { padding-top: 3.6rem; }
  .stApp h1 { font-size: 1.8rem; }
  .ws-gauges { grid-template-columns: repeat(3, 1fr); gap: 1.2rem 0.4rem; }
  .ws-mark small { display: block; margin: 0.2rem 0 0; }
}
"""


def css() -> str:
    return "<style>" + CSS % {**PALETTE, "sans": FONT_SANS, "serif": FONT_SERIF} + "</style>"
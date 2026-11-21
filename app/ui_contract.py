"""Frontend view of the backend file contract.

Every file name, column name, metric key and unit the UI depends on lives here.
If UI_CONTRACT.md differs from what is written below, edit this file only.

Provenance of each assumption:
  [repo]     stations.csv header on main
  [prompts]  "Shared interface contract" section of PROMPTS.md
  [brief]    frontend brief (file list, data_mode, WATERSHED_DATA_DIR)
  [assumed]  not yet confirmed against UI_CONTRACT.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

ENV_VAR = "WATERSHED_DATA_DIR"            # [brief]
DEFAULT_DIR_NAME = "outputs"               # [brief]

STATIONS_FILE = "stations.csv"             # [brief]
HISTORY_FILE = "history_daily.csv"         # [brief]
PREDICTIONS_FILE = "test_predictions.csv"  # [brief]
METRICS_FILE = "metrics.json"              # [brief]
FEATURES_FILE = "feature_catalog.json"     # [brief]
ALL_FILES = (STATIONS_FILE, HISTORY_FILE, PREDICTIONS_FILE, METRICS_FILE, FEATURES_FILE)

REAL_DATA_MODE = "real"                    # [brief] anything else triggers the warning


@dataclass(frozen=True)
class Col:
    required: bool
    kind: str  # "str" | "float" | "date"
    aliases: tuple = ()
    unit: str = ""
    source: str = "assumed"


# stations.csv: one row per reservoir.
STATION_COLS = {
    "station_id": Col(True, "str", ("station", "id", "cdec_id"), source="repo"),
    "reservoir_name": Col(True, "str", ("name", "station_name"), source="repo"),
    "lat": Col(True, "float", ("latitude",), "degrees north", "repo"),
    "lon": Col(True, "float", ("longitude", "lng"), "degrees east", "repo"),
    "capacity_af": Col(True, "float", ("capacity", "capacity_acre_feet"), "acre-feet", "repo"),
}

# history_daily.csv: one row per reservoir per calendar day. Storage may be blank
# (or the row absent) on days CDEC has no value. Nothing is interpolated.
HISTORY_COLS = {
    "station_id": Col(True, "str", ("station",)),
    "date": Col(True, "date", ("obs_date", "day")),
    "storage_af": Col(True, "float", ("storage", "storage_acre_feet", "storage_15"), "acre-feet"),
    "storage_pct": Col(False, "float", (), "% of capacity in force"),
    "swe_basin_in": Col(False, "float", (), "inches"),
    "precip_mm": Col(False, "float", ("prectotcorr", "precip", "precip_mm_day", "rain_mm", "prectotcorr_mm"), "mm/day"),
    "t2m_c": Col(False, "float", ("t2m", "temp_c", "temperature_c", "t2m_mean_c"), "degrees C"),
    "rh2m_pct": Col(False, "float", ("rh2m", "rh_pct", "humidity_pct", "relative_humidity"), "%"),
}

# test_predictions.csv: one row per reservoir per forecast origin in the test split.
# "date" is the forecast origin; "target_date" is the date the 30-day change is measured to.
# All *_pp columns are percentage points of documented capacity:
#   100 * (storage at target_date - storage at date) / capacity_af
PREDICTION_COLS = {
    "station_id": Col(True, "str", (), source="prompts"),
    "reservoir_name": Col(False, "str", (), source="prompts"),
    "date": Col(True, "date", ("origin_date", "forecast_origin"), source="prompts"),
    "target_date": Col(True, "date", (), source="prompts"),
    "target_delta_pp": Col(True, "float", (), "pp of capacity", "prompts"),
    "pred_delta_pp": Col(True, "float", (), "pp of capacity", "prompts"),
    "lower_90_pp": Col(True, "float", (), "pp of capacity", "prompts"),
    "upper_90_pp": Col(True, "float", (), "pp of capacity", "prompts"),
    "baseline_persistence_pp": Col(False, "float", (), "pp of capacity", "prompts"),
    "baseline_trend_pp": Col(False, "float", (), "pp of capacity", "prompts"),
    "baseline_seasonal_pp": Col(False, "float", (), "pp of capacity", "prompts"),
    "baseline_doy_pp": Col(False, "float", (), "pp of capacity", "repo v2"),
    "storage_af": Col(False, "float", (), "acre-feet", "repo v2"),
    "storage_pct": Col(False, "float", (), "% capacity in force", "repo v2"),
    "capacity_af": Col(False, "float", (), "acre-feet at origin", "repo v2"),
}

HORIZON_DAYS = 30          # [prompts] target_date - date
INTERVAL_LEVEL = 0.90      # [prompts] lower_90_pp / upper_90_pp


@dataclass(frozen=True)
class ModelSpec:
    key: str          # internal id
    label: str        # shown in the UI
    metric_key: str   # key in metrics.json holding test MAE
    pred_col: str     # column in test_predictions.csv
    color: str        # token name in ui_theme.PALETTE
    dash: str         # plotly dash style
    blurb: str


MODELS = (
    ModelSpec("ml", "ML model", "ML", "pred_delta_pp", "model", "solid",
              "Gradient-boosted regressor trained on storage history, CDEC snowpack, season and reservoir identity (v2)."),
    ModelSpec("persistence", "Persistence", "Persistence", "baseline_persistence_pp", "persistence", "dot",
              "Predicts no change: storage in 30 days equals storage today."),
    ModelSpec("trend", "30-day trend", "30-day trend", "baseline_trend_pp", "trend", "dash",
              "Repeats the change observed over the previous 30 days."),
    ModelSpec("seasonal", "Monthly seasonal", "Historical seasonal", "baseline_seasonal_pp", "seasonal", "dashdot",
              "Average 30-day change for the origin month in training years."),
    ModelSpec("doy", "Smoothed DOY seasonal", "Smoothed DOY seasonal", "baseline_doy_pp", "doy", "longdash",
              "Gaussian-smoothed historical seasonal change (15-day day-of-year kernel)."),
)
MODEL_BY_KEY = {m.key: m for m in MODELS}
BASELINES = tuple(m for m in MODELS if m.key != "ml")

# metrics.json scalar keys. [prompts] unless marked.
METRIC_KEYS = {
    "data_mode": "data_mode",                            # [brief]
    "coverage": "interval_90_coverage",                  # fraction 0..1
    "halfwidth": "interval_90_halfwidth_pp",
    "train_count": "train_count",
    "val_count": "val_count",
    "test_count": "test_count",
    "test_station_count": "test_station_count",
    "note": "note",
}
# [assumed] keys that may hold bootstrap comparisons, checked in order.
BOOTSTRAP_KEYS = ("bootstrap", "bootstrap_comparisons", "bootstrap_mae_diff", "mae_diff_bootstrap")
# [assumed] optional free-text lists.
LIMITATIONS_KEYS = ("limitations", "known_limitations")
ASSUMPTIONS_KEYS = ("assumptions", "prediction_time_assumptions")
SOURCES_KEYS = ("sources", "data_sources")
WEATHER_LAG_KEYS = ("weather_lag_days", "weather_availability_lag_days")

# feature_catalog.json: [assumed] list of objects, or {"features": [...]}, or {name: description}.
FEATURE_FIELD_ALIASES = {
    "name": ("name", "feature", "column"),
    "description": ("description", "desc", "definition"),
    "group": ("group", "category", "family"),
    "source": ("source", "dataset"),
    "unit": ("unit", "units"),
    "window_days": ("window_days", "window", "aggregation_window_days"),
    "lag_days": ("lag_days", "lag", "min_lag_days", "availability_lag_days"),
    "available_at_prediction_time": ("available_at_prediction_time", "prediction_time_available", "available"),
    "used_by_primary_model": ("used_by_primary_model",),
}


def find_column(columns, canonical: str, spec: Col):
    """Return the actual column name matching canonical or an alias (case-insensitive)."""
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in (canonical,) + tuple(spec.aliases):
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None
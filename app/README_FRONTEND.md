# WatershedShift frontend

Streamlit app over the file-based backend contract. Owns `app.py`, `app/`, and `.streamlit/config.toml` only.

## Run

```
pip install -r app/requirements-ui.txt
python app/make_sample_data.py                          # synthetic fixtures, skip if they already exist
WATERSHED_DATA_DIR=app/sample_data streamlit run app.py # development
streamlit run app.py                                    # real pipeline outputs in outputs/
```

`WATERSHED_DATA_DIR` may be absolute or relative (resolved from the working directory, then the repo root).
Data reloads automatically when any of the five files changes.

## Views

| View | Needs | Shows |
| --- | --- | --- |
| Overview | stations.csv, history_daily.csv | End-of-record gauges, map (click to select), reservoir facts, network table |
| Reservoir explorer | stations.csv, history_daily.csv | Storage in AF or % of capacity, time windows, weather panels, CSV download |
| Backtesting | test_predictions.csv | Held-out timeline with 90% band and baselines, predicted vs observed, error by method, single-forecast inspector |
| Model and research | metrics.json and/or test_predictions.csv | Verdict, scores, per-reservoir errors, coverage, bootstrap, feature catalog, sources, limits, diagnostics |

A view whose files are missing shows what is missing and how to produce it. Nothing is substituted.
If `metrics.json["data_mode"]` is not `"real"`, or is absent, every view shows a banner, a fixed bottom ribbon,
and a watermark on every chart.

## Contract assumptions

All names live in `app/ui_contract.py`, tagged by provenance. **UI_CONTRACT.md was not available when this was
built; reconcile that file first.**

| File | Confirmed from | Assumed |
| --- | --- | --- |
| stations.csv | repo header: station_id, reservoir_name, lat, lon, capacity_af | none |
| test_predictions.csv | PROMPTS.md shared contract | `date` is the origin, `target_date = date + 30` days |
| metrics.json | PROMPTS.md keys (ML, Persistence, 30-day trend, Historical seasonal, interval_90_*, counts, note) | `data_mode`, `bootstrap` block, `limitations`, `assumptions`, `weather_lag_days` |
| history_daily.csv | nothing | station_id, date, storage_af, precip_mm, t2m_c, rh2m_pct (aliases accepted, e.g. PRECTOTCORR) |
| feature_catalog.json | nothing | list of {name, description, group, source, unit, window_days, lag_days} or {"features": [...]} |

Bootstrap convention assumed: difference = baseline MAE minus ML MAE, so positive means ML has lower error.
Entries named `ml_minus_baseline` are negated. If a bootstrap sign disagrees with the point MAEs, the app flags it
rather than correcting it.

## Check outputs before integrating

```
python app/check_contract.py app/sample_data
python app/check_contract.py outputs --require-real
```

Exit code 1 if a required file is unusable or, with `--require-real`, data_mode is not "real". Alias matches,
duplicate dates, out-of-range coordinates, interval inversions and metrics.json vs predictions disagreements are
reported as notes. The same report appears in the app under Model and research, Data contract diagnostics.

## Fixture scenarios

`python app/make_sample_data.py --force --scenario {unclear,worse,better}` sets where the stand-in model lands
relative to the seasonal baseline, to exercise each verdict state. Fixtures are fully synthetic and labelled so.

## Module layout

`app/` is intentionally not a package (an `app/__init__.py` would collide with `app.py`). `app.py` puts `app/` on
`sys.path`; modules use `ui_` and `view_` prefixes to avoid name clashes with `src/`.

- `ui_contract.py` names, units, model specs
- `ui_io.py` loading, validation, summaries (no Streamlit; shared with the checker)
- `ui_metrics.py` MAE, coverage, verdict, bootstrap parsing, cross-checks (no Streamlit)
- `ui_theme.py` tokens, CSS, Plotly layout and watermark
- `ui_charts.py` Plotly figure builders
- `ui_components.py` banners, pickers, gauges, version-tolerant widget helpers
- `view_*.py` the four pages; `ui_nav.py` page registry

## Not supported, by design

Live forecasts, satellite imagery or CNN models, water-safety or water-quality assessment, reservoir siting.


## V2 integration notice (2026-11-21)

This frontend has been adapted for the frozen v2 engineering contract. The default strongest baseline is read
from `bootstrap.best_baseline`; its v2 predictions are `baseline_doy_pp`. Bootstrap estimates are interpreted
with the backend sign convention, and all three preregistered block lengths can be displayed.

The primary v2 model uses storage history, CDEC SWE, seasonality and reservoir identity. NASA POWER weather
is available for historical context and secondary models only. Historical storage percentages retain the
backend's date-specific capacity normalization (Oroville reporting change on 2024-07-02).

**Deployment:** Copy the `app/` Python modules and root `app.py` into the merged engineering repository.
Preserve the engineering repository's `app/sample_data/` files (they satisfy schema v1.0); do not copy the
older fixture folder from Claude Web. Install `app/requirements-ui.txt` as well as the engineering dependencies.
Run `WATERSHED_DATA_DIR=outputs streamlit run app.py` after placing the privately supplied real v2 output
files in `outputs/`. Run `WATERSHED_DATA_DIR=app/sample_data streamlit run app.py` for synthetic development.
No raw CDEC files or real outputs should be committed to public GitHub while reuse conditions are unresolved.
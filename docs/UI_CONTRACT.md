# WatershedShift UI and engineering interface contract (schema v1.0)

Owner: Claude Code (engineering). Consumers: Claude Web / Muse (UI).
Machine-checkable version: `src/contract.py` (`python -m src.contract <dir>` validates a directory).
Schema v1.0 is unchanged by the v2 model revision. All v2 changes are **additive** (section 3.7).

## 1. How the UI gets data

- **File-based and read-only. There is no API server, and the UI never needs to import the model.**
- The engineering pipeline writes 5 files into one directory. The UI reads them with `pandas.read_csv` / `json.load`.
- Directory selection:
  - the env var `WATERSHED_DATA_DIR` if set, otherwise `outputs/` (real pipeline outputs, git-ignored, produced locally);
  - for development without real outputs: `WATERSHED_DATA_DIR=app/sample_data`.
- **Always check `metrics.json["data_mode"]`.** If it is `"SYNTHETIC_SAMPLE"`, show a persistent, prominent banner: *"SYNTHETIC SAMPLE DATA: not real observations, predictions or accuracy."* Only `"real"` may be presented as results.
- If any required file is missing, show a clear missing-data state. Never fall back to synthetic data silently.
- Ignore unknown extra columns and keys. Removing or renaming a required field would bump `schema_version`.

## 2. Conventions

| Item | Convention |
|---|---|
| Dates | ISO `YYYY-MM-DD` strings, calendar days (CDEC daily = Pacific local day; NASA POWER requested in LST). No times, no time zones. |
| Storage | `storage_af` in **acre-feet** (CDEC sensor 15, daily, end-of-day). |
| `*_pct` | percent **of the reservoir capacity in force on that date** (normally 0 to 100; can slightly exceed 100 during flood encroachment). For ORO, the capacity is 3,537,577 AF through 2024-07-01 and 3,424,753 AF from 2024-07-02. |
| `*_pp` | **percentage points of capacity** (for example, +3.0 means storage rose by 3% of capacity). *Not* percent of current storage. |
| Rain | mm/day (NASA POWER `PRECTOTCORR`), a point or grid-cell estimate, **not** watershed-average rainfall. |
| Temperature | deg C (NASA POWER `T2M`, 2 m air temperature). |
| Humidity | % (NASA POWER `RH2M`). |
| Snow | `swe_basin_in` in inches: the mean of core CDEC snow pillows in the upstream basin (sensor 82). |
| Missing values | empty cell (`NaN`). Storage is **never interpolated**; missing days may be absent rows or blank values. |
| Station IDs | CDEC 3-letter codes (real: `SHA`, `ORO`, `FOL`, `NML`, `DNP`, `EXC`); synthetic fixtures use `SYN1..SYN4`. |

## 3. Files

### 3.1 `stations.csv`: reservoir metadata (one row per candidate station)

| column | type | description |
|---|---|---|
| station_id | str | CDEC station code |
| reservoir_name | str | display name |
| lat, lon | float | decimal degrees (WGS84) of the station or dam; also the NASA POWER query point |
| capacity_af | int | current documented capacity (acre-feet) |
| capacity_source | str | where the capacity was verified |
| status | str | `included` (used in the model) or `excluded` |
| first_date, last_date | date | first and last day with a valid storage observation |
| n_storage_days | int | count of days with valid storage |
| missing_storage_pct | float | % of days without valid storage |
| *(optional)* exclusion_reason, operator | str | |

### 3.2 `history_daily.csv`: historical daily observations (joined dataset, 2001 to 2025 in v2)

| column | type | description |
|---|---|---|
| station_id | str | |
| date | date | |
| storage_af | float | observed storage, acre-feet (blank = missing) |
| storage_pct | float | 100 x storage_af / capacity in force on that date |
| rain_mm | float | daily precipitation at the station point (NASA POWER) |
| temp_c | float | daily mean 2 m temperature |
| humidity_pct | float | daily mean 2 m relative humidity |
| *(optional, v2)* swe_basin_in | float | basin snow water equivalent index, inches (blank when fewer than 2/3 of core pillows report) |

Use this file for "storage over time", "weather context" and "snowpack context" charts. It holds observed data only, never forecasts.

### 3.3 `test_predictions.csv`: held-out 30-day forecasts (historical backtest)

One row per (station, forecast origin). Origins are weekly Mondays in the **test period only** (2024-01-01 to 2025-12-01).

| column | type | description |
|---|---|---|
| station_id | str | |
| reservoir_name | str | |
| date | date | forecast origin t (only information available by t is used) |
| target_date | date | t + 30 days |
| target_delta_pp | float | **observed** 100 x (storage(t+30) - storage(t)) / capacity in force at t |
| pred_delta_pp | float | primary ML prediction (v2: gradient boosting on storage history, snowpack and season) |
| lower_90_pp, upper_90_pp | float | nominal 90% prediction interval for the primary prediction (v2: width depends on season) |
| baseline_persistence_pp | float | baseline: no change (always 0) |
| baseline_trend_pp | float | baseline: repeat the previous 30-day change |
| baseline_seasonal_pp | float | baseline: mean historical change for this station and calendar month (training data only) |
| *(optional)* storage_af, capacity_af, storage_pct | | storage at the origin |

Derived for display:
- Predicted level at the target date (% capacity) = `storage_pct + pred_delta_pp`.
- Observed level at the target date = `storage_pct + target_delta_pp`.
- Error = `pred_delta_pp - target_delta_pp`.

**Note:** consecutive weekly origins have overlapping 30-day windows, so errors are serially correlated. Do not present each row as an independent trial.

### 3.4 `metrics.json`: evaluation summary

```jsonc
{
  "schema_version": "1.0",
  "data_mode": "real" | "SYNTHETIC_SAMPLE",
  "generated_utc": "2026-11-21T12:00:00Z",
  "ML": 2.31,                         // test MAE of the primary model, pp of capacity
  "Persistence": 3.10,
  "30-day trend": 3.45,
  "Historical seasonal": 2.80,        // monthly climatology baseline
  "interval_90_coverage": 0.87,       // fraction of test targets inside [lower_90, upper_90]
  "interval_90_halfwidth_pp": 4.1,    // v1: constant half-width. v2: MEAN half-width (per-row widths vary by season)
  "train_count": 600, "val_count": 150, "test_count": 291, "test_station_count": 3,
  "note": "free-text caveats, display verbatim",
  "split": {"train_target_end_exclusive": "...", "val_start": "...", "val_target_end_exclusive": "...",
            "test_start": "...", "test_target_end_inclusive": "...", "horizon_days": 30,
            "weather_lag_days": 7, "sampling": "weekly (Mondays)"},
  "per_station": {"SHA": {"n": 97, "ML": 2.1, "Persistence": 3.0, "30-day trend": 3.3,
                          "Historical seasonal": 2.6, "interval_90_coverage": 0.88}},
  "bootstrap": {"method": "...", "n_boot": 2000, "block_weeks": 5, "best_baseline": "...",
                "mae_diff_ml_minus_best_baseline": {"estimate": -0.49, "ci95_low": -0.9, "ci95_high": -0.1}}
}
```
The values above are illustrative only.
- `mae_diff_ml_minus_best_baseline` < 0 means ML beat the best baseline.
- If the CI includes 0, the UI should say the difference is **not clearly distinguishable**.
- If ML loses, show that plainly.
- Always read split dates and the name of the best baseline from this file; never hard-code them.

### 3.5 `feature_catalog.json`: model inputs

`{"schema_version": "1.0", "features": [{"name", "description", "units", "source", "min_lag_days"}, ...]}`. `min_lag_days` is how many days before the forecast origin the feature's latest input ends.

### 3.6 Additive extras in v1 outputs (historical; kept for reference)

- v1 `stations.csv`: `exclusion_reason`, `operator`.
- v1 `test_predictions.csv`: `pred_ridge_pp`.
- v1 `metrics.json`: `"Ridge"`, `"validation"`, `bootstrap.mae_ci95`, `bootstrap.mae_diff_ml_minus_best_baseline.share_boot_ml_better`.

Frozen v1 results are in `results/v1/` (public, without raw storage columns).

### 3.7 Additive fields in v2 outputs (current `outputs/`, `model_version: "v2"`)

**`metrics.json`**

| key | meaning |
|---|---|
| `model_version` | `"v2"` |
| `evaluation_mode` | `"final"` (the single preregistered held-out run) |
| `"Smoothed DOY seasonal"` | test MAE of the day-of-year climatology baseline (**the strongest baseline in v2**) |
| `"HGB (v1 features)"`, `"HGB + SWE"`, `"HGB no weather"`, `"Ridge"` | test MAE of secondary models (ablations, not selected) |
| `interval_90_halfwidth_by_season_pp` | median half-width per origin season (DJF, MAM, JJA, SON) |
| `interval_90_mean_width_pp` | mean full width (upper minus lower) |
| `per_station[s]` | adds every model above plus `interval_mean_width_pp` |
| `per_season`, `per_water_year` | same structure as `per_station`, keyed by season or by water year (Oct to Sep; WY2026 is partial) |
| `bootstrap.best_baseline` | `"Smoothed DOY seasonal"` (selected on development data) |
| `bootstrap.by_comparison_and_block` | for "ML minus Smoothed DOY seasonal" and "ML minus Historical seasonal (monthly)": `{ "5_weeks" \| "9_weeks" \| "13_weeks": {estimate, ci95_low, ci95_high, share_better} }` |
| `v1_vs_v2` | paired comparison on rows with identical targets: `n_common_rows`, `v1_gbm_mae`, `v2_primary_mae`, `bootstrap_v2_minus_v1` (5/9/13-week CIs) |
| `v1_reference` | the complete frozen v1 `metrics.json` |
| `split.snow_lag_days`, `split.data_period` | 1; ["2001-01-01", "2025-12-31"] |

**`test_predictions.csv`:** `baseline_doy_pp`, `pred_hgb_v1_features_pp`, `pred_hgb_swe_pp`, `pred_hgb_no_weather_pp`, `pred_ridge_pp`, `season`.

**`feature_catalog.json`:** `model_version` and per feature `used_by_primary_model` (true or false). The v2 primary model uses storage, SWE and season features. The NASA POWER weather features are listed with `used_by_primary_model: false`.

**`history_daily.csv`:** `swe_basin_in`.

**Recommended UI additions (all optional):**
- show v1 vs v2 side by side using `v1_reference` and `v1_vs_v2`;
- show the three block-length CIs;
- use `baseline_doy_pp` as the "strongest baseline" line;
- show the snowpack (`swe_basin_in`) context chart;
- show the per-season interval widths.

Display the v2 disclosure from `note` verbatim. It states that the v2 design came after the v1 test results on the same period were seen.

## 4. Feature status

| Status | Feature |
|---|---|
| **Implemented and evaluated** | CDEC storage 2001 to 2025 (6 reservoirs, verified); historical capacity normalization; NASA POWER weather; CDEC snow-pillow SWE (basin index, 14-day change, anomaly); persistence, trend, monthly and day-of-year seasonal baselines; Ridge; HistGradientBoosting variants; season-conditional 90% intervals; moving-block bootstrap CIs (5, 9, 13 weeks); rolling water-year development evaluation (`results/v2/development/report.md`) |
| **Investigated and not used** | CDEC reservoir inflow (a mass-balance quantity, not an independent measurement) and releases (operator decisions) |
| **Future research only (not implemented)** | satellite-derived features, image models or CNNs |
| **Not supported: do not show or promise** | satellite imagery or CNN predictions, real-time or operational forecasts, forecasts beyond the data end date, flood, drought or drinking-water safety assessments, reservoir siting |

## 5. File ownership (to avoid conflicts)

| Path | Owner | Others may |
|---|---|---|
| `app.py`, `app/` (including `app/sample_data/` from v2 onward), `.streamlit/`, `docs/UI_*.md` other than this file, report and handoff drafts | **UI (Claude Web / Muse)** | engineering will not edit |
| `src/`, `tests/`, `data/`, `outputs/`, `models/`, `results/`, `config/`, `stations.csv`, `docs/UI_CONTRACT.md`, `docs/preregistration.md`, `data/provenance.md` | **Engineering (Claude Code)** | the UI reads only; request changes in `STATUS.md` |
| `requirements.txt` | Engineering | the UI asks in `STATUS.md` to add a package |
| `README.md` | Engineering; the UI may edit the "Demo app" section | |
| `STATUS.md` | shared | append under your own heading only |

Do not create `app/__init__.py` (it would shadow `app.py` as an import name). Work on a separate branch (for example `ui`) and merge through a pull request.

## 6. Quick start for the UI

```bash
python -m src.make_sample_data            # (re)generate synthetic fixtures (v1 schema subset)
WATERSHED_DATA_DIR=app/sample_data streamlit run app.py
python -m src.contract app/sample_data    # validate a data directory against this contract
python -m src.contract outputs            # validate the real v2 outputs
```

```python
import os, json, pandas as pd
from pathlib import Path
D = Path(os.environ.get("WATERSHED_DATA_DIR", "outputs"))
stations = pd.read_csv(D / "stations.csv", parse_dates=["first_date", "last_date"])
history  = pd.read_csv(D / "history_daily.csv", parse_dates=["date"])
preds    = pd.read_csv(D / "test_predictions.csv", parse_dates=["date", "target_date"])
metrics  = json.loads((D / "metrics.json").read_text())
features = json.loads((D / "feature_catalog.json").read_text())["features"]
synthetic = metrics["data_mode"] != "real"
is_v2 = metrics.get("model_version") == "v2"
```

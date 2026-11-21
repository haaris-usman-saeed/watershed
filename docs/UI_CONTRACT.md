# WatershedShift — UI ↔ Engineering Interface Contract (schema v1.0)

Owner: Claude Code (engineering). Consumers: Claude Web / Muse (UI).
Machine-checkable version: `src/contract.py` (`python -m src.contract <dir>` validates a directory).

## 1. How the UI gets data

- **File-based, read-only. No API server, no Python import of the model required.**
- The engineering pipeline writes 5 files into one directory. The UI reads them with `pandas.read_csv` / `json.load`.
- Directory selection (UI should implement this):
  - env var `WATERSHED_DATA_DIR` if set, else `outputs/` (real pipeline outputs, git-ignored, produced locally),
  - for development before real outputs exist: `WATERSHED_DATA_DIR=app/sample_data`.
- **Always check `metrics.json["data_mode"]`.** If it is `"SYNTHETIC_SAMPLE"` the UI must show a persistent, prominent banner: *"SYNTHETIC SAMPLE DATA — not real observations, predictions or accuracy."* Only `"real"` may be presented as results.
- If any required file is missing, show a clear missing-data state. Never fall back to synthetic data silently.
- The UI must ignore unknown extra columns/keys (engineering may add fields; removing/renaming a required field bumps `schema_version`).

## 2. Conventions

| Item | Convention |
|---|---|
| Dates | ISO `YYYY-MM-DD` strings, calendar days (CDEC daily = Pacific local day; NASA POWER requested in LST). No times, no time zones. |
| Storage | `storage_af` in **acre-feet** (CDEC sensor 15, daily). |
| `*_pct` | percent **of documented reservoir capacity** (0–100+; can slightly exceed 100 in encroachment). |
| `*_pp` | **percentage points of capacity** (e.g. +3.0 = storage rose by 3% of capacity). *Not* percent of current storage. |
| Rain | mm/day (NASA POWER `PRECTOTCORR`), point/grid-cell estimate, **not** watershed-average rainfall. |
| Temperature | °C (NASA POWER `T2M`, 2 m air temp). |
| Humidity | % (NASA POWER `RH2M`). |
| Missing values | empty cell (`NaN`). Storage is **never interpolated**; missing days may be absent rows or blank values. |
| Station IDs | CDEC 3-letter codes (real: e.g. `SHA`, `ORO`, `FOL`); synthetic fixtures use `SYN1..SYN4`. |

## 3. Files

### 3.1 `stations.csv` — reservoir metadata (one row per candidate station)

| column | type | description |
|---|---|---|
| station_id | str | CDEC station code |
| reservoir_name | str | display name |
| lat, lon | float | decimal degrees (WGS84) of the station/dam; also the NASA POWER query point |
| capacity_af | int | documented capacity used as denominator (acre-feet) |
| capacity_source | str | where capacity was verified (URL or short citation) |
| status | str | `included` (used in model) or `excluded` (failed verification / coverage) |
| first_date, last_date | date | first/last day with a valid storage observation |
| n_storage_days | int | count of days with valid storage |
| missing_storage_pct | float | % of days in [first_date, last_date] without valid storage |
| *(optional)* exclusion_reason | str | why a station was excluded |

### 3.2 `history_daily.csv` — historical daily observations (joined dataset)

| column | type | description |
|---|---|---|
| station_id | str | |
| date | date | |
| storage_af | float | observed storage, acre-feet (blank = missing) |
| storage_pct | float | 100 × storage_af / capacity_af |
| rain_mm | float | daily precipitation at station point (NASA POWER) |
| temp_c | float | daily mean 2 m temperature |
| humidity_pct | float | daily mean 2 m relative humidity |

Use for “storage over time” and “weather context” charts. Observed data only — no forecasts in this file.

### 3.3 `test_predictions.csv` — held-out 30-day forecasts (historical backtest)

One row per (station, forecast origin). Origins are weekly Mondays in the **test period only**.

| column | type | description |
|---|---|---|
| station_id | str | |
| reservoir_name | str | |
| date | date | forecast origin *t* (only information available by *t* used; weather ends at *t*−7d) |
| target_date | date | *t* + 30 days |
| target_delta_pp | float | **observed** 100 × (storage(t+30) − storage(t)) / capacity |
| pred_delta_pp | float | ML (gradient boosting) prediction |
| lower_90_pp, upper_90_pp | float | nominal 90% prediction interval for the ML prediction |
| baseline_persistence_pp | float | baseline: no change (always 0) |
| baseline_trend_pp | float | baseline: repeat the previous 30-day change |
| baseline_seasonal_pp | float | baseline: mean historical change for this station & calendar month (training data only) |
| *(optional)* storage_af | float | storage at origin |
| *(optional)* capacity_af | int | |
| *(optional)* storage_pct | float | storage at origin, % capacity |

Derived for display: predicted level at target date (% capacity) = `storage_pct + pred_delta_pp`; observed = `storage_pct + target_delta_pp`. Error = `pred_delta_pp − target_delta_pp`.
**Note:** consecutive weekly origins have overlapping 30-day windows, so errors are serially correlated; do not present each row as an independent trial.

### 3.4 `metrics.json` — evaluation summary

```jsonc
{
  "schema_version": "1.0",
  "data_mode": "real" | "SYNTHETIC_SAMPLE",
  "generated_utc": "2026-11-21T12:00:00Z",
  "ML": 2.31,                         // test MAE, pp of capacity
  "Persistence": 3.10,                // test MAE
  "30-day trend": 3.45,
  "Historical seasonal": 2.80,
  "interval_90_coverage": 0.87,       // fraction of test targets inside [lower_90, upper_90]
  "interval_90_halfwidth_pp": 4.1,    // symmetric half-width if constant (may become per-row later)
  "train_count": 600, "val_count": 150, "test_count": 291, "test_station_count": 3,
  "note": "free-text caveats — display verbatim",
  "split": {
    "train_target_end_exclusive": "2023-01-01",
    "val_start": "2023-01-01", "val_target_end_exclusive": "2024-01-01",
    "test_start": "2024-01-01", "test_target_end_inclusive": "2025-12-31",
    "horizon_days": 30, "weather_lag_days": 7, "sampling": "weekly (Mondays)"
  },
  "per_station": {
    "SHA": {"n": 97, "ML": 2.1, "Persistence": 3.0, "30-day trend": 3.3,
            "Historical seasonal": 2.6, "interval_90_coverage": 0.88}
  },
  "bootstrap": {
    "method": "moving-block bootstrap over weekly test dates (stations resampled jointly per date)",
    "n_boot": 2000, "block_weeks": 5,
    "best_baseline": "Historical seasonal",
    "mae_diff_ml_minus_best_baseline": {"estimate": -0.49, "ci95_low": -0.9, "ci95_high": -0.1}
  }
}
```
(Values above are illustrative only.) `mae_diff_ml_minus_best_baseline` < 0 means ML beat the best baseline; if the CI includes 0 the UI should say the difference is **not clearly distinguishable**. If ML loses, the UI must show that plainly.
Split dates are the current plan and may move after the data audit — always read them from this file, never hard-code.

### 3.5 `feature_catalog.json` — model inputs

`{"schema_version": "1.0", "features": [{"name", "description", "units", "source", "min_lag_days"}, ...]}` —
`min_lag_days` = how many days before the forecast origin the feature’s latest input ends. Current features: `storage_pct, prev7_pct, prev30_pct, prev60_pct, rain_7d_mm, rain_30d_mm, temp_30d_c, humidity_30d_pct, season_sin, season_cos` (+ station one-hot, not listed).

### 3.6 Additive extras present in real outputs (v1.0, optional for the UI)

- `stations.csv`: `exclusion_reason`, `operator`.
- `test_predictions.csv`: `pred_ridge_pp` (Ridge regression prediction).
- `metrics.json`:
  - `"Ridge"` (test MAE);
  - `"validation"` (validation MAEs, GBM grid, selected params, `best_baseline_on_validation`);
  - `bootstrap.mae_ci95` (95% CI for each model's MAE);
  - `bootstrap.mae_diff_ml_minus_best_baseline.share_boot_ml_better`.

Real-data caveat to surface in the UI: the ORO (Oroville) series has a reporting-basis break on 2024-07-02 (DWR storage-table change). ORO test windows spanning that date are excluded, so ORO has fewer test rows (88 vs 101). See `data/provenance.md`.

## 4. Feature status

| Status | Feature |
|---|---|
| **Implemented in starter, not yet verified on real data** | CDEC + NASA POWER fetch; weekly features; 3 baselines; gradient boosting; constant-width empirical 90% interval; `test_predictions.csv` + `metrics.json` core keys |
| **In development (this sprint, engineering)** | verified stations/capacities; `stations.csv`, `history_daily.csv`, `feature_catalog.json` outputs; `per_station` metrics; block-bootstrap CI on MAE difference; `schema_version`/`data_mode`/`split` keys; provenance doc |
| **Possible stretch (only if core validated)** | one satellite-derived scalar feature (e.g. surface water area). Would appear only as a new row in `feature_catalog.json`; no imagery files will be provided |
| **Not supported — do not show or promise** | CNN / image models, satellite imagery viewer, real-time or operational forecasts, forecasts beyond the data end date, flood/drought/water-safety warnings, reservoir siting |

## 5. File ownership (to avoid conflicts)

| Path | Owner | Others may… |
|---|---|---|
| `app.py`, `app/` (except `app/sample_data/`), `.streamlit/`, `docs/UI_*.md` other than this file, report/handoff drafts | **UI (Claude Web / Muse)** | engineering will not edit |
| `src/`, `tests/`, `data/`, `outputs/`, `models/`, `stations.csv`, `docs/UI_CONTRACT.md`, `app/sample_data/`, `data/provenance.md` | **Engineering (Claude Code)** | UI reads only; request changes in `STATUS.md` |
| `requirements.txt` | Engineering | UI asks in `STATUS.md` to add a package |
| `README.md` | Engineering (sections 1–7); UI may add a “Demo app” section | |
| `STATUS.md` | shared | append under your own heading only |

Do not create `app/__init__.py` (it would shadow `app.py` as an import name). Work on a separate branch (e.g. `ui`) and merge via PR.

## 6. Quick start for the UI

```bash
python -m src.make_sample_data            # (re)generate synthetic fixtures
WATERSHED_DATA_DIR=app/sample_data streamlit run app.py
python -m src.contract app/sample_data    # validate a data dir against this contract
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
```

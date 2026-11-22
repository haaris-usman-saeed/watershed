# WatershedShift v3 preregistration: Models A, B, B2 and C

Committed before any v3 model is scored on the 2026 test period. The guard in `src/v3_experiment.py --final` refuses to run unless:
- this file and `config/v3_plan.json` are committed and unmodified;
- the SHA-256 of every listed v3 code file equals the value in `config/v3_plan.json`;
- no earlier final run record exists.

## What was seen before this plan

- **2024-2025 (the v2 test period):** model performance there has been seen, so it is **not** an independent test in v3. In v3 it is part of the training and development data and is reported only as historical context.
- **2026 observations:** CDEC storage and snow observations from January to November 2026 were downloaded on 2026-11-22. Their last 180 days were plotted on the Forecast page. **No model's 2026 errors were computed.** The only 2026 model outputs are the six Model A forecasts issued on 2026-11-22, whose targets lie after the test window.

## Target, origins and data

- **Target:** `target_delta_pp`, the 30-day change in storage in percentage points of the capacity in force at the origin. It is the exact frozen v2 definition (`src.features.construct_features`).
- **Origins:** Monday origins. The storage rules are those of v2: the capacity history, series breaks and spike review.
- **Storage and snow:** processed v2 data through 2025-12-31, plus the CDEC download of 2026-11-22 (`data/live/2026-11-22/`) for 2026. That download reproduced the archived 2025 storage on all 2,190 station-days.
- **Weather (B):** NASA POWER at the dam point, downloaded fresh in v3 for 2001-01-01 to 2026-11-21. It has the v2 feature definitions and the v2 7-day lag.
- **Weather (B2):** NASA POWER averaged over grid cells inside the upstream box (`src/v3_basins.py`), with the same definitions and lag.
- **Forecasts (C):** the GEFSv12 control member, 00Z on the origin Monday, averaged over the upstream box (`src/v3_gefs.py`). Reforecast for 2001-2019, operational from 2020-09-24.
  - An operational cycle is used only if every file's S3 publication time is no later than the end of the origin Monday in Pacific Standard Time (init + 32 h), the earliest moment a Monday-origin forecast can be issued.
  - Later uploads are archive backfills with no evidence of real-time availability, and they are excluded: 4 cycles (2020-10-19, 2020-12-14, 2021-02-22, 2021-03-01). This rule was set from publication metadata before any v3 model was fitted.
- **Row set:** rows produced by `construct_features` on the combined daily data (the frozen row rule requires storage, dam-point weather and the target).

## Models (all fixed; no tuning in v3)

Every ML model is `HistGradientBoostingRegressor` with the frozen v2 hyperparameters: learning_rate 0.03, max_iter 150, max_depth None, min_samples_leaf 30, l2_regularization 1.0, random_state 42. Each model uses its listed inputs plus the six station one-hot columns.

| Model | Inputs |
|---|---|
| A (v3 refit) | storage_pct, prev7_pct, prev30_pct, prev60_pct, swe_basin_in, swe_change_14d_in, swe_anom_in, season_sin, season_cos |
| B | A + rain_7d_mm, rain_30d_mm, temp_30d_c, humidity_30d_pct (dam point, 7-day lag) |
| B2 | A + the same four computed from upstream-box means |
| C | A + gefs_p_d1_7_mm, gefs_p_d8_16_mm |
| A (frozen) | the unchanged v2 artifact (trained on targets before 2024-01-01). A reference only; not refit. |

Baselines are recomputed from the same training rows: persistence (0), monthly seasonal, and smoothed day-of-year seasonal (sigma 15 days).

- **Snow anomaly climatology:** built from snow-index days before the training cutoff of each fold or run.
- **Model C rows:** Model C is trained and evaluated only on rows whose GEFS cycle is complete. Rows between 2020-01-01 and 2020-09-23 have no GEFSv12 product, so Model C is unavailable there.

## Development folds (reporting and interval calibration only)

- **Folds:** expanding window by water year, WY2008-2025. Fold W trains on targets before Oct 1 of W-1 and evaluates origins in W (the v2 rule).
- **No selection:** no model, feature or hyperparameter is chosen from these folds.
- **Reporting:** paired MAE (A vs B, A vs B2, B vs B2 on common rows; A vs C on rows where C is available), plus results by station, season and water year.
- **Not independent:** WY2024 and WY2025 are part of these folds but are not independent (see above).

## Prediction intervals (model-specific)

For each model, the 90% interval uses the `sym_season` method: the 90th percentile (method "higher") of that model's own absolute out-of-fold residuals from the development folds, computed per origin season. If a season has fewer than 150 residuals, the global value is used. Model A's bands are never reused for B, B2 or C.

## Final test (run once)

- **Training:** all rows with `target_date < 2026-01-01`.
- **Test:** origins from 2026-01-05 with `target_date <= 2026-11-20`, the last storage date in the frozen download.
- **Hypotheses:**
  - **H1 (primary):** Model C has lower MAE than Model A (v3 refit), on test rows where both are available.
  - **H2:** Model B has lower MAE than Model A.
  - **H3:** Model B2 has lower MAE than Model B.
- **Decision rule:** a hypothesis is "supported" only if the 95% moving-block bootstrap CI of the paired MAE difference is entirely below zero for **all three** block lengths (5, 9 and 13 weeks; 2,000 resamples; seed 42; reservoirs resampled jointly by date). Otherwise it is "not established". No winner is declared from point estimates alone.
- **Also reported:**
  - each model against the smoothed day-of-year baseline and against frozen Model A;
  - interval coverage and mean width per model, overall and per station (Folsom included);
  - availability per model;
  - MAE per station and per season.
- **Expected power:** about 42 weekly origins (about 250 rows) will give wide intervals, so "not established" is a likely, honest outcome.

## Use of the final models on the website

- The website serves **Model A (frozen v2 artifact)** together with **B, B2 and C as trained for the final test** (targets before 2026-01-01).
- The training cutoff of each model is displayed next to it.
- No model is retrained on the 2026 test outcomes in v3.

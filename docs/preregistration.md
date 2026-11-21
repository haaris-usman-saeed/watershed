# WatershedShift v2 preregistration (frozen before the held-out evaluation)

Date: 2026-11-21. Branch `review-v2`. Base commit before this file: `8f28f0c`.

This file fully specifies the single v2 evaluation on the 2024-2025 held-out period. It is committed **before** `python -m src.train_v2 --final` is run. The script refuses to run unless this file is committed and unmodified, and unless the SHA-256 hashes below match the configuration and code exactly.

## 0. Disclosure: the test period is not fully independent

The 2024-2025 test period was already used to evaluate v1 (`git tag v1-results`), and the v1 test results were seen before v2 was designed. The v2 changes were motivated by an independent review and by development-data findings, not by v2 test results. Even so, **the v2 configuration was developed after observing v1 test performance on this same period, so the period is not completely independent of the revision process.**

v2 is evaluated on it exactly once. No v2 setting will be changed after its test metrics are seen. A third configuration will not be evaluated on 2024-2025 without explicit approval from the project owner.

## 1. Frozen files

| file | SHA-256 |
|---|---|
| `config/v2_frozen.json` | `02cfcfd007b0513a50d00b60354c31d102d4452f381dfcce329f692e866c230c` |
| `src/train_v2.py` | `3d2f6b0730255c2ad6dfce1ec2e0988f65f48c222535ba17a786e679467c3149` |
| `src/models.py` | `416210cfc76d1908feff580dc18929b22fc45dac578eaa35fa6112a4c29ade97` |
| `src/features.py` | `68912938f3512b719c2d448d276a3faa532b0629aea007688c553eb686afda3a` |
| `src/evaluation.py` | `da29e5683c29c18b6831d603b468f3309da22cc21ccf01457db6ba417761a01d` |
| `src/build_dataset.py` | `b40a6d9b1fd3bcb77688a727d0ed1e2669a06f150889729fb37f31d42b94125c` |

The raw inputs are those listed in `data/raw_manifest.jsonl` (188 downloads with SHA-256).

## 2. Dataset period and eligibility

- **Period:** 2001-01-01 to 2025-12-31 (daily). The first usable weekly origin is 2001-03-05, after the 60-day feature history.
- **Reservoirs:** SHA, ORO, FOL, NML, DNP, EXC.
- **Eligibility rule:** `status == included` in `data/processed/coverage.csv`, which requires at most 10% of storage days missing. All six are included (maximum 0.43%). Excluded stations cannot enter feature construction (`src.features.filter_eligible`).
- **Training rows:** every weekly row with target_date < 2024-01-01 (7,076 rows).
- **Test rows:** origin >= 2024-01-01 and target_date <= 2025-12-31. This is 593 rows: 101 for each reservoir except ORO, which has 88 because windows spanning 2024-07-02 are excluded. It is the same test period as v1.

## 3. Data-cleaning rules (`src/build_dataset.py`)

1. The CDEC daily date is taken from `DATE TIME` (end-of-day storage). `OBS DATE` is ignored.
2. Rows must have sensor 15, duration D and units AF, or the build fails.
3. Non-numeric values stay missing. Storage is never interpolated or filled.
4. Values <= 0 or > 110% of the capacity in force are set to missing.
5. Identical duplicate dates collapse to one row. Conflicting duplicates become missing.
6. One-day reversing jumps > 2 pp are adjudicated in `data/spike_review.csv` against the elevation sensor. Artifacts and unreviewed candidates become missing; physical extremes are kept.
7. Capacity denominators follow `data/capacity_history.csv`: ORO is 3,537,577 AF through 2024-07-01 and 3,424,753 AF from 2024-07-02. Targets and change features use the capacity in force at the origin.
8. No target or storage-change feature window may span a series break in `data/series_breaks.csv` (ORO 2024-07-02).
9. NASA POWER `-999` and implausible values become missing.
10. Snow pillow (sensor 82) handling:
    - Values below -0.5 in or above 250 in become missing.
    - Values from -0.5 to 0 in become 0.
    - Missing is never zero.
    - Core pillows: at least 75% of WY2002-2023 snow seasons with at least 80% daily availability.
    - Basin index: mean of reporting core pillows when at least 2/3 of them report; otherwise missing.

## 4. Features and target

- **Target:** y = 100 x (storage[t+30] - storage[t]) / capacity in force at t. Exact calendar-day match. Origins t are Mondays.
- **Primary model features:**
  - `storage_pct`, `prev7_pct`, `prev30_pct`, `prev60_pct` (storage up to and including t);
  - `swe_basin_in` (t-1), `swe_change_14d_in` (t-1 minus t-15), `swe_anom_in` (t-1 index minus a smoothed day-of-year climatology computed from snow index dates < 2024-01-01, Gaussian sigma 10 days);
  - `season_sin`, `season_cos`;
  - station one-hot.
- NASA POWER weather is **not** an input of the primary model. This follows the predetermined development rule below.

## 5. Models (all fit on the same training rows)

| name | definition |
|---|---|
| **Primary: HGB SWE no weather** | `HistGradientBoostingRegressor(learning_rate=0.03, max_iter=150, max_depth=None, min_samples_leaf=30, l2_regularization=1.0, random_state=42)`, primary features above |
| HGB (v1 features) | same estimator; storage + weather + season + station |
| HGB + SWE | same estimator; storage + weather + SWE + season + station |
| HGB no weather | same estimator; storage + season + station |
| Ridge | StandardScaler + RidgeCV(alphas = logspace(-2, 3, 20)); v1 features |

**Why these settings.** They come from development data only (`results/v2/development/report.md`):
- **Hyper-parameters:** chosen per variant from a predetermined 8-point grid by pooled MAE on rolling tuning folds WY2008-2015.
- **Primary variant:** the predetermined rule was "the lowest pooled MAE on confirmation folds WY2016-2023 among HGB variants". HGB SWE no weather scored 3.160 pp, HGB + SWE 3.168, HGB (v1 features) 3.231 and HGB no weather 3.300. The top two are effectively tied, and the rule is applied as written.

## 6. Baselines (fit on the same training rows as the models)

1. **Persistence:** 0.
2. **30-day trend:** `prev30_pct`.
3. **Monthly seasonal:** mean training target by station and origin month. This is the v1 "Historical seasonal".
4. **Smoothed DOY seasonal:** per station, a Gaussian-kernel (sigma 15 days, circular) weighted mean of training targets by origin day of year.

The **strongest baseline is fixed in advance as Smoothed DOY seasonal**, because it had the lowest pooled MAE on confirmation folds WY2016-2023 (4.108 vs monthly seasonal 4.209).

## 7. Prediction interval

- **Method:** `sym_season`. The half-width is the 90th percentile (numpy `method="higher"`) of absolute out-of-fold residuals of the primary model from development folds WY2008-2023, computed separately for each origin season (DJF, MAM, JJA, SON), with a global fallback if a group has fewer than 150 residuals.
- **Why this method:** on development folds WY2012-2023, each calibrated only on earlier folds, it was the narrowest method with pooled coverage in [0.87, 0.93]. It achieved coverage 0.888 and mean width 12.2 pp.
- **Known development weakness:** per-reservoir coverage was uneven (FOL 0.75, NML 0.98).
- **Caveat:** nominal 90% coverage is **not guaranteed**. Errors are serially correlated, and the climate is non-stationary.

## 8. Metrics and uncertainty (all reported whatever their direction)

- MAE in pp of capacity: overall, by reservoir, by origin season and by water year (WY2024, WY2025, WY2026 partial).
- Interval coverage and mean width: overall, by reservoir and by season.
- **Primary comparison:** primary MAE minus Smoothed DOY seasonal MAE.
- **Secondary comparisons:**
  - primary minus Monthly seasonal (the v1 comparator);
  - paired v2 primary vs v1 GBM, on test rows whose target definition is identical in both versions. That is all rows except ORO origins before 2024-07-02, where v1 used the post-2024 capacity.
- **Bootstrap:** moving-block bootstrap over weekly origin dates, with all reservoirs on a date resampled jointly. Block lengths are 5, 9 and 13 weeks (fixed in advance); 2,000 replicates; seed 42; two-sided 95% percentile CI. "Statistically distinguishable" will be claimed only if the CI excludes 0 at all three block lengths.
- **Development expectation, stated in advance:** on confirmation folds the primary model beat Smoothed DOY seasonal by 0.95 pp (95% CI about [-1.28, -0.63]). A smaller or reversed difference on 2024-2025 will be reported as observed.

## 9. Outputs

- `results/v2/final/metrics.json`, `results/v2/final/test_predictions_public.csv` (no raw storage columns) and `results/v2/final/RUN_RECORD.json`.
- The five UI contract files in `outputs/` (schema v1.0, additive fields only).
- v1 results remain in `results/v1/` unchanged.

## 10. What happens next

- **If v2 performs worse than v1 or the baselines:** the outcome is documented as is. Possible reasons are investigated using development data only, and no v2 retuning happens on the test period.
- **After the run:** an identical-configuration rerun is allowed only to verify reproducibility, and it must reproduce the same numbers.

# 30-day research forecast: feasibility and implementation notes

Written before any forecasting code was added (branch `feature/30-day-forecast`, from `main` at `d72cd68`), and updated as the work progressed. The frozen v2 evaluation is not touched.

## 1. Feasibility assessment (Phase 1 audit)

| Question | Finding |
|---|---|
| Where does the trained model exist? | Nowhere. `src/train_v2.py` fits the primary HGB inside `run()` and discards it after predicting the 593 test rows. `models/forecast.joblib` is an older v1 artifact (git-ignored) and is **not** the v2 primary model. |
| Are model weights persisted? | No. Only predictions (`results/v2/final/test_predictions_public.csv`) and metrics are archived. |
| How can the original HGB be reconstructed? | Exactly as `train_v2.run("final")` does: `train_v2.load(2024-01-01, 2024-01-01, 2025-12-31)` gives the 7,076 training rows (target date before 2024-01-01) and the snow anomaly from `snow_climatology(snow_index, 2024-01-01)`; then `hgb(hp).fit(design(train, FEATURE_SETS["storage_snow"], stations), y)` with the frozen hyperparameters. HGB is deterministic here (`random_state=42`; early stopping is off below 10,000 rows, `n_iter_ = 150`). **Checked during the audit: the refit reproduces all 593 archived test predictions to a maximum absolute difference of 1.8e-15 pp.** |
| Which training data are required? | The private processed files `data/processed/weekly_features.csv` and `data/processed/snow_basin_index.csv` (built from raw CDEC files by `src.build_dataset` and `src.features`), plus `data/processed/snow_daily.csv` for the core-pillow rule. The interval needs the tracked `results/v2/development/oof_predictions.csv`. None of these may be committed (CDEC terms unresolved). |
| How are features built without a target? | `src/features.construct_features` drops rows without storage at t+30, so it cannot score a live origin. A separate builder (`src/forecast_live.py`) computes the same 15 inputs from storage up to t and the snow index up to t-1 only, with the same capacity, series-break, anomaly, season and one-hot rules. It is tested for equality with the historical builder on every 2024-2025 test origin. |
| Which recent CDEC products exist? | Checked on 2026-11-22. **Sensor 15 (daily storage):** values through 2026-11-20 for the stations probed, with `---` for the two most recent days (reporting lag of 1 to 2 days). **Sensor 82 (daily adjusted SWE):** values through 2026-11-21 for 41 of the 51 candidate pillows. |
| Can recent sensor 82 be treated as available at forecast time? | **Partly, with a disclosed caveat.** Sensor 82 is a revised product. Recent values exist one day after the fact, but in November 2026 only about half carry the `r` (revised) flag; the rest are blank or `o`. The training data are mostly fully revised values. So a live forecast uses SWE that may still be revised, which is a distribution shift the backtest did not measure. This is reported with every forecast. Early-season SWE (November) is small, so the effect is likely limited now and larger in winter and spring. |
| Model inputs | Exactly the frozen 15 columns: `storage_pct, prev7_pct, prev30_pct, prev60_pct, swe_basin_in, swe_change_14d_in, swe_anom_in, season_sin, season_cos, station_DNP, station_EXC, station_FOL, station_NML, station_ORO, station_SHA`. NASA POWER weather is **not** an input and is not fetched. (Training rows were filtered on weather availability by `construct_features`; that is a row filter on the frozen training set, not an input, so inference does not need weather.) |
| What runs where? | **Locally (trusted, private):** model build and verification, CDEC refresh, feature building, scoring, snapshot export. **Streamlit:** read-only rendering of a validated JSON snapshot. It never fits, never unpickles a model and never downloads CDEC data. |
| Deployment | The snapshot contains CDEC-derived values, so it cannot be committed or put on a public deployment while CDEC terms are unresolved. A local or private deployment works today. A public deployment shows the forecast page only as an explained "no snapshot" state, or with the committed synthetic fixture under a synthetic-data banner. See section 4. |

**Conclusion: feasible with the original model.** The model is reconstructed bit-for-bit, current storage and SWE observations exist, and no input needs to be invented or substituted. The remaining limits are scientific (SWE revision status, no prospective validation, Folsom under-coverage) and legal (CDEC redistribution), not technical.

## 2. Design

```
python -m src.forecast_model --build      # trusted: refit, verify vs archive, write models/v2_forecast/
python -m src.forecast_live               # refresh CDEC, build features, score, write forecasts/
streamlit run app.py                      # page "30-Day Forecast" reads forecasts/latest.json
```

- `src/forecast_model.py`: rebuilds the frozen primary model and its fixed companions (core snow pillows, snow climatology, season interval offsets, per-station historical coverage). Refuses to write the artifact unless the refit reproduces every archived 2024-2025 prediction within 1e-9 pp. Writes `models/v2_forecast/model.joblib` and `model_card.json` (training cutoff, feature order, config hash, source commit, data provenance, creation time, artifact SHA-256). Loading checks the SHA-256 against the card first.
- `src/forecast_live.py`: refresh, inference-time features, scoring and the snapshot. Raw responses go to `data/live/` with their own fetch log, never into `data/raw/` or the tracked raw manifest. Cleaning reuses `src.build_dataset.load_cdec` and `load_snow` unchanged.
- `src/forecast_schema.py`: pure-Python snapshot validation, shared by the pipeline (before writing) and the page (before displaying).
- `app/view_forecast.py`: the fifth page.

## 3. Rules fixed in advance (not tuned on any outcome)

- **Origin:** the latest Monday t on or before the issue date minus one day (end-of-day storage must be complete) with a valid storage value. Earlier Mondays are tried only within 14 days of the issue date; an origin more than 7 days old is flagged stale; none within 14 days means "Forecast unavailable".
- **Target:** exactly t + 30 calendar days, and it must be after the issue date.
- **Completeness:** all 15 inputs must be present. HGB could score missing SWE, but the forecast is refused instead.
- **Plausibility:** projections are never clipped. A projected storage below 0% or above 100% of documented capacity is flagged as unsuitable for operational interpretation.
- **Interval:** the frozen `sym_season` 90% offsets from the development out-of-fold residuals (WY2008-2023). Retrospectively calibrated; future coverage unknown; Folsom reached 0.75 historically.

## 4. Deployment

See README section "Generating 30-Day Research Forecasts". In short: build and refresh locally, then run Streamlit locally or on a private host with `WATERSHED_FORECAST_FILE` pointing at the snapshot. A public deployment must not receive the snapshot until the owner confirms CDEC redistribution terms; until then it can show only the documented unavailable state.

## 5. Implementation log

| Step | Result |
|---|---|
| Audit | No persisted v2 model. Exact refit confirmed: 7,076 training rows, 150 iterations, 593/593 predictions reproduced (1.8e-15 pp). |
| Live availability probe (2026-11-22) | Sensor 15 through 2026-11-20. Sensor 82 through 2026-11-21 for 41 of 51 pillows. November 2026 sensor 82 flags: about half `r`, the rest blank or `o`. |
| `src/forecast_model.py` | Builds and verifies the artifact plus its card. `--verify` re-checks the hash and the predictions. |
| `src/forecast_live.py` | Refresh with retries and timeouts, header validation, a same-day cache, and a fetch log with SHA-256. Cleaning reuses `src.build_dataset`. Builds the basin index from the frozen core pillows, then features, scoring, intervals, plausibility and freshness flags, then a versioned record. |
| `src/forecast_schema.py` | Validation shared by the pipeline and the page. Checks: arithmetic consistency, origin + 30, Monday origin, target after issue, the frozen feature order, no weather input, the interval and Folsom disclosures, `is_heldout_test_prediction: false`, and expiry. |
| First real run | Five forecasts from origin 2026-11-16 and one stale (DNP, origin 2026-11-09). All source hashes verified. The 2025 storage overlap was unchanged. |
| `app/view_forecast.py` | Cards, chart (observed up to t solid, later observations dotted and unused, forecast dashed, 90% wedge plus error bar at day 30), snowpack inputs, freshness and provenance facts, limitations, and a JSON download. |
| Problems fixed during build | The chart window was too wide to see the forecast, so it is now 90 days before the origin. The `?station=` link was ignored, so the page now uses the shared station state. The capacity line flattened the chart, so it is shown only near the data. |

## 6. Final checklist

- [x] Frozen code, config, preregistration and results unchanged (checked with `git diff` against `e47deab`, `f21bf25` and `main`, and against the `RUN_RECORD.json` hashes)
- [x] No new model and no retraining on 2024-2025 outcomes; the refit equals the archived predictions
- [x] Exactly the 15 frozen inputs; no weather
- [x] Inference never reads t+30 storage, storage after t, or SWE after t-1 (tests)
- [x] Missing inputs refuse the forecast; no interpolation and no fallback to old forecasts
- [x] Target = origin + 30 days and after the issue date (tests and validation)
- [x] pp to acre-feet uses the capacity in force on the origin date (tests)
- [x] Intervals: frozen `sym_season` offsets, retrospective, Folsom warning retained (tests)
- [x] Records are marked as prospective research forecasts, never held-out test rows (tests and validation)
- [x] Real forecast produced from live CDEC data on 2026-11-22
- [x] Page renders real, synthetic, missing, invalid and expired states (AppTest and browser)
- [x] No raw, derived, model or forecast files committed (`.gitignore` and a CI check)
- [ ] Public deployment of real forecasts: blocked on CDEC terms (owner)
- [ ] Prospective validation: first targets on 2026-12-09 and 2026-12-16

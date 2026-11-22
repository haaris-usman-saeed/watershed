# WatershedShift: 30-day reservoir storage change for six California reservoirs (historical backtest)

WatershedShift is an independently assembled, quality-audited joint dataset for SHA, ORO, FOL, NML, DNP and EXC, covering 2001 to 2025. It combines three sources:
- **CDEC daily measured reservoir storage**;
- **CDEC snow-pillow snow water equivalent**;
- **NASA POWER daily weather**.

The models forecast the **30-day change in storage, in percentage points (pp) of the capacity in force**, using only information available at the forecast origin. They are evaluated against strong baselines with rolling water-year folds and one preregistered held-out run (2024 to 2025).

> This is a research prototype and historical backtest. It is **not** an operational forecast, and not a flood, drought or drinking-water safety tool.

## Where things are

| What | Where |
|---|---|
| Status and checkpoint log | `STATUS.md` |
| Data provenance, verification, cleaning rules | `data/provenance.md`, `data/coverage_table.md` |
| Frozen v1 results (2018-2025 data) | `results/v1/` (git tag `v1-results`) |
| v2 development evaluation (rolling water years) | `results/v2/development/report.md` |
| v2 preregistration (committed before the test run) | `docs/preregistration.md`, `config/v2_frozen.json` |
| v2 held-out results | `results/v2/final/metrics.json`, `results/v2/final/RUN_RECORD.json` |
| UI and output schema | `docs/UI_CONTRACT.md` |
| Relation to prior work | `docs/originality_check.md` |
| Guide for assessors (bundle, checksums, frozen rerun) | `ASSESSORS.md` |
| Source-linked report facts | `results/v2/report_facts.md`, `results/v2/report_facts.json` |
| Reproducibility record | `docs/reproducibility.md` |
| Merge plan for the owner | `docs/merge_plan.md` |
| 30-day research forecasts (experimental) | section 10 below, `docs/FORECAST_IMPLEMENTATION.md` |

## 1. Install (verified on macOS, Python 3.14.5)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # exact versions of the verified run: requirements-lock.txt
```

## 2. Reproduce end to end

Internet access is needed for steps 1 to 3. The whole run takes about 10 minutes.

```bash
python -m src.fetch --start 2001-01-01 --end 2025-12-31        # 1. CDEC storage + NASA POWER -> data/raw/ (unmodified bytes)
python -m src.fetch_snow --start 2001-01-01 --end 2025-12-31   # 2. CDEC snow pillows -> data/raw/snow/
python -m src.fetch_wsihist                                    # 3. DWR water year types (descriptive grouping only)
# optional verification: python -m src.fetch_resopsus && python -m src.compare_resopsus  (299 MB download)
python -m src.build_dataset                                    # 4. offline raw -> data/processed/ + audit + outputs/stations.csv, history_daily.csv
python -m src.features                                         # 5. weekly table + snow basin index
python -m src.experiments                                      # 6. rolling water-year development evaluation (pre-2024 data only)
python -m src.dev_report                                       # 7. results/v2/development/report.md
python -m src.train_v2 --final                                 # 8. preregistered v2 held-out run -> results/v2/final/, outputs/
python -m pytest -q tests                                      # 9. tests
python -m src.contract outputs                                 # 10. validate UI contract
python -m src.make_bundle                                      # 11. deterministic private bundle -> dist/
python -m src.make_bundle --verify dist/watershedshift_v2_bundle.tar.gz
python -m src.ui_export                                        # 12. additive UI metadata + dist/outputs_v2_ui.zip (local only)
python -m src.report_facts --check                             # 13. report facts match the evidence
```

Notes on these steps:
- **Step 8** refuses to run unless `docs/preregistration.md` is committed and the hashes of the config and code match it. Re-running it with the identical frozen configuration is allowed, to check reproducibility. A different configuration needs explicit owner approval.
- **Step 12** must be re-run after any rerun of step 8, because step 8 rewrites `outputs/` without the additive handoff keys.
- **Raw data stays on the local machine** (git-ignored) because the repository is public and CDEC redistribution terms are unconfirmed. `data/raw_manifest.jsonl` (tracked) lists every download with its URL, retrieval time and SHA-256.
- **CDEC flakiness:** CDEC sometimes returns empty responses. The fetchers retry and cache completed requests, so they can be re-run safely.

## 3. Data sources (verified 2026-11-21)

| Source | What | Link |
|---|---|---|
| CDEC (CA Dept of Water Resources) | Daily reservoir storage, sensor 15, acre-feet, end-of-day | https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet |
| CDEC snow pillows | Daily snow water content, sensor 82 (adjusted), inches; 51 candidate pillows in the upstream basins, 3 to 11 core pillows per reservoir | same endpoint; station lists from https://cdec.water.ca.gov/dynamicapp/staSearch |
| CDEC metadata and capacities | Coordinates, operators, sensors; Daily Reservoir Storage Summary | https://cdec.water.ca.gov/dynamicapp/staMeta?station_id=SHA , https://cdec.water.ca.gov/reportapp/javareports?name=RES |
| DWR water year types | Sacramento and San Joaquin Valley indices (descriptive only) | https://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST |
| NASA POWER daily point API v2.10.5 | T2M, PRECTOTCORR, RH2M (local solar time) | https://power.larc.nasa.gov/docs/services/api/temporal/daily/ |

Key verification findings:
- The CDEC `DATE TIME` field is end-of-day storage. `OBS DATE` is operator dependent and was the source of a one-day shift in the starter code.
- Oroville's capacity changed from 3,537,577 to 3,424,753 AF when DWR adopted a new storage table on 2024-07-01. The daily CDEC series switches on 2024-07-02. v2 uses the capacity in force on each date, and no window spans the change.
- Spike candidates were adjudicated against the independent elevation sensor. Two artifacts were removed; six flood-day extremes at Folsom were confirmed and kept.
- **ResOpsUS cross-check (Steyaert et al. 2022):** partly independent, because four of its six storage records (SHA, ORO, NML, EXC) come from CDEC. Details are in `results/v2/resopsus_check/report.md`.
  - The two independent sources agree with our dates: FOL (Bureau of Reclamation) is 99.6% identical on the same dates, and DNP (USGS gauge) agrees within a median of 0.04 pp.
  - ResOpsUS's Oroville record is predominantly offset by one day relative to our end-of-day dates.
  - For McClure (EXC), the offset changes across periods: it is aligned at zero lag in about April 2003 to May 2014, and one day off in 2001 to early 2003 and from July 2014.
  - Both match CDEC `OBS DATE` keying. The authors' intended date convention is unknown, so we do not claim an error.
  - The offsets matter mainly for daily-resolution analysis.

### Attribution and terms

**CDEC (wording PENDING OWNER APPROVAL):**
> Reservoir storage and snow observations were obtained from the California Department of Water Resources, California Data Exchange Center (CDEC), retrieved November 21, 2026. Observations may be provisional or revised. CDEC/DWR does not endorse this research prototype. Raw CDEC records are not redistributed in this public repository.

No specific reuse licence for CDEC data has been verified. Do not redistribute raw or CDEC-derived records (including the private bundle and `outputs_v2_ui.zip`) until the owner confirms the terms.

**NASA POWER** (requested acknowledgement):
> These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program.

**ResOpsUS** (used only to verify storage, never as a model input):
- Citation: Steyaert, Condon, Turner and Voisin (2022), *Scientific Data*, DOI: 10.1038/s41597-022-01134-7.
- The release used is Zenodo record 5893641, licensed CC BY 4.0. It is kept locally and not redistributed here.

## 4. Method (v2)

- **Rows and target:** reservoir x Monday origin t. Target = 100 x (storage[t+30] - storage[t]) / capacity in force at t, with exact day matching. Targets are never interpolated.
- **Features and prediction-time rules:**
  - storage level and 7/30/60-day changes, up to t;
  - snow (basin SWE index, 14-day change and anomaly), up to t-1;
  - NASA POWER weather (7- and 30-day aggregates), ending at t-7;
  - season terms and a station one-hot.
  All rules are tested by perturbing future data.
- **Models and baselines:**
  - baselines: persistence; 30-day trend; monthly seasonal; smoothed day-of-year seasonal;
  - Ridge;
  - four HistGradientBoosting variants: v1 features, no weather, + SWE, SWE without weather.
- **Development evaluation:** 16 expanding-window water-year folds (WY2008 to WY2023). Hyper-parameters were tuned on WY2008-2015, and comparison and selection used WY2016-2023.
- **Interval:** season-conditional 90% interval from rolling out-of-fold residuals.
- **Uncertainty:** moving-block bootstrap over weekly dates (5, 9 and 13 weeks; reservoirs resampled jointly).
- **Final test:** one preregistered run on 2024-2025.

## 5. Results

### Held-out 2024-2025 (n = 593)

| Model | v1 (data 2018-2025) | v2 (data 2001-2025) |
|---|---|---|
| Primary ML | GBM, storage + weather: **2.81** | HGB, storage + SWE: **2.33** |
| Smoothed DOY seasonal | n/a | 2.97 (strongest baseline, chosen on development data) |
| Monthly seasonal | 3.16 | 2.98 |
| 30-day trend | 4.80 | 4.79 |
| Persistence | 5.79 | 5.78 |
| 90% interval coverage, mean width | 96.3%, 20.0 pp | 92.1%, 13.1 pp |

- **v2 ML minus strongest baseline: -0.64 pp.** 95% CI [-1.02, -0.35] (5-week blocks), [-1.07, -0.34] (9), [-1.10, -0.34] (13).
- **v1 ML minus its best baseline:** -0.34 pp, 95% CI [-0.80, +0.03] (5 weeks); [-0.90, +0.11] (9); [-0.93, +0.11] (13).
- **Paired v2 vs v1 ML** on 571 rows with identical targets: -0.47 pp, CI [-0.68, -0.29] (5 weeks), [-0.69, -0.27] (9), [-0.67, -0.26] (13).
- **Folsom interval under-coverage:** only about 75% of Folsom's 2024-2025 outcomes fell inside the nominal 90% interval (coverage 0.75). The other five reservoirs range from 0.91 to 0.99. The UI export marks FOL `interval_reliable: false`, which is a retrospective warning, not a recalibration.

**Test-period disclosure.** The 2024-2025 period was evaluated for v1 and subsequently for a preregistered v2 configuration. The v2 model was selected using pre-2024 development data, but the original test results had already been examined when the scientific revisions were planned. The v2 evaluation therefore provides additional evidence rather than a fully independent replication. See `docs/preregistration.md`.

**Primary model.**
- The v2 primary model ("HGB SWE no weather", chosen by the preregistered rule) uses reservoir history, reservoir identity, seasonality and CDEC SWE. The exact columns are in `results/v2/report_facts.md`, fact 7.
- NASA POWER weather is included in the assembled dataset and in the secondary model variants, but not in the selected primary model.
- The longer, corrected historical training record accounts for most of the improvement relative to v1. A secondary HGB using the v1 feature set but the 2001+ training data scored 2.32 on the same test.
- The independent predictive value of adding SWE or weather is not conclusively established. The development ablation CIs include 0.
- That secondary HGB (2.32) and HGB + SWE (2.32) are marginally better than the primary (2.33). The primary model was not reselected after the test.

### Development, confirmation folds WY2016-2023 (n = 2,500)

| Model | MAE (pp) |
|---|---|
| HGB SWE no weather | 3.16 |
| HGB + SWE | 3.17 |
| HGB (v1 features) | 3.23 |
| HGB no weather | 3.30 |
| Ridge | 3.90 |
| Smoothed DOY | 4.11 |
| Monthly seasonal | 4.21 |

- The ML models beat the strongest baseline in 15 of 16 water years.
- The added value of weather (-0.07) and of SWE given weather (-0.06) is small, and their CIs include 0.

## 6. Limitations

- **Retrospective data:**
  - CDEC values are provisional and revised;
  - snow sensor 82 is an adjusted, after-the-fact product;
  - NASA POWER archives are MERRA-2, while real-time values are GEOS-IT.
  None of these are as-of-time archives.
- **Interval calibration:** intervals are miscalibrated per reservoir (FOL about 75% coverage, NML 98%). Nominal coverage is not guaranteed under serial correlation and climate non-stationarity.
- **Not modelled:** reservoir operations (releases, flood-control rules, allocations). CDEC inflow is a mass-balance quantity and was not added.
- **Scope:** six reservoirs and one two-year test period that was also used for v1.
- **Spatial meaning:** the snow index is a point index from pillows, not basin-volume SWE. NASA POWER is grid-cell weather, not basin rainfall.

## 7. Scope

The project covers tabular multisource forecasting only. **There is no satellite-image, CNN, operational-forecast or reservoir-siting capability.** The 30-day research forecast in section 10 is experimental and is not an operational forecast. Satellite features remain future research.

## Demo app

Run `streamlit run app.py` (owned by the UI team; see `docs/UI_CONTRACT.md`). Use `WATERSHED_DATA_DIR=app/sample_data` for the **synthetic** UI fixtures. The fifth page, **30-Day Forecast**, is described in section 10.

## 8. Suggested handoff

Send the repository or app link, together with the limitations above, to an appropriate water-resources group. Keep proof of sending. Do not imply that a regulator has endorsed the model.

## 9. Agent workflow

See `PROMPTS.md`. Engineering (Claude Code) owns `src/`, `tests/`, `data/`, `results/`, `config/`, `outputs/` and the engineering docs. The UI team (Claude Web / Muse) owns `app.py` and `app/`.

## 10. Generating 30-Day Research Forecasts

This adds new forecasts for future dates to the historical backtest. It uses the **frozen v2 primary model, unchanged**, and the latest eligible CDEC observations. It is an **experimental research forecast**, not operational water-management advice. Design notes and the feasibility audit are in `docs/FORECAST_IMPLEMENTATION.md`.

**What is predicted.** The change in reservoir **storage** over the next 30 days, in percentage points of the capacity in force on the origin date. It is converted to acre-feet and to projected storage. It is not a water-surface elevation: +5 pp means storage rises by 5% of documented capacity, not 5 feet.

**Observations required** (all from CDEC, no weather):
- daily storage (sensor 15) on the origin date t and on t-7, t-30 and t-60;
- the basin snow index (sensor 82, mean of the frozen core pillows, at least 2/3 reporting) on t-1 and t-15.

If any of these are missing, the forecast for that reservoir is refused. An older forecast is never substituted.

**How the model is obtained.** `python -m src.forecast_model --build` refits the frozen primary model:
- the same code path as `src.train_v2`;
- the same 7,076 training rows, with targets before 2024-01-01;
- the same hyperparameters, feature order and snow climatology.

It writes the artifact only if the refit reproduces all 593 archived 2024-2025 test predictions within 1e-9 pp. The measured difference is 1.8e-15 pp. No 2024-2025 outcome is used for fitting. The artifact goes to `models/v2_forecast/` (git-ignored), with a model card recording:
- training cutoff and feature order;
- config hash and source commit;
- SHA-256 of every training input;
- creation time and artifact SHA-256.

Loading checks the artifact hash against the card before anything is unpickled.

**How to generate a forecast** (needs the private processed data for the build step, and network access to cdec.water.ca.gov):

```bash
python -m src.forecast_model --build     # once; --verify re-checks it later
python -m src.forecast_live              # refresh CDEC, build features, score, write the snapshot
```

`src.forecast_live` does the following:
- downloads about two calendar years of storage and SWE into `data/live/<issue date>/`, with a fetch log (URL, time, SHA-256);
- cleans the data with the unchanged historical rules in `src.build_dataset`;
- builds the 15 inputs with a prediction-time feature builder that never needs storage at t+30;
- scores each reservoir;
- writes the record to `forecasts/records/<forecast id>.json`, which is never overwritten, and copies it to `forecasts/latest.json`.

Options:
- `--offline <dir>` rebuilds from cached downloads without network access;
- `--as-of` sets an earlier issue date, but never a future one.

**How the forecast date is chosen.** The origin is the latest Monday on or before the issue date minus one day that has all inputs. Mondays up to 14 days old are considered. An origin more than 7 days old is flagged stale. The target is exactly origin + 30 calendar days, and it must be after the issue date.

**How often it can be updated.**
- A new origin becomes available each week, once CDEC reports Monday's end-of-day storage, usually by Tuesday or Wednesday.
- Rerunning later in the same week keeps the origin but may pick up revised values; each run is a new record.

**What the interval means.**
- It uses the frozen `sym_season` 90% offsets: the 90th percentile of absolute out-of-fold errors from the development folds (WY2008-2023), for the origin's season.
- It is **retrospectively calibrated, and its future coverage is not established.**
- In the 2024-2025 backtest, Folsom intervals covered only about 75% of outcomes, and the Folsom forecast carries that warning.

**Why this is not an operational tool.**
- The model knows only recent storage, snowpack and season. It does not represent planned releases, flood-control rules, demands or precipitation forecasts.
- It has not been prospectively validated.
- Recent sensor 82 SWE values may still be revised, while training used mostly revised values.
- The capacity denominator is assumed unchanged through the target date.
- Projections outside 0-100% of capacity are shown unclipped and flagged.

**Viewing it.** `streamlit run app.py` and open **30-Day Forecast**. The page is read-only: it validates the snapshot with `src.forecast_schema` and displays only a valid, unexpired one. It never fits a model, loads weights or downloads data. It looks for the snapshot in this order:
1. `WATERSHED_FORECAST_FILE`;
2. `forecasts/latest.json`;
3. `forecast_snapshot.json` in the UI data directory.

The synthetic fixture in `app/sample_data/` (`python -m src.make_forecast_fixture`) is shown only under a synthetic-data banner.

**Deploying privately.**
- *Local or private host:* run the two commands above on an authorised machine and start Streamlit there. Point `WATERSHED_FORECAST_FILE` at the snapshot if it lives elsewhere. Re-run `src.forecast_live` weekly, for example from cron on Tuesday.
- *Public deployment (blocker):* the snapshot, the live downloads and the model artifact all contain CDEC-derived values and are git-ignored. While CDEC redistribution terms are unresolved, they must not be committed or uploaded to a public host. A public deployment built from this repository therefore shows the page's explained "No forecast snapshot" state, or the synthetic fixture under its banner. Enabling real forecasts publicly needs the owner to confirm CDEC terms first. It then needs a private snapshot source, such as a private bucket read through Streamlit secrets, which is **not implemented**.

**Real-world test (2026-11-22): succeeded.** `python -m src.forecast_live` was run against live CDEC data on 2026-11-22:
- Five reservoirs (SHA, ORO, FOL, NML, EXC) received forecasts from origin Monday 2026-11-16 to target 2026-12-16.
- Don Pedro (DNP) received a forecast from origin 2026-11-09, flagged stale, because fewer than 2 of its 3 core pillows reported after 2026-11-12.
- All 14 downloaded files matched their logged SHA-256.
- The re-downloaded 2025 storage was identical to the archived 2025 data on every station-day.

These numbers are private and are not reproduced here. They are new forecasts, not held-out test results, and they do not change any v2 result.

## 11. v3: weather-aware models and interactive forecasting

v3 adds three models alongside the unchanged frozen Model A. It also adds a **Generate Forecast** button that fetches fresh data and runs all four models. Everything in v3 is new work; the frozen v2 results are untouched.

| Model | Adds to A | Data |
|---|---|---|
| A | (frozen v2: storage history, snowpack, season, reservoir) | CDEC |
| B | Recent weather at the dam (7-day lag) | NASA POWER point |
| B2 | Recent weather averaged over the upstream area (7-day lag) | NASA POWER grid cells in the upstream box |
| C | Forecast precipitation for days 1-7 and 8-16 over the upstream area | NOAA GEFSv12 control member, 00Z on the origin Monday (reforecast 2001-2019, operational from 2020-09-24) |

**Key documents**
- `docs/v3/MODEL_C_FEASIBILITY.md`: which forecast archives exist and what Model C may use.
- `docs/v3/PREREGISTRATION_V3.md` and `config/v3_plan.json`: the fixed comparison plan.
- `docs/DATA_PUBLICATION_AUDIT.md`: who may publish what.
- `docs/v3/DEPLOYMENT.md`: how to run the interactive site.

**Rebuilding the models and results**

```bash
python -m src.v3_gefs --history          # GEFS basin precipitation, about 20 GB streamed, small CSV kept
python -m src.v3_power --end YYYY-MM-DD  # NASA POWER dam points and upstream-box cells
python -m src.v3_dataset                 # private weekly table (CDEC-derived)
python -m src.v3_experiment --dev        # development folds WY2008-2025, model-specific intervals
python -m src.v3_experiment --final      # the one preregistered 2026 test; writes models/v3/
python -m src.v3_release --version YYYY-MM-DD   # public data release (NOAA and NASA-derived only)
```

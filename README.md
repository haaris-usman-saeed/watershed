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
python -m src.build_dataset                                    # 4. offline raw -> data/processed/ + audit + outputs/stations.csv, history_daily.csv
python -m src.features                                         # 5. weekly table + snow basin index
python -m src.experiments                                      # 6. rolling water-year development evaluation (pre-2024 data only)
python -m src.dev_report                                       # 7. results/v2/development/report.md
python -m src.train_v2 --final                                 # 8. preregistered v2 held-out run -> results/v2/final/, outputs/
python -m pytest -q tests                                      # 9. tests
python -m src.contract outputs                                 # 10. validate UI contract
python -m src.make_bundle                                      # 11. deterministic private bundle -> dist/
python -m src.make_bundle --verify dist/watershedshift_v2_bundle.tar.gz
```

Notes on these steps:
- **Step 8** refuses to run unless `docs/preregistration.md` is committed and the hashes of the config and code match it. Re-running it with the identical frozen configuration is allowed, to check reproducibility. A different configuration needs explicit owner approval.
- **Raw data stays on the local machine** (git-ignored) because the repository is public and CDEC redistribution terms are unconfirmed. `data/raw_manifest.jsonl` (tracked) lists every download with its URL, retrieval time and SHA-256.
- **CDEC flakiness:** CDEC sometimes returns empty responses. The fetchers retry and cache completed requests, so they can be re-run safely.

## 3. Data sources (verified 2026-11-21)

| Source | What | Link |
|---|---|---|
| CDEC (CA Dept of Water Resources) | Daily reservoir storage, sensor 15, acre-feet, end-of-day | https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet |
| CDEC snow pillows | Daily snow water content, sensor 82 (adjusted), inches; 51 candidate pillows in the upstream basins, 3 to 12 core pillows per reservoir | same endpoint; station lists from https://cdec.water.ca.gov/dynamicapp/staSearch |
| CDEC metadata and capacities | Coordinates, operators, sensors; Daily Reservoir Storage Summary | https://cdec.water.ca.gov/dynamicapp/staMeta?station_id=SHA , https://cdec.water.ca.gov/reportapp/javareports?name=RES |
| DWR water year types | Sacramento and San Joaquin Valley indices (descriptive only) | https://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST |
| NASA POWER daily point API v2.10.5 | T2M, PRECTOTCORR, RH2M (local solar time) | https://power.larc.nasa.gov/docs/services/api/temporal/daily/ |

Key verification findings:
- The CDEC `DATE TIME` field is end-of-day storage. `OBS DATE` is operator dependent and was the source of a one-day shift in the starter code.
- Oroville's capacity changed from 3,537,577 to 3,424,753 AF when DWR adopted a new storage table on 2024-07-01. The daily CDEC series switches on 2024-07-02. v2 uses the capacity in force on each date, and no window spans the change.
- Spike candidates were adjudicated against the independent elevation sensor. Two artifacts were removed; six flood-day extremes at Folsom were confirmed and kept.

**Terms:** NASA POWER asks for this acknowledgement: "These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program." CDEC data are public and provisional. No explicit open licence was found, so cite CDEC/DWR, and do not redistribute raw records without confirming the terms.

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
- **Paired v2 vs v1 ML** on 571 rows with identical targets: -0.47 pp, CI [-0.69, -0.26] (5 weeks).
- **Disclosure:** v2 was designed after the v1 test results on the same period were seen, so the period is not fully independent of the revision. See `docs/preregistration.md`.
- **What drove the improvement:** a secondary HGB using the v1 feature set but the 2001+ training data scored 2.32 on the same test. Most of the improvement over v1 therefore comes from the longer, corrected training history, not from the snow features.

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

The project covers tabular multisource forecasting only. **There is no satellite-image, CNN, operational-forecast or reservoir-siting capability.** Satellite features remain future research.

## Demo app

Run `streamlit run app.py` (owned by the UI team; see `docs/UI_CONTRACT.md`). Use `WATERSHED_DATA_DIR=app/sample_data` for the **synthetic** UI fixtures.

## 8. Suggested handoff

Send the repository or app link, together with the limitations above, to an appropriate water-resources group. Keep proof of sending. Do not imply that a regulator has endorsed the model.

## 9. Agent workflow

See `PROMPTS.md`. Engineering (Claude Code) owns `src/`, `tests/`, `data/`, `results/`, `config/`, `outputs/` and the engineering docs. The UI team (Claude Web / Muse) owns `app.py` and `app/`.

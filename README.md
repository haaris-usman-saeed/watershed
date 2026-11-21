# WatershedShift — 30-day reservoir storage change, California (historical backtest)

This project builds an independently assembled joint dataset from **CDEC daily measured reservoir storage** for 6 verified California reservoirs and **NASA POWER daily weather**, covering 2018–2025. It forecasts the **30-day change in storage, in percentage points (pp) of documented capacity**, using only information available at the forecast origin. Models are evaluated against simple baselines on held-out 2024–2025 dates.

> This is a research prototype and historical backtest. It is **not** an operational forecast or a water-safety tool.

Current results are in `STATUS.md` (checkpoint C2). Data provenance is in `data/provenance.md`, the coverage audit in `data/coverage_table.md`, and the UI/output schema in `docs/UI_CONTRACT.md`.

## 1. Install (verified on macOS, Python 3.14.5)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # exact versions of the verified run: requirements-lock.txt
```

## 2. Reproduce end to end (about 3–5 minutes; internet required for step 1)

```bash
python -m src.fetch --start 2018-01-01 --end 2025-12-31   # 1. raw downloads -> data/raw/ (unmodified bytes + fetch_log.jsonl)
python -m src.build_dataset                             # 2. offline raw -> data/processed/daily_joined.csv + audit
python -m src.features                                  # 3. weekly table -> data/processed/weekly_features.csv
python -m src.train                                     # 4. baselines + Ridge + GBM -> outputs/, models/
python -m pytest -q tests                               # 5. leakage + contract tests
python -m src.contract outputs                          # 6. validate outputs against UI contract v1.0
```

Steps 2–6 are offline, so the whole dataset can be rebuilt from `data/raw/` without downloading again. Raw data, processed data, outputs and models are git-ignored and stay on the local machine. `data/raw_manifest.jsonl` (tracked) lists every download URL, retrieval time and SHA-256 hash. CDEC sometimes returns empty responses; `src.fetch` retries and caches completed station-years, so it can safely be re-run.

## 3. Data sources (verified 2026-11-21)

| Source | What | Link |
|---|---|---|
| CDEC (CA Dept of Water Resources) | Daily reservoir storage, sensor 15, acre-feet, end-of-day | https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet |
| CDEC station metadata / capacities | Coordinates, operator, sensor list; capacities from the Daily Reservoir Storage Summary | https://cdec.water.ca.gov/dynamicapp/staMeta?station_id=SHA , https://cdec.water.ca.gov/reportapp/javareports?name=RES |
| NASA POWER daily point API v2.10.5 | T2M (°C), PRECTOTCORR (mm/day), RH2M (%), local solar time | https://power.larc.nasa.gov/docs/services/api/temporal/daily/ |

Stations: SHA (Shasta), ORO (Oroville), FOL (Folsom), NML (New Melones), DNP (Don Pedro), EXC (McClure). All six cover 2018-01-01 to 2025-12-31, with at most 2 missing storage days per station. Verification found and fixed three problems in the starter: the CDEC date field was misread (one-day shift for 3 stations), and the ORO and EXC capacities were wrong. It also found a **reporting-basis break at ORO on 2024-07-02**, when DWR adopted a new elevation–storage table. No target or change-feature spans that date. See `data/provenance.md`.

**Weather caveat:** NASA POWER gives a coarse reanalysis grid-cell value at the dam coordinate. It is **not watershed-average rainfall**. The 2018–2025 archive comes from MERRA-2, while near-real-time values come from GEOS-IT, so the retrospective series differs from what was available in real time.

**Terms:** NASA POWER asks for this acknowledgement: *"These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program."* CDEC data are public and provisional. No explicit open licence was found, so cite CDEC/DWR. This repository does not redistribute raw records.

## 4. Method

- **Rows:** reservoir × Monday forecast origin *t*.
- **Target:** `100 × (storage[t+30] − storage[t]) / capacity`, matched on exact calendar days and never interpolated.
- **Features:**
  - storage level and 7/30/60-day storage changes, up to *t*;
  - 7- and 30-day rain, and 30-day temperature and humidity, ending at *t*−7 (the observed POWER latency is about 3 days);
  - seasonality terms and a station one-hot.
- **Timing caveat:** CDEC publishes storage[*t*] 1–2 days later, so the effective lead time from issue is about 28–29 days.
- **Splits:**
  - train: targets before 2023-01-01;
  - validation: 2023;
  - test: origins from 2024-01-01 with targets up to 2025-12-31.

  Windows that cross a split boundary are dropped. Hyper-parameters, the interval width and the "best baseline" are chosen on validation only.
- **Models:**
  - baselines: persistence (0 change), 30-day trend, and per-station monthly climatology;
  - Ridge regression;
  - HistGradientBoosting (selected from a 5-setting grid).
- **Uncertainty:**
  - 90% interval = ±(90th percentile of absolute validation errors), with test coverage reported;
  - 95% CIs from a **moving-block bootstrap over weekly dates** (5-week blocks, stations resampled together). Overlapping 30-day windows mean rows are not independent.

## 5. Results (test 2024–2025, n = 593; from `outputs/metrics.json`)

| Model | Test MAE (pp of capacity) |
|---|---|
| Gradient boosting | **2.81** |
| Historical seasonal (best baseline on validation) | 3.16 |
| Ridge | 3.41 |
| 30-day trend | 4.80 |
| Persistence | 5.79 |

GBM minus the best baseline is **−0.34 pp, with a 95% block-bootstrap CI of [−0.80, +0.03]**. The CI includes 0, so the gain is suggestive, not conclusive. GBM is **worse than the seasonal baseline at ORO and SHA**. The 90% interval (±10.0 pp) covers 96.3% of test targets: it is conservative and too wide for operational use. Per-station results are in `STATUS.md`.

## 6. Limitations

This is a single temporal holdout of two years. The weather data are a retrospective, revised point series rather than as-of-time archives. Reservoir operations (releases, flood-control rules, allocations) are not modelled. There are no snowpack or inflow-forecast inputs. Six reservoirs is a small sample. Interval calibration comes from one extreme wet year (2023).

## 7. Scope

The project covers tabular multisource forecasting only. **There is no satellite-image, CNN or reservoir-siting component.** Any satellite-derived feature would be added later as a clearly separate, evaluated experiment.

## Demo app

`streamlit run app.py` (owned by the UI team; see `docs/UI_CONTRACT.md`). Use `WATERSHED_DATA_DIR=app/sample_data` for the **synthetic** UI fixtures.

## 8. Suggested handoff

Send the working app link or repository, with a short summary of limitations, to an appropriate water-resources group or to the California Department of Water Resources. Keep proof of sending. Do not imply that a regulator has endorsed the model.

## 9. Agent workflow (one-day sprint)

See `PROMPTS.md`. Claude Code owns data acquisition, cleaning, training, tests and reproducibility (`src/`, `tests/`, `data/`, `stations.csv`, `docs/UI_CONTRACT.md`). The UI team (Muse / Claude Web) owns `app.py` and `app/`. Status is shared through `STATUS.md`. The originality search is in `docs/originality_check.md`.

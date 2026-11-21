# WatershedShift — 24-hour proof-of-concept starter

**Objective:** An independently assembled California reservoir-date ML dataset that combines CDEC daily *measured storage* with NASA POWER weather. Forecast the storage change over the next 30 days, measured in **percentage points of documented capacity**, based only on past information. This starter is not yet a validated model or operational forecast.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Acquire data (internet required)

```bash
python -m src.fetch --start 2018-01-01 --end 2025-12-31 --stations SHA ORO FOL
# after verifying those stations, rerun without --stations for all six
python -m src.fetch --start 2018-01-01 --end 2025-12-31
```

**Sources:**
- California Data Exchange Center (CDEC), daily storage (sensor 15, acre-feet): https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet
- NASA POWER daily point API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/

The API download script preserves raw CDEC CSVs and NASA JSON responses in `data/raw/`. Network access was **not** available in the environment where this starter was assembled, so the live retrieval must be tested and debugged locally. Station coordinates and capacity values in `stations.csv` are *initial candidates*, not fully verified official station metadata; cross-check them before claiming physical capacity-based metrics. If a station lacks sensor-15 daily values, replace or drop it after inspecting its station metadata. NASA POWER point weather is a coarse grid-cell estimate, **not rainfall measured over the contributing watershed**.

## 3. Construct training rows

```bash
python -m src.features
python -m tests.test_features
```

- Observation: reservoir × Monday date.
- Target: 30-day observed storage change divided by reservoir capacity × 100.
- Weather inputs stop at least 7 days before the prediction origin (conservative availability buffer).
- Daily storage is never interpolated for input or target construction.
- Features include 7/30-day rainfall, 30-day temperature and humidity, storage lags and seasonality.

## 4. Train and evaluate

```bash
python -m src.train
streamlit run app.py
```

Training labels finish before 2023-01-01; 2023 is validation; 2024–2025 is held-out test. A 30-day *outcome embargo* prevents leakage across split boundaries. Compare **unchanged storage**, **30-day trend**, **seasonal average**, and gradient boosting. Metric: MAE in percentage points of capacity. Interval half-width is an empirical 90th percentile of 2023 absolute validation errors, with coverage checked in the test period. Actual 90% coverage is not guaranteed. Test predictions and metrics save to `outputs/`.

**Important data limitation:** The dataset uses retrospectively available NASA POWER values, which may have been revised since the original dates. Do not claim fully operational as-of-time accuracy until archived release versions and real-world availability have been checked. Similar caution applies to CDEC corrections. Choose a genuine real-time backtest only if source-version archives can be reconstructed.

## 5. Dataset-creation documentation for the assignment

Record a `provenance.md` including URLs, acquisition timestamps, source licenses, field definitions, units, raw date ranges, missingness by reservoir, weather-grid uncertainty, station ID mapping and the exact transformation steps. Disclose that this is a newly *constructed joint dataset*, not newly measured physical observations. Perform a short literature search to establish that this exact assembled set has not evidently been used previously for ML. There is no guarantee of novelty simply from joining two public data sources.

## 6. Hard stop / pivot rule

If CDEC does not yield multiple years of usable storage for at least 3 pilot reservoirs, stop and verify station/sensor metadata rather than moving onto CNN or app features. If the trained model fails to beat persistence/trend, report the negative result and do not imply added value.

## 7. Scope of this starter

This is **tabular multisource forecasting**. It deliberately does not pretend to include satellite image analysis, watershed forest changes, a CNN or site selection. Add image-derived landscape features **only after** the real-data pipeline is working and report them as a clearly separate experiment.

## 8. Suggested handoff

Deliver the working app link or repository to an appropriate water-resources group or the California Department of Water Resources, along with a short summary of limitations. Keep proof of sending. Do not imply a regulator has endorsed the model.

## 9. Agent workflow (one-day sprint)

For the current assignment split, see `PROMPTS.md`: Claude Code owns data acquisition, cleaning, training, tests and reproducibility. Muse owns project coordination, Streamlit UX, and the report/handoff skeleton. The user owns integration approvals and real sending; ChatGPT can audit model/evaluation files shared in chat. Agents should share status through `STATUS.md` in a common repository or the user must relay updates. No autonomous agent-to-agent coordination is assumed.

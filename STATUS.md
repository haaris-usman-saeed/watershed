# WatershedShift — STATUS

Shared file. Append under your own heading only.

## Engineering (Claude Code) — branch `engineering`

### 2026-11-21 — kickoff
**Findings from starter review**
- Pipeline: `src/fetch.py` (CDEC CSVDataServlet sensor 15 + NASA POWER daily point) → `src/features.py` (weekly Monday rows, 30-day target, weather lagged 7 d) → `src/train.py` (3 baselines + HistGradientBoosting, constant-width 90% interval from 2023 validation errors).
- Starter was never run against real data. `stations.csv` coordinates/capacities are unverified candidates.
- ~~First live probe: SHA daily returned an empty CSV~~ → resolved: transient CDEC empty responses; SHA daily sensor 15 exists (1985–present). Fetch now retries.
- Starter gaps to fix: no per-station metrics, no bootstrap CI, test split filter `target_date <= 2025-12-31` silently depends on fetch end date; raw NASA JSON overwritten without timestamp; no provenance/coverage outputs.

**Plan**: (1) verify CDEC daily storage per station & capacities, NASA POWER params → C1 coverage table; (2) fix features/splits, add per-station metrics + block bootstrap → C2; (3) clean-env reproduction, README, provenance → C3.

### UI contract published
- `docs/UI_CONTRACT.md` (schema v1.0), validator `src/contract.py`, synthetic fixtures in `app/sample_data/` (clearly labelled `SYNTHETIC_SAMPLE`, station IDs `SYN1..SYN4`).
- UI owns `app.py` and `app/` (except `app/sample_data/`). Engineering will not edit `app.py`.

### C1 — real data verified (2026-11-21)
- **Live downloads succeeded for all 6 stations** (SHA, ORO, FOL, NML, DNP, EXC): CDEC sensor 15 daily, 2018-01-01 → 2025-12-31, plus NASA POWER T2M/PRECTOTCORR/RH2M. 48 CDEC + 6 POWER raw files, about 2 MB, kept **locally only** in `data/raw/` (git-ignored). URLs and SHA-256 hashes are in `data/raw_manifest.jsonl`.
- Joined daily dataset: **17,532 station-days** (6 × 2,922). Storage missing: SHA 0, ORO 1, FOL 0, NML 0, DNP 2, EXC 2 days (max 0.068%). Weather missing: 0. Duplicates: 0. Full table: `data/coverage_table.md`.
- **Corrections to the starter:**
  1. CDEC `OBS DATE` is inconsistent by operator (D vs D+1). The code now uses `DATE TIME`, which is end-of-day storage, verified against hourly data. The starter would have shifted ORO, DNP and EXC by one day.
  2. ORO capacity changed 3,537,577 → **3,424,753**; EXC 1,025,000 → **1,024,600**. FOL and NML coordinates were corrected from CDEC metadata.
  3. Fetching is now split into a network step (`src.fetch`, raw bytes saved unchanged, per-year requests with retries) and an offline step (`src.build_dataset`). CDEC sometimes returns empty bodies, which the retries handle.
- **Data issue found:** ORO has a **reporting-basis break on 2024-07-02**. DWR adopted a new elevation–storage table, producing a −123k AF artificial step. Values are not adjusted. The break is logged in `data/series_breaks.csv`, and C2 will exclude any target or change-feature window that spans it.
- Latency observed today: CDEC 1–2 days; POWER about 3 days (recent days come from GEOS-IT, the archive from MERRA-2, so retrospective ≠ real-time). Details are in `data/provenance.md`.
- Usable for modelling: **all 6 reservoirs, full 2018–2025**, minus ORO windows that span the break.
- Next (C2): weekly targets with series-break masking; baselines + GBM; per-station metrics; block-bootstrap CI.

### C2 — baselines + ML on held-out 2024–2025 (2026-11-21)
Weekly Monday origins, 30-day target in pp of capacity, weather lagged 7 d, no window spans the ORO break. Splits: train = targets < 2023-01-01 (**1,486** rows); validation = 2023 (**288**); test = origins ≥ 2024-01-01 with targets ≤ 2025-12-31 (**593** rows, 6 reservoirs). Hyper-parameters, the interval width and the "best baseline" were all chosen on **validation only**. The final model was refit on all pre-2024 outcomes and scored once on test.

| Test MAE (pp of capacity) | all | DNP | EXC | FOL | NML | ORO | SHA |
|---|---|---|---|---|---|---|---|
| **GBM (ML)** | **2.81** | 1.76 | 2.31 | 4.53 | 1.15 | 3.83 | 3.43 |
| Ridge | 3.41 | 2.12 | 2.98 | 6.17 | 1.91 | 3.43 | 3.84 |
| Historical seasonal (best baseline on validation) | 3.16 | 2.56 | 3.21 | 5.47 | 1.18 | **3.23** | **3.30** |
| 30-day trend | 4.80 | 2.79 | 4.33 | 9.24 | 1.29 | 5.32 | 5.89 |
| Persistence (0 change) | 5.79 | 3.86 | 5.60 | 9.81 | 1.85 | 7.21 | 6.60 |
| n test rows | 593 | 101 | 101 | 101 | 101 | 88 | 101 |

- **ML − best baseline = −0.34 pp**, 95% moving-block bootstrap CI **[−0.80, +0.03]** (2,000 reps, 5-week blocks over dates, stations resampled together). ML is better in 97% of replicates, but the **CI includes 0**: the improvement is suggestive, not conclusive. **ML loses to the seasonal baseline at ORO and SHA** and ties at NML.
- 90% interval: ±9.99 pp (90th percentile of absolute 2023 validation errors). Test coverage is **96.3%**, so it over-covers: 2023 was an extreme wet year, which makes the interval wide and of limited practical use. Per-station coverage ranges from 91% to 100%.
- Leakage controls tested in `tests/test_features.py`: perturbing all data after the origin leaves features unchanged; the last 7 days of weather are unused; missing targets are dropped, never filled; no window spans a series break. The split ordering is asserted in `src/train.py`.
- Outputs (`outputs/`, git-ignored, produced locally) pass `python -m src.contract outputs`. Extra columns/keys beyond contract v1.0: `pred_ridge_pp`, `storage_pct`, and `Ridge`, `validation`, `bootstrap.mae_ci95` in metrics. These are additive; the UI can ignore them.

### C3 — reproducibility, provenance, README (2026-11-21)
- **Clean reproduction passed:** a fresh `git clone -b engineering` into an empty directory, a new venv and `pip install -r requirements.txt`, then fetch → build_dataset → features → train → pytest → contract. Result: `daily_joined.csv` identical, every metric identical (ML 2.8144, seasonal 3.1580, coverage 0.9629, n_test 593), **7/7 tests pass**, contract OK. Exact package versions are in `requirements-lock.txt` (Python 3.14.5, macOS).
- README rewritten with verified commands, sources, method, results and limitations. Provenance: `data/provenance.md`. Coverage: `data/coverage_table.md`. Originality search: `docs/originality_check.md` (CDEC + ML for California reservoirs has precedent; no identical CDEC + POWER benchmark found in a limited search, which does not prove novelty).
- Final output schema = **UI contract v1.0, unchanged**. Real outputs add optional extras (see `docs/UI_CONTRACT.md` §3.6). **Muse / Claude Web:** run the pipeline locally, or use `app/sample_data` (synthetic), until real outputs are shared. `outputs/` is git-ignored by design.

**Unresolved / open**
- The ML gain over the seasonal baseline is not statistically clear (CI includes 0), and ML is worse at ORO and SHA.
- The 90% interval is too wide (±10 pp) because it was calibrated on 2023. A per-station or seasonally scaled interval could help, but would need a fresh validation design, not test tuning.
- No as-of-time weather archive; POWER is a retrospective MERRA-2 series.
- Only a deprecation warning remains: NumPy "generic timedelta unit" on Python 3.14 in the tests. It is harmless.
- The CDEC licence wording needs owner sign-off.
- Stretch (satellite feature) **not started**: by the priority order, it comes only after review.
- The UI's `app.py` has not been checked against the extended outputs by engineering (owned by the UI team).

## Engineering v2 revision (Claude Code), branch `review-v2`

### Checkpoint A: v1 preserved and audited (2026-11-21)
- v1 is frozen as git tag `v1-results` (commit 01e43cb, pushed). Public artifacts are in `results/v1/`: `metrics.json`, `feature_catalog.json`, `stations.csv`, `config_and_hashes.json` (GBM params, interval half-width, splits, SHA-256 of every local v1 output), and `test_predictions_public.csv`. The public predictions file drops the raw storage columns because the repository is public and CDEC redistribution terms are unconfirmed. Full-resolution v1 outputs are kept locally in `results/v1/local_full/` (git-ignored).
- v1 headline, unchanged: GBM 2.81 pp, monthly seasonal 3.16 pp, difference -0.34 pp, 95% CI [-0.80, +0.03] (5-week blocks), interval coverage 96.3%.
- v1 difference CI at the predetermined block lengths: 5 weeks [-0.80, +0.03]; 9 weeks [-0.90, +0.11]; 13 weeks [-0.93, +0.11]. The shared bootstrap in `src/evaluation.py` reproduces the original 5-week CI exactly.
- Diagnostics are in `results/v1/diagnostics.md` and are descriptive only, not used to tune v2:
  - By season, v1 GBM's advantage comes from summer (JJA: 1.70 vs 3.08 seasonal) and autumn (SON: 1.82 vs 2.08). It is slightly worse than seasonal in winter (DJF: 4.20 vs 4.07) and spring (MAM: 3.61 vs 3.51).
  - By fullness, errors are largest when storage is below 40% of capacity (n=13; GBM 8.16, seasonal 8.49, coverage 69%).
  - ORO before the storage-table change (n=22): GBM 4.11 vs seasonal 2.15. After (n=66): 3.74 vs 3.60.
- Data audit of the original 2018-2025 data: no duplicate dates, no unit mismatches (every row is AF, sensor 15, duration D), 2 isolated spikes, flags e/r counted. See `data/coverage_table.md`.

### Checkpoint B: historical expansion and reporting-basis corrections (2026-11-21)
- **Period extended to 2001-01-01 through 2025-12-31 for all six reservoirs** (102 new CDEC station-years plus POWER 2001-2025). The joined daily table has 54,786 station-days. Missing storage: SHA 0, ORO 1, FOL 0, NML 0, DNP 36, EXC 39 days (max 0.43%; longest gap 6 days). No duplicates, no unit mismatches, no weather gaps. Details are in `data/coverage_table.md`.
- **Date convention re-verified for 2003 and 2010**: daily D equals hourly 00:00 of D+1 for SHA, ORO, FOL, NML and EXC.
- **Oroville capacity normalization**: `data/capacity_history.csv` gives 3,537,577 AF through 2024-07-01 and 3,424,753 AF from 2024-07-02. In the raw records, hourly sensor 15 steps by -110,466 AF at 2024-07-01 00:00, and the daily series switches on 2024-07-02. This matches DWR's July 1, 2024 effective date. `storage_pct` and targets now use the capacity in force; window exclusion across the break is unchanged.
- **Extremes versus artifacts**: spike candidates are now adjudicated against the independent elevation sensor (`data/spike_review.csv`). DNP 2019-09-26 and EXC 2022-05-22 are artifacts and set to missing. Six FOL flood-day candidates (2005-12-30, 2016-12-15/16, 2017-01-07/11, 2017-02-10) are physical and **kept**; the v1 rule would have deleted them. A dry-day shift scan found only multi-day flood-release sequences (ORO February 2017 spillway emergency, FOL 2006 and 2016-17) and no other reporting breaks.
- **Snow**: 51 CDEC daily snow pillows (sensor 82, adjusted SWE, inches) were downloaded for the upstream basins, 2001-2025, and QC'd (missing is never treated as zero). Core pillows were picked by a predetermined availability rule using WY2002-2023 only: SHA 7, ORO 8, FOL 12, NML 5, DNP 3, EXC 3.
- DWR water year types (WSIHIST) were downloaded for descriptive grouping only.
- Regression tests: `tests/test_build.py` (8 tests) covering the date convention, unit and sensor rejection, no interpolation, duplicate handling, spike adjudication, historical capacity, and capacity and break consistency.
- Raw manifest: 188 downloads with SHA-256. Provenance has been rewritten (`data/provenance.md`); v1 provenance and coverage are archived in `results/v1/`.

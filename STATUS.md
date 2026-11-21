# WatershedShift: STATUS

Shared file. Append under your own heading only.

## Engineering (Claude Code): branch `engineering`

### 2026-11-21: kickoff
**Findings from starter review**
- Pipeline: `src/fetch.py` (CDEC CSVDataServlet sensor 15 + NASA POWER daily point) → `src/features.py` (weekly Monday rows, 30-day target, weather lagged 7 d) → `src/train.py` (3 baselines + HistGradientBoosting, constant-width 90% interval from 2023 validation errors).
- Starter was never run against real data. `stations.csv` coordinates/capacities are unverified candidates.
- ~~First live probe: SHA daily returned an empty CSV~~ → resolved: transient CDEC empty responses; SHA daily sensor 15 exists (1985-present). Fetch now retries.
- Starter gaps to fix: no per-station metrics, no bootstrap CI, test split filter `target_date <= 2025-12-31` silently depends on fetch end date; raw NASA JSON overwritten without timestamp; no provenance/coverage outputs.

**Plan**: (1) verify CDEC daily storage per station & capacities, NASA POWER params → C1 coverage table; (2) fix features/splits, add per-station metrics + block bootstrap → C2; (3) clean-env reproduction, README, provenance → C3.

### UI contract published
- `docs/UI_CONTRACT.md` (schema v1.0), validator `src/contract.py`, synthetic fixtures in `app/sample_data/` (clearly labelled `SYNTHETIC_SAMPLE`, station IDs `SYN1..SYN4`).
- UI owns `app.py` and `app/` (except `app/sample_data/`). Engineering will not edit `app.py`.

### C1: real data verified (2026-11-21)
- **Live downloads succeeded for all 6 stations** (SHA, ORO, FOL, NML, DNP, EXC): CDEC sensor 15 daily, 2018-01-01 → 2025-12-31, plus NASA POWER T2M/PRECTOTCORR/RH2M. 48 CDEC + 6 POWER raw files, about 2 MB, kept **locally only** in `data/raw/` (git-ignored). URLs and SHA-256 hashes are in `data/raw_manifest.jsonl`.
- Joined daily dataset: **17,532 station-days** (6 × 2,922). Storage missing: SHA 0, ORO 1, FOL 0, NML 0, DNP 2, EXC 2 days (max 0.068%). Weather missing: 0. Duplicates: 0. Full table: `data/coverage_table.md`.
- **Corrections to the starter:**
  1. CDEC `OBS DATE` is inconsistent by operator (D vs D+1). The code now uses `DATE TIME`, which is end-of-day storage, verified against hourly data. The starter would have shifted ORO, DNP and EXC by one day.
  2. ORO capacity changed 3,537,577 → **3,424,753**; EXC 1,025,000 → **1,024,600**. FOL and NML coordinates were corrected from CDEC metadata.
  3. Fetching is now split into a network step (`src.fetch`, raw bytes saved unchanged, per-year requests with retries) and an offline step (`src.build_dataset`). CDEC sometimes returns empty bodies, which the retries handle.
- **Data issue found:** ORO has a **reporting-basis break on 2024-07-02**. DWR adopted a new elevation-storage table, producing a −123k AF artificial step. Values are not adjusted. The break is logged in `data/series_breaks.csv`, and C2 will exclude any target or change-feature window that spans it.
- Latency observed today: CDEC 1-2 days; POWER about 3 days (recent days come from GEOS-IT, the archive from MERRA-2, so retrospective ≠ real-time). Details are in `data/provenance.md`.
- Usable for modelling: **all 6 reservoirs, full 2018-2025**, minus ORO windows that span the break.
- Next (C2): weekly targets with series-break masking; baselines + GBM; per-station metrics; block-bootstrap CI.

### C2: baselines + ML on held-out 2024-2025 (2026-11-21)
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

### C3: reproducibility, provenance, README (2026-11-21)
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
- **Snow**: 51 CDEC daily snow pillows (sensor 82, adjusted SWE, inches) were downloaded for the upstream basins, 2001-2025, and QC'd (missing is never treated as zero). Core pillows were picked by a predetermined availability rule using WY2002-2023 only: SHA 7, ORO 8, FOL 11, NML 5, DNP 3, EXC 3 (corrected in the Phase 1 handoff; this line originally said FOL 12, a transcription error; the data and model always used 11).
- DWR water year types (WSIHIST) were downloaded for descriptive grouping only.
- Regression tests: `tests/test_build.py` (8 tests) covering the date convention, unit and sensor rejection, no interpolation, duplicate handling, spike adjudication, historical capacity, and capacity and break consistency.
- Raw manifest: 188 downloads with SHA-256. Provenance has been rewritten (`data/provenance.md`); v1 provenance and coverage are archived in `results/v1/`.

### Checkpoint C: environmental features, ablations, rolling water-year evaluation (2026-11-21)
- Weekly table: 7,693 rows (2001-03 to 2025-12, 6 reservoirs). The development set has 7,076 rows with origin and target before 2024-01-01 (corrected; an earlier draft of this line said 7,224 without computing it); the hold-out is never loaded by `src/experiments.py` (asserted).
- **SWE features** (CDEC snow pillows, t-1 and earlier):
  - `swe_basin_in`: mean of core pillows; missing when fewer than 2/3 of core pillows report.
  - `swe_change_14d_in`.
  - `swe_anom_in`: anomaly against a smoothed day-of-year climatology computed per fold from training years only.
- Basin index availability is 94-100% of weekly rows. Leakage tests cover the snow lag, missing-not-zero, the 2/3 rule and station eligibility.
- **Inflow and release not added**: CDEC "inflow" equals outflow plus storage change (mass balance; residual 0.2 to 0.3 cfs at DNP and EXC, correlation > 0.986 elsewhere), so it is not an independent measurement. Releases are operator decisions with no future information available at t. Both are documented, omitted and left as future work.
- **Rolling evaluation**: 16 expanding-window water-year folds, WY2008-2023 (4,988 forecasts; corrected in the Phase 1 handoff, this line originally said 4,995). HGB hyper-parameters were tuned on WY2008-2015 only. Models were compared on confirmation folds WY2016-2023 (2,500 forecasts). Full report: `results/v2/development/report.md`.

| Pooled MAE, pp | tuning WY08-15 | confirmation WY16-23 |
|---|---|---|
| HGB SWE no weather | 2.60 | **3.16** |
| HGB + SWE (weather + SWE) | 2.60 | 3.17 |
| HGB (v1 features) | 2.68 | 3.23 |
| HGB no weather | 2.85 | 3.30 |
| Ridge | 3.31 | 3.90 |
| Smoothed DOY seasonal (strongest baseline) | 3.24 | 4.11 |
| Monthly seasonal | 3.35 | 4.21 |
| 30-day trend | 3.99 | 5.17 |
| Persistence | 4.89 | 5.76 |

- **Primary vs strongest baseline** (confirmation): -0.95 pp. 95% CI [-1.23, -0.68] (5-week blocks), [-1.26, -0.64] (9), [-1.28, -0.63] (13).
- **Wins**: HGB SWE variants beat the strongest baseline in 15 of 16 water years. The exception is WY2008, where the smoothed DOY baseline wins (1.98 vs 2.12/2.16). Ridge wins only 8 of 16. In WY2021 (critically dry) the 30-day trend baseline (1.88) beat every model.
- **Ablations** (confirmation, MAE difference, 5/9/13-week CIs):
  - Weather: -0.07 [-0.18, +0.03]. Small and not distinguishable from 0.
  - SWE given weather: -0.06 [-0.16, +0.04]. Not distinguishable from 0.
  - SWE instead of weather: -0.14 [-0.28, +0.01] (5 weeks), +0.03 at the upper end for 13 weeks. Suggestive only.
- **By season** (confirmation): gains are largest in summer drawdown (JJA 1.52 vs 2.99) and smallest in winter storms (DJF 5.35 vs 6.45). SWE helps most in MAM and JJA (snowmelt).
- **By reservoir** (confirmation): HGB SWE no weather beats the strongest baseline at all six (DNP 2.25 vs 3.12; EXC 3.44 vs 4.60; FOL 4.99 vs 6.08; NML 1.67 vs 2.70; ORO 3.65 vs 4.59; SHA 2.96 vs 3.55).
- **By water year class** (confirmation): wet 4.07 vs 5.40; below normal 3.15 vs 4.02; dry 2.38 vs 3.03.
- **Primary model by the predetermined rule: "HGB SWE no weather"** (storage + SWE + season, no NASA POWER). It is effectively tied with HGB + SWE (3.160 vs 3.168).

### Checkpoint D: uncertainty evaluation and frozen v2 configuration (2026-11-21)
- **Interval methods** (primary model, folds WY2012-2023, each calibrated only on earlier folds):

| method | coverage | mean width (pp) |
|---|---|---|
| sym_global | 0.885 | 13.04 |
| sym_season | 0.888 | 12.22 |
| sym_station | 0.887 | 12.89 |
| asym_global | 0.886 | 13.27 |
| asym_season | 0.883 | 12.64 |

  - The predetermined rule (narrowest method with coverage in [0.87, 0.93]) picks `sym_season`. Its widths by season: about 17 pp in DJF, 16 in MAM and 8 in JJA and SON, compared with v1's constant 20 pp.
  - Known weakness: per-reservoir coverage is uneven (FOL 0.75, NML 0.98). Reservoir x season groups would be too small for stable calibration in the early folds.
- **v2 frozen** in `docs/preregistration.md`, together with `config/v2_frozen.json` and the SHA-256 hashes of the config and of the code in `src/train_v2.py`, `models.py`, `features.py`, `evaluation.py` and `build_dataset.py`.
  - Primary model: HGB SWE no weather. Strongest baseline: Smoothed DOY seasonal. Test period: 2024-2025 (593 rows, the same as v1).
  - The preregistration discloses that the period is not fully independent, because v1 results on it were seen.
- **Guards** (tested in `tests/test_v2_guards.py`): `--final` refuses unless the preregistration is committed and the hashes match, and refuses a different configuration after a first run.
- **Rehearsal** on development data only (train before 2023, evaluate 2023): the identical code path runs end to end. ML 3.02 pp vs Smoothed DOY 4.83 pp; coverage 0.892; the UI files validate. Results are in `results/v2/rehearsal/`.

### Checkpoint E: single v2 held-out evaluation, reproducibility, submission package (2026-11-21)
- **One preregistered run.** `python -m src.train_v2 --final` was run once at commit e47deab. Its hashes match `docs/preregistration.md`, and `results/v2/final/RUN_RECORD.json` records it. Nothing was changed after the metrics were seen.

| Held-out 2024-2025 (n=593), MAE pp | v1 (frozen) | v2 |
|---|---|---|
| Primary ML | 2.81 (GBM, storage + weather) | **2.33** (HGB, storage + SWE) |
| HGB with v1 features (2001+ training) | n/a | 2.32 |
| HGB + SWE (weather + SWE) | n/a | 2.32 |
| HGB no weather | n/a | 2.45 |
| Ridge | 3.41 | 3.35 |
| Smoothed DOY seasonal | n/a | 2.97 (strongest baseline, fixed in advance) |
| Monthly seasonal | 3.16 | 2.98 |
| 30-day trend | 4.80 | 4.79 |
| Persistence | 5.79 | 5.78 |
| 90% interval coverage / mean width | 96.3% / 20.0 pp | 92.1% / 13.1 pp |

- **v2 ML minus Smoothed DOY seasonal:** -0.64 pp. 95% CI [-1.02, -0.35] (5-week blocks), [-1.07, -0.34] (9), [-1.10, -0.34] (13). The CI excludes 0 at all three preregistered block lengths.
- **v2 ML minus Monthly seasonal:** -0.64 pp. CI [-1.01, -0.37] / [-1.05, -0.37] / [-1.08, -0.38].
- **Paired v2 vs v1 GBM** (571 rows with identical target definitions; ORO old-table rows excluded): 2.30 vs 2.76, difference -0.47 pp. CI [-0.68, -0.29] / [-0.69, -0.27] / [-0.67, -0.26].
- **By reservoir** (v2 ML vs smoothed DOY): DNP 1.45 vs 2.21; EXC 1.85 vs 2.87; FOL 4.17 vs 5.39; NML 0.95 vs 1.06; ORO 3.21 vs 3.47; SHA 2.49 vs 2.87. ML is better at all six.
- **By season:** DJF 3.46 vs 3.89; MAM 2.96 vs 3.15; JJA 1.41 vs 2.95; SON 1.55 vs 1.98.
- **By water year:** WY2024 2.09 vs 2.62; WY2025 2.55 vs 3.33; WY2026 (partial, n=54) 2.09 vs 2.33.
- **Interval coverage by reservoir:** DNP 0.99, EXC 0.96, FOL 0.75, NML 0.98, ORO 0.91, SHA 0.93. FOL under-covers, as the development folds predicted. Median half-width by season: DJF 10.2, MAM 8.6, JJA 3.7, SON 3.9 pp.
- **Interpretation (descriptive):** HGB with the v1 feature set but 2001+ training (2.32) performs like the primary. Most of the improvement over v1 comes from the longer, corrected training history, not from snow. SWE helps when weather is absent (2.33 vs 2.45) but adds about nothing on top of weather. The primary was chosen by the preregistered rule and was not reselected.
- **Disclosure:** v2 was designed after the v1 results on this test period were seen. The period is not fully independent of the revision.
- **Submission bundle:** `python -m src.make_bundle` writes `dist/watershedshift_v2_bundle.tar.gz` (git-ignored). It is byte-deterministic, carries SHA256SUMS and a LIMITATIONS.txt, and is checked with `--verify`. Raw data is excluded unless `--include-raw` is passed.
- **Frontend compatibility:** UI contract schema v1.0 is unchanged; v2 fields are additive (`docs/UI_CONTRACT.md` section 3.7). Streamlit AppTest of the unmodified `app.py` against the v2 outputs gives no exceptions; it shows ML 2.33 and coverage 92.1%.
- **Dashes:** em and en dashes were removed from engineering docs, comments and STATUS.md. `PROMPTS.md` (owner), `app.py` and `app/` (frontend) and the archived `results/v1/provenance.md` (v1 preserved verbatim) were left unchanged.
- **Clean-environment reproducibility (verified)**: a fresh `git clone -b review-v2` (commit cf7d4bc) into an empty directory, a new venv and `pip install -r requirements.txt`, then every step in the README, including **re-downloading all raw data**.
  - Processed datasets: `daily_joined.csv`, `snow_daily.csv`, `snow_basin_index.csv` and `weekly_features.csv` are **byte-identical** to the original run.
  - Development results: all files in `results/v2/development/` were regenerated with **no git diff**, so the rolling evaluation is deterministic.
  - Frozen v2 rerun: the guard allowed it as an identical-configuration rerun, and the headline was **identical** (ML 2.332333455339604, coverage 0.9207419898819561). The only change in `results/v2/final/metrics.json` was the `generated_utc` timestamp.
  - Tests and contract: **25/25 tests pass**; the outputs contract is OK; the bundle verifies (60 files).
  - Raw files: 177 of 182 re-downloaded files were byte-identical. The 5 that differed are NASA POWER JSON responses whose only difference is the server's `times` (request processing time) field; every data value was identical.
- **Not done or open**:
  - The CDEC licence wording still needs owner sign-off.
  - No value-level comparison against ResOpsUS yet.
  - Per-reservoir interval calibration is uneven (FOL).
  - A third held-out configuration needs owner approval.
  - The UI should switch its "best baseline" card to `bootstrap.best_baseline`.

### Follow-up: independent cross-check against ResOpsUS (2026-11-21)
- Downloaded ResOpsUS.zip (Zenodo 5893641, 298.6 MB, CC-BY 4.0) with owner approval. It was md5-verified against Zenodo, kept local and git-ignored, with its SHA-256 in the manifest. Only CSV members are read, in memory; nothing else in the archive is loaded or executed.
- All six reservoirs are in ResOpsUS. Daily storage was compared for 2001 to 2020 (`python -m src.compare_resopsus`; report in `results/v2/resopsus_check/report.md`).
- **Independent sources agree with us**:
  - FOL (Bureau of Reclamation): 99.6% of 6,574 days identical on the same date.
  - DNP (USGS 11287500): median difference 891 AF (0.04 pp). The best alignment is the same date (daily-change correlation 0.96, against 0.80 and 0.82 at plus or minus one day).
- **CDEC-sourced records**:
  - SHA and NML are identical on the same dates.
  - ORO (99.5%) and EXC (98.9%) match ResOpsUS only when our raw CDEC rows are keyed by `OBS DATE`. ResOpsUS therefore inherits the operator- and time-dependent `OBS DATE` offset: ORO is one day off throughout, and EXC is one day off in 2001-2002 and from mid-2014.
  - This independently confirms that keying on `DATE TIME` (end-of-day) was the right correction.
- Also found: the `OBS DATE` offset for USBR stations starts changing in late 2025 (first D+1 on 2025-09-24). Our pipeline is unaffected. The provenance statement has been corrected.
- No data, model or v2 result was changed. Tests: 27 pass (2 new for the comparison utilities).

## Final submission readiness (Claude Code), branch `review-v2`

### Phase 1: report facts (2026-11-21)
- `results/v2/report_facts.md` and `results/v2/report_facts.json` list 15 facts, each with value, units, exact source files (tracked or local), and kind (observed, computed or documented configuration). They are generated by `python -m src.report_facts` from existing files only; nothing was retrained. `python -m src.report_facts --check` regenerates them and fails on any difference. `tests/test_report_facts.py` checks them against the tracked frozen metrics, config and RUN_RECORD (runs without raw data).
- Known values checked against sources: dates 2001-01-01 to 2025-12-31 for all six reservoirs; 54,786 station-days; SWE lag 1 day; development folds WY2008-2023; 593 test rows; ML 2.3323 pp; Smoothed DOY seasonal 2.9694 pp. All confirmed.
- **Two documentation errors found and corrected (no data or result changed):**
  - Core snow pillows for FOL are **11**, not 12. Recomputed with `core_snow_stations` and read from `snow_basin_index.csv`; the model always used 11. Corrected in README, `data/provenance.md` and Checkpoint B above.
  - Rolling development forecasts are **4,988** (2,488 tuning + 2,500 confirmation), not 4,995. `metrics.json` `val_count` and `results/v2/development/summary.json` already said 4,988. Corrected in Checkpoint C above.
- ORO break windows: 13 weekly origins (2024-06-03 to 2024-08-26) are excluded because their window spans the 2024-07-02 break, which is why ORO has 88 test rows instead of 101.

### Phase 2: reproducibility evidence verified (2026-11-21)
- `docs/reproducibility.md` records the frozen configuration identity, the single held-out run and the earlier clean-clone reproduction (commit `cf7d4bc`), using existing evidence only. **No new fresh-clone run was performed for this handoff.**
- Checked now: the config and the five frozen code files hash to the preregistered values and the RUN_RECORD values; `git diff e47deab HEAD` is empty for all of them; `results/v2/final/` has not changed since `f21bf25`; tag `v1-results` (`01e43cb`) is on the remote and `results/v1/` was only ever added to.
- Since `cf7d4bc`, `src/` has only gained three separate scripts (ResOpsUS fetch and comparison, report facts) that no pipeline step imports.
- Gaps flagged: the clean-clone terminal logs were not archived as files, and reproduction has only been tested on macOS with Python 3.14.5.

### Phase 3: assessor guide and data package (2026-11-21)
- `ASSESSORS.md` (about one page) covers: software, how to get the private bundle, safe extraction, a table of included and excluded files, checksum verification, offline reconstruction, a frozen rerun from the pinned commit, metric comparison, tests and the contract validator, metadata-only differences, and agency revisions.
- **Gap found and fixed:** the earlier bundle contained only `src/contract.py`, not the pipeline. `src/make_bundle.py` (the existing generator, reused) now also includes `src/`, `tests/`, `app/sample_data/`, `ASSESSORS.md`, `docs/reproducibility.md` and `PROMPTS.md`, plus `SOURCE_COMMIT.txt` naming the commit and whether the working tree was clean.
- `ASSESSORS.md` says plainly that the bundle is **not self-contained for the frozen rerun**: the preregistration guard needs a git checkout, so assessors check out the pinned commit and copy the bundle's data into it. Raw agency responses are excluded unless the owner builds with `--include-raw` after the CDEC terms are confirmed.
- New read-only helper `src/compare_metrics.py` compares a reproduced `metrics.json` with the archived one number by number, ignoring `generated_utc`.
- The bundle is rebuilt and verified in the final verification step, after the last commit, so that `SOURCE_COMMIT.txt` names the final commit.

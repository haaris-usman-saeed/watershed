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
  - ORO (99.5%) and EXC (98.9%) match ResOpsUS only when our raw CDEC rows are keyed by `OBS DATE`. This is consistent with ResOpsUS following the operator- and time-dependent `OBS DATE` offset; the authors' intended convention is unknown (wording refined in Phase 5). ORO is predominantly one day off. EXC is one day off from 2001 to about March 2003 and from July 2014, and aligned from about April 2003 to May 2014.
  - Together with the FOL and DNP agreement, this supports keying on `DATE TIME` (end-of-day). Four of the six ResOpsUS records come from CDEC, so the check is only partly independent.
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

### Phase 4: final UI handoff (2026-11-21)
- New `src/ui_export.py` adds three pieces of additive metadata to `outputs/metrics.json` and `outputs/feature_catalog.json`. They are derived from the frozen config and the frozen test results, with no rerun:
  - `primary_model_features`: 15 exact columns, namely storage x4, SWE x3, season x2 and station one-hot x6; identical in both files.
  - `backtest`: `is_live_forecast: false`, evaluation period 2024-01-01 to 2025-12-31.
  - `per_station[s].interval_reliable` and an `interval_reliability` block: threshold 0.85 on the frozen test coverage, with a caveat. FOL is false; DNP, EXC, NML, ORO and SHA are true.
- Before writing, the script checks that every existing value equals `results/v2/final/metrics.json`. It also recomputes ML MAE and per-reservoir coverage from `test_predictions.csv`.
- The `features` array and the three CSV files are byte-unchanged, and the NASA POWER weather entries remain `used_by_primary_model: false`.
- `src/contract.py` checks the new keys only when present. A new test confirms that the synthetic fixtures still validate and that a wrong flag or `is_live_forecast: true` is rejected. The real outputs and `app/sample_data` both validate. Contract schema v1.0 is unchanged; the fields are documented in `docs/UI_CONTRACT.md` section 3.8.
- **UI package:** `dist/outputs_v2_ui.zip` (git-ignored, local only). It holds exactly the five files plus a README covering real backtest data, `WATERSHED_DATA_DIR`, the primary inputs, the strongest baseline and sign convention, the Folsom warning, and weather as context only. The ZIP is byte-deterministic, and `python -m src.ui_export --verify` checks its contents, the contract and the frozen numbers.
- **Checked against the frontend package the owner supplied** (`app/check_contract.py`, run read-only from a scratch copy, not added to the repo): PASSED with `--require-real`. It raised three integration notes for Claude Web, which are now written into the ZIP README:
  - it compares ML with the monthly baseline rather than `bootstrap.best_baseline`;
  - it cannot parse our `bootstrap` block, so it says no uncertainty was reported;
  - its "storage above 102% of capacity" note covers 239 ORO days before 2024-07-02, measured against the current capacity; the UI should use `storage_pct`.
- No change to `app.py`, `app/` or any frozen modelling file.

### Phase 5: final wording and attribution (2026-11-21)
- **ResOpsUS wording** was made precise in README, `data/provenance.md`, `docs/originality_check.md`, the Follow-up section above and `results/v2/resopsus_check/report.md`. The report now has an Interpretation section generated by `src/compare_resopsus.py`; only that text changed, and every number and CSV is identical.
  - **Oroville** is predominantly one day off relative to our end-of-day dates. The one-day match is at least 0.91 in every year except 2008 (0.62).
  - **McClure (EXC)** is one day off from 2001 to about March 2003. It is predominantly aligned at zero lag from about April 2003 to May 2014 (2004-2013 aligned), mixed in the June 2014 transition, and one day off from July 2014.
  - The authors' intended date convention is unknown, and no error is claimed. The offsets matter mainly for daily-resolution analysis.
  - The comparison is described as only partly independent, because four of the six records come from CDEC.
- **Primary model description** added to the README: it uses reservoir history, reservoir identity, seasonality and CDEC SWE. NASA POWER weather appears in the dataset and the secondary variants only. The longer, corrected record explains most of the gain over v1, and the independent value of SWE or weather is not conclusively established. The marginally better secondary models (2.32) are named, and the primary model was not reselected.
- **Test disclosure:** the exact required paragraph is in README section 5. It replaces the shorter disclosure bullet; the preregistration text is unchanged.
- **Folsom under-coverage (0.75)** now has its own bullet in the README results, in addition to Limitations.
- **Attribution:** a new README section, "Attribution and terms", contains:
  - the CDEC wording, marked **PENDING OWNER APPROVAL**;
  - a statement that no specific CDEC reuse licence has been verified;
  - the NASA POWER acknowledgement, unchanged;
  - the ResOpsUS citation (Steyaert, Condon, Turner and Voisin 2022, DOI 10.1038/s41597-022-01134-7; Zenodo record 5893641, CC BY 4.0).
  Provenance and the UI contract point to this section.
- Another stale count was corrected: the provenance line on the raw manifest said 187 entries. The file has 189: 188 pipeline inputs, as at `e47deab` and in the preregistration, plus 1 ResOpsUS entry.
- No v1 or v2 metric changed. The README results tables are untouched.

### Phase 6: CI and merge preparation (2026-11-21)
- Confirmed tracked in PR #2, not recreated: `results/v2/resopsus_check/report.md` and `src/compare_resopsus.py`.
- **CI added** in `.github/workflows/ci.yml`: Ubuntu, Python 3.14, `pip install -r requirements-lock.txt`. It runs:
  - all tests;
  - the contract check on `app/sample_data`;
  - a read-only check that the frozen config and code hashes equal `RUN_RECORD.json`;
  - a check that no raw, processed, outputs or dist files are tracked.

  It never downloads data, never runs `src.train_v2 --final` and never writes to the config. Every step was simulated locally in a data-free clone and passes; the first GitHub run is checked after the push.
- **No test is excluded from CI.** All 30 tests pass without local raw or processed data. One test, `tests/test_contract.py::test_real_outputs_follow_contract_if_present`, validates `outputs/` only when it exists, so in CI it checks nothing. The real outputs are validated locally (`python -m src.contract outputs`, `python -m src.ui_export --verify`).
- **Merge plan** in `docs/merge_plan.md` (owner actions, 7 steps): merge PR #1 with a merge commit; retarget PR #2 to `main`; resolve any conflicts without touching frozen files; run checks; merge PR #2; integrate the frontend; tag `submission-v2`. Branch ancestry was checked: `main` has nothing that `engineering` lacks, and `engineering` is an ancestor of `review-v2`. The PR #2 description has been updated with the plan. Neither PR was merged.

## 30-day research forecast feature (branch `feature/30-day-forecast`, 2026-11-22)

The frozen v2 evaluation is untouched. The frozen code and config are byte-identical to `e47deab`, `results/v2/final` and `results/v1` are identical to `f21bf25`, and the frozen hashes still equal `RUN_RECORD.json`. Nothing was rerun under `src.train_v2`.

### Phase 1: audit and feasibility
- **Done:** wrote `docs/FORECAST_IMPLEMENTATION.md`.
- **Findings:**
  - No v2 model weights existed; `models/forecast.joblib` is a v1 artifact.
  - The primary model can be refit exactly from the private processed data.
  - CDEC serves storage through 2026-11-20 and sensor 82 through 2026-11-21.
- **Problem found:** recent sensor 82 values are only partly flagged as revised, so the live SWE is not identical in kind to the mostly revised training values. This is disclosed in every forecast.

### Phase 2: prediction-time features
- **Done:** `src/forecast_live.inference_features` reads storage on dates up to t and SWE up to t-1 only. It never needs t+30.
- **Tests:**
  - On all 593 frozen test origins, its 9 non-station inputs equal `weekly_features.csv` plus `add_snow_anomaly` within 1e-9.
  - The model-input order equals `models.design()`.

### Phase 3: model reconstruction
- **Done:** `src/forecast_model.py --build` refits with the frozen code path and refuses to write unless every archived prediction is reproduced.
- **Result:**
  - 593/593 reproduced, max difference 1.8e-15 pp.
  - Artifact SHA-256 `00eec470...`.
  - The interval offsets in the card equal the archived interval widths for every season.
  - The artifact is git-ignored. Loading refuses a tampered file.

### Phase 4: data refresh
- **Done:** downloads go to `data/live/<issue date>/` with their own fetch log. They never touch `data/raw/` or `data/raw_manifest.jsonl`. Cleaning reuses `build_dataset.load_cdec` and `load_snow` unchanged.
- **Real run on 2026-11-22:**
  - 14 files, all hashes verified.
  - The 2025 storage overlap equals the archived data on all 2,190 station-days.

### Phase 5: forecasts
- **Done:** real forecasts were issued 2026-11-22.
  - SHA, ORO, FOL, NML and EXC: origin 2026-11-16, target 2026-12-16.
  - DNP: origin 2026-11-09, flagged stale. Its basin index lacked 2 of 3 core pillows after 2026-11-12.
- No projection fell outside 0-100%. FOL carries the interval caution.
- Records are written to `forecasts/records/` and are never overwritten.

### Phase 6: Streamlit page
- **Done:** added `app/view_forecast.py` and registered it as the fifth page in `app.py`. That is a UI-team file, edited at the owner's explicit request.
- The page is read-only and validated. It shows dedicated states for:
  - no snapshot;
  - an invalid snapshot;
  - an expired forecast;
  - a reservoir that is unavailable;
  - stale data;
  - a synthetic fixture.
- **Checked:** in a browser against the real snapshot.

### Phase 7: deployment
- **Done:** documented in README section 10.
- **Blocker:** the snapshot, the downloads and the model artifact are CDEC-derived, so a public deployment cannot show real forecasts until the owner confirms CDEC terms. A private snapshot source is also needed and is not implemented.

### Phase 8: tests
- **Added:**
  - `tests/test_forecast.py`: 29 tests. 5 need private data or the artifact and are skipped in CI.
  - `tests/test_forecast_page.py`: 4 headless page tests with Streamlit AppTest.
- **Full suite:** 63 passed locally.
- **Other checks:**
  - `src.contract` passes on `outputs` and `app/sample_data`.
  - `app/check_v2_integration.py outputs` passes.
  - `report_facts --check` and `ui_export --verify` pass.
- **CI:** now installs plotly for the page tests, checks that no live, forecast or model files are tracked, and checks that the synthetic fixture is reproducible.

### Phase 9: documentation
- **Done:** README section 10, `docs/FORECAST_IMPLEMENTATION.md` (log and checklist), this entry.

### Remaining blockers
- CDEC redistribution terms, which block public deployment of real forecasts.
- No prospective validation yet. The first targets fall on 2026-12-09 (DNP) and 2026-12-16 (the rest).
- Sensor 82 revision status at forecast time.

### Follow-up: plainer site guidance (2026-11-22)
- **Overview:**
  - Added "How to read this site", a plain description of the five pages.
  - The storage heading now names the last day of the historical dataset (Dec 31, 2025) instead of "end of the record".
  - Added "Latest observed storage" from the CDEC refresh, shown only when a valid, unexpired snapshot exists. Otherwise a caution says the gauges are not today's levels.
- **Forecast page:** added "Who this forecast helps". It covers:
  - what the model accounts for;
  - what it cannot see (future storms, planned releases);
  - that it is not a tool for operators;
  - where it is useful: research, a benchmark, public understanding.
- **Tests:** 3 new page tests; 66 passed locally.

## v3: weather-aware models and interactive forecasting (branch `feature/v3-forecast-platform`, 2026-11-22)

This branch is based on `feature/30-day-forecast` (PR #4, open, CI passing). It must be merged after PR #4, or retargeted to `main` once PR #4 is merged. The frozen v2 files are unchanged, and the frozen hashes equal `RUN_RECORD.json`.

### Phase 0: protection
- The 2026-11-22 Model A record is read-only on disk. Its SHA-256 is in the tracked ledger `docs/v3/issued_forecast_ledger.jsonl`, and a test checks it.
- Model A is the unchanged v2 artifact.

### Phase 3 (done first): data rights and forecast-archive audits
- **Data rights:** `docs/DATA_PUBLICATION_AUDIT.md` and `config/data_release_policy.json`.
  - NOAA GEFS (operational and reforecast) and NASA POWER: confirmed redistributable.
  - CDEC: unresolved. DWR's Conditions of Use contain only a disclaimer, and CDEC pages carry "Copyright © State of California". A permission request is drafted.
- **Weather-forecast archives:** `docs/v3/MODEL_C_FEASIBILITY.md`.
  - Verified directly on AWS: GEFSv12 reforecast 2000-2019 (00Z daily, 5 members, to day 16) and operational GEFSv12 from 2020-09-23.
  - The two use the same APCP encoding. Operational S3 publication times are recorded.

### Phase 1: models and data
- **Model B** reproduces the v2 HGB + SWE comparison exactly: 2.3231 against Model A's 2.3323 on 2024-2025, a marginal difference already published.
- **Model B2:** upstream-box NASA POWER means. The 2001-2025 POWER re-download is identical to the v2 weather on all 54,786 overlapping days.
- **Model C:** basin-box GEFS precipitation for days 1-7 and 8-16, from the control member.
  - 1,313 Monday cycles from 2001 to 2026 were downloaded in byte ranges.
  - One reforecast cycle (2007-06-04) has a malformed index and is recorded as incomplete.
- **v3 weekly table:** identical to the v2 table on all 7,693 shared rows. It adds 24 December 2025 rows and 251 rows with 2026 origins.

### Phase 2: scientific comparison (run once, 2026-11-22)
- **Development folds (WY2008-2025):**
  - C vs A on paired rows: 2.717 vs 2.832.
  - B vs A: 2.802 vs 2.803.
  - B2 vs B: 2.809 vs 2.802.
- **2026 test (251 rows):** A 2.349, B 2.339, B2 2.364, C 2.316, frozen A 2.355, seasonal baseline 2.882.
  - H1, H2 and H3 are all "not established": every 95% CI includes zero.
  - Interval coverage is 0.93-0.94 overall, but only 0.69-0.74 at Folsom.

### Phases 4-6: interactive forecasting
- `src/v3_service.py`:
  - a shared cache in 6-hour (CDEC), 12-hour (POWER) and per-cycle (GEFS) buckets;
  - coalescing of identical requests, upstream rate limiting and a cooldown on forced refreshes;
  - cache entries hash-checked, and records written once;
  - a policy gate for public deployments.
- **Forecast page:** a "Generate a forecast" tab with cards for Models A, B, B2 and C. The chart shows only the day-30 endpoint with error bars; no straight-line path is drawn. It also has the model comparison, weather charts (observed for B, forecast for C), data freshness, provenance and a download. The issued v2 record is in a second tab.
- **Live demo:** Oroville on 2026-11-22 in the local app, with all four models available.

### Phase 7: tests
- 96 tests passing locally before the final run.
- The CI simulation in a clean clone is recorded in the PR.

### Remaining blockers
- See `docs/v3/V3_REPORT.md`: CDEC public display rights, public hosting of the model artifacts, and persistent records on Streamlit Community Cloud.

### PR #5 review fixes (2026-11-22)
- **CI failure fixed:** `test_external_api_failure_gives_no_forecast` failed on the clean runner because the missing-artifact guard ran before the mocked CDEC timeout.
  - The service tests now inject a controlled mock model, so the CDEC failure path really runs.
  - A separate test drives the real loading path with no artifacts installed and checks that nothing is fetched.
  - The service's safety checks are unchanged.
- **Public gate widened:** with `WATERSHED_DEPLOYMENT=public`, `app.py` renders no page content from real (CDEC-derived) outputs, and the "Issued record" tab hides real snapshots. Both are tested.
- **Comparison table:** now shows paired, same-row MAE against Model A, with the preregistered verdicts displayed on the page. README and report wording no longer imply any unsupported superiority.
- **Artifact provisioning:** `src/v3_artifacts.py` packs and installs hash-pinned artifacts and installs them at app start when configured. Tested.
- **Merge sequence:** simulated without conflicts. CI now re-runs when a PR is retargeted. Details in `docs/v3/LAUNCH_AND_MERGE.md`.

## Merge and public-launch preparation (2026-11-22)

- **Merged:** PR #4 into `main` (merge commit `32d6122`). PR #5 was retargeted to `main`, its CI run 44997714780 passed (97 passed, 11 skipped), and it was merged (merge commit `f972621`). The `main` push CI is green for both.
- **Post-merge checks on `main`, all passing:**
  - full suite: 109 passed;
  - frozen v2 unchanged, with hashes equal to `RUN_RECORD.json`;
  - UI contracts, v2 integration, report facts and the UI zip;
  - Model A reproduces 593/593 predictions;
  - the v3 artifacts reproduce the 2026 predictions;
  - the 2026 metrics and verdicts are unchanged;
  - public-release checksums;
  - the ledger: both issued records are unchanged and read-only.
- **Publication:** CDEC flags set to `true` on the owner's reported authorization (`docs/DATA_PUBLICATION_AUDIT.md`, section 7); the written permission is still to be archived. Release `2026-11-22.2` includes the CDEC-derived data, and CI enforces the policy against every published file.
- **Model bundle:** `dist/watershedshift_models_7ad581d9f585.tar.gz`, SHA-256 `7ad581d9f58594050688ec33cb810ff5ff513f2df3e233cc8fc1160eb193b97d`. It is byte-reproducible, self-checking and **not uploaded**.
- **Clean deployment run locally in public mode** (fresh clone, bundle served over HTTP):
  - all four models generated real forecasts for Folsom and Shasta, the second served from the shared cache;
  - all five pages rendered;
  - on a 375 px phone viewport, cards stack with no horizontal scroll;
  - a wrong pin, a missing URL, no configuration and a truncated bundle are each refused safely.
- **Bugs found and fixed during this work:**
  - the Model and Research page crashed on real data (missing chart symbol, from before this work);
  - an empty `outputs/` folder in a clone hid the public release;
  - a truncated bundle raised an unhandled error.
- **Public website not verified:** its URL is not in the repository and there is no host access. The site cannot forecast publicly until PR #6 is merged and the bundle is uploaded and configured.

## Engineering (Claude Code): recovery, reliability, prospective protocol and v4 (2026-11-23 to 2026-11-24)

- **Recovery review (2026-11-23, read-only).**
  - All frozen v2 and v3 hashes intact; both issued records match the ledger.
  - The model bundle **is uploaded**: GitHub Release `models-2026-11-22`, a pre-release whose digest equals the pinned SHA-256. This supersedes "not uploaded" above.
  - Public website still **not verified**: no URL available.
- **Merged** (each with the owner's explicit approval):
  - PR #7, `CLAUDE.md` and `docs/PROJECT_HANDOFF.md` (2026-11-24 06:32 UTC, `0d524a5`);
  - PR #10, prospective evaluation and scoring protocol v1.0 with reference scorer `src/prospective.py`, and no scheduler (2026-11-24 09:37 UTC, `e79c54b`). This is before the first target date, 2026-12-09.
- **Open for review:**
  - PR #8: retry for failed model installs; NASA POWER no longer delays Model A; write-once records; GitHub records repository backend, chosen by the owner, which needs the owner's repository and token;
  - PR #9: v4 NeuralHydro research. Shared LSTM/GRU/TCN on a daily, leakage-tested pipeline. They match HGB with the frozen v2 configuration on the development folds but do not beat it;
  - the docs-consistency PR, which adds this entry.
- **Tests on `main`:** 132 passed with the private data; 118 passed and 13 skipped without it.
- **Current state and priorities:** `docs/PROJECT_HANDOFF.md`.

### 2026-11-24 (later): all open PRs merged

- **Merged, each with the owner's explicit approval:**
  - PR #11, docs consistency (`57cb7d7`);
  - PR #8, forecast reliability and GitHub record store (`50c83c6`);
  - PR #12, official weekly issuance command, manual with no scheduler (`e3e3466`);
  - PR #9, v4 NeuralHydro research, not deployed (`1accaa6`).
- **Testing before merging:** PRs #8 and #9 were tested merged onto the then-current `main` locally first.
- **No PRs open.**
- **Waiting on the owner:**
  - the GitHub records repository and token, needed before the first official issue;
  - the public Streamlit URL, needed for the deployment check.
- **Next due:** provisional scores for the two pre-protocol records, from 2026-12-11 (DNP) and 2026-12-18 (the others).


### 2026-11-25: v4 N4 built and evaluated on development folds (branch `research/v4-n4`, not merged)

- **Built:** N4, a water-balance-structured encoder-decoder LSTM (`WaterBalanceED` in `src/v4/nets.py`, composite loss in `src/v4/train.py`), with offline tests in `tests/test_v4.py`.
- **Plan fixed before any run:** `docs/v4/N4_PLAN.md`, 2026-11-25 04:00 UTC.
- **Tuning folds WY2012–2015:** selected `wb_noaux` (water-balance layer, no inflow and outflow loss terms) with a 30-day history.
- **Confirmation folds WY2016–2023, 2,500 identical rows:**
  - N4 3-seed ensemble: 3.084 pp;
  - HGB SWE no weather per-fold refit: 3.160 pp;
  - difference −0.076 pp, **not established** (95% CIs span zero at 5-, 9- and 13-week blocks).
- **Other findings:** the water-balance layer adds no measurable accuracy (−0.008 against the direct-head ablation). The intervals undercover (90% nominal gives 0.840).
- **Not frozen, not deployed.** Nothing from v4 is on the website, per the owner's decision of 2026-11-24.
- **Details:** `docs/v4/RESULTS_N4.md`.

### 2026-11-27: v5 benchmark: HGB v5 and the retraining of every implemented model (branch `research/v5-benchmark`, not committed)

- **Plan fixed before any v5 result:** `docs/v5/BENCHMARK_PLAN.md`, SHA-256 `53aa2df4…`. Seven stage manifests record it, with UTC run times from 05:12 to 08:52.
- **New data:** CDEC inflow and outflow (sensors 76/23) for 2024-01-01 to 2026-11-20, downloaded 2026-11-27. Private, like all flow data.
- **Consolidated dataset:**
  - `data/v5/daily_panel.parquet`, 56,730 × 18;
  - `data/v5/features_daily.parquet`, 56,730 × 31.
  Both are private. Before 2024 the panel is identical to the v4 panel, and its Monday rows are identical to the v3 weekly table.
- **Run:**
  - 13 tabular configurations plus 4 baselines;
  - 14 network configurations × 3 seeds;
  - an equal-budget search of 20 configurations for each of 6 algorithms on WY2012–2015;
  - tuned models with 3 seeds;
  - post-hoc windows HELD (2024–2025) and T26 (2026).

  That is 1,060 saved networks and 196 saved tabular models, under `models/v5/` (git-ignored).
- **Reproduction:**
  - the frozen v2 artifact on HELD (2.3323), v3 Models A, B and B2 on T26, and the v4 networks are all reproduced exactly;
  - v3 Model C differs, as expected: v5 trains on all rows with GEFS missing where unavailable.
- **Confirmation folds WY2016–2023, 2,500 identical rows (exploratory):**
  - lowest MAE: N4 wb_noaux with full information, 2.981 pp;
  - HGB v5 [all], 3.074;
  - HGB[S,N,C,R] reference, 3.160.
- **Planned comparisons** (block bootstrap; Holm correction within each family): only two are supported.
  - HGB v5 beats storage-only HGB[S,C,R] by 0.226 pp.
  - Tuned HGB v5 beats tuned Ridge by 0.652 pp.
  - Not established: HGB v5 against the reference (−0.086), and every network against HGB v5. The lowest-error model differs in each population.
- **Not frozen, not deployed, not on the website.**
- **Details:** `docs/v5/RESULTS.md`.
- **Draft prospective protocol:** `docs/v5/PROSPECTIVE_PROTOCOL_DRAFT.md`. It is not in force; adopting it is the owner's decision.

### 2026-11-27: website: frozen v2 HGB is the only forecasting model, plus a retrospective Model playground (branch `web/hgb-main-playground`, not committed)

Owner's request: the site's forecasts use only the HGB, and the other nine study models go in a separate testing playground. After asking for a recommendation, the owner chose the frozen v2 HGB (not HGB v5 or N4) as the main model.

- **Generate Forecast** (`app/view_forecast_v3.py`):
  - runs only HGB[S,N,C,R] · frozen v2 artifact; v3 Models B, B2 and C are no longer shown;
  - `src/v3_service.py` gains `ForecastService(model_keys=...)`. The default is still all four, so `src/official.py` and the prospective protocol are unchanged. With `("A",)` the service never fetches NASA POWER or GEFS;
  - no frozen file changed: v2 hashes and v3 `code_sha256` were checked;
  - no new model bundle or secret is needed.
- **Live check** (local, private mode, 2026-11-27 ~21:50 UTC): all six reservoirs returned `ok`. Origin 2026-11-23 (Monday), target 2026-12-23, only CDEC fetched. Example: SHA 50.0% → 53.7% (+3.72 pp). **The public URL has not been checked.**
- **Model playground** (`app/view_playground.py`, data in `src/v5/playground.py`):
  - shows the other nine models against the HGB on confirmation WY2016–2023 (exploratory) and on 2024–2025 and 2026 (post-hoc);
  - reads only tracked `results/v5` aggregates: metrics, breakdowns and paired comparisons. Nothing is fitted, re-scored or tested in the app;
  - model 6 is the frozen artifact on 2024–2025 and 2026, and the per-fold refit HGB[S,N,C,R] on confirmation;
  - per-row replays need the private `data/v5/preds/all_predictions.parquet`. On a public deployment, the five flow-using models (Ridge[all], LSTM, GRU, TCN, N4) are withheld, because CDEC sensors 76/23 are not in `config/data_release_policy.json`.
- **Owner decision recorded:** v4/v5 models may appear in the retrospective playground only. None makes live forecasts.
- **Tests:** `tests/test_playground.py` (18) and updated `tests/test_forecast_page.py`. Full suite: 198 passed, 5 skipped. Contract, frozen-hash, `v3_release --check` and fixture checks all OK.
- **Merge dependency:** the playground reads `results/v5/`, which arrives with PR #15 (stacked on PR #14). This branch is cut from `research/v5-benchmark`.

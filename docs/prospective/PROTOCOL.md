# Prospective evaluation and scoring protocol, version 1.0

**Status.**
- Drafted 2026-11-24.
- **It takes effect when merged into `main`.** That must happen before 2026-12-09, the first target date of an issued forecast, so the protocol is fixed before any of the outcomes it scores exist.

**Machine-readable form.** `config/prospective_protocol.json` holds the SHA-256 of this document and of the fixed baseline table. The reference scorer `src/prospective.py` refuses to run if either file differs. Changes are made only as described in section 7.

## 1. Purpose and scope

The aim is to measure how the **frozen** WatershedShift models perform on forecasts issued *before* their outcomes exist.

| Model | Version | Role |
|---|---|---|
| A | frozen v2 (`models/v2_forecast`) | **Primary** |
| B, B2, C | v3 final models (`models/v3`) | Secondary |

- No model is retrained, recalibrated or replaced because of prospective results.
- v4 research models are out of scope. Including one would need a new protocol version, committed before that model's first official forecast, after the model is frozen.
- This protocol covers statistical accuracy only. Forecasts are not reservoir-operation simulations and are not for operational or safety decisions.

## 2. Record classes

| Class | What | Scored? | In the primary analysis? |
|---|---|---|---|
| Pre-protocol issued | The two records in `docs/v3/issued_forecast_ledger.jsonl`, both issued 2026-11-22 before their outcomes existed but before this protocol | Yes, with the same rules | **No.** Reported separately as descriptive results |
| Official weekly | Records issued under section 3 and listed in `docs/prospective/official_ledger.jsonl` | Yes | Yes |
| Visitor | Forecasts generated on the website by anyone pressing Generate Forecast | **Never** | Never |

The pre-protocol records are:
- `wsf-v2-2026-11-22-20261122T052508Z`: Model A, all six reservoirs. Targets are 2026-12-09 at DNP and 2026-12-16 elsewhere.
- `wsf-v3-ORO-20261122T064412Z`: Models A, B, B2 and C at Oroville, target 2026-12-16.

## 3. Official weekly issuance

1. **Cadence and origin.**
   - One official record per week, covering all six reservoirs: SHA, ORO, FOL, NML, DNP, EXC.
   - The origin is the **Monday immediately before the issue day**.
2. **Issue window.** From Tuesday 00:00 UTC to Wednesday 23:59 UTC of the same week, which leaves time for CDEC to post Monday's end-of-day storage.
3. **Models in each record.**
   - A, B, B2 and C, each with its 90% interval (`sym_season`, model-specific) and an explicit available or unavailable status with reasons.
   - Inputs follow the frozen information rules: storage to the origin, snow to origin−1, NASA POWER weather to origin−7, and the GEFS cycle of the origin Monday only if it was published before the issue time.
4. **No substitution.** If the week's Monday lacks complete inputs for a reservoir, the record holds an explicit **no-forecast** entry for it, with reasons. An older origin is never used in an official record. This is stricter than the website, which may fall back up to 14 days.
5. **Missed weeks.** If no record is issued within the window, a "missed week" line with the reason is appended to the official ledger. Missed weeks count against availability.
6. **Immutability and evidence.**
   - Each official record has `record_type` `official_weekly_forecast`.
   - It is written once to the persistent record store (the owner-chosen GitHub records repository) and never changed.
   - Within the issue window, its SHA-256, issue time and origin are appended to `docs/prospective/official_ledger.jsonl` in a commit to the main repository. The commit time is independent evidence that the forecast preceded its outcome.
7. **Who and how.**
   - The owner, or an operator the owner designates, issues the record manually.
   - The issuing command will be added in a separate PR after the persistent store (PR #8) is merged and verified.
   - **No scheduler is used until this protocol is merged and the storage is verified.** A scheduler, if later approved, must produce records identical in form to manual ones.
8. **First official week.** The first Monday origin after this protocol is merged and the issuing command exists. Weeks before that are not "missed".

## 4. Scoring

1. **Observation.** CDEC daily reservoir storage (sensor 15), retrieved at scoring time.
   - The raw response is kept with its SHA-256, under `data/prospective/raw/` on the scoring machine.
   - The score records the URL and the SHA-256.
2. **Realised change.** Exactly the training target:

   100 × (S[target] − S[origin]) / capacity in force at the origin

   - The capacity is taken from the forecast record.
   - Both storages come from the same retrieval.
   - Exact calendar days only; never interpolated.
3. **Two stages.**

   | Stage | Allowed from | Use |
   |---|---|---|
   | Provisional | target date + 2 days | Monitoring only |
   | **Settled** | **target date + 30 days** | Every analysis in section 5 |

   If a settled observation is missing, scoring may be retried until target + 90 days, after which the entry is permanently **unscorable**.
4. **Unscorable entries** are never imputed. An entry is unscorable when CDEC storage is missing on the origin or target date, or when a series break (`data/series_breaks.csv`) lies between them.
5. **Per model and reservoir:**
   - forecast change;
   - error (forecast − realised) and absolute error;
   - whether the realised change falls inside the 90% interval;
   - level error (projected percentage of capacity − realised percentage of capacity).
6. **Origin revision.** Settled S[origin] minus the storage the forecast used at issue time. It is flagged when larger than 1% of capacity. The forecast is still scored against the settled values, with the flag reported.
7. **Baselines on the same rows:**
   - **Fixed smoothed day-of-year seasonal baseline.** `config/prospective_doy_baseline.csv` is built once from weekly rows with target date before 2026-01-01, with a Gaussian kernel (σ = 15 days) on the origin day-of-year, per reservoir. It is identical to `src.models.baselines` (checked to within 5e-7) and is hash-registered.
   - **Persistence** (zero change).
8. **Score files are write-once,** one per record, reservoir and stage, under `results/prospective/scores/<stage>/<record>/`. They are never edited. A later CDEC revision shows up by comparing stages, not by rewriting a score.

## 5. Preregistered analyses

All primary and secondary analyses use **settled scores of official weekly records** only.

**P1 (primary): is Model A more accurate than the seasonal baseline?**
- **Sample:** the first **52 official weekly origins** with settled scores, all reservoirs pooled, rows where Model A is available.
- **Statistic:** paired MAE difference, A minus the fixed day-of-year baseline.
- **Uncertainty:** moving-block bootstrap over origin weeks (all reservoirs on a date resampled together), with blocks of 5, 9 and 13 weeks, 2,000 replicates, seed 42. This uses `src.evaluation`, unchanged.
- **Decision:** *supported* only if the upper end of the 95% CI is below zero at all three block lengths. Otherwise *not established*.

**P2 (primary): interval coverage.**
- Empirical coverage of Model A's 90% intervals on the P1 rows, overall and by reservoir, with a 95% moving-block bootstrap CI using 13-week blocks.
- Folsom is reported explicitly. Its intervals undercovered in 2024–2025 and in 2026.

**Secondary (same sample rule, decision rule and labelling):**

| ID | Comparison | Repeats |
|---|---|---|
| S1 | C vs A | v3 H1 |
| S2 | B vs A | v3 H2 |
| S3 | B2 vs B | v3 H3 |

Each is computed on rows where both models are available. A secondary result never overrides a primary one.

**Descriptive:**
- MAE by reservoir and by season;
- availability for each model, as the share of scheduled official reservoir-weeks with an available forecast;
- unscorable and missed weeks;
- origin revisions;
- the two pre-protocol records, reported separately.

**Timing.**
- The P1/P2 analysis runs **once**, when the first 52 official origins have settled scores. The scorer refuses to run it earlier.
- Interim summaries are descriptive, at most quarterly, and labelled *interim*. There are no interim significance claims and no stopping rules.

**Availability caveat.** MAE is computed over available forecasts only, and availability is always reported next to it. If Model A is available for fewer than 90% of scheduled reservoir-weeks, the P1 result is reported with that caveat.

## 6. Integrity rules

- Forecast records (pre-protocol and official) are read, never modified.
  - Their SHA-256 must match the ledger, or the scorer refuses.
  - Ledgers are append-only.
- **Frozen research is untouched.** No frozen model, configuration, preregistration, result or public data release is changed by this protocol or its scorer.
- **No reselection.** Prospective outcomes are never used to select, tune or recalibrate any frozen model, or to choose among analyses after the fact.
- **No rewriting of claims.** Results are reported as *supported* or *not established* under section 5. Retrospective reforecasts and backtests are never presented as prospective results.

## 7. Amendments

Any change needs a new protocol version (1.1, 1.2, …). It must be committed with:
- a dated entry here explaining the change;
- an updated `config/prospective_protocol.json` with new hashes.

An amendment applies only to forecasts **issued after** it is committed. It can never change how already-issued forecasts are scored or analysed.

(No amendments yet.)

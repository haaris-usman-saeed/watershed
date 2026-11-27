# v5 benchmark plan: HGB v5 and the standardised retraining of every implemented model

Written 2026-11-27 (UTC) **before any v5 model was trained or scored.** The only data step run before writing it was the CDEC flow download for 2024-01-01 to 2026-11-20 (`python -m src.v5.flows_ext --fetch`), which computes coverage and nothing else. This file is not edited after the first v5 run. Every v5 run manifest records its SHA-256; a later change would show. Changes go in a dated "Amendments" section at the end.

## 1. Scope and rules

- **Nothing frozen changes.** No frozen model, configuration, preregistration, result, issued record, public release or v4 result is modified, refit or rerun. Every model here is a **new per-fold refit** of an existing *configuration*, saved under `models/v5/`. The frozen v2 artifact (`models/v2_forecast/model.joblib`) is only a reference, using its recorded predictions.
- **Nothing goes on the website.** The 2026-11-24 owner decision about v4 applies to v5 as well.
- **Code.** New code goes in `src/v5/`. It reuses v2, v3 and v4 functions read-only.
- **Where outputs go:**
  - aggregates: `results/v5/`;
  - per-row predictions and private tables: `data/v5/` (git-ignored);
  - models: `models/v5/` (git-ignored).
- **Flow data.** CDEC flow data (sensors 76 and 23) is not covered by the data release policy. Only aggregates are written to tracked files.

## 2. Target, origins, information rule

- **Target:** y = 100 × (S[t+30] − S[t]) / C[t], in pp of the capacity in force at the origin, as in v2. Every evaluation row is a v2 Monday row (storage, dam-point weather and target present).
- **Information at origin t:**
  - storage and flows up to *t*;
  - basin snow up to *t* − 1;
  - NASA POWER weather up to *t* − 7;
  - GEFS: the latest *usable* Monday 00Z cycle issued at or before *t*, used under the v3 rule. That means a complete cycle; an operational cycle must also be published by init + 32 h. The age *t* − init must be 0–6 days, otherwise GEFS is missing.
  - On Monday evaluation rows the age is 0, which is exactly the v3 feature.
  - Reforecasts (2001–2019) are retrospective and labelled so. No GEFSv12 product exists for 2020-01-06 to 2020-09-21.
- **Missing data.** Missing values are never filled with invented observations:
  - HGB uses native missing-value handling;
  - neural networks use explicit masks;
  - Ridge uses training-fold median imputation with missing-indicator columns. This is documented as a representation difference.

## 3. Consolidated dataset

**The daily panel** (`src/v5/data.py`) has one row per reservoir-day from 2001-01-01, built from:
- the v4 sources;
- the 2026 CDEC storage download used by v3 (`data/live/2026-11-22`);
- the v3 snow index;
- the 2024–2026 flows;
- the eligible GEFS cycles aligned as in §2.

**Development stages** use a panel truncated before 2024-01-01; it must be identical to the v4 panel (checked). Only the retrospective stage builds the panel through 2026-11-20.

**The tabular feature table** holds the v2 feature definitions computed on every day, plus box weather, GEFS and the four flow features. Its Monday rows must equal `data/v3/private/weekly_v3.csv` on every shared column (checked).

## 4. Models

**Notation.** Information codes: S storage, N snow, Wd dam weather, Wu upstream weather, G GEFS, Q flows, C calendar, R reservoir.

**Fixed settings:**
- "Frozen HGB" means the v2 hyperparameters: lr 0.03, max_iter 150, max_depth None, min_samples_leaf 30, l2 1.0, max_leaf_nodes 31, random_state 42.
- "v4 defaults" for neural networks means:
  - architecture: hidden 64, 1 layer, embedding 4, dropout 0.2;
  - training: AdamW with lr 1e-3, weight decay 1e-4, batch 256, at most 30 epochs, patience 4;
  - history L = 30, every 2nd daily origin;
  - the N4 loss weights from `docs/v4/N4_PLAN.md`.

**Baselines,** refit per fold on the same weekly training rows: persistence; 30-day trend; monthly climatology; DOY kernel climatology (σ = 15 days).

**HGB v5, the full-information model, HGB[S,N,Wd,Wu,G,Q,C,R]:** 29 inputs.

| Group | Columns |
|---|---|
| Storage | storage_pct, prev7_pct, prev30_pct, prev60_pct |
| Snow | swe_basin_in, swe_change_14d_in, swe_anom_in (fold-specific climatology) |
| Dam weather | rain_7d_mm, rain_30d_mm, temp_30d_c, humidity_30d_pct |
| Upstream weather | the same four, as `_box` |
| GEFS | gefs_p_d1_7_mm, gefs_p_d8_16_mm |
| Flows | inflow_7d_pp, inflow_30d_pp, outflow_7d_pp, outflow_30d_pp |
| Calendar | season_sin, season_cos |
| Reservoir | six station one-hots |

Each flow feature is the mean daily flow over the 7 or 30 days ending at *t*, in percent of capacity per day. It is missing unless every day of the window is observed. These are the four features of the v4 post-hoc flow ablation.

**Experiment B: historical configurations.** These are the original feature sets, retrained on identical folds and rows.

| ID | Model | Origin |
|---|---|---|
| B-HGB-SWdCR | HGB[S,Wd,C,R], frozen HGB | v1/v2 "HGB", storage and weather |
| B-HGB-SCR | HGB[S,C,R] | v2 "HGB no weather" |
| B-HGB-SNCR | HGB[S,N,C,R] | v2 primary "HGB SWE no weather" and v3 Model A. **The HGB reference.** |
| B-HGB-SNWdCR | HGB[S,N,Wd,C,R] | v2 "HGB + SWE" and v3 Model B |
| B-HGB-SNWuCR | HGB[S,N,Wu,C,R] | v3 Model B2 |
| B-HGB-SNGCR | HGB[S,N,G,C,R] | v3 Model C. Trained on all rows with GEFS missing where unavailable; this differs from the v3 rule, which used GEFS rows only |
| B-HGB-SNQCR | HGB[S,N,Q,C,R] | v4 post-hoc flow ablation |
| HGB-V5 | HGB[S,N,Wd,Wu,G,Q,C,R] | new |
| B-RIDGE-SWdCR | Ridge[S,Wd,C,R] (StandardScaler + RidgeCV, logspace(−2, 3, 20)) | v1 Ridge |
| B-RIDGE-SCR | Ridge[S,C,R] | v4 ridge autoregression baseline |

Neural networks, v4 defaults, original v4 inputs: daily S, Q, N, Wd, Wu and C sequences with masks, plus reservoir embedding and log capacity. **No GEFS.**
- N1 LSTM; N1 without flows; N2 GRU; N3 TCN;
- N4 `wb_noaux` (the v4-selected N4); N4 `wb_balanced`; N4 `wb_primary`; N4 `wb_strict`; N4 `direct` (ablation).

**Experiment A: controlled information.** Every algorithm receives S, N, Wd, Wu, G, Q, C and R.

| ID | Model | Representation |
|---|---|---|
| A-RIDGE | Ridge[all] | the 29 HGB v5 columns, with median imputation and missing indicators |
| HGB-V5 | HGB[all], weekly Monday training rows | 29 columns |
| A-HGB-DAILY | HGB[all] · daily | the same 29 columns plus `gefs_age_days`, trained on every 2nd daily origin with target before the fold, which is the neural networks' sample set (fit plus early-stopping days) |
| A-N1, A-N2, A-N3, A-N4, A-N4-DIRECT | N1, N2, N3, N4 `wb_noaux`, N4 `direct`, each [all] | the original daily sequences plus GEFS as static inputs: d1–7 and d8–16 (arcsinh(mm)/3), an availability mask and age/7 |

**Representation differences, stated rather than hidden:**
- Tabular models see a 60-day storage change. The networks see only the last 30 days of daily storage.
- Tabular weather covers days *t* − 36 to *t* − 7. The networks see daily weather for *t* − 29 to *t* − 7.
- The networks see daily flow sequences. The tabular models see 7- and 30-day means.
- The networks train on daily origins (about 3.5 times as many samples, heavily overlapping) and use the last 365 days before the fold for early stopping. A-HGB-DAILY isolates that difference for HGB.
- Ridge needs imputation; the others do not.
- The networks are trained with pinball loss and score the median. HGB and Ridge are trained with squared error and score the mean.

## 5. Folds and populations

| Population | Training | Evaluation | Status |
|---|---|---|---|
| Tuning, WY2012–2015 | targets before 1 Oct of W − 1 | 1,248 Monday rows | used for hyperparameter selection |
| Confirmation, WY2016–2023 | same rule | 2,500 Monday rows | comparison only; **exploratory**, because these folds were used in v2–v4 |
| HELD, origins 2024-01-01 to 2025-12-01 | targets before 2024-01-01 | 593 rows | **post-hoc retrospective**: viewed in v2; never used for selection |
| T26, origins 2026-01-05 to 2026-10-19 | targets before 2026-01-01 | 251 rows | **post-hoc retrospective**: viewed in v3; never used for selection |

- **Neural networks:** fit on origins whose target precedes the fold start minus 365 days (purged), and early-stop on the 365 days of origins before the fold. Every 2nd daily origin is used.
- **The retrospective stage runs only after** every selection is written to `results/v5/selection.json`. Its SHA-256 is recorded in the retrospective manifest.

**Subsets reported for every population:**
- (a) all rows;
- (b) rows with GEFS available;
- (c) rows with every input group observed (complete information).

Every model predicts every row, so (a) is both the common-row set and each model's maximum eligible set.

## 6. Two comparisons of hyperparameters

1. **Frozen:** every model is run with the hyperparameters it was originally frozen or defaulted with (above).
2. **Equal budget.** Each Experiment A algorithm (Ridge[all], HGB v5, N1, N2, N3, N4, all [all]) gets **20 configurations**:
   - configuration 0 is the frozen one;
   - the other 19 come from a seeded random search (numpy seed 42 + algorithm index);
   - each is run on tuning folds WY2012–2015 with seed 0;
   - the selection rule is the lowest pooled MAE on the 1,248 tuning Monday rows;
   - the selected configuration is then run with 3 seeds on confirmation folds and retrospective windows.

**Search spaces:**
- **Ridge:** fixed α ∈ logspace(−3, 4, 19), plus the frozen RidgeCV.
- **HGB:**
  - learning_rate log-uniform in [0.01, 0.2];
  - max_iter ∈ {100, 200, 300, 500, 800};
  - max_leaf_nodes ∈ {7, 15, 31, 63};
  - max_depth ∈ {None, 4, 6, 8};
  - min_samples_leaf ∈ {10, 20, 30, 50, 100};
  - l2 ∈ {0, 0.1, 1, 3, 10};
  - max_features ∈ {0.5, 0.75, 1.0}.
- **LSTM and GRU:**
  - hidden ∈ {32, 64, 128}; layers ∈ {1, 2}; dropout ∈ {0, 0.1, 0.2, 0.3};
  - lr ∈ {3e-4, 1e-3, 3e-3}; weight decay ∈ {1e-5, 1e-4, 1e-3};
  - L ∈ {30, 60, 90}; batch ∈ {128, 256, 512}.
- **TCN:** the same, but kernel ∈ {2, 3, 5} replaces layers.
- **N4:** the same as LSTM, plus candidate ∈ {wb_noaux, wb_balanced, wb_primary, wb_strict}.
- Epochs (30) and patience (4) are fixed for every network.

## 7. Statistical validation

**Test.** Paired moving-block bootstrap over origin dates. Reservoirs are resampled jointly, blocks are 5, 9 and 13 weeks, with 2,000 replicates and seed 42 (the index generator of `src.evaluation`). The statistic is a contrast of per-row absolute errors.

**p-value.** Two-sided percentile p = min(1, 2 · min((#d* ≥ 0 + 1)/(B + 1), (#d* ≤ 0 + 1)/(B + 1))). The test uses p_max, the maximum over the three block lengths.

**Holm correction** is applied within each family. A difference is **supported** only if its Holm-adjusted p_max < 0.05; otherwise it is "not established". Observed differences are always reported alongside, labelled "observed".

**Family F (frozen hyperparameters, confirmation folds, 2,500 rows).** Negative values favour the first model.
- **P1** HGB v5 − HGB[S,N,C,R].
- **P2** Ridge competitiveness under identical information, a difference-in-differences: [Ridge[all] − HGB v5] − [Ridge[S,C,R] − HGB[S,C,R]].
- **P3** A-N1 − HGB v5.
- **P4** A-N2 − HGB v5.
- **P5** A-N3 − HGB v5.
- **P6** A-N4 − HGB v5.
- **P7** HGB v5 − HGB[S,C,R]: do all the added variables improve on storage, calendar and reservoir alone?
- **P8** A-N4 − A-N4-DIRECT: the water-balance structure with information, encoder and decoder held fixed.

**Family T (tuned, confirmation folds).**
- T1 tuned HGB v5 − HGB[S,N,C,R] (frozen reference);
- T2 tuned Ridge[all] − tuned HGB v5;
- T3–T6 tuned N1, N2, N3, N4 − tuned HGB v5.

**Secondary comparisons** are reported without adjustment and labelled exploratory:
- each Experiment B HGB configuration vs HGB[S,N,C,R];
- each algorithm's [all] version vs its original version;
- A-HGB-DAILY vs HGB v5;
- each tuned model vs its frozen version;
- the N4 variants vs N4 `direct` and vs N1.

**Retrospective windows.** The same contrasts are reported there, labelled post-hoc. They support no claim.

## 8. Metrics

For every model and population:
- n, MAE, RMSE, bias (mean of prediction − observation), R² (around the population mean), median and maximum absolute error;
- skill relative to DOY climatology, 1 − MAE/MAE_DOY, and ΔMAE against HGB[S,N,C,R];
- MAE by reservoir, by water-year class (wet = W/AN, dry = BN/D/C, by the Sacramento or San Joaquin index) and by regime (rapid fill y ≥ +10 pp, rapid drawdown y ≤ −10 pp, other);
- training-row, validation-row and test-row counts;
- training MAE;
- training time and inference time;
- trainable parameters for networks; trees and leaves for HGB.

**Intervals:**
- native network quantiles (50/80/90/95 % coverage and width);
- for **every** model, a chronological split-conformal 90 % interval: fold W uses the 90th percentile (method "higher") of that model's |residuals| on all earlier folds from WY2012. It is reported on confirmation folds WY2016–2023 and on the retrospective windows, which are calibrated on all earlier development folds.

## 9. Prospective confirmation

The confirmation and retrospective results here are not independent evidence. Any candidate must be frozen first: code hash, configuration, weights and seeds. It is then scored only on Monday origins issued after the freeze. A draft protocol is written in `docs/v5/PROSPECTIVE_PROTOCOL_DRAFT.md`. It takes effect only if the owner adopts it as a new prospective-protocol version. `docs/prospective/PROTOCOL.md` is not edited.

## Amendments

**2026-11-27: Monday-row feature values come from the v3 table (reproducibility fix, selects nothing).**
- **What was seen.** The first tabular run (about 05:00–05:03 UTC; the rerun started at 05:07:38 UTC and overwrote its manifest) used feature values recomputed from the daily panel. These differ from the stored v2/v3 values by at most about 1e-13, which is floating-point order of operations. That was enough to move HGB bin thresholds: the HGB[S,N,C,R] per-fold refit scored 3.1631 on confirmation folds instead of the v4 value, 3.1598.
- **The fix.** v2 Monday rows now take every shared column from `data/v3/private/weekly_v3.csv`, which is bit-identical to the v2 table. Rows and missingness are unchanged. The refit then matches v4 to 4e-15.
- **What it reveals.** HGB results move by about 0.003 pp under 1e-13 input perturbations. This is reported as a numerical-sensitivity finding.
- **Integrity.** Nothing was selected from the first run. Its outputs were overwritten.

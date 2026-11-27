# v5 results memo: HGB v5 and the standardised retraining of every implemented model

Written 2026-11-27 (UTC), after every stage had run.

**Every number below comes from files written by the code.** The full generated tables are in [`results/v5/RESULTS_TABLES.md`](../../results/v5/RESULTS_TABLES.md); the statistics are in `results/v5/stats/`; the figures are in `results/v5/figures/`.

**Plan.** [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md), SHA-256 `53aa2df4…24f7`. It was recorded unchanged in all seven stage manifests (`results/v5/manifests/`).

**Status of the evidence:**
- **Confirmation folds WY2016–2023: exploratory.** These folds were already used in v2–v4.
- **HELD (2024–2025) and T26 (2026): post-hoc retrospective.** Both periods were viewed before. Hyperparameters were selected on WY2012–2015 only, and the retrospective stage ran only after `results/v5/selection.json` existed.
- Nothing here is an independent test, and nothing here is deployed.

## 1. Answers

**1. Which model had the lowest error?**
- **Confirmation folds:** N4 `wb_noaux` with full information (A-N4), N4-wb_noaux[S,Q,N,Wd,Wu,G,C,R]-seq30, 3-seed ensemble: **MAE 2.981 pp**. The equal-budget search selected exactly this configuration (configuration 0), so the tuned N4 is the same model and gives the same predictions.
- **The leader changes across populations:**
  - HELD (post-hoc): N4 `direct` on the original inputs, 2.189;
  - T26 (post-hoc): N4 `wb_strict` on the original inputs, 2.089;
  - tuning folds: GRU [all], 2.064.

  No single model is best everywhere.

**2. Is that improvement statistically supported? No.**
- **Against HGB v5 (planned comparison P6):** −0.093 pp. The 95 % CIs at 5-, 9- and 13-week blocks all include 0, and the Holm-adjusted p is 1.0.
- **Against the existing reference HGB[S,N,C,R] (per-fold refit):** −0.179 pp, with CIs [−0.41, +0.02], [−0.45, +0.03] and [−0.47, +0.05]. This contrast was added after the results were seen, so it is labelled exploratory.
- **HGB v5 against HGB[S,N,C,R] (P1):** −0.086 pp, not established. In the post-hoc windows it was −0.065 pp on HELD (not established) and +0.092 pp on T26 (worse; not established).
- **Only two planned comparisons are supported, and neither is about the best model:**
  - **P7:** HGB v5 beats storage-only HGB[S,C,R] by **0.226 pp** (Holm p = 0.040).
  - **T2:** tuned HGB v5 beats tuned Ridge [all] by **0.652 pp** (Holm p = 0.006).

**3. Which information helped? Under HGB, snow and GEFS. Under the networks, flows.**

| Information added | Under HGB (confirmation) | Under the networks (confirmation) |
|---|---|---|
| Snow, from HGB[S,C,R] to HGB[S,N,C,R] | −0.140 pp (not established, p = 0.099) | — |
| GEFS, from HGB[S,N,C,R] to HGB[S,N,G,C,R] | −0.113 pp (not established, p = 0.096) | static inputs: −0.078 (N1), −0.079 (N2), −0.015 (N3), −0.103 (N4); none established |
| Dam weather, upstream weather, flows | +0.008, +0.014, +0.004 pp: no visible gain | — |
| Removing flows from N1 | — | MAE rises by **0.343 pp** (N1 3.224 against 3.567; all three CIs exclude 0, unadjusted) |

- **HGB v5 with everything (3.074)** is no better than HGB[S,N,G,C,R] (3.047): −0.027 pp, not established.
- **Why the flows do nothing for HGB.** The 7- and 30-day storage changes already approximately equal the summed net flow over the same windows.

**4. Was the extra complexity justified? Not on this evidence.**
- **The cost.** HGB v5 trains in 0.26 s on one thread for the largest fold. N4 takes 26.8 s per fit, about 100 times longer, and the ensemble needs 3 fits.
- **The gain.** N4 has 50,340 parameters; HGB v5 has 150 trees with about 4,650 leaves. N4's advantage over HGB v5 is not established on the confirmation folds, is −0.005 pp on HELD, and is −0.272 pp on T26. The T26 difference is post-hoc and not established after Holm (p = 0.084).
- **Tuning did not help either.** Every tuned model differs from its frozen version by less than 0.05 pp on the confirmation folds, and none of those differences is established.

## 2. What was run

- **Experiment B tabular stage.** 10 tabular configurations plus 4 baselines (see the model registry, `src/v5/registry.py`). Each is a per-fold refit with frozen hyperparameters:
  - 12 development folds;
  - 2 retrospective windows.
- **Experiment A tabular models:**
  - Ridge [all];
  - HGB v5 trained on weekly rows;
  - HGB v5 trained on daily origins.
- **Networks.** 14 network configurations × 3 seeds × 14 folds, giving 588 frozen network runs:
  - Experiment B, original inputs: N1, N1 without flows, N2, N3, and N4 `wb_noaux` / `wb_balanced` / `wb_primary` / `wb_strict` / `direct`;
  - Experiment A, with GEFS: N1, N2, N3, N4 `wb_noaux`, N4 `direct`.
- **Equal-budget search.** 20 configurations per algorithm for 6 algorithms, on tuning folds WY2012–2015 with seed 0:
  - 304 network runs, plus 40 tabular configurations, in-process;
  - each selected configuration then ran with 3 seeds on all 14 folds (168 network runs).
- **Totals.** 1,060 trained networks and 196 tabular models, all saved:
  - `models/v5/nn/<stage>/<model>/<fold>_s<seed>.pt`, each with its job specification;
  - `models/v5/tabular/<model>/<fold>.joblib`.
- **Compute.** The runs went through 9 concurrent single-thread workers, from 05:12 to 08:52 UTC. A separate controlled timing benchmark is in `results/v5/compute_benchmark.csv`: one fit per model, on one thread, with nothing else running.
- **Commands.** See [README.md](README.md). Per-row predictions are in `data/v5/preds/all_predictions.parquet`, which is private.

### Reproduction checks (all passed)

| Check | Result |
|---|---|
| v5 daily panel before 2024 vs `data/v4/daily_panel.csv` | identical: 50,400 rows, every column |
| v5 Monday rows vs `data/v3/private/weekly_v3.csv` | same 7,968 rows; every shared column within 1e-13 before the switch to stored values, identical after |
| HGB[S,N,C,R] per-fold refit vs the v4 confirmation predictions | max \|diff\| 4e-15 (MAE 3.1598) |
| HGB[S,N,C,R] trained on targets before 2024-01-01 vs the **frozen v2 artifact** on HELD | max \|diff\| 1.8e-15 (MAE 2.3323) |
| v2 final HGB (v1 features), HGB + SWE, HGB no weather, Ridge and baselines on HELD | all within 4e-15 |
| v3 final Models A, B and B2 on T26 | within 1.8e-15 (2.3485, 2.3386, 2.3637) |
| v3 Model C on T26 | **differs** (v5 2.298, v3 2.316): v5 trains on all rows with GEFS missing where unavailable, while v3 used GEFS rows only |
| N1, N2, N3 and the N4 variants vs v4 per-seed predictions | within 8e-7, from the float32 CSV precision of v4 |
| Tuned N4 vs frozen A-N4 (the same configuration, rerun) | identical predictions: training is deterministic |

**A numerical-sensitivity finding.** Recomputing the features changes values by at most about 1e-13, from floating-point order. That alone moved the HGB[S,N,C,R] confirmation MAE from 3.1598 to 3.1631. HGB differences of a few thousandths of a pp are therefore within the algorithm's numerical noise. The plan amendment of 2026-11-27 records this.

## 3. Dataset and row accounting

The details are in `results/v5/dataset/` (accounting, missingness, GEFS and flow availability, per-fold rows) and in [FEATURES.md](FEATURES.md).

| Quantity | Value |
|---|---|
| Reservoir-days (full panel, 2001-01-01 to 2026-11-20) | 56,730 (6 × 9,455); 50,400 before 2024 |
| Labelled daily origins (30-day target exists) | 56,363; 50,074 with target before 2024 |
| v2 Monday rows | 7,968: tuning 1,248, confirmation 2,500, HELD 593, T26 251, development total 7,076 |
| Monday rows in no population (target crosses a boundary) | 48 |
| Raw variables | 13: storage, capacity, SWE, 6 POWER weather, inflow, outflow; plus 2 GEFS per Monday cycle |
| Engineered tabular features | 29 in HGB v5 (+ `gefs_age_days` for daily training) |
| GEFS available | 96.8 % of Monday rows; confirmation 2,248 of 2,500; no product 2020-01-06 to 2020-09-21 |
| Flows observed | 2001–2023: inflow 98.1 %, outflow 96.6 %; 2024–2026: 99.4 %, 99.7 % |
| All inputs observed | tuning 1,102 of 1,248; confirmation 1,972 of 2,500; HELD 500 of 593; T26 204 of 251 |
| Independent information, confirmation | 417 origin dates; about 97 non-overlapping 30-day windows per reservoir (about 583 in total, fewer given correlation between reservoirs) |

**Training rows per fold** (`results/v5/dataset/fold_rows.csv`, `results/v5/row_accounting.csv`):

| Training regime | Range across WY2012–WY2023 | HELD | T26 |
|---|---|---|---|
| Tabular, weekly rows | 3,250 → 6,686 | 7,076 | 7,693 |
| Tabular, daily origins | 11,632 → 23,668 | 25,036 | 27,215 |
| Networks: fitting sequences (every 2nd day, purged) | 10,536 → 22,570 | 23,944 | 26,117 |
| Networks: early-stopping sequences (the 365 days before the fold) | 1,995–2,010 | — | — |

**Validation rows.** Tabular models with frozen hyperparameters use none: HGB early stopping is inactive, and RidgeCV uses efficient leave-one-out internally. Their hyperparameters come from the tuning folds, a separate population.

## 4. Consolidated results, confirmation folds WY2016–2023 (2,500 rows, the primary population)

**Columns.** MAE, RMSE and bias (prediction − observation) are in pp. "Training rows" is the range across the 8 folds. "Train s" is the total training time across the 8 folds, with networks counted over 3 seeds on 9 parallel single-thread workers; it is indicative only, and §8 has controlled timings. Rows are sorted by MAE; selected rows are shown here and every model is in the generated tables.

| Model | Code | Training rows | Validation rows | MAE | RMSE | Bias | R² | Train s |
|---|---|---|---|---|---|---|---|---|
| A-N4 (= tuned N4) | N4-wb_noaux[S,Q,N,Wd,Wu,G,C,R]-seq30 | 14,906–22,570 | 2,006–2,010 | **2.981** | 4.988 | −0.711 | 0.618 | 1,793 |
| B-HGB-SNGCR | HGB[S,N,G,C,R] | 4,498–6,686 | 0 | 3.047 | **4.949** | −0.219 | **0.624** | 4.3 |
| A-HGB-DAILY | HGB[S,N,Wd,Wu,G,Q,C,R] · daily | 15,998–23,668 | 0 | 3.060 | 5.003 | −0.241 | 0.616 | 5.7 |
| A-N4-DIRECT | N4-direct[S,Q,N,Wd,Wu,G,C,R]-seq30 | 14,906–22,570 | 2,006–2,010 | 3.062 | 5.085 | −0.824 | 0.603 | 1,939 |
| A-N2 | GRU[S,Q,N,Wd,Wu,G,C,R]-seq30 | 14,906–22,570 | 2,006–2,010 | 3.067 | 5.119 | −0.601 | 0.598 | 761 |
| T-HGB-V5 | HGB[all] · tuned | 4,498–6,686 | 0 | 3.068 | 4.990 | −0.156 | 0.618 | 12.6 |
| **HGB-V5** | **HGB[S,N,Wd,Wu,G,Q,C,R]** | 4,498–6,686 | 0 | 3.074 | 4.991 | −0.198 | 0.618 | 4.7 |
| B-N4-NOAUX | N4-wb_noaux[S,Q,N,Wd,Wu,C,R]-seq30 (v4) | 14,906–22,570 | 2,006–2,010 | 3.084 | 5.142 | −0.785 | 0.594 | 1,871 |
| T-N1 | LSTM[all]-seq60 · tuned | 14,906–22,570 | 2,006–2,010 | 3.112 | 5.151 | −0.737 | 0.593 | 1,845 |
| T-N2 | GRU[all]-seq30 · tuned | 14,906–22,570 | 2,006–2,010 | 3.112 | 5.168 | −0.814 | 0.590 | 4,906 |
| A-N1 | LSTM[S,Q,N,Wd,Wu,G,C,R]-seq30 | 14,906–22,570 | 2,006–2,010 | 3.145 | 5.242 | −0.728 | 0.578 | 895 |
| **B-HGB-SNCR** | **HGB[S,N,C,R] (reference)** | 4,498–6,686 | 0 | 3.160 | 5.186 | −0.217 | 0.587 | 4.1 |
| B-N1 | LSTM[S,Q,N,Wd,Wu,C,R]-seq30 (v4 N1) | 14,906–22,570 | 2,006–2,010 | 3.224 | 5.351 | −0.750 | 0.560 | 775 |
| B-HGB-SCR | HGB[S,C,R] | 4,498–6,686 | 0 | 3.300 | 5.268 | −0.095 | 0.574 | 4.2 |
| A-N3 | TCN[S,Q,N,Wd,Wu,G,C,R]-seq30 | 14,906–22,570 | 2,006–2,010 | 3.306 | 5.397 | −0.796 | 0.553 | 3,770 |
| B-N1-NOFLOW | LSTM[S,N,Wd,Wu,C,R]-seq30 | 14,906–22,570 | 2,006–2,010 | 3.567 | 5.819 | −0.752 | 0.480 | 646 |
| A-RIDGE | Ridge[S,N,Wd,Wu,G,Q,C,R] | 4,498–6,686 | 0 | 3.671 | 5.779 | −0.373 | 0.487 | 0.2 |
| B-RIDGE-SCR | Ridge[S,C,R] | 4,498–6,686 | 0 | 3.846 | 6.110 | −0.304 | 0.427 | 0.04 |
| BL-DOY | DOY climatology | 4,498–6,686 | 0 | 4.108 | 6.152 | −0.570 | 0.419 | — |
| BL-PERS | persistence | — | — | 5.761 | 8.089 | −0.529 | −0.004 | — |

**Additional metrics, confirmation folds:**

| Model | Median AE | Max AE | Skill vs DOY | ΔMAE vs HGB[S,N,C,R] | Seed MAE mean ± SD | Training MAE | Parameters |
|---|---|---|---|---|---|---|---|
| A-N4 | 1.568 | 34.35 | 0.274 | −0.179 | 3.072 ± 0.049 | 2.197 | 50,340 |
| HGB-V5 | 1.688 | 33.48 | 0.252 | −0.086 | — | 1.664 | 150 trees, 4,650 leaves |
| T-HGB-V5 | 1.674 | 33.36 | 0.253 | −0.092 | — | 1.528 | 300 trees, 12,994 leaves |
| A-N2 | 1.653 | 33.96 | 0.253 | −0.093 | 3.118 ± 0.046 | 2.311 | 23,969 |
| B-HGB-SNCR | 1.697 | 34.49 | 0.231 | 0 | — | 1.844 | 150 trees, 4,650 leaves |
| A-RIDGE | 2.290 | 43.50 | 0.106 | +0.512 | — | 2.958 | 38 coefficients |

**Bias.** The networks score the median of their forecast distribution and have a bias of −0.6 to −1.0 pp. HGB and Ridge score the mean, with a bias of about −0.2 pp. This matches the skew of the target: the development rows have mean +0.53 pp and median −0.44 pp.

**The ensemble.** The 3-seed ensemble beats the average single seed by about 0.05–0.11 pp. For example, A-N4 scores 3.072 per seed on average and 2.981 as an ensemble.

**Training against confirmation MAE.** HGB fits its training rows far more closely than the networks do: 1.66 in training against 3.07 on confirmation for HGB v5, compared with 2.20 against 2.98 for N4, which stops early.

### Other populations (each reported separately; never pooled)

| Model | Tuning WY2012–2015 (1,248) | Confirmation complete-information subset (1,972) | HELD, post-hoc (593) | T26, post-hoc (251) |
|---|---|---|---|---|
| A-N4 | 2.094 | **3.039** | 2.262 | 2.169 |
| HGB-V5 | 2.212 | 3.109 | 2.267 | 2.441 |
| T-HGB-V5 | 2.146* | 3.088 | 2.267 | 2.273 |
| B-HGB-SNGCR | 2.338 | 3.119 | 2.285 | 2.298 |
| B-HGB-SNCR | 2.520 | 3.269 | 2.332 | 2.349 |
| Frozen v2 artifact | — (trained on these rows) | — | 2.332 | 2.355 |
| A-N1 | 2.169 | 3.236 | 2.310 | 2.162 |
| A-N2 | **2.064** | 3.171 | 2.284 | 2.383 |
| B-HGB-SCR | 2.731 | 3.388 | 2.450 | 2.174 |
| A-RIDGE | 2.605 | 3.654 | 3.231 | 3.077 |
| BL-DOY | 3.263 | 4.177 | 2.969 | 2.882 |
| Lowest MAE in this population | A-N2, 2.064 | A-N4, 3.039 | B-N4-DIRECT, 2.189 | B-N4-STRICT, 2.089 |

\* Tuned models were selected on the tuning folds, so their tuning-fold numbers are optimistic (`selected_on_these_rows` in the CSV).

The GEFS-available confirmation subset (2,248 rows) is in the generated tables. A-N4 scores 3.103 there, T-HGB-V5 3.109 and HGB-V5 3.135.

## 5. Statistical validation

**Method.** Paired moving-block bootstrap over origin dates, with all reservoirs on the same date resampled together:
- blocks of 5, 9 and 13 weeks; 2,000 replicates; seed 42;
- the decision uses p_max, the largest two-sided p across the three block lengths;
- Holm correction within each family;
- "supported" means the Holm-adjusted p_max is below 0.05.

**Family F: frozen hyperparameters, confirmation folds.** Negative values favour the first model.

| ID | Comparison | Observed (pp) | 95 % CI (9-week blocks) | Holm p | Verdict |
|---|---|---|---|---|---|
| P1 | HGB v5 − HGB[S,N,C,R] | −0.086 | [−0.245, +0.055] | 1.000 | not established |
| P2 | Ridge gap change: [Ridge[all] − HGB v5] − [Ridge[S,C,R] − HGB[S,C,R]] | +0.051 | [−0.200, +0.299] | 1.000 | not established |
| P3 | LSTM [all] − HGB v5 | +0.071 | [−0.120, +0.246] | 1.000 | not established |
| P4 | GRU [all] − HGB v5 | −0.007 | [−0.206, +0.167] | 1.000 | not established |
| P5 | TCN [all] − HGB v5 | +0.231 | [+0.022, +0.432] | 0.238 | not established (all three unadjusted CIs exclude 0) |
| P6 | N4 [all] − HGB v5 | −0.093 | [−0.292, +0.078] | 1.000 | not established |
| P7 | HGB v5 − HGB[S,C,R] | **−0.226** | [−0.382, −0.080] | **0.040** | **supported: HGB v5 better** |
| P8 | N4 wb_noaux [all] − N4 direct [all] | −0.081 | [−0.153, −0.012] | 0.238 | not established (all three unadjusted CIs exclude 0) |

**Family T: tuned models, confirmation folds:**

| ID | Comparison | Observed (pp) | 95 % CI (9-week blocks) | Holm p | Verdict |
|---|---|---|---|---|---|
| T1 | tuned HGB v5 − HGB[S,N,C,R] | −0.092 | [−0.285, +0.097] | 1.000 | not established |
| T2 | tuned Ridge [all] − tuned HGB v5 | **+0.652** | [+0.337, +0.959] | **0.006** | **supported: HGB v5 better** |
| T3 | tuned LSTM − tuned HGB v5 | +0.044 | [−0.168, +0.223] | 1.000 | not established |
| T4 | tuned GRU − tuned HGB v5 | +0.044 | [−0.190, +0.236] | 1.000 | not established |
| T5 | tuned TCN − tuned HGB v5 | +0.255 | [−0.014, +0.516] | 0.350 | not established |
| T6 | tuned N4 − tuned HGB v5 | −0.087 | [−0.307, +0.097] | 1.000 | not established |

**Secondary comparisons** (unadjusted and exploratory; full list in `results/v5/stats/comparisons.csv`):
- **Where all three CIs exclude 0:**
  - N1 with flows beats N1 without flows by −0.343 pp;
  - N4 `wb_noaux` beats N1 on the original inputs by −0.140 pp;
  - Ridge [all] beats the v1 Ridge by −0.226 pp;
  - every model listed beats DOY climatology, by 0.44–1.13 pp.
- **Everything else is not established:**
  - every Experiment B HGB configuration against the reference;
  - the effect of GEFS within each network;
  - daily-origin against weekly training for HGB (−0.015 pp);
  - every tuned model against its frozen version;
  - every N4 water-balance variant against N4 `direct` on the original inputs.

**Post-hoc windows** (the same families; supports no claim):
- **HELD:** only T2 (tuned Ridge worse than tuned HGB v5, +1.043 pp) passes Holm.
- **T26:** T2 (+0.897 pp) and P3 pass Holm. In P3, LSTM [all] beats HGB v5 by 0.278 pp, Holm p = 0.016. On HELD the same comparison is +0.043 pp, in the opposite direction.
- **What this shows.** The 2026 result does not replicate, and with 42 origin dates the bootstrap is coarse.

## 6. Experiment B: information or algorithm? (Figure 2)

With HGB and frozen hyperparameters:
- snow and GEFS are the only additions with a visible gain on the tuning and confirmation folds;
- dam weather, upstream weather and flows change the MAE by less than 0.015 pp on confirmation;
- in the post-hoc windows the ordering is unstable. On T26, storage-only HGB[S,C,R] (2.174) beats every other HGB configuration, including HGB v5 (2.441). This is a 251-row window.

**The answer.** Across HGB configurations, MAE varies by 0.25 pp on confirmation (3.047 to 3.300). Across algorithms given the same full information, it varies by 0.69 pp (2.981 to 3.671); most of that range is Ridge against the rest. Without Ridge, HGB and the networks fall within 0.33 pp (2.981 to 3.306), and the only established differences involve Ridge or storage-only information. So the evidence separates **linear from nonlinear** (supported) and **storage-only from added snow and forecast information** (supported for HGB v5 against HGB[S,C,R]). It does not separate HGB from the networks.

## 7. Breakdowns (Figures 4 and 8)

**By reservoir (confirmation MAE, pp):**
- Folsom is hardest. N4 [all] scores 4.42, HGB v5 4.79 and the reference 4.99.
- New Melones is easiest, and there HGB beats every network: HGB 1.63–1.71 against networks 1.92–2.08.
- Shasta and Folsom gain most from full information: HGB v5 against the reference is 2.67 against 2.96 at Shasta, and 4.79 against 4.99 at Folsom.

**By water-year class:**
- **Wet years (W/AN, 942 rows):** the reference HGB is lowest (4.07). HGB v5 scores 4.11 and N4 4.21.
- **Dry years (BN/D/C, 1,558 rows):** the networks are lower. N4 scores 2.24 and the GRU 2.25, against HGB v5 2.45 and the reference 2.61.

**By regime:**
- **Rapid fill (≥ +10 pp, 281 rows):** tuned HGB v5 is best (8.61); N4 scores 9.66 and N1 10.58.
- **Rapid drawdown (≤ −10 pp, 140 rows):** N4 is best (4.46), against HGB v5 4.69.
- **Other rows (2,079):** the networks other than the TCN score 1.96–2.02, against HGB 2.17–2.24.

The networks win on ordinary and dry-year rows; HGB wins on large filling events. These differences were not tested.

## 8. Intervals and compute (Figures 6 and 7)

**Intervals.**
- **Chronological conformal 90 % intervals,** applied identically to every model, cover 85.6–88.3 % of confirmation rows (below the nominal 90 %, consistent with drift between earlier and later years) and 89–97 % in the post-hoc windows. Mean widths are 12.0–14.6 pp for the models and 16.5–23.2 pp for the baselines.
- **Native 5–95 % network quantiles** are narrower (10.1–12.5 pp) and cover only 82.8–87.3 %. The native intervals are not calibrated.

**Controlled training cost** (WY2023 training set, one fit, one thread; `results/v5/compute_benchmark.csv`):

| Model | Training time | Inference (per 1,000 rows) |
|---|---|---|
| Ridge | 0.009–0.015 s | — |
| HGB[S,N,C,R] | 0.16 s | — |
| HGB v5 | 0.26 s | 5.4 ms |
| Tuned HGB v5 | 0.56 s | — |
| GRU [all] | 3.4 s | — |
| LSTM [all] | 5.2 s | — |
| Tuned LSTM | 19.3 s | — |
| N4 [all] | 26.8 s | 36 ms |
| TCN [all] | 28.0 s | 141 ms |

## 9. Equal-budget tuning (Figure 5)

**Selected on the tuning folds** (`results/v5/selection.json`; seed 0):

| Algorithm | Selected configuration | Tuning MAE (pp) |
|---|---|---|
| Ridge | α = 681 | 2.578 |
| HGB | lr 0.025, 300 trees, 63 leaves, min leaf 100, l2 0.1, max_features 0.75 | 2.146 |
| LSTM | hidden 32, 2 layers, L = 60 | 2.078 |
| GRU | hidden 64, 2 layers, lr 3e-4, batch 128 | 2.083 |
| TCN | hidden 32, kernel 2 | 2.175 |
| N4 | the frozen defaults | 2.151 |

**The gains did not carry over to the confirmation folds.** The tuned GRU scores 3.112 there against 3.067 for the frozen GRU; the tuned Ridge is worse than RidgeCV. The frozen GRU's 3-seed tuning MAE (2.064) equals the tuned GRU's (2.065), which shows that single-seed selection mostly picks seed noise.

**What "equal budget" means here.** Each algorithm got the same number of configurations, not the same compute.

## 10. Limitations and work not completed

**Limitations:**
- **No independent test.** The confirmation folds are reused and the retrospective windows were viewed before. [PROSPECTIVE_PROTOCOL_DRAFT.md](PROSPECTIVE_PROTOCOL_DRAFT.md) sets out a freeze-then-score protocol: at least 150 scored origin dates, about 3 years. It needs the owner's adoption as a new protocol version.
- **Unequal information content.** Experiment A gives every algorithm the same information *sources*, but not identical content: 60-day storage for tabular models against 30 days for networks, explicit snow anomaly, weather windows, and daily against aggregated flows ([FEATURES.md](FEATURES.md) §4).
- **GEFS has gaps.** Only Monday 00Z cycles exist. Non-Monday training origins use cycles 1–6 days old. There is no GEFS product for 2020-01 to 2020-09. Reforecasts (2001–2019) are retrospective.
- **Flow data quality.** 2024–2026 flows are provisional CDEC values. Inflow is computed by the operator.
- **Unclassified 2026.** `data/water_year_types.csv` has no water-year type for 2026, so T26 rows are unclassified by wet and dry.
- **Search limits.** The search used 20 configurations per algorithm with single-seed selection, and the information representation itself (history length for frozen networks, the feature set) was not searched.
- **Timings.** The Train s columns come from parallel runs and are indicative. Only `compute_benchmark.csv` is controlled.
- **Plan changes.** One amendment to the plan (the reproducibility fix) and one contrast added after the results were seen (§1, item 2). Both are labelled.

**Not done:**
- no v4/v5 model has been frozen for prospective issuance;
- the GEFS-rows-only training variant of HGB[S,N,G,C,R] (the v3 rule) was not run separately;
- there are no per-model cards beyond `model_inventory.csv` and `hyperparameter_registry.json`.

# v4 experimental protocol (WatershedShift NeuralHydro)

Written on 2026-11-24 **before any v4 neural result was seen**. Changes to this protocol are appended in a dated "Amendments" section, never rewritten.

## 1. What v4 is and is not

v4 is a separate research programme. It does not change, retrain or replace anything frozen:
- the v1, v2 and v3 models;
- their configurations, preregistrations and results;
- the issued forecasts and public data releases.

The deployed site keeps serving the frozen v2 Model A and the v3 models. **The frozen Model A remains the validated fallback and the research reference.**

No v4 model is deployed until all of the following hold:
1. it has been frozen;
2. it has been evaluated on genuinely independent future data under a written protocol;
3. the owner has approved it.

## 2. Data used for development

- **Cutoff.** Only data dated before **2024-01-01**: every origin, every 30-day target window and every training label. The code asserts this in the panel builder, the sequence builder and the experiment loader.
- **Never loaded by v4 development code:**
  - 2024–2025, the v2 held-out period;
  - 2026, the v3 preregistered test.
- **No tuning on 2024–2025 or 2026.** Any later look at those periods is labelled **post-hoc**, never used for selection, and cannot support a claim of improvement.
- **Inputs.** All daily, built by `src/v4/dataset.py`:

| Input | Source |
|---|---|
| CDEC storage | sensor 15 |
| CDEC inflow and outflow | sensors 76 and 23, fetched by `src/v4/flows.py` for 2001–2023 |
| Basin snow index | sensor 82, core pillows |
| Weather at the dam point and the upstream-box mean | NASA POWER |
| Seasonality | calendar |

- **Information rule** (`src/v4/sequences.py`, tested):
  - storage and flows are used up to the origin day *t*;
  - snow up to *t*−1;
  - weather up to *t*−7;
  - nothing after *t*;
  - storage across a series break is masked.
- **Missing values** have explicit masks and are never filled.

## 3. Target

The target is the same as in v2 and v3: the 30-day storage change, 100 × (S[t+30] − S[t]) / capacity in force at *t*.
- It uses exact calendar days and is never interpolated.
- A window that crosses a series break produces no sample.
- The v4 target is checked to equal the v2 `target_delta_pp` on every evaluation row.

## 4. Folds, selection and evaluation rows

**Folds.** For development water year *W* (Oct 1 of *W*−1 to Sep 30 of *W*):
- training samples have their target date before the fold starts, which is the v2 rule;
- the last 365 days of origins before the fold are the early-stopping set;
- the fitting set is purged so that no target window overlaps the early-stopping set;
- overlapping sequences are never split at random.

**Daily origins** are used for training: every second day (`STRIDE = 2`). The Monday-origin evaluation is kept for direct comparison.

**Evaluation rows** are the v2 Monday rows (`data/processed/weekly_features.csv`, origin and target before 2024-01-01). Every model and baseline is scored on exactly the same rows.

**Selection.**

| Folds | Use |
|---|---|
| Tuning, WY2008–2015 | History length, and later any architecture or hyperparameter search. The first stage uses WY2012–2015, the most recent tuning folds, to limit compute. |
| Confirmation, WY2016–2023 | Comparison only. Nothing is chosen on these folds. |

**Initial network settings are fixed defaults, not searched:**
- architecture: hidden 64, 1 layer, reservoir embedding 4, dropout 0.2;
- training: AdamW, learning rate 1e-3, weight decay 1e-4, batch 256, up to 30 epochs, patience 4, gradient clipping 1.0.

**Seeds.** Every neural configuration is run with 3 seeds (0, 1, 2). Each seed's MAE is reported with the mean and SD. The seed ensemble (the mean of the seed medians) is reported separately.

## 5. Baselines

All baselines are refit per fold on the same weekly rows as v2 and scored on identical rows:
- persistence;
- 30-day trend;
- monthly seasonal;
- smoothed day-of-year seasonal (the v2 strongest baseline);
- ridge autoregression: storage level, the 7-, 30- and 60-day changes, season and reservoir;
- **HGB with the frozen v2 features and hyperparameters**, the v2 primary configuration.

## 6. Metrics and uncertainty

**Point accuracy.**
- MAE in percentage points: overall, by reservoir, by water year, by season, by water-year type, and for rapid filling and drawdown (|change| ≥ 10 pp).

**Probabilistic accuracy.** Quantiles 0.025, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95 and 0.975 give:
- 50%, 80%, 90% and 95% intervals, each with its coverage and mean width;
- the mean pinball loss;
- an approximate CRPS, computed as twice the mean pinball loss over the 9 levels;
- results reported overall and by reservoir. Folsom is examined specifically, for coverage gained without simply widening the intervals.

**Paired differences.** Moving-block bootstrap over origin dates, using `src.evaluation` unchanged:
- 5-, 9- and 13-week blocks, 2,000 replicates, seed 42;
- a difference counts as supported only when its 95% CI excludes zero at all three block lengths.

**Robustness at inference.** Each source group (flows, snow, weather) is removed in turn, and the MAE change is measured.

## 7. Freezing and independent evaluation

- **Freeze before evaluation.** A v4 candidate is frozen before any genuinely independent evaluation: its code hash, configuration, weights and seeds are recorded.
- **The independent evaluation** uses forecasts issued after the freeze. Its protocol is preregistered before those outcomes exist.
- **The MAE goal.** An MAE below 1.8 pp is an aspirational target. It is not assumed, and it is not claimed unless it is observed under this protocol.

## Amendments

**2026-11-24: N3 TCN added.** The TCN was added to the confirmation-fold comparison when that stage was launched, before any confirmation result was seen. It uses the same defaults, except that its depth follows the receptive field.

**2026-11-24: flow ablation (post-hoc).** Two models were run *after* the confirmation-fold comparison had been seen:
- HGB with the frozen v2 configuration plus four recent-flow features;
- N1 trained without flow inputs.

They are labelled **post-hoc**. They explain where the neural models' accuracy comes from and select nothing.

**2026-11-24: one seed in the history-length study.** The history-length selection used one seed (0), as specified, and the margins were small. A repeat with 3 seeds on tuning folds is planned before any further length decision.

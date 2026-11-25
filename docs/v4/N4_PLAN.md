# N4 development plan (fixed before any N4 result)

Written on 2026-11-25 (UTC), after the N4 code passed its offline tests on synthetic data, and **before N4 was run on any real reservoir data.** This file is not edited after the first N4 run. The tuning and confirmation summaries record its SHA-256 (`provenance_sha256`), so any later change would show. Changes go in a dated amendment in `PROTOCOL.md`.

Everything in `PROTOCOL.md` still applies:
- development data before 2024-01-01 only;
- the water-year folds;
- the v2 Monday evaluation rows;
- 3 seeds;
- the moving-block bootstrap;
- no use of 2024–2025 or 2026.

## 1. What N4 is

The model is `WaterBalanceED` in `src/v4/nets.py`, as designed in `ARCHITECTURE.md`. It is a shared encoder-decoder LSTM for all six reservoirs.

| Part | What it does |
|---|---|
| Encoder | Takes the N1 inputs over the past window: storage, inflow, outflow, snow and weather, each with a mask, plus the day of year, the reservoir embedding and log capacity. |
| Decoder | Runs over days 1 to 30 after the origin. Its inputs are known in advance only: the calendar and the lead time. |
| Daily heads | Inflow, release (≥ 0) and loss (≥ 0), in percentage points of capacity per day, plus a bounded residual. |
| Water-balance layer | dS(k) = Σ (inflow − release − loss) + r(k), with \|r(k)\| ≤ r_max. |
| Quantiles | The median is dS(30) itself. The other quantiles are non-crossing gaps added around it. |

- **Network size** uses the v4 defaults: hidden 64, 1 layer, embedding 4, dropout 0.2.
- **Training** uses the v4 defaults: AdamW, learning rate 1e-3, weight decay 1e-4, batch 256, up to 30 epochs, patience 4, clip 1.0, every second daily origin.
- **Early stopping** uses the same criterion as N1–N3: the pinball loss of the 30-day change on the early-stopping year.
- **Not included in this first build,** to keep N4 comparable with N1–N3:
  - GEFS in the decoder;
  - source-dropout training;
  - feasibility by projection;
  - conformal calibration;
  - static basin attributes.

### Composite loss

| Term | Definition |
|---|---|
| quantile | Pinball loss of the 30-day change |
| storage | L1 on the daily trajectory, where labelled |
| inflow, outflow | L1 on the cumulative 30-day flows, where the full window is labelled |
| feasible | Hinge on projected storage below 0% or above 100% of capacity |
| residual | Mean \|r\| |

The design's separate "balance" and "residual" penalties are the same quantity in this implementation, so they form one term.

## 2. Candidates (all fixed now)

| Name | Water-balance layer | r_max (pp) | Weights: quantile, storage, inflow, outflow, feasible, residual |
|---|---|---|---|
| `wb_balanced` | yes | 1.0 | 1, 1, 0.5, 0.5, 1, 0.5 |
| `wb_primary` | yes | 1.0 | 1, 0.25, 0.1, 0.1, 1, 0.5 |
| `wb_strict` | yes | 0 (exact accounting identity) | 1, 1, 0.5, 0.5, 1, — |
| `wb_noaux` | yes | 1.0 | 1, 1, 0, 0, 1, 0.5 |
| `direct` (ablation, never eligible as N4) | no: direct daily-change head | — | 1, 1, —, —, 1, — |

## 3. Selection, on tuning folds WY2012–2015 only

Run by `python -m src.v4.experiment n4-tune`.

1. **Step 1: loss weights.** Run all five candidates with a 30-day history and seeds 0, 1 and 2.
   - The N4 configuration is the eligible candidate (the first four) with the **lowest 3-seed-ensemble MAE** on the Monday rows of WY2012–2015.
2. **Step 2: history length.** Run the step-1 winner at 90 and 365 days, with the same seeds.
   - The history length with the lowest 3-seed-ensemble MAE among 30, 90 and 365 days is selected.
3. **References on the same rows,** none of them eligible:
   - N1 LSTM, 30 days, 3 seeds;
   - the per-fold baselines;
   - HGB SWE no weather, refit per fold.
4. **No other choice is made.** Nothing is chosen from confirmation folds.

## 4. Comparison, on confirmation folds WY2016–2023

Run by `python -m src.v4.experiment n4-confirm`.

**Models**, each with 3 seeds and the selected history length, on the 2,500 v2 Monday rows:
- the selected N4;
- `wb_strict`, if it was not selected;
- the `direct` ablation.

**Reference: the "HGB SWE no weather" per-fold refit.**
- Estimator: scikit-learn `HistGradientBoostingRegressor` with the frozen v2 hyperparameters: learning_rate 0.03, max_iter 150, max_depth None, min_samples_leaf 30, l2_regularization 1.0, random_state 42.
- Inputs: the 15 v2 inputs (storage %, 7/30/60-day storage changes, basin SWE, 14-day SWE change, SWE anomaly against a fold-specific climatology, season sine and cosine, and six reservoir one-hots).
- Training data: refit for each water year W on v2 weekly rows with target date before Oct 1 of W−1.
- **This is not the frozen v2 artifact.**
- It must reproduce the `compare` stage exactly; the code checks this.

**Other comparisons, on identical rows:**
- the smoothed DOY seasonal baseline;
- the N1 and N2 3-seed ensembles from `compare`.

**Decision rule** (`PROTOCOL.md` §6), using the moving-block bootstrap over origin dates with 5-, 9- and 13-week blocks, 2,000 replicates and seed 42:
- "N4 more accurate than X" is **supported** only if the 95% CI of the MAE difference (N4 − X) is below zero at all three block lengths.
- Otherwise it is **not established**.

The owner's condition for revisiting the website is: N4 built, N4 beats the HGB SWE no weather refit on development folds, and N4 frozen. This plan reports the evidence for the second part and decides nothing about the website.

**Also reported:**
- intervals and coverage;
- MAE by reservoir, water year, season, water-year type and regime;
- robustness with each source removed at inference;
- the water-balance components on evaluation rows:
  - the mean and 95th-percentile \|r(30)\|;
  - the mean learned loss;
  - the MAE of the cumulative inflow and release heads against operator-computed labels.

## 5. What a result here can and cannot mean

- These are **development** results on 2016–2023. They say nothing about 2024–2026 or about future accuracy.
- Even a supported result does not freeze or deploy anything. Freezing needs a separate decision, and a recorded code hash, configuration, weights and seeds (`PROTOCOL.md` §7).
- "Water-balance-structured" is the correct description of N4.
  - `wb_strict` enforces an exact accounting identity between *predicted* inflow, release and loss.
  - That is not a guarantee that real water is conserved, because the inflow labels are operator-computed.

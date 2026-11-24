# v4 initial development results (2026-11-24)

**Scope.** These are development-only results: chronological folds before 2024. They are not test results, not evaluated on 2024–2026, and say nothing about future accuracy.

All numbers come from `results/v4/development/*/summary.json` and are tabulated in [RESULTS_TABLES.md](RESULTS_TABLES.md), which `python -m src.v4.report` generates. The protocol is in [PROTOCOL.md](PROTOCOL.md).

## Summary

1. **The baselines reproduce v2 exactly.** On the 2,500 confirmation-fold rows, HGB with the frozen v2 configuration scores 3.1598 pp and the smoothed seasonal baseline 4.1083, identical to `results/v2/development/summary.json`. The comparison frame is the same as v2's.
2. **History length.** On tuning folds WY2012–2015 (LSTM, seed 0) the predetermined rule chose a **30-day** history:

   | History | MAE (pp) |
   |---|---|
   | 30 days | 2.224 |
   | 90 days | 2.359 |
   | 180 days | 2.385 |
   | 365 days | 2.282 |

   This rests on one seed, and the 365-day result was close, so it is not strong evidence for short contexts.
3. **Confirmation folds WY2016–2023, identical rows, 3 seeds.** The neural models perform at the level of gradient boosting. They do not beat it.

   | Model | Seed ensemble MAE (pp) | Per-seed mean ± SD (pp) | vs HGB, 95% CIs at 5/9/13-week blocks |
   |---|---|---|---|
   | N2 shared GRU | **3.146** | 3.193 ± 0.037 | −0.014: not established (all CIs span 0) |
   | HGB, frozen v2 config | 3.160 | (deterministic) | reference |
   | N1 shared LSTM | 3.224 | 3.284 ± 0.074 | +0.064: not established |
   | N3 TCN | 3.321 | 3.386 ± 0.076 | +0.161: not established |
   | Smoothed DOY seasonal | 4.108 | | |

   Every neural model beats the seasonal baseline (by 0.79–0.96 pp, every CI below zero). None is shown to beat HGB.
4. **What makes the networks competitive is the flow data, not the architecture.** This ablation is post-hoc (see Amendments in `PROTOCOL.md`).
   - Without inflow and outflow inputs, the LSTM falls to 3.567 pp, significantly worse than HGB: +0.407, every CI above zero.
   - With flows, the LSTM is 0.343 pp better than without (every CI below zero).
   - Adding the same recent-flow information to HGB changes nothing: 3.164 vs 3.160.

   HGB already extracts this information from its engineered storage-change and snow-anomaly features. Note that a 30-day window cannot see HGB's 60-day storage change, and the network receives raw snow rather than a snow anomaly.
5. **Errors are complementary.**

   | Where | Neural models (GRU) better | HGB better |
   |---|---|---|
   | Reservoirs | Folsom (4.73 vs 4.99), Oroville, Shasta | New Melones (1.99 vs 1.67), Don Pedro |
   | Seasons | Spring | Summer |
   | Water years | Critical (2.03 vs 2.85), and WY2020–2022 | Wet years and WY2016, 2017, 2019, 2023 |
   | Regimes | | Rapid filling (9.24 vs 10.74 pp) |

   A combination of the two is therefore a natural candidate. It must be specified on the tuning folds before it is evaluated; choosing it from these confirmation results would be selection on viewed folds.
6. **Probabilistic outputs undercover.** For the GRU ensemble:

   | Nominal interval | Coverage |
   |---|---|
   | 50% | 0.445 |
   | 80% | 0.756 |
   | 90% | 0.860 |
   | 95% | 0.925 |

   Mean pinball loss 0.83 pp; approximate CRPS 1.65 pp.
   - **Folsom:** 90% coverage is 0.87, with a mean width of 18.3 pp. Whether this improves on HGB-based intervals without merely being wider needs an HGB interval on identical rows, which is not done yet.
   - Quantile regression alone is not calibrated here. The planned fix is chronological conformal calibration on earlier folds.
7. **Robustness.** These models were trained without source dropout. Removing a source at inference raises the GRU ensemble's MAE:

   | Source removed | MAE (pp), from 3.15 |
   |---|---|
   | Flows | 3.94 |
   | Snow | 3.54 |
   | Weather | 3.31 |

   The models depend on flows, so source-dropout training is needed before any operational use.
8. **Compute is small.**

   | Model | Parameters | Seconds per fold-seed on CPU |
   |---|---|---|
   | LSTM | 28,897 | about 10 |
   | GRU | 22,945 | about 10 |
   | TCN | 47,393 | about 80 |

   The full confirmation stage (72 fold-seed runs plus baselines) took 41 minutes on an Apple M5 CPU. Memory stays well under 1 GB, because windows are gathered on the fly.

## Training behaviour

Loss curves: `results/v4/development/compare/loss_curves_*.png` and `context/loss_curves_WY2015.png`.
- **Training loss falls smoothly.** The early-stopping loss is noisy, because it is a single water year and so depends on the regime.
- **Fold WY2018** stops on WY2017, the wettest year in the record, so its early-stopping loss stays high and erratic.
- **Fold WY2019** stopped after one epoch.
- **Mean best epoch is about 7–9.**

A one-year early-stopping set is a weakness of this first protocol. Next time, the epoch count or the early-stopping design should be chosen on the tuning folds.

## What this does and does not show

- It **does** show:
  - working, leakage-tested daily-sequence infrastructure;
  - shared recurrent models that match a strong gradient-boosting model under an identical chronological protocol;
  - flow data (inflow and outflow) that is informative for sequence models;
  - a water balance that closes well enough to support physics-informed designs (`FLOW_DATA.md`).
- It does **not** show:
  - that any neural model is more accurate than HGB;
  - anything about 2024–2026;
  - any progress toward the 1.8 pp aspirational target. For reference, HGB scores 3.16 pp on these harder confirmation years against 2.33 pp on 2024–2025. MAEs from different periods are not comparable.

## Next experiments (proposed; tuning folds WY2008–2015 for every choice)

1. **N4, the physics-informed multi-task model** (`ARCHITECTURE.md`). Inflow and release heads with a water-balance integration layer. The flow results suggest this is where sequence models can add something HGB cannot: an explicit split between hydrology and operations.
2. **Close the information gap with HGB.** Add origin-level context (snow anomaly against a fold-specific climatology, the 60-day storage change, or a longer history with a separate summary encoder). Then re-run the history-length study with 3 seeds.
3. **Calibration.** Chronological conformal adjustment of the quantiles, with HGB intervals on identical rows for the Folsom comparison.
4. **Source-dropout training**, evaluated with each source removed and with delayed observations.
5. **A preregistered HGB + neural combination**, specified on the tuning folds.
6. **Hyperparameter search** (hidden size, dropout, learning rate, epochs) on tuning folds only, with 3 seeds.
7. **N5 and N6** only if N4 and the steps above justify them.

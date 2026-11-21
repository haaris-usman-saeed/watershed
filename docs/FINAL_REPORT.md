# WatershedShift
## An audited dataset and historical 30-day reservoir storage-change benchmark

**Project type:** Original data construction, supervised regression, historical evaluation and research handoff  
**Scope:** Six California reservoirs, January 2001 to December 2025  
**Repository:** https://github.com/haaris-usman-saeed/watershed

### Research question and contribution
Can historical reservoir conditions predict the change in storage over the next 30 days more accurately than simple seasonal rules? WatershedShift builds an independently assembled, quality-audited dataset and evaluates this question using chronological backtesting. It is a research tool, not an operational reservoir forecast, drought-warning system or reservoir-management recommendation.

The original contribution lies in the *construction and verification* of a consistent reservoir-weather-snow forecasting table, rather than the invention of reservoir forecasting itself. Existing measurements were joined with documented date conventions, reporting-basis changes, prediction-time rules and reproducible ground-truth labels. The final dataset contains **54,786 station-days** across Shasta, Oroville, Folsom, New Melones, Don Pedro and McClure; **76 storage days remain missing**, rather than being interpolated.

### Data collection and quality audit
Daily reservoir storage comes from California DWR's CDEC sensor 15. The dataset also includes NASA POWER point-reanalysis precipitation, temperature and humidity, and adjusted CDEC snow-water equivalent (SWE) observations from **51 candidate snow pillows**, of which **37** meet a predetermined basin-coverage criterion. The SWE index is a mean across available qualifying pillows, not basin-total snow water volume.

We verified that CDEC's `DATE TIME` field consistently represents the end-of-day storage observation. `OBS DATE` shifts by station and sometimes by year. We corrected Lake Oroville's reporting-basis transition by using 3,537,577 acre-feet before July 2, 2024, and 3,424,753 afterwards, excluding prediction windows crossing the transition. Suspected one-day spikes were checked against independent CDEC elevation measurements; two artifacts were discarded and flood-related extremes retained.

An external comparison with **ResOpsUS** covered all six reservoirs for 2001-2020. Independent Bureau of Reclamation data aligned with **99.6% of Folsom dates within a stated tolerance**; USGS Don Pedro records aligned best at zero-day lag, with median storage difference of 891 acre-feet. Comparisons also found that the ResOpsUS Oroville and McClure series have date labels offset from the consistent end-of-day convention for substantial periods. This validates processing choices without treating same-source CDEC comparisons as independent physical measurements.

### Prediction task and methodology
For each reservoir and Monday origin date *t*, the target is `100 x [storage(t+30) - storage(t)] / capacity(t)`, measured in **percentage points (pp) of capacity**. Missing storage is never filled. The primary model is `HistGradientBoostingRegressor` using four storage-history variables, three SWE variables ending at *t-1*, two seasonal terms and six reservoir indicators. **NASA POWER weather is not an input to the selected primary model**, but is retained in the dataset and used in secondary models. Adjusted SWE and retrospective weather archives may have been revised after the modeled dates.

The pipeline uses **7,076 pre-2024 training rows** and **16 expanding-window water-year development folds (WY2008-WY2023, 4,988 forecasts)**. Hyperparameters were chosen using WY2008-WY2015, with primary variant and seasonal baseline selected on WY2016-WY2023. The **593 held-out 2024-2025 predictions** were evaluated under a committed v2 preregistration. Comparators include persistence, extrapolated 30-day trend, monthly climatology, smoothed day-of-year climatology, Ridge and alternative gradient-boosting feature sets. Forecasts have overlapping 30-day outcomes, so paired uncertainty is estimated with a two-sided moving-block bootstrap across weekly dates at 5-, 9- and 13-week block lengths.

---

### Results

| Historical test, 2024-2025 | v1 | v2 |
|---|---:|---:|
| Primary ML MAE (pp of capacity) | 2.814 | **2.332** |
| Strongest development-selected seasonal baseline MAE (pp) | 3.158 | **2.969** |
| ML minus baseline (pp) | -0.344 | **-0.637** |
| 95% CI, 5-week blocks | [-0.804, +0.030] | **[-1.020, -0.348]** |
| Nominal 90% interval coverage | 96.3% | 92.1% |
| Mean full interval width (pp) | 20.0 | 13.1 |

The v2 primary model reduces MAE by **21.5% compared with the preregistered smoothed day-of-year baseline**. The paired 95% confidence interval remains below zero at 5, 9 and 13 weeks: [-1.020, -0.348], [-1.068, -0.341] and [-1.103, -0.341] pp. It beats that baseline at **all six reservoirs** in the 2024-2025 test and **15 of 16 development water years**, although the development folds informed model selection and are not independent test replications.

### Interpretation and limits
The most informative ablation is that a secondary v2 model retaining the original weather-based feature set but trained on the longer 2001+ history achieves **2.318 pp MAE**, essentially matching the primary model's 2.332 pp. The expanded and corrected training history therefore appears to explain most of the v1-to-v2 improvement. Neither SWE nor weather has a conclusively established incremental advantage across model specifications and development periods. The selected SWE-based model remains the preregistered primary model; the lower-MAE secondary result is not substituted after test inspection.

Prediction intervals are season-conditional, calibrated from rolling out-of-fold development residuals. Overall nominal 90% coverage is **92.1%**, but Folsom reaches only **75.2%** and New Melones approximately 98.0%, illustrating uneven reservoir-level calibration. The model omits actual release decisions and operating rules, uses revised historical meteorological and snowpack products, and is evaluated on only six reservoirs. Its output must not be interpreted as a real-time forecast.

**Test-set reuse disclosure:** The 2024-2025 period was evaluated for v1 and subsequently for a preregistered v2 configuration. The v2 model was selected using pre-2024 development data, but the original test results had already been examined when the scientific revisions were planned. The v2 evaluation therefore provides additional evidence rather than a fully independent replication.

### Reproducibility and intended beneficiary
Public code, frozen metrics, preregistration, coverage audit, provenance, tests and assessor instructions are available in the repository. A deterministic private bundle contains processed data, features, predictions and checksums; raw CDEC responses are excluded pending verification of applicable reuse terms. The clean-clone reproduction and offline GitHub CI were documented and passed. The Streamlit frontend is designed to display historical reservoir behavior and model comparisons; its final real-data integration remains a separate acceptance step.

The immediate research handoff is a concise data-quality note for the **ResOpsUS maintainers** covering date-label convention changes at Oroville and McClure. The note should be accompanied by the reproducible comparison report at `results/v2/resopsus_check/report.md`; proof of sending must be retained separately after delivery.

### Sources

California DWR, California Data Exchange Center (CDEC), daily storage and adjusted SWE, retrieved November 21, 2026. NASA Langley Research Center POWER Project, daily point meteorology. Steyaert, J.C., Condon, L.E., Turner, S.W.D. and Voisin, N. (2022), *Scientific Data* 9, 34, https://doi.org/10.1038/s41597-022-01134-7; dataset https://doi.org/10.5281/zenodo.5893641 (CC BY 4.0). Project source audit: `results/v2/report_facts.md`, `docs/preregistration.md`, and `results/v2/final/metrics.json`.
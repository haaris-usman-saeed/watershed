# v5 features: HGB v5 inputs, flow engineering and the network representation

The machine-readable version is `results/v5/feature_inventory.csv`. Availability counts are in `results/v5/dataset/`.

## 1. HGB v5: HGB[S,N,Wd,Wu,G,Q,C,R], 29 inputs

**The convention.** Every value is known at the end of origin day *t*, the Monday whose storage is the starting point of the 30-day target.

| Group | Feature | Definition | Units | Window |
|---|---|---|---|---|
| S | storage_pct | 100 · S(t) / C(t) | % of capacity | *t* |
| S | prev7_pct, prev30_pct, prev60_pct | 100 · (S(t) − S(t − k)) / C(t), only within one series segment | pp | *t* − k … *t* |
| N | swe_basin_in | basin snow index (core pillows, CDEC sensor 82) | inches | *t* − 1 |
| N | swe_change_14d_in | SWE(t − 1) − SWE(t − 15) | inches | *t* − 15 … *t* − 1 |
| N | swe_anom_in | SWE(t − 1) minus the smoothed day-of-year climatology (σ = 10 days), built only from days before the fold | inches | *t* − 1 |
| Wd | rain_7d_mm, rain_30d_mm | NASA POWER precipitation at the dam point, summed | mm | *t* − 13 … *t* − 7; *t* − 36 … *t* − 7 |
| Wd | temp_30d_c, humidity_30d_pct | mean 2-m temperature and relative humidity at the dam point | °C, % | *t* − 36 … *t* − 7 |
| Wu | the same four with `_box` | means over the upstream-box grid cells (`src/v3_basins.py`) | as above | as above |
| G | gefs_p_d1_7_mm, gefs_p_d8_16_mm | GEFSv12 control member, 00Z on the origin Monday, upstream-box mean precipitation | mm | forecast for *t* + 1 … *t* + 7 and *t* + 8 … *t* + 16 |
| Q | inflow_7d_pp, inflow_30d_pp | see §2 | % of capacity per day | *t* − 6 … *t*; *t* − 29 … *t* |
| Q | outflow_7d_pp, outflow_30d_pp | see §2 | % of capacity per day | *t* − 6 … *t*; *t* − 29 … *t* |
| C | season_sin, season_cos | sin and cos of 2π · doy / 365.25 | — | *t* |
| R | station_DNP … station_SHA | one-hot reservoir indicators | 0/1 | — |

**Monday-row values.** On Monday rows, every column shared with v2 and v3 takes its value from `data/v3/private/weekly_v3.csv`, which is bit-identical to the v2 table. The reason is in the 2026-11-27 amendment to `BENCHMARK_PLAN.md`. The flow features and `gefs_age_days` are computed by `src/v5/data.py`.

**GEFS eligibility.** This is the v3 rule:
- complete cycles only;
- an operational cycle is used only if every file was published by init + 32 h;
- reforecast cycles cover 2001–2019 and are **retrospective**, not issued in real time;
- there is no product for 2020-01-06 to 2020-09-21.

On Monday rows, GEFS is available for 96.8 % of rows. On the confirmation folds it is available for 2,248 of 2,500 rows.

## 2. Flow features (Q)

**Source.** CDEC daily reservoir inflow (sensor 76) and outflow (sensor 23):
- 2001–2023 come from the v4 download of 2026-11-24;
- 2024-01-01 to 2026-11-20 come from the v5 download of 2026-11-27.

**Processing.** Values are parsed and cleaned by `src.v4.flows`:
- outflow below 0 is treated as missing;
- |flow| above 10⁶ cfs is treated as missing;
- negative computed inflow is kept, because it is what the operator reports.

The data is private: sensors 76 and 23 are not covered by `config/data_release_policy.json`.

**The definition:**

$$ \text{inflow\_k d\_pp}(t) = \frac{1}{k} \sum_{j=t-k+1}^{t} \frac{100 \times Q_\text{in}(j)\,[\text{cfs}] \times 1.983471\,[\text{AF per cfs-day}]}{C(j)\,[\text{AF}]}, \quad k \in \{7, 30\} $$

`outflow_k d_pp` uses the same formula with Q_out.

**Missing data.** A feature is missing unless every day of its window is observed. HGB routes missing values natively; Ridge imputes them, with an indicator column.

**Units.** Percent of capacity per day. Multiplying by 30 converts a 30-day mean into an approximate 30-day volume in pp, the unit of the target.

**Windows.** Both windows end at the origin day *t*. This is the information rule for storage, because a day's inflow and outflow are reported with that day's storage. The 7-day window captures the current regime. The 30-day window matches the forecast horizon and smooths day-to-day operating changes.

**Scientific justification:**
1. **The water balance closes.** Over 30 days, observed inflow minus outflow reproduces the storage change to within 0.04–0.33 pp (`docs/v4/FLOW_DATA.md`). The target is therefore, to a good approximation, future cumulative inflow minus future cumulative release.
2. **Inflow persists.** Recession limbs and snowmelt pulses change slowly. Recent inflow is a direct measure of the current hydrological state, which snow and weather indices measure only indirectly.
3. **Release persists.** Outflow is an operator decision: irrigation deliveries, minimum flows, flood-control releases. Day to day these follow schedules, so the recent release level is informative.
4. **Collinearity.** `prev7_pct` and `prev30_pct` already equal, approximately, the summed net flow over the same windows. The flow features add the *split* between the inflow and release components, not a new total.

**Limitations:**
- Inflow is operator-computed from storage change plus releases at most dams; at Don Pedro the daily residual is exactly 0. It is therefore not an independent measurement.
- Don Pedro (about 90 %) and McClure outflow (90 %) have gaps before 2024.
- Values for 2024–2026 are provisional and may be revised.

## 3. Network representation (Experiment A: N1–N4 [all])

| Element | Representation |
|---|---|
| Per-day inputs, days *t* − 29 … *t* (history L = 30) | 10 channels × (value · mask), 10 masks, day-of-year sine and cosine = 22 inputs |
| Channels and lags | storage % (*t*); inflow and outflow as arcsinh of pp per day (*t*); SWE/20 (*t* − 1); dam and box precipitation as arcsinh(mm)/3, temperature/20, RH/100 (*t* − 7) |
| Static inputs | log10(capacity) − 6; in Experiment A, plus GEFS d1–7 and d8–16 (arcsinh(mm)/3), an availability flag and age/7 |
| Reservoir | 4-dimensional learned embedding |
| N4 decoder (days *t* + 1 … *t* + 30) | day-of-year sine and cosine and lead k/30: calendar only |

**GEFS on non-Monday training origins.** Networks (and A-HGB-DAILY) train on every second daily origin. On a non-Monday origin they use the latest eligible Monday cycle issued 1–6 days earlier, together with its age. This uses only information available at that origin. Every evaluation row is a Monday, where the age is 0, so evaluation inputs equal the tabular GEFS features.

## 4. Representation differences

| Difference | Tabular models | Networks |
|---|---|---|
| Storage history | 7-, 30- and 60-day changes | daily storage over the last 30 days (L = 30) or 60–90 days (when tuned): no 60-day change at L = 30 |
| Weather | 7- and 30-day aggregates over *t* − 36 … *t* − 7 | daily values for *t* − 29 … *t* − 7 at L = 30 |
| Flows | 7- and 30-day means | daily series |
| Snow anomaly | explicit `swe_anom_in` (fold climatology) | not given; must be learned from daily SWE and day of year |
| Training samples | v2 Monday rows (A-HGB-DAILY: every 2nd day) | every 2nd daily origin plus a 365-day early-stopping year |
| Loss and point forecast | squared error, mean | pinball loss (N4: composite), median |
| Missing data | HGB native; Ridge imputation with indicators | value 0 with mask 0 |

**What this means for the comparison.** The information *sources* are the same in Experiment A. The information *content* is not exactly equal, and the comparison does not claim it is.

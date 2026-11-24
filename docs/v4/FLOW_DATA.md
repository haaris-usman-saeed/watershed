# Flow data availability and water-balance audit (2001–2023)

**Source.** CDEC daily reservoir inflow (sensor 76) and outflow (sensor 23), fetched on 2026-11-24 by `python -m src.v4.flows --fetch`:
- 6 reservoirs × 2 sensors × 5-year chunks, 60 files;
- raw bytes and SHA-256 in `data/v4/raw/` (git-ignored).

**Publication status.** Sensors 76 and 23 are **not** covered by `config/data_release_policy.json`, so nothing derived from them is published. Only the aggregate statistics below are recorded, in `results/v4/flow_audit.json`.

**Checked but not available.** Full natural flow (sensor 8) has daily values only at New Melones in the probed periods, so it is not used.

## Coverage, share of days 2001-01-01 to 2023-12-31

| Reservoir | Inflow (76) | Outflow (23) | Days with negative inflow |
|---|---|---|---|
| SHA | 0.998 | 0.998 | 0.0002 |
| ORO | 1.000 | 0.993 | 0.0042 |
| FOL | 0.999 | 0.999 | 0.0000 |
| NML | 0.996 | 0.999 | 0.0006 |
| DNP | 0.905 | 0.906 | 0.0431 |
| EXC | 0.988 | 0.900 | 0.0001 |

## Water-balance closure

**Daily closure** compares the residual S(t) − S(t−1) − (inflow − outflow) × 1.983471 AF/cfs-day with capacity.

**30-day closure** compares the v2/v3 target, 100 × (S[t+30] − S[t]) / capacity, with 100 × Σ over days t+1..t+30 of (inflow − outflow) × 1.983471 / capacity.

| Reservoir | Daily residual, median abs (% cap) | Daily residual, p95 abs (% cap) | 30-day MAE, flows vs storage (pp) | 30-day bias, flows − storage (pp) | 30-day correlation |
|---|---|---|---|---|---|
| SHA | 0.0045 | 0.0138 | 0.167 | +0.167 | 0.99993 |
| ORO | 0.0045 | 0.0201 | 0.129 | +0.127 | 0.99994 |
| FOL | 0.0074 | 0.0244 | 0.279 | +0.275 | 0.99981 |
| NML | 0.0041 | 0.0121 | 0.149 | +0.149 | 0.99964 |
| DNP | 0.0000 | 0.0005 | 0.039 | −0.011 | 0.99944 |
| EXC | 0.0001 | 0.0444 | 0.326 | −0.197 | 0.97955 |

## What this means

1. **The physical decomposition holds almost exactly in the data.** Over 30 days, observed inflow minus outflow reproduces the storage change to within 0.04–0.33 pp, which is small next to the target's spread and the 2.3 pp MAE of the current models.

   Forecasting the 30-day storage change is therefore, to a very good approximation, forecasting **cumulative inflow** (hydrology: snowmelt, rain, upstream regulation) and **cumulative release** (operations: flood-control space, contracts, temperature and downstream requirements), plus a small loss term.

2. **The positive bias at SHA, ORO, FOL and NML** (inflow − outflow slightly larger than the storage change) is consistent with evaporation and seepage, which are not in "outflow". A learned non-negative loss term, or a fixed seasonal evaporation estimate, is needed to close the balance.

3. **Inflow is not an independent measurement.** At Don Pedro the daily residual is exactly zero: its reported inflow is computed from storage change plus outflow. Elsewhere it is very likely computed the same way, with evaporation handled differently. Auxiliary inflow labels are therefore "operator-computed inflow". They are consistent with storage by construction and cannot be used as independent validation of the water balance.

4. **Outflow is the operator's decision.** Its history is informative (operations are persistent and rule-based), but its future depends on decisions no meteorological input determines. That is the main reason a pure rainfall-runoff network is not enough for managed storage.

5. **Gaps.** Don Pedro (inflow and outflow about 90%) and McClure (outflow 90%) have gaps, so masks and missing-source training are needed. Auxiliary losses are applied only where the full 30-day label window is observed.

## Consequences for the architecture

- **N4, the physics-informed multi-task model, is supported.** All six reservoirs have inflow and outflow labels for auxiliary heads, and the water balance closes well enough for a water-balance integration layer with a small learned loss term (`ARCHITECTURE.md`).
- **N5, the mass-conserving recurrent model, is feasible** as an encoder over the observed window, or as a decoder driven by *predicted* inflow and release. Its "mass conservation" means the accounting identity between predicted terms, not a guarantee about real water.
- **Realised future inflow or release is never an input.** It is used only as a training label.

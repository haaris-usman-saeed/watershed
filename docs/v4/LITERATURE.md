# Research foundations and how they transfer to managed reservoir storage

These are methodological notes on three papers, each followed by what transfers to WatershedShift and what does not. The summaries describe each paper's published method at the level needed for design decisions. Check details against the papers before citing them.

## Kratzert et al. (2019), regional LSTMs on large samples

*Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets.* HESS 23, 5089–5110. doi:10.5194/hess-23-5089-2019.

**The paper's method**
- **One model across many basins.** A single LSTM is trained jointly on many catchments: the CAMELS US basins, with daily meteorological forcings and static catchment attributes. It predicts daily streamflow.
- **Static attributes.** The **Entity-Aware LSTM (EA-LSTM)** feeds the static attributes into the input gate. The dynamic inputs then drive the cell state, and the static inputs decide which parts of the state each basin uses. This gives an interpretable catchment embedding.
- **Result.** Regional models outperformed basin-by-basin models and benchmark hydrological models, so the information shared across basins helps each one.

**Transfer to WatershedShift**
- **Adopted: one shared network for all six reservoirs** with a reservoir embedding and static inputs (`SharedRNN`, N1/N2). Static attributes are concatenated at every step and at the head. The EA-LSTM input-gate form is a planned variant.
- **Limit: there are 6 reservoirs, not hundreds of basins.** With so few entities, an embedding can memorise each reservoir instead of learning transferable behaviour, so the embedding is small (4 dimensions). Real catchment attributes (upstream-box area, elevation, snow fraction) are planned to replace pure identity.
- **Limit: the target is managed storage, not natural streamflow.** Releases are operator decisions driven by flood-control rules, contracts, temperature management and downstream demands, so the rainfall-to-storage relationship is not purely physical. Section 4 below and `ARCHITECTURE.md` handle this by separating inflow from release.

## Hoedt et al. (2021), MC-LSTM: Mass-Conserving LSTM

ICML 2021, PMLR 139. https://proceedings.mlr.press/v139/hoedt21a.html

**The paper's method**
- **Two kinds of input.** *Mass inputs* (for example precipitation) are conserved quantities. *Auxiliary inputs* only control the gates.
- **Mass accounting by construction.**
  - The cell state holds mass.
  - A normalised input gate distributes incoming mass across cells.
  - A column-normalised redistribution matrix moves mass between cells.
  - An output gate removes a fraction of each cell's mass as outflow.

  Because every gate is normalised, mass in = mass out + change in stored mass, exactly.
- **Result.** In rainfall-runoff modelling on CAMELS it was competitive with an unconstrained LSTM, and better for extreme flows in some analyses. The constraint costs little accuracy and adds physical consistency.

**Transfer to WatershedShift**
- **A reservoir is literally a mass store.** S[t+1] = S[t] + inflow − release − losses.
- **The candidate MC-LSTM (N5) mapping:**
  - storage, in AF, as the cell state, or as a dedicated conserved cell;
  - inflow as the mass input;
  - release and losses as the output-gate flux;
  - weather, snow, season and reservoir as auxiliary inputs.
- **The flow audit (`FLOW_DATA.md`) makes this feasible.** Daily inflow and outflow exist at all six reservoirs, and close the 30-day balance to 0.04–0.33 pp.
- **The catch: inflow is not known at forecast time.** In MC-LSTM the mass input is observed. Here, future inflow has to be *forecast*, and realised future inflow must never be an input. So a direct MC-LSTM can only be a *historical encoder* (conserving mass over the observed window) or a *decoder fed by predicted inflow*. The second option enforces the accounting identity between predicted quantities. It does not guarantee physical mass conservation.

## Lim et al. (2021), Temporal Fusion Transformers

*Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.* International Journal of Forecasting 37(4), 1748–1764. doi:10.1016/j.ijforecast.2021.03.012.

**The paper's method.** TFT separates its inputs into:
- static covariates;
- past observed inputs;
- **known future inputs** (calendar, planned events, forecasts).

Its components:
- gated residual networks and variable-selection networks, which weight inputs at each step;
- static covariate encoders, which condition the variable selection, the LSTM initial state and the enrichment layers;
- an LSTM encoder-decoder for local temporal processing;
- interpretable multi-head self-attention for long-range dependence;
- direct multi-horizon **quantile outputs** trained with the pinball loss.

**Transfer to WatershedShift**
- **Adopted now:**
  - multi-quantile outputs with the pinball loss (all v4 heads);
  - the split between past-observed inputs and known future inputs, which shapes the flagship encoder-decoder.
- **Later candidates:** variable selection and static conditioning of the initial state.
- **Data-size caution.** TFT has many parameters relative to this dataset. There are about 50,000 daily origins, but they are highly autocorrelated. Only 6 entities and about 23 water years give few independent hydrological situations. A compact TFT (N6) is tried only if the simpler recurrent models show that more capacity helps on the tuning folds.
- **Known future inputs here:**
  - the calendar, always;
  - archived GEFS forecasts, which exist only for Monday cycles and only from 2001, with a reforecast/operational split (`docs/v3/MODEL_C_FEASIBILITY.md`).

  Daily training origins therefore lack GEFS on non-Mondays. The options are to mask it (with an availability indicator) or to restrict GEFS-using models to Monday origins.

## Cross-cutting design decisions for v4

1. **The target stays the 30-day change in percentage points of capacity,** so every v4 result is comparable with v2 and v3.
2. **Inputs never include realised future inflow, precipitation or releases.** Future-known inputs are limited to the calendar and forecasts published before the issue time.
3. **Complexity is earned.** N1/N2 (small shared recurrent networks) are run first. Larger or more constrained models (TCN, physics-informed multi-task, MC-LSTM, TFT) are added only when the tuning folds justify them, and always against identical-row baselines including the frozen-configuration HGB.
4. **Physical claims are limited to what the equations enforce.** "Water-balance consistent" means the predicted storage change equals predicted inflow − release − losses by construction. It does not mean real-world mass is conserved.

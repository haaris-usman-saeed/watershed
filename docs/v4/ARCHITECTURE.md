# Flagship design: shared physics-informed encoder-decoder LSTM (proposal)

**Status: proposed, not yet implemented or evaluated.** It builds on the working shared LSTM and GRU (N1/N2) in `src/v4/nets.py`. It is justified by the flow audit (`FLOW_DATA.md`): daily inflow and outflow are available at all six reservoirs, and close the 30-day water balance to within 0.04–0.33 pp.

## Inputs, by what is known at the issue time

| Group | Content | Availability rule |
|---|---|---|
| Static | Reservoir embedding; log capacity; upstream-box area, mean elevation, snow-dominated fraction (planned) | Always |
| Past observed (encoder) | Storage; inflow; outflow; basin snow; dam-point and upstream-box precipitation, temperature and humidity | Up to *t* for storage and flows, *t*−1 for snow, *t*−7 for weather, each with a mask |
| Known future (decoder, days 1–30) | Calendar (day-of-year), always. GEFS precipitation days 1–16 only where a cycle was published before the issue time, with an availability flag. Climatological precipitation as a fallback | Never realised future weather, inflow or release |
| Operations (encoder) | Release history, recent release changes, storage relative to the seasonal flood-control envelope (planned, if a verified envelope source exists) | Up to *t* |

## Network

```
past inputs [L days] -> LSTM encoder ---------> h_t, c_t
static + embedding --------------------------+   (initial state and context)
known future [30 days] -> LSTM decoder (initialised from h_t, c_t, conditioned on static)
   per day k = 1..30:
     inflow head   q_in(k)   (AF/day as pp of capacity; small negative values allowed: operator-computed inflow)
     release head  q_out(k)  >= 0     (softplus)
     loss head     l(k)      >= 0     (softplus; evaporation and seepage)
water-balance integration (deterministic layer):
     dS_wb(k) = sum over j <= k of [ q_in(j) - q_out(j) - l(j) ]
constrained residual correction:
     dS(k) = dS_wb(k) + r(k),  |r(k)| <= r_max (tanh-bounded), penalised
feasibility: 0 <= S(t) + dS(k) <= 100% of capacity (penalty; projection as an ablation)
probabilistic heads: quantiles of the 30-day change dS(30) (and of cumulative inflow and release)
```

**Point and trajectory outputs.**
- The primary output is the 30-day change dS(30), in percentage points of capacity.
- The daily trajectory dS(k) is trained where daily storage labels exist; they exist almost everywhere.

**What is and is not guaranteed.**
- **Guaranteed (accounting).** Without the residual, the predicted storage change equals the predicted inflow minus the predicted release and losses, exactly.
- **Not guaranteed.** With the residual term, that identity holds only up to |r| ≤ r_max. Neither form guarantees that *real* water is conserved: the inflow labels are operator-computed, and the losses are learned.

  So the documentation and the UI must say "water-balance-structured", not "mass-conserving", unless the residual is removed and that model is the one being described.

## Composite training objective

L = w_s·L_storage + w_q·L_quantile + w_in·L_inflow + w_out·L_outflow + w_wb·L_balance + w_f·L_feasible + w_r·L_residual

| Term | Definition | Applied when |
|---|---|---|
| `L_storage` | L1 on the daily trajectory dS(k) | That day's storage label exists and is in the same segment |
| `L_quantile` | Pinball loss of the dS(30) quantiles | Always (the primary target) |
| `L_inflow`, `L_outflow` | L1 on the cumulative 30-day inflow and outflow (pp of capacity) | The full 30-day label window is observed |
| `L_balance` | Penalty on the residual r(k) | Always |
| `L_feasible` | Hinge on storage below 0% or above 100% of capacity | Always |
| `L_residual` | Magnitude penalty on r | Always |

**Normalisation.** Every flow and storage quantity is in percentage points of capacity per day or cumulative. The fixed transforms are in `src/v4/sequences.py`.

**Loss weights** are chosen on the tuning folds (WY2008–2015) only.

## Ablations (each against N1 and the frozen-configuration HGB, on identical rows and multiple seeds)

1. Encoder only (N1), the reference.
2. With flow inputs vs without them. This measures whether past inflow and release history helps at all.
3. Multi-task heads (inflow and outflow) with the water-balance layer vs a direct 30-day head.
4. With vs without the residual correction, and with different r_max.
5. With vs without the feasibility penalty, and penalty vs projection.
6. With vs without the auxiliary losses (each term separately).
7. With vs without GEFS days 1–16 in the decoder: Monday origins only, labelled reforecast or operational.
8. Source dropout during training vs none, evaluated with each source removed at inference.

## Risks

- **Release behaviour is non-stationary.** Operating rules, biological opinions and drought operations change over time. Chronological folds expose this; it must not be hidden by random splits.
- **Inflow labels are computed by operators,** so the inflow head can learn reporting conventions rather than hydrology. This is especially true at Don Pedro.
- **Small number of reservoirs.** Six entities limit what a shared model can learn across reservoirs, which keeps the number of parameters small.

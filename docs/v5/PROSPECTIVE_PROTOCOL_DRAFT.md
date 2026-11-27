# Draft: prospective confirmation protocol for v5 candidates

**Status: draft, not in force.**
- It takes effect only if the owner adopts it as a new version of the prospective protocol, with the hashes recorded in `config/`, before the first forecast it covers is issued.
- `docs/prospective/PROTOCOL.md` and `config/prospective_protocol.json` are **not** changed by this draft.
- No scheduler is added.

## 1. Why a prospective test is needed

Every v5 result is retrospective:
- **Confirmation folds WY2016–2023** were used in v2, v3 and v4.
- **HELD (2024–2025)** was viewed in v2.
- **T26 (2026)** was viewed in v3.

Hyperparameters were selected on WY2012–2015 only. Even so, the choice of what to compare was made with knowledge of earlier results on all these periods. Only forecasts issued after a model is frozen are independent.

## 2. What is frozen before the first issued forecast

**For each candidate:**
- the model ID from `results/v5/model_inventory.csv`;
- the SHA-256 of `src/v5/*.py`, `src/v4/nets.py` and `src/v4/train.py`;
- the hyperparameters (from `results/v5/selection.json` for tuned models);
- the training cutoff;
- the seeds;
- the SHA-256 of every weight file.

**Two training regimes:**
- **Refit for issuance:** trained once on all targets before the freeze date, and saved.
- **No re-training** during the evaluation period, unless a later protocol version allows it.

**Candidates.** At most three, chosen by the owner. Example: HGB v5 (frozen hyperparameters), tuned HGB v5, and the best network. The reference is always the deployed frozen v2 artifact (HGB[S,N,C,R] · artifact ≤2023), together with DOY climatology.

## 3. Issuance and scoring

**Issuance.**
- Monday origins.
- Issued manually by the existing official issuance command, extended to the candidates by a separately reviewed code change.
- Each record is appended to the official ledger with its input snapshot hash.

**Inputs.** Only data available at issue time:
- CDEC storage and flows up to the origin;
- snow up to *t* − 1;
- POWER weather up to *t* − 7;
- the GEFS 00Z Monday cycle, as published by issue time.

If an input is missing at issue time, the record says so; it is never back-filled.

**Scoring.**
- Observed 30-day storage change from CDEC (sensor 15), retrieved after the target date and stored with its SHA-256.
- One score per record, written once.

## 4. Analysis (fixed now)

**The primary comparison.** For each candidate:
- the paired MAE difference against the frozen v2 artifact on identical issued rows;
- moving-block bootstrap (5-, 9- and 13-week blocks, 2,000 replicates, seed 42);
- Holm correction over the candidates;
- "supported" only if the Holm-adjusted p (the largest over block lengths) is below 0.05.

**When it runs.** Not before 150 scored origin dates (about 900 rows, about 3 years). The first analysis is at that milestone. There are no interim looks with decision power.

**Why so long.** The 2026 test gave 95 % CIs of about ±0.05 to ±0.15 pp between close HGB variants. CI width shrinks roughly with the square root of the number of rows, and the rows are dependent. This is an order-of-magnitude argument, not a power calculation.

**Secondary analyses** are reported but cannot support a claim:
- RMSE, bias and R²;
- MAE by reservoir, water-year class and regime;
- coverage and width of 90 % intervals.

## 5. What would end the protocol early

- A data or code defect that invalidates issued records. It is documented, and the protocol is restarted as a new version.
- A source discontinued (CDEC, POWER or GEFS). The affected candidate is stopped; past records stay.

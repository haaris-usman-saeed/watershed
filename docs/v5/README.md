# WatershedShift v5: HGB v5 and the standardised model benchmark

**Status: research only.**
- Nothing from v5 is deployed or shown on the website.
- Nothing frozen was changed: models, results, issued records, public releases and v4 results are all untouched.
- Every v5 model is a new per-fold refit of an existing configuration, or a new configuration, saved under `models/v5/` (git-ignored).

| Document | Content |
|---|---|
| [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) | the plan, written before any v5 result; SHA-256 recorded in every run manifest |
| [FEATURES.md](FEATURES.md) | HGB v5 inputs, flow feature engineering, network representation, representation differences |
| [RESULTS.md](RESULTS.md) | the research memo: what was run, the numbers, the statistics and what they mean |
| [PROSPECTIVE_PROTOCOL_DRAFT.md](PROSPECTIVE_PROTOCOL_DRAFT.md) | draft protocol for genuinely independent confirmation; not in force until the owner adopts it |

## How it was run

Every command runs in the v4 environment (`.venv-v4`, which is `requirements-v4.txt`: the lock file plus PyTorch 2.15.0).

1. Download the CDEC flows for 2024-01-01 to 2026-11-20. This step uses the network.

```bash
.venv-v4/bin/python -m src.v5.flows_ext --fetch
```

2. Build the panel and features, with their checks against v4 and v3.

```bash
.venv-v4/bin/python -m src.v5.data
```

3. Compute the dataset and training-row accounting.

```bash
.venv-v4/bin/python -m src.v5.accounting
```

4. Run the frozen tabular models and baselines on tuning and confirmation folds.

```bash
.venv-v4/bin/python -m src.v5.run tabular --which dev
```

5. Run the whole benchmark chain: frozen networks, then search, selection, tuned models, and the retrospective windows.

```bash
./data/v5/chain.sh
```

6. Run the tuned configurations on the tuning folds. This is needed for conformal calibration.

```bash
.venv-v4/bin/python -m src.v5.run tuned --which tune --workers 9 --threads 1
```

7. Run the controlled compute benchmark.

```bash
OMP_NUM_THREADS=1 .venv-v4/bin/python -m src.v5.compute
```

8. Build the tables, statistics and figures.

```bash
.venv-v4/bin/python -m src.v5.report
```

The offline tests use synthetic data and need no PyTorch:

```bash
.venv/bin/python -m pytest -q tests/test_v5.py
```

## Outputs

| Path | Content | Tracked |
|---|---|---|
| `results/v5/dataset/` | dataset manifest (input and output hashes, equality checks), accounting, missingness, GEFS and flow availability, per-fold row counts | yes |
| `results/v5/metrics/` | one table per population, plus breakdowns by reservoir, water-year class, regime and fold | yes |
| `results/v5/stats/` | planned comparisons (families F and T, Holm), secondary and post-hoc comparisons | yes |
| `results/v5/model_inventory.csv`, `feature_inventory.csv`, `hyperparameter_registry.json`, `row_accounting.csv`, `selection.json`, `tuning/` | inventories, the search log and the selection | yes |
| `results/v5/figures/` | figures, PNG and SVG | yes |
| `results/v5/manifests/` | per-stage manifests: plan and code SHA-256, dataset hashes, UTC times | yes |
| `data/v5/` | private panel, features, flow downloads, per-row predictions, run logs | no (CDEC-derived) |
| `models/v5/` | every trained model: tabular `.joblib` per fold; network `state_dict` per fold and seed | no |

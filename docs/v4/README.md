# WatershedShift v4: NeuralHydro (research programme)

v4 is a separate deep-learning research programme for 30-day managed-reservoir storage forecasting. It works on its own branch, in its own environment and with its own data folders.

**Nothing frozen is touched.** The v1, v2 and v3 models, results, preregistrations, issued forecasts and public releases stay as they are. **The frozen v2 Model A remains the validated fallback and the research reference.** No v4 model is deployed.

| Document | Content |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | Experimental integrity rules, written before any result was seen |
| [LITERATURE.md](LITERATURE.md) | Kratzert et al. 2019, Hoedt et al. 2021 and Lim et al. 2021, and what transfers to managed reservoirs |
| [FLOW_DATA.md](FLOW_DATA.md) | CDEC inflow and outflow availability, and the water-balance audit |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Flagship design (N4): a physics-informed encoder-decoder LSTM, its objective and ablations |
| [N4_PLAN.md](N4_PLAN.md) | N4 candidates, selection rule and comparisons, fixed before any N4 run |
| [RESULTS_N4.md](RESULTS_N4.md) | N4 development results (tuning and confirmation folds) |
| [RESULTS_INITIAL.md](RESULTS_INITIAL.md) | First chronological development results, and what they do and do not show |
| [RESULTS_TABLES.md](RESULTS_TABLES.md) | Every result table, generated from `results/v4/` by `python -m src.v4.report` |

## Environment (separate from the deployed app)

```bash
python3.14 -m venv .venv-v4
```

```bash
.venv-v4/bin/pip install -r requirements-v4.txt
```

`requirements-v4.txt` is the app's `requirements-lock.txt` plus PyTorch 2.15.0 and its dependencies. The app, CI and the Streamlit deployment keep using `requirements.txt` and `requirements-lock.txt`, so they are unaffected. `.venv-v4/` is git-ignored.

## Reproduce (on the trusted build machine, which holds the private processed data)

```bash
.venv-v4/bin/python -m src.v4.flows --fetch
```

```bash
.venv-v4/bin/python -m src.v4.dataset
```

```bash
.venv-v4/bin/python -m src.v4.flows --audit
```

```bash
.venv-v4/bin/python -m src.v4.experiment context
```

```bash
.venv-v4/bin/python -m src.v4.experiment compare --length 30 --models lstm gru tcn
```

```bash
.venv-v4/bin/python -m src.v4.experiment ablation --length 30
```

```bash
.venv-v4/bin/python -m src.v4.experiment n4-tune
```

```bash
.venv-v4/bin/python -m src.v4.experiment n4-confirm
```

```bash
.venv-v4/bin/python -m src.v4.report > docs/v4/RESULTS_TABLES.md
```

```bash
.venv-v4/bin/python -m pytest -q tests/test_v4.py
```

What each step does:
1. `flows --fetch` downloads CDEC inflow and outflow for 2001–2023 (network; about 60 small requests).
2. `dataset` builds the daily panel, `data/v4/daily_panel.csv`.
3. `flows --audit` writes `results/v4/flow_audit.json`.
4. `experiment context` runs the history-length study on tuning folds WY2012–2015.
5. `experiment compare` runs the main comparison on confirmation folds WY2016–2023 at the selected length (30 days).
6. `experiment ablation` (post-hoc) runs HGB with flow features and the LSTM without flows.
7. `experiment n4-tune` selects the N4 configuration and history length on tuning folds WY2012–2015 (47 minutes), as fixed in N4_PLAN.md.
8. `experiment n4-confirm` compares the selected N4, its strict and direct variants and the baselines on confirmation folds WY2016–2023 (29 minutes).
9. `report` regenerates the result tables.
10. The tests run offline on synthetic data. Without PyTorch, the network tests skip; CI runs the rest.

The full set of stages took about 2.8 hours on an Apple M5 CPU.

## Layout

| Path | Content |
|---|---|
| `src/v4/flows.py` | CDEC inflow and outflow fetch, parsing, cleaning, water-balance audit |
| `src/v4/dataset.py` | Daily panel: storage, snow, weather (dam point and upstream box), flows; full calendar; series-break segments |
| `src/v4/sequences.py` | Leakage-safe windows, masks, targets (30-day change, daily trajectory, cumulative flows), chronological folds, v2 Monday-row matching (numpy only) |
| `src/v4/nets.py` | N1 shared LSTM, N2 shared GRU, N3 TCN, N4 water-balance encoder-decoder; non-crossing quantile head; pinball loss |
| `src/v4/train.py` | Seeded training, early stopping, loss curves, timings; N4 composite loss |
| `src/v4/experiment.py` | Development stages and identical-row baselines (including the frozen-configuration HGB) |
| `results/v4/` | Aggregate results (tracked) |
| `data/v4/` | Panel, raw flows, per-row predictions (private, git-ignored) |

## Status

- **N1 (LSTM), N2 (GRU) and N3 (TCN)** are implemented and evaluated on the development folds (RESULTS_INITIAL.md). They match, but do not beat, HGB with the frozen v2 configuration on identical rows.
- **N4, the water-balance-structured encoder-decoder,** is implemented and evaluated (RESULTS_N4.md).
  - Its 3-seed ensemble has the lowest point MAE on the 2016–2023 development folds: 3.084 pp, against 3.160 for the HGB SWE no weather per-fold refit.
  - That difference is **not established** under the protocol's decision rule.
  - The water-balance layer itself adds no measurable accuracy.
  - N4 is **not frozen** (no weights saved) and **not deployed**. By the owner's decision of 2026-11-24, nothing from v4 appears on the website.
- **N5** is designed. **N6** (a compact TFT) is deferred until the simpler models justify more capacity.

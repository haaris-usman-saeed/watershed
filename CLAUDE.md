# CLAUDE.md: WatershedShift

Read this file first in every session. Then read:
- [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md) for the current state, open problems and priorities;
- the last section of [STATUS.md](STATUS.md) for the most recent checkpoint.

## What this project is

WatershedShift forecasts the **30-day change in reservoir storage** at six California reservoirs: SHA, ORO, FOL, NML, DNP and EXC.
- The change is measured in percentage points of the capacity in force at the origin.
- Origins are Mondays.
- The models are scikit-learn `HistGradientBoostingRegressor` with frozen hyperparameters.

It has three layers, in the order they were built:

1. **v2, the frozen research submission.** "HGB SWE no weather", preregistered and evaluated once on 2024–2025 (n = 593). It was clearly better than the seasonal baselines.
2. **v3, the preregistered extension.** Models A, B, B2 and C, tested once on 2026 (251 rows). All three hypotheses are **not established**.
3. **The website.** A Streamlit app (`app.py`) with historical backtest pages and a Generate Forecast page. That page serves the frozen v2 Model A plus the v3 Models B, B2 and C, using live CDEC, NASA POWER and NOAA GEFS data.

The owner makes every scientific, publication and merge decision. The engineering agent (you) implements, verifies and reports.

## Non-negotiable rules

These hold even if a later instruction seems to imply otherwise. If a task appears to need one of them broken, stop and ask the owner.

**Frozen models and results**
1. **Never retrain, refit or modify the frozen models, and never rerun the final evaluations.**
   - `python -m src.train_v2 --final` and `python -m src.v3_experiment --final` are one-shot and guarded.
   - Do not work around their guards.
2. **Never edit frozen or hashed files.** They are listed in the next section.
   - Editing any of them breaks a guard, a test or CI, even for a typo.
   - Fix stale wording elsewhere: in a new document, or by appending to STATUS.md or the ledger.
3. **No new model selection on viewed test sets.**
   - 2024–2025 and the 2026 test window (origins from 2026-01-05, targets up to 2026-11-20) have both been seen.
   - Any new analysis of them must be labelled **post-hoc**. Only analysis registered *before* the outcomes exist may be called **preregistered**.

**Records and published data**

4. **Issued records are immutable.**
   - [docs/v3/issued_forecast_ledger.jsonl](docs/v3/issued_forecast_ledger.jsonl) is append-only; never rewrite an existing line.
   - The records under `forecasts/records*/` are read-only (mode 0444).
5. **Published data is write-once.**
   - Never change, delete or replace anything under `public_data/<version>/`. A correction is a new version built with `python -m src.v3_release --version <new>`.
   - Never replace a GitHub Release asset. A new bundle gets a new tag, and both secrets are updated together.
6. **Publication safeguards.**
   - [config/data_release_policy.json](config/data_release_policy.json) changes only on the owner's explicit instruction.
   - Never loosen it, or remove the `WATERSHED_DEPLOYMENT=public` policy gate, without verified authority.
   - Do not invent names, dates or licence conditions for data permissions.

**Honest claims**

7. **Retrospective is not real time.**
   - GEFS reforecasts (2001–2019) were never issued in real time. They are always labelled `reforecast` / "retrospective".
   - Backtests and reforecasts are never presented as issued forecasts.
8. **Forecasts are not operations.** The models predict observed storage change. They do not simulate or advise reservoir operations.
9. **No invented results.**
   - Report only what was run and observed, with the command.
   - Do not claim Model C (or any model) is more accurate. Every v3 verdict is "not established".
   - Do not describe the public website as working until it has been checked on its public URL ([docs/v3/LAUNCH_AND_MERGE.md §4](docs/v3/LAUNCH_AND_MERGE.md)).

**Ownership, git and deployment**

10. **UI ownership.**
    - `app.py`, `app/`, `.streamlit/` and `docs/UI_*` belong to the UI owner, except `docs/UI_CONTRACT.md`. Change them only when the owner explicitly asks.
    - No arbitrary frontend changes.
11. **Git.**
    - Work on a branch, never directly on `main`.
    - Commit, push or open a PR only when the owner asks.
    - **Never merge a PR without the owner's explicit authorization for that PR.**
    - Never commit `.claude/`, secrets, or anything CI forbids: `data/raw`, `data/processed`, `data/live`, `data/v3`, `outputs`, `dist`, `models`, `forecasts`.
12. **Deployment fixes.**
    - Do not redesign or retrain models to fix a deployment error.
    - Do not add a scheduler for official forecasts until the prospective protocol and its storage design are agreed and committed.

## Frozen and protected files

| Path | Why | What enforces it |
|---|---|---|
| `docs/preregistration.md`, `config/v2_frozen.json`, `src/train_v2.py` | v2 preregistration | SHA-256 in `results/v2/final/RUN_RECORD.json`, checked by CI and `src.forecast_model` |
| `src/models.py`, `src/features.py`, `src/evaluation.py`, `src/build_dataset.py` | Shared by v2 and v3 | v2 run record and `config/v3_plan.json` |
| `docs/v3/PREREGISTRATION_V3.md`, `config/v3_plan.json` | v3 preregistration | `code_sha256` and `plan_sha256` in `results/v3/final/RUN_RECORD.json` |
| `src/v3_basins.py`, `src/v3_gefs.py`, `src/v3_power.py`, `src/v3_dataset.py`, `src/v3_experiment.py`, `src/forecast_live.py`, `src/forecast_model.py` | Listed in the v3 plan | `config/v3_plan.json` `code_sha256` |
| `results/v1/`, `results/v2/final/`, `results/v3/final/` | Frozen results | Guards, and tests in `tests/test_v2_guards.py`, `tests/test_report_facts.py` and `tests/test_v3.py` |
| `docs/v3/issued_forecast_ledger.jsonl` | Proof that issued records are unaltered | Append-only; `tests/test_v3.py` checks every record hash |
| `public_data/*` | Released datasets | `python -m src.v3_release --check` in CI |
| Tags `v1-results`, `models-2026-11-22` and the release asset | Published | Never moved or replaced |

Changing a file in the v3 list (including `forecast_live.py` and `forecast_model.py`) breaks the v3 guard, so a bug there needs the owner's decision first. Usually the fix belongs in a new module or in `src/v3_service.py`, which is not frozen.

## Environment and commands

- **Python.** The local virtualenv is `.venv` (Python 3.14.5). CI uses Python 3.14 with `requirements-lock.txt`.
- **Tests.** `.venv/bin/python -m pytest -q tests`. Tests write only to temporary directories.
  - Without the private data: 107 passed, 12 skipped.
  - On this machine, with the private data: 109 or more pass. Some tests then rebuild Model A and compare all 593 predictions, which takes longer.
- **The other CI checks** (see `.github/workflows/ci.yml`):
  - `python -m src.contract app/sample_data`;
  - the frozen-hash check against `results/v2/final/RUN_RECORD.json`;
  - `python -m src.v3_release --check`;
  - `python -m src.make_forecast_fixture`, then `git diff --exit-code app/sample_data/forecast_snapshot.json`.
- **App.** `.venv/bin/streamlit run app.py`. Local preview configurations are in `.claude/launch.json`, which is untracked; `bundle-server` and `clean-public` point to old scratch paths.
- **Model bundle.**
  - `python -m src.v3_artifacts pack --out dist/` (trusted machine only) gives a new SHA-256 if any model file changed.
  - `install --url … --sha256 …` is what deployments run, through `ensure_artifacts()`.
- **Never run casually:** `--final` runs, `src.v3_release --version` (it creates a permanent release), `gh release create/upload`, `gh pr merge`.

## Repository map

- `src/`: the pipeline. Engineering owns it.
  - v1/v2: `fetch*`, `build_dataset`, `features`, `experiments`, `train_v2`, `models`, `evaluation`.
  - Live Model A: `forecast_model`, `forecast_live`.
  - v3: `v3_basins`, `v3_power`, `v3_gefs`, `v3_dataset`, `v3_experiment`, `v3_release`, `v3_artifacts`, and `v3_service` (the live forecast service: cache, rate limits, models, records).
- `app.py` and `app/`: the Streamlit UI, owned by the UI owner. The pages are overview, explorer, backtest, research, the forecast page (issued record) and `view_forecast_v3` (Generate Forecast).
- `config/`: frozen plans and the data release policy.
- `results/`: frozen results.
- `public_data/`: write-once data releases.
- `docs/`: method, audit, contract and v3 documents.
- `tests/`: offline tests.
- **Git-ignored and only on the trusted build machine:** `data/raw`, `data/processed`, `data/v3`, `data/live`, `models/`, `dist/`, `forecasts/`, `outputs/`. They are not in the repository and cannot be recovered from it.

## Working conventions

- **STATUS.md** is shared. Append a dated section under your own heading; never rewrite earlier entries.
- **Current state** goes in `docs/PROJECT_HANDOFF.md`.
  - Update its "Current state" section and its *Last verified* line whenever the state changes, for example a PR merged, a deployment checked or a record issued.
  - History stays in STATUS.md.
- **Every behaviour change gets an offline test.** Tests never touch the network.
- **Dates are absolute** (YYYY-MM-DD) and **times are UTC**.
- **State uncertainty plainly** ("not established", "unverified", "owner-reported") rather than rounding it up.
- **Before a PR,** run the full test suite and the CI checks above, and confirm `git status` shows no frozen or ignored path.

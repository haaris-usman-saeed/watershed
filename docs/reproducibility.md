# Reproducibility record (v2)

This record summarizes existing evidence. No new fresh-clone run was performed for this handoff.

## Frozen configuration and single held-out run

| Item | Value | Source |
|---|---|---|
| Frozen configuration | `config/v2_frozen.json`, SHA-256 `02cfcfd007b0513a50d00b60354c31d102d4452f381dfcce329f692e866c230c` | `docs/preregistration.md`, `results/v2/final/RUN_RECORD.json` |
| Preregistration and frozen code committed | commit `e47deabb647f0cd99d60103ea754a4b7e4b5cd00` | `git log -- docs/preregistration.md` |
| Single held-out run | `python -m src.train_v2 --final` at commit `e47deab`, 2026-11-21T10:46:10Z | `RUN_RECORD.json`, `results/v2/final_run_stdout.txt` |
| Frozen code hashes | `src/train_v2.py` `3d2f6b07...`, `src/models.py` `416210cf...`, `src/features.py` `68912938...`, `src/evaluation.py` `da29e568...`, `src/build_dataset.py` `b40a6d9b...` (full values in the sources) | `RUN_RECORD.json`, `docs/preregistration.md` |
| Headline | ML 2.332333455339604 pp; Smoothed DOY seasonal 2.969354735255069 pp; coverage 0.9207419898819561 | `RUN_RECORD.json`, `results/v2/final/metrics.json` |

Checked for this handoff (2026-11-21):
- The working-tree SHA-256 of the config and the five frozen code files equals both the preregistration table and `RUN_RECORD.json` (`results/v2/report_facts.json` fact 15; `python -m src.train_v2` performs the same check before any run).
- `git diff e47deab HEAD` is empty for the config, the preregistration and the five frozen files.
- `results/v2/final/` was written in one commit (`f21bf25`) and has not been modified since.
- v1 is preserved: tag `v1-results` (commit `01e43cb`) is on the remote, and `results/v1/` files were only ever added (commits `8f69ad5`, `7b6f041`), never modified.

## Previous verified clean-clone reproduction

Performed once, before the ResOpsUS follow-up, and recorded in `STATUS.md` (Checkpoint E):
- A fresh `git clone -b review-v2` at commit `cf7d4bc`, a new venv, `pip install -r requirements.txt`, and every README step, including re-downloading all raw data.
- `daily_joined.csv`, `snow_daily.csv`, `snow_basin_index.csv` and `weekly_features.csv` were byte-identical to the original run.
- `results/v2/development/` regenerated with no git diff.
- The frozen v2 rerun (allowed as an identical-configuration rerun) reproduced the headline exactly; only `generated_utc` changed in `metrics.json`.
- 25 of 25 tests passed at that commit; the outputs contract validated; the bundle verified.
- 177 of 182 re-downloaded raw files were byte-identical; the 5 that differed were NASA POWER responses whose only difference is the server `times` field.

Since `cf7d4bc`, the only changes under `src/` are three new, separate scripts (`src/fetch_resopsus.py`, `src/compare_resopsus.py`, `src/report_facts.py`). None of them is imported by the pipeline or writes to `data/processed/`, `results/v2/final/` or `results/v2/development/`. The test count has grown from 25 to the current total through new tests only.

## Gaps

- The terminal logs of the clean-clone run were not archived as files. The evidence is the `STATUS.md` record above, plus the regenerated files matching the committed ones.
- The clean clone used the same machine (macOS, Python 3.14.5). Reproduction on another operating system or Python version has not been tested. The exact package versions are in `requirements-lock.txt`.
- Re-downloading from CDEC or NASA POWER later may give different raw bytes, because agencies revise provisional data (see `ASSESSORS.md`).

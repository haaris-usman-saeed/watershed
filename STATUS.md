# WatershedShift — STATUS

Shared file. Append under your own heading only.

## Engineering (Claude Code) — branch `engineering`

### 2026-11-21 — kickoff
**Findings from starter review**
- Pipeline: `src/fetch.py` (CDEC CSVDataServlet sensor 15 + NASA POWER daily point) → `src/features.py` (weekly Monday rows, 30-day target, weather lagged 7 d) → `src/train.py` (3 baselines + HistGradientBoosting, constant-width 90% interval from 2023 validation errors).
- Starter was never run against real data. `stations.csv` coordinates/capacities are unverified candidates.
- First live probe: CDEC `SHA` sensor 15 returns **monthly** (`dur_code=M`) values, but `dur_code=D` returns an empty CSV (header only). Investigating station sensor listing before going further — do not assume daily storage exists for every station.
- Starter gaps to fix: no per-station metrics, no bootstrap CI, test split filter `target_date <= 2025-12-31` silently depends on fetch end date; raw NASA JSON overwritten without timestamp; no provenance/coverage outputs.

**Plan**: (1) verify CDEC daily storage per station & capacities, NASA POWER params → C1 coverage table; (2) fix features/splits, add per-station metrics + block bootstrap → C2; (3) clean-env reproduction, README, provenance → C3.

### UI contract published
- `docs/UI_CONTRACT.md` (schema v1.0), validator `src/contract.py`, synthetic fixtures in `app/sample_data/` (clearly labelled `SYNTHETIC_SAMPLE`, station IDs `SYN1..SYN4`).
- UI owns `app.py` and `app/` (except `app/sample_data/`). Engineering will not edit `app.py`.

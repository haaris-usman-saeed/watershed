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

### C1 — real data verified (2026-11-21)
- **Live downloads succeeded for all 6 stations** (SHA, ORO, FOL, NML, DNP, EXC): CDEC sensor 15 daily, 2018-01-01 → 2025-12-31, plus NASA POWER T2M/PRECTOTCORR/RH2M. 48 CDEC + 6 POWER raw files, about 2 MB, kept **locally only** in `data/raw/` (git-ignored). URLs and SHA-256 hashes are in `data/raw_manifest.jsonl`.
- Joined daily dataset: **17,532 station-days** (6 × 2,922). Storage missing: SHA 0, ORO 1, FOL 0, NML 0, DNP 2, EXC 2 days (max 0.068%). Weather missing: 0. Duplicates: 0. Full table: `data/coverage_table.md`.
- **Corrections to the starter:**
  1. CDEC `OBS DATE` is inconsistent by operator (D vs D+1). The code now uses `DATE TIME`, which is end-of-day storage, verified against hourly data. The starter would have shifted ORO, DNP and EXC by one day.
  2. ORO capacity changed 3,537,577 → **3,424,753**; EXC 1,025,000 → **1,024,600**. FOL and NML coordinates were corrected from CDEC metadata.
  3. Fetching is now split into a network step (`src.fetch`, raw bytes saved unchanged, per-year requests with retries) and an offline step (`src.build_dataset`). CDEC sometimes returns empty bodies, which the retries handle.
- **Data issue found:** ORO has a **reporting-basis break on 2024-07-02**. DWR adopted a new elevation–storage table, producing a −123k AF artificial step. Values are not adjusted. The break is logged in `data/series_breaks.csv`, and C2 will exclude any target or change-feature window that spans it.
- Latency observed today: CDEC 1–2 days; POWER about 3 days (recent days come from GEOS-IT, the archive from MERRA-2, so retrospective ≠ real-time). Details are in `data/provenance.md`.
- Usable for modelling: **all 6 reservoirs, full 2018–2025**, minus ORO windows that span the break.
- Next (C2): weekly targets with series-break masking; baselines + GBM; per-station metrics; block-bootstrap CI.

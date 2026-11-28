# WatershedShift: project handoff

**Last verified:** 2026-11-27, against `main` at `da1101d` (after PRs #13–#19). Before that: 2026-11-24 against `1accaa6` (after PRs #7, #10, #11, #8, #12 and #9). The public website (https://watershed-lwe3carmkq2mgurch5jurm.streamlit.app) was checked on 2026-11-27 (section 9). The first version was a read-only recovery review on 2026-11-23 against `0063cc0`: nothing was retrained, re-evaluated or republished.

This document holds everything needed to take over the project without reconstructing it from the history.
- Rules for agents are in [CLAUDE.md](../CLAUDE.md).
- The history of checkpoints is in [STATUS.md](../STATUS.md).
- When the state described here changes, update the relevant section and the *Last verified* line in the same PR.

Contents:
1. The project in brief
2. Current state
3. History
4. Repository architecture
5. Models
6. Data and publication
7. Model bundle and release
8. The live forecast service
9. Deployment
10. Issued forecasts and prospective evaluation
11. Known bugs and weaknesses
12. Documents that contradict the current state
13. Priorities
14. Owner-only actions and missing items
15. Assets that exist only on the trusted build machine
16. How to re-verify this document
17. Glossary

---

## 1. The project in brief

**Question.** How much will storage at a reservoir change over the next 30 days?

**Reservoirs.** Six in California:

| Code | Reservoir | Operator |
|---|---|---|
| SHA | Shasta | US Bureau of Reclamation |
| ORO | Oroville | DWR |
| FOL | Folsom | US Bureau of Reclamation |
| NML | New Melones | US Bureau of Reclamation |
| DNP | Don Pedro | Listed as USGS/USACE in the CDEC operator field |
| EXC | McClure | Merced Irrigation District |

**Target.**
- The storage change from the origin to origin + 30 days, in percentage points of the capacity in force at the origin.
- Origins are Mondays.
- Storage values are CDEC daily values (sensor 15).
- Oroville's capacity changed on 2024-07-02 (3,537,577 to 3,424,753 AF). Windows that span a series break are excluded (`data/series_breaks.csv`).

**Model family.**
- scikit-learn `HistGradientBoostingRegressor`.
- Frozen hyperparameters: learning rate 0.03, 150 iterations, no depth limit, 30 minimum samples per leaf, L2 1.0, random state 42.

**Results**

| Stage | Test | Result | Where it is recorded |
|---|---|---|---|
| v1 (frozen tag `v1-results`) | 2024–2025 | GBM 2.81 pp against seasonal 3.16 pp | `results/v1/` |
| **v2 (frozen submission)** | 2024–2025, n = 593, evaluated once | ML MAE **2.332**, smoothed seasonal baseline 2.969, historical seasonal 2.977. Every bootstrap CI is below zero. 90% interval coverage 0.921; Folsom about 0.75. | `results/v2/final/` |
| **v3 (preregistered extension)** | 2026, 251 rows, evaluated once | A (v3 refit) 2.349, B 2.339, B2 2.364, **C 2.316**, A (frozen v2) 2.355, smoothed seasonal baseline 2.882. **H1 (C<A), H2 (B<A) and H3 (B2<B) are all "not established."** | `results/v3/final/` |

Notes on these results:
- **v2 disclosure:** the v1 results for 2024–2025 had been seen before v2 was designed (`docs/preregistration.md`).
- **v2 development folds:** the confirmation folds WY2016–2023 (n = 2,500) are descriptive.
- **v3 intervals:** 90% intervals (`sym_season`, built from each model's own development residuals) covered 0.93–0.94 overall in 2026, but only 0.69–0.74 at Folsom.
- **v3 against the seasonal baseline:** every model's MAE was 0.53–0.57 pp lower than the baseline's in 2026, but with 42 weekly origins this is **not established** on 2026 alone.

**Not established or not validated**
- Whether any v3 model is better than another.
- Model C in real time: about 80% of its training rows used reforecasts, but live forecasts use operational GEFS.
- Any prospective (genuinely future) accuracy: no forecast has been scored yet.
- The upstream boxes are rectangles, not delineated watersheds.
- Recent sensor-82 snow values and NASA POWER near-real-time values may be revised.

**Wording rules.**
- No model is the winner.
- Reforecasts are labelled "retrospective, not issued in real time".
- Forecasts are statistical predictions of observed storage change. They are not reservoir-operation simulations and not a water-safety tool.

## 2. Current state (2026-11-27)

| Item | State | Evidence |
|---|---|---|
| `main` | `da1101d` = `origin/main`. PRs #1–#19 merged; none open | `git ls-remote`, `gh pr list` |
| CI on `main` | Green (run 46084465213, on `da1101d`) | GitHub Actions |
| Frozen v2 and v3 material | Intact | All 6 v2 hashes match `RUN_RECORD.json`. All 12 v3 hashes match `config/v3_plan.json`, and the plan hash matches the v3 run record. |
| Issued forecast records | 2 records, both unaltered | Local and public copies match `docs/v3/issued_forecast_ledger.jsonl` |
| Public data releases | 2 versions, both intact | `python -m src.v3_release --check` |
| Model bundle | Uploaded as a **pre-release** | Tag `models-2026-11-22`; GitHub's digest equals the pinned SHA-256; publicly downloadable |
| Public website | **Checked 2026-11-27, about 23:30 UTC**, at https://watershed-lwe3carmkq2mgurch5jurm.streamlit.app. Generate Forecast worked for all six reservoirs. After PRs #18 and #19 (about 23:50 UTC), Model playground replays work in all three windows with all ten models | Section 9; STATUS.md 2026-11-27 |
| Website forecast model | Generate Forecast runs only HGB[S,N,C,R] · frozen v2 artifact (owner decision 2026-11-27, PR #16). v3 Models B, B2 and C are no longer shown on the site; `src/official.py` still issues all four | `app/view_forecast_v3.py`, `ForecastService(model_keys=("A",))` |
| Model playground | New page (PR #16): the other nine study models against the HGB on past windows only, read from tracked `results/v5` aggregates. Per-row replays read the tracked export `results/v5/playground/retro_predictions.csv` (no flow, weather or snow values) | `app/view_playground.py`, `src/v5/playground.py` |
| v4 N4 and v5 benchmark | Merged as research (PRs #14, #15); nothing frozen, nothing makes live forecasts. v5 results on 2024–2025 and 2026 are post-hoc | `docs/v4/RESULTS_N4.md`, `docs/v5/RESULTS.md` |
| CDEC publication permission | Owner-reported, **not archived**: storage and snow (2026-11-22); inflow and outflow for derived products and display, not raw (2026-11-27, policy `2026-11-27.1`) | `docs/DATA_PUBLICATION_AUDIT.md` §7 and §8; `docs/permissions/` does not exist |
| Licence and citation | **None** | No `LICENSE`, no `CITATION.cff` |
| Prospective scoring protocol | **In effect:** v1.0, merged 2026-11-24 09:37 UTC (`e79c54b`), before the first target date | `docs/prospective/PROTOCOL.md`; its hashes are verified by `src/prospective.py` |
| Official weekly issuance | None yet. The command `src/official.py` is merged (PR #12; [ISSUING.md](prospective/ISSUING.md)). Issuing needs the owner's GitHub records repository and token, set in the issuing machine's shell | `docs/prospective/official_ledger.jsonl` (empty) |
| Tests on `main` | With the private data: 201 passed and 5 skipped in the app environment (the PyTorch tests skip), 2026-11-27 after PR #16. The v4 environment was not re-run after PRs #14–#16 | `pytest` |

## 3. History

| PR | Branch | Merge commit | Merged (UTC) | Content |
|---|---|---|---|---|
| #1 | `engineering` | `7965d3d` | 2026-11-21 21:41 | Verified CDEC and NASA POWER pipeline, leakage-safe 30-day model, UI contract (v1) |
| #2 | `review-v2` | `abff902` | 2026-11-21 21:42 | v2: 2001–2025 data, snowpack, rolling water-year evaluation, preregistered held-out run |
| #3 | `ui-integration` | `d72cd68` | 2026-11-21 22:10 | Streamlit UI on the frozen v2 results |
| #4 | `feature/30-day-forecast` | `32d6122` | 2026-11-22 07:19 | 30-day research forecast: frozen v2 Model A, live CDEC, Forecast page |
| #5 | `feature/v3-forecast-platform` | `f972621` | 2026-11-22 07:20 | v3: Models B, B2 and C, preregistered 2026 comparison, Generate Forecast, data rights audit |
| #6 | `feature/v3-public-launch` | `0063cc0` | 2026-11-22 07:35 | Public launch: CDEC publication on owner authorization, full public release, model bundle, deployment config |
| #7 | `docs/project-handoff` | `0d524a5` | 2026-11-24 06:32 | `CLAUDE.md` and this handoff document |
| #10 | `protocol/prospective-scoring` | `e79c54b` | 2026-11-24 09:37 | Prospective evaluation and scoring protocol v1.0 and reference scorer (no scheduler) |
| #11 | `docs/consistency` | `57cb7d7` | 2026-11-24 09:54 | Docs consistency: publication, bundle, deployment and protocol status |
| #8 | `fix/forecast-reliability` | `50c83c6` | 2026-11-24 09:55 | Install retries; NASA POWER never delays Model A; write-once records; GitHub record store |
| #12 | `feature/official-issuance` | `e3e3466` | 2026-11-24 10:03 | Official weekly issuance command (manual, no scheduler) and operator guide |
| #9 | `research/v4-neuralhydro` | `1accaa6` | 2026-11-24 10:05 | v4 NeuralHydro research: daily neural pipeline, shared LSTM/GRU/TCN, initial development results |
| #13 | `docs/status-refresh` | `0b50bc0` | 2026-11-24 10:14 | Docs: status after PRs #12 and #9 |
| #14 | `research/v4-n4` | `91d8242` | 2026-11-27 22:47 | v4 N4: water-balance-structured encoder-decoder, development results |
| #15 | `research/v5-benchmark` | `cb62c75` | 2026-11-27 22:54 | v5: HGB v5 and the standardised benchmark of every implemented model (research) |
| #16 | `web/hgb-main-playground` | `a29b876` | 2026-11-27 23:00 | Website: frozen v2 HGB as the only forecasting model, Model playground, straight-line guide on the forecast charts |
| #17 | `docs/status-after-14-15-16` | `462e4b3` | 2026-11-27 23:14 | Docs: status after PRs #14, #15 and #16 |
| #18 | `web/playground-replays` | `cd4e278` | 2026-11-27 23:37 | Playground replays published (owner-reported CDEC flow permission, policy `2026-11-27.1`); README links to the live site; public-site check recorded |
| #19 | `fix/playground-cache-refresh` | `da1101d` | 2026-11-27 23:47 | Playground caches keyed on file stamps, so deployed files are not hidden by stale cached results |

Tags and branches:
- **Tags:** `v1-results` (`01e43cb`) and `models-2026-11-22`. The `submission-v2` tag proposed in `docs/merge_plan.md` was never created.
- **Old branches:** all merged; they remain on the remote. Local `review-v2` has 7 commits not pushed to its remote. This is harmless; do not push it.

## 4. Repository architecture

**Historical pipeline (v1/v2)**
- `src/fetch.py`, `fetch_snow.py` and `fetch_wsihist.py` download raw data into `data/raw/` and record every download in `data/raw_manifest.jsonl`.
- `src/build_dataset.py` builds the joined daily data.
- `src/features.py` builds the weekly Monday rows and the 30-day target.
- `src/experiments.py` runs the experiments, and `src/train_v2.py` runs the guarded final evaluation. `src/models.py` and `src/evaluation.py` supply the models, baselines, bootstrap and intervals.
- Outputs: `results/` and `outputs/`. `src/ui_export.py` writes the UI handoff package, and `src/contract.py` checks the UI contract files.

**Live Model A**
- `src/forecast_model.py` rebuilds the frozen v2 model into `models/v2_forecast/`. It refuses unless the rebuilt model reproduces all 593 archived predictions.
- `load_artifact()` checks the model card, the artifact hash, the frozen hashes and the feature order.
- `src/forecast_live.py` refreshes CDEC data and writes a write-once snapshot (`forecasts/latest.json` and `forecasts/records/`).

**v3**
- Data: `src/v3_basins.py` (upstream boxes), `v3_power.py` (NASA POWER), `v3_gefs.py` (GEFSv12 reforecast and operational, by byte range, with publication times) and `v3_dataset.py` (the weekly v3 table).
- `src/v3_experiment.py` runs the development folds and the guarded `--final` run, which writes `models/v3/` and `results/v3/final/`.
- `src/v3_release.py` builds the write-once public data releases, filtered by `config/data_release_policy.json`. Its `--check` is run in CI.
- `src/v3_artifacts.py` packs and installs the hash-pinned model bundle and runs the self-check.
- `src/v3_service.py` is the live forecast service (section 8).

**UI (owned by the UI owner)**
- `app.py` and `app/`:
  - `view_overview`, `view_explorer`, `view_backtest` and `view_research` are the historical pages;
  - `view_forecast` shows the issued Model A record;
  - `view_forecast_v3` is Generate Forecast, which since PR #16 runs only the frozen v2 HGB;
  - `view_playground` is the Model playground (retrospective comparison of the other nine study models).
- When `outputs/` is absent, the backtest pages read the newest `public_data/*/ui_outputs`.

**Tests and CI**
- `tests/` contains 206 tests, all offline (5 skip without PyTorch or the private data).
- `.github/workflows/ci.yml` runs on Python 3.14 with `requirements-lock.txt`:
  - pytest;
  - the contract check on `app/sample_data`;
  - the frozen v2 hash check;
  - a check that no private paths are tracked;
  - `v3_release --check`;
  - a reproducibility check on the forecast fixture.

**Dependencies.**
- `requirements.txt` pins `scikit-learn==1.9.1` and `joblib==1.6.0`, plus `eccodes` and `eccodeslib`. It does not include `boto3`.
- `requirements-lock.txt` pins everything (numpy 2.5.5, streamlit 1.67.0).
- The `Dockerfile` uses `python:3.14-slim`.

**v4 research programme (NeuralHydro; merged as PR #9 on 2026-11-24; not deployed)**
- Separate deep-learning research in `src/v4/`, `tests/test_v4.py`, `docs/v4/`, `results/v4/` and `requirements-v4.txt`, in its own `.venv-v4` with PyTorch.
- Development data stops before 2024-01-01. It touches nothing frozen and is not deployed.
- See `docs/v4/README.md`. N4 (water-balance-structured encoder-decoder) was merged as PR #14 (`docs/v4/RESULTS_N4.md`).

**v5 benchmark (merged as PR #15 on 2026-11-27; research only)**
- `src/v5/`, `tests/test_v5.py`, `docs/v5/`, `results/v5/`: HGB v5 and the retraining of every implemented model on one dataset, with frozen and equal-budget tuning, paired block bootstrap and Holm correction.
- Plan fixed before any result (`docs/v5/BENCHMARK_PLAN.md`, hashed in every manifest). Results on 2024–2025 and 2026 are post-hoc.
- `src/v5/playground.py` serves the Model playground from the tracked aggregates.

## 5. Models

| Model | Inputs | Training targets | Train rows | On the website? | Needs at forecast time |
|---|---|---|---|---|---|
| **A (frozen v2)** `models/v2_forecast/` | 15 features: storage ×4, SWE ×3, season ×2, station one-hot ×6 | before 2024-01-01 | (v2 training set) | **Yes**, shown as "Model A" | CDEC storage and snow |
| A (v3 refit) `models/v3/model_A.joblib` | same as frozen A | before 2026-01-01 | 7,693 | **No.** Bundled and self-checked only. | — |
| B | A + dam-point NASA POWER weather (T2M, PRECTOTCORR, RH2M), lagged 7 days | before 2026-01-01 | 7,693 | Yes | CDEC + POWER |
| B2 | A + mean POWER weather over an upstream box (4–12 grid cells, rule fixed in advance) | before 2026-01-01 | 7,693 | Yes | CDEC + POWER |
| C | A + GEFSv12 control member (00Z on the origin Monday), upstream-box precipitation for days 1–7 and 8–16 | before 2026-01-01 | 7,436 | Yes | CDEC + a GEFS cycle published before the issue time |

**How Model C's training data was assembled**
- 2001–2019: reforecasts, never issued in real time.
- 2020-01-01 to 2020-09-23: no data.
- From 2020-09-24: operational GEFS.
- Excluded: 4 backfilled operational cycles. One cycle (2007-06-04) is malformed.
- Days 17–30 are never filled.

See `docs/v3/MODEL_C_FEASIBILITY.md`.

**Intervals.** `sym_season` offsets per model, by season, from each model's own development out-of-fold residuals (stored in `models/v3/model_cards.json`).

## 6. Data and publication

**Release policy.** `config/data_release_policy.json`, version `2026-11-22.2`.

| Source | Status in policy | Basis |
|---|---|---|
| CDEC storage (sensor 15) and snow (sensor 82) | raw, derived and public display all `true` | **Owner-reported authorization** (audit §7): "DWR CDEC and everyone involved has given me permission go public." No agent has seen the correspondence. |
| NASA POWER | confirmed | Published terms. POWER *requests* a notification email when a dataset is released; whether it was sent is unknown. |
| NOAA GEFSv12, operational and reforecast | confirmed | NODD terms. Attribution requested, no implied endorsement, derived values must not be called unaltered NOAA data. |

Attribution strings are in the policy file. Full audit: `docs/DATA_PUBLICATION_AUDIT.md`.

**Public releases (write-once)**
- `public_data/watershedshift-public-data-2026-11-22/`: 6 files, NOAA and NASA only.
- `public_data/watershedshift-public-data-2026-11-22.2/`: 27 files.
  - CDEC daily storage, snow pillows and basin snow index;
  - the v3 weekly features, GEFS and POWER tables;
  - audits and metadata;
  - research predictions (v3 development out-of-fold and 2026 test);
  - both issued forecast records;
  - `ui_outputs/`;
  - `MANIFEST.json` (with a citation string) and `SHA256SUMS`.

**Gaps**
- No licence for the code or the data.
- No `CITATION.cff`.
- The permission correspondence is not archived (`docs/permissions/`).
- The raw CDEC responses (about 29 MB) are permitted but not attached to any release.

## 7. Model bundle and release

| | |
|---|---|
| Release | `models-2026-11-22`, a **pre-release** |
| Asset | `watershedshift_models_7ad581d9f585.tar.gz`, 1,240,630 bytes |
| SHA-256 | `7ad581d9f58594050688ec33cb810ff5ff513f2df3e233cc8fc1160eb193b97d` |
| URL | `https://github.com/haaris-usman-saeed/watershed/releases/download/models-2026-11-22/watershedshift_models_7ad581d9f585.tar.gz` (public, no sign-in) |
| Contents | `models/v2_forecast/{model.joblib,model_card.json}`, `models/v3/model_{A,B,B2,C}.joblib`, `models/v3/model_cards.json`, `MANIFEST.json` |
| Built with | scikit-learn 1.9.1, numpy 2.5.5, joblib 1.6.0 |

**What `MANIFEST.json` records:** each file's SHA-256, self-check reference predictions (16 synthetic rows per model, tolerance 1e-9) and the library versions above.

**When `install` refuses:**
- the hash differs from the pinned value;
- there is an extra, linked or path-traversing member;
- a file differs from the manifest;
- the release policy blocks public display.

**Other facts**
- The bundle is byte-reproducible from `models/` with `python -m src.v3_artifacts pack`.
- The local `dist/` copy is identical to the release asset.
- The release notes contain the three deployment values (URL, SHA-256, `public`). None of them is secret.
- **To change models:** publish under a new tag and update both secrets together. Never replace the asset.

## 8. The live forecast service

**Startup**
- `app/view_forecast_v3.py` calls `ensure_artifacts()` once per process, inside `st.cache_resource`. If the models are missing and both `WATERSHED_ARTIFACT_URL` and `WATERSHED_ARTIFACT_SHA256` are set, it downloads and installs the bundle.
- Each model must then pass the self-check.

**Each Generate Forecast request** (`src/v3_service.py`, `ForecastService.forecast`)

Since PR #16 the website creates the service with `model_keys=("A",)`: only step 1 runs, and NASA POWER and GEFS are never fetched. The default (all four models, all three steps) is unchanged and is what `src/official.py` uses.

1. **CDEC** storage and snow, cached in 6-hour buckets.
   - If this fails, or Model A cannot load, **no forecast** is produced.
2. **NASA POWER**, cached in 12-hour buckets, fetched in the background alongside CDEC.
   - If it fails, or takes longer than 45 s (`WATERSHED_POWER_WAIT_S`), **B and B2 are unavailable** for that forecast. A and C proceed. A slow download continues for later forecasts; after a failure, new attempts wait 5 minutes.
3. **GEFS:** the 00Z cycle of the origin Monday, cached per cycle.
   - If that cycle is missing or incomplete, or was not published before the issue time, **C is unavailable**.

**Shared mechanics**
- One disk cache (`WATERSHED_CACHE_DIR`, default `data/v3/cache`) with a SHA-256 for every entry, and request coalescing.
- A token bucket of 6 upstream jobs per minute.
- A 10-minute cooldown on forced refreshes.
- In public mode, the policy gate runs before any fetch or display.

**Records**
- Each forecast is saved once, write-once, by `RecordStore`. Record IDs end with 8 hex characters of the record's SHA-256, and the page states where the record went.
- Locally: `WATERSHED_RECORD_DIR` (default `forecasts/records_v3`), mode 0444.
- In a GitHub repository when `WATERSHED_RECORD_GITHUB_REPO` and `WATERSHED_RECORD_GITHUB_TOKEN` are set. This is the backend the owner chose on 2026-11-24. Records are created through the contents API without a blob SHA, so GitHub refuses to overwrite them, and each record is one commit. Setup: `docs/v3/DEPLOYMENT.md`.
- In S3 when `WATERSHED_RECORD_BUCKET` is set, with a conditional write (`If-None-Match: *`). This needs `boto3`, which is not in the requirements: current `botocore` requires `python-dateutil<3`, but the lock pins 4.0.0.
- **On Streamlit Community Cloud the disk is ephemeral.** Without the GitHub (or bucket) backend configured, records disappear on restart.

## 9. Deployment

**Status: checked on the public URL, 2026-11-27, about 23:30 UTC.**
- URL: https://watershed-lwe3carmkq2mgurch5jurm.streamlit.app (Streamlit Community Cloud, deployment mode `public`; the README links to it).
- Generate Forecast (frozen v2 HGB only) gave origin 2026-11-23 → target 2026-12-23, CDEC as the only source, for all six reservoirs:
  - SHA: 50.0 → 53.7%;
  - FOL: 38.3 → 44.2%, with the Folsom interval caution shown;
  - NML: 63.4 → 65.3%;
  - ORO: 42.3 → 45.4%;
  - EXC: 49.0 → 51.4%;
  - DNP: 72.8 → 74.7%.
- A repeat press showed "(cached)". On a 375 px phone width the cards stack.
- The Model playground renders its tables and charts.
- **Replays, checked after PRs #18 and #19 were deployed (about 23:50 UTC):** "Replay a past 30-day forecast" works in the confirmation, 2024–2025 and 2026 windows, with all ten models and none withheld.
- Right after PR #18, the confirmation window still showed a "not available" result that had been cached before the deploy. PR #19 keys the playground caches on file stamps. The redeploy cleared it without a reboot.
- Not checked: the app's logs and settings, and whether forecast records persist (the page reports where each record went).
- The only end-to-end run was the previous engineer's local, data-free, public-mode deployment with the bundle served from localhost (STATUS.md, 2026-11-22). That run produced forecasts with all four models and rendered all five pages.

**Intended settings (Streamlit Community Cloud)**
- Repository and file: `haaris-usman-saeed/watershed`, branch `main`, file `app.py`.
- Python: 3.12 or 3.13 (Advanced settings).
- Secrets: `WATERSHED_DEPLOYMENT="public"`, `WATERSHED_ARTIFACT_URL` and `WATERSHED_ARTIFACT_SHA256` (section 7). Template: `.streamlit/secrets.toml.example`. Full guide: `docs/v3/DEPLOYMENT.md`.

**Acceptance check before calling the site live:** `docs/v3/LAUNCH_AND_MERGE.md` §4 (updated for PR #16).
- The frozen v2 HGB forecast is available for each reservoir, with CDEC as the only source.
- The Model playground renders.
- The origin is a Monday and the target is 30 days later.
- A second press shows "(cached)".
- The layout works on a phone.

**Known deployment risks**
- **Python version:** every verified run used Python 3.14.5, so the 3.12/3.13 path has never run. PyPI has cp312 and cp313 wheels for all compiled pins. The self-check would refuse any model whose predictions change.
- **Install failure persists:** a failed install is cached until reboot, and the page shows a misleading message (bug 1).
- **Slow POWER:** a hanging POWER service can delay every forecast (bug 2).
- **Records:** they do not persist without a bucket.

## 10. Issued forecasts and prospective evaluation

| ID | Issued (UTC) | Models | Origin → target | SHA-256 |
|---|---|---|---|---|
| `wsf-v2-2026-11-22-20261122T052508Z` | 2026-11-22 05:25:08 | A (frozen v2), all six reservoirs | DNP 11-09 → **12-09**; others 11-16 → **12-16** | `81433f59…b98e` |
| `wsf-v3-ORO-20261122T064412Z` | 2026-11-22 06:44:12 | A (frozen v2), B, B2, C | ORO 11-16 → **12-16** | `5fe960ad…4fbe` |

**Where the copies are**
- Local: `forecasts/records/` and `forecasts/records_v3/` (mode 0444).
- Public: `public_data/watershedshift-public-data-2026-11-22.2/issued_forecasts/`.
- Full hashes: `docs/v3/issued_forecast_ledger.jsonl`.

**Stale ledger note.** The ledger note for the v2 record still says "private", but the record is published.
- It is deliberately left unchanged. Appending a correction line would add a second entry for the same `forecast_id`, which the scorer and the tests would read as a duplicate record.
- The correct fact is recorded here and in `README.md`.

**Protocol v1.0 is in effect.** It was merged on 2026-11-24 (`e79c54b`), before the first target date, so it is genuinely pre-specified. The files are `docs/prospective/PROTOCOL.md`, `config/prospective_protocol.json` and the reference scorer `src/prospective.py`. The persistent store (GitHub records repository) was merged with PR #8 and becomes active once the owner configures it. **There is still no scheduler.** Official records are issued manually with `src/official.py` ([ISSUING.md](prospective/ISSUING.md)). The protocol defines:
- official weekly issuance, with no scheduler yet;
- provisional scores from target + 2 days and settled scores from target + 30 days;
- a fixed seasonal baseline;
- P1 (Model A vs seasonal) and P2 (90% coverage) on the first 52 official weeks, with S1–S3 secondary;
- the two pre-protocol records, reported separately.

## 11. Known bugs and weaknesses

Found in the 2026-11-23 review. Line numbers drift, so functions are named instead. Items marked **Fixed** were fixed by PR #8 (merged 2026-11-24).

1. **Fixed.** **`app/view_forecast_v3.py` `service()`** (UI-owned):
   - `ensure_artifacts()` runs inside `st.cache_resource`, so a transient download failure leaves the process without models until a reboot.
   - Its status string is discarded, so visitors see "run `python -m src.forecast_model --build`" instead of the real reason.
2. **Fixed.** **`src/v3_service.py` `fetch_power`:** 47 jobs, each with 3 tries, a 90 s timeout and backoff, run 8 at a time. If POWER *hangs* rather than fails, the whole forecast, including Model A, can stall for many minutes. Fast failures are handled and tested.
3. **Shared rate limit:** CDEC, POWER and GEFS share one token bucket. An incomplete GEFS cycle is refetched on every request, which can use up the budget and rate-limit a later CDEC refresh, blocking every model.
4. **Fixed.** **`src/v3_artifacts.py` `install`:** it reads `blobs["MANIFEST.json"]` *before* checking that the manifest is present.
   - An archive without a manifest raises `KeyError` instead of "REFUSED".
   - `ensure_artifacts()` catches it; the CLI does not.
5. **`RecordStore`:**
   - **Fixed:** S3 is not truly write-once: any `head_object` error leads to a plain `put_object`.
   - **Fixed in code; needs owner setup:** persistent storage. The owner chose a GitHub records repository (2026-11-24), and the backend is implemented. It needs the repository and a fine-grained token in the Streamlit secrets. The S3 path still needs `boto3`, which conflicts with the locked `python-dateutil==4.0.0`.
   - **Fixed:** Record IDs have one-second resolution, so two forecasts for the same reservoir in the same second lose one.
   - **Open (prospective protocol):** Visitor and official forecasts share one record type and one store.
6. **`fetch_cdec`** writes raw responses to `data/v3/cache/cdec_raw/` whatever `WATERSHED_CACHE_DIR` is set to, and never prunes them.
7. **GEFS publication times:** failed HEAD requests are ignored, so a recorded publication time can be earlier than the true one.
8. **Fixed.** **Rerun saves:** every Streamlit rerun calls `record_store().save(r)`. The duplicate save is silently ignored.
9. **Stale UI text** (UI-owned; needs the owner's approval):
   - `app/view_forecast.py` still says CDEC terms are unresolved.
   - The Folsom caution cites only the 2024–25 coverage (about 75%), not 2026 (0.69–0.74).

## 12. Documents that contradict the current state

The docs-consistency PR (2026-11-24) fixed the stale publication and deployment statements:
- in `README.md`, which also gains a section separating what is frozen from what is new, and a section on prospective evaluation;
- in `ASSESSORS.md`, `data/provenance.md`, `docs/UI_CONTRACT.md` (wording only), `docs/FORECAST_IMPLEMENTATION.md`, `docs/v3/V3_REPORT.md`, `docs/v3/LAUNCH_AND_MERGE.md`, `docs/v3/DEPLOYMENT.md` (Python versions actually verified), `docs/DATA_PUBLICATION_AUDIT.md` (status notes) and `docs/merge_plan.md` (marked historical);
- in a `.gitignore` comment.

**Still stale, and deliberately not changed there:**

| File | Stale statement | Why it stays |
|---|---|---|
| `app/view_forecast.py` | CDEC terms "unresolved" (two UI strings) | UI-owned: needs the owner's approval (bug 9) |
| `app/README_FRONTEND.md` | Real outputs not to be committed "while reuse conditions are unresolved" | UI-owned |
| `docs/UI_CONTRACT.md` §4 table | "forecasts beyond the data end date" not supported | A dated note now explains the owner-approved research forecast page. Changing the rule needs the UI owner |
| `src/make_bundle.py` | Bundle README text: raw responses excluded because terms were "not confirmed" | Code that generates bundle text; change it in a functional PR |
| `docs/v3/issued_forecast_ledger.jsonl` | v2 record "is private" | Append-only; see section 10 |
| `STATUS.md` (old entries) | Blockers and "not uploaded" | Append-only; a new entry records the current state |
| `docs/prospective/PROTOCOL.md` | "takes effect when merged" | Hash-registered. Still accurate as the record of when it took effect |

## 13. Priorities

1. **Verify the deployment:** done 2026-11-27 (§9), including the playground replays after PRs #18 and #19. Re-check after any deploy that changes the pages. Fix only deployment problems; never retrain to fix them.
2. **Documentation consistency:** done by the docs-consistency PR (section 12). The UI strings remain, pending the owner.
3. **Make forecasting resilient:** bugs 1, 2 and 4 are fixed (PR #8). Still open: bug 3 (shared rate limit), bugs 6 and 7, and a documented table of which models are available for each failure.
4. **Prospective evaluation:**
   - the protocol is in effect (section 10);
   - the official issuing command is merged (PR #12). Issuing starts once the owner configures the records repository and token;
   - provisional scores of the two November records are due from 2026-12-11 (DNP) and 2026-12-18 (the others).
5. **Publication:**
   - licence;
   - `CITATION.cff`;
   - research report updates;
   - every new analysis labelled preregistered or post-hoc.

## 14. Owner-only actions and missing items

- Access to the Streamlit app's logs or settings, or screenshots of them. The URL is recorded in section 9.
- The written CDEC/DWR/operator permission, archived under `docs/permissions/` (sender, date, scope, conditions), covering storage and snow (§7) and inflow and outflow (§8). If it has conditions, update the policy and attribution text to match.
- Licence choice for the code and the data, and the author list for `CITATION.cff`.
- Persistent forecast records: create the GitHub records repository, protect its branch, and add a fine-grained token limited to it to the Streamlit secrets (`docs/v3/DEPLOYMENT.md`, "Persistent records in a GitHub repository").
- Whether the bundle release should stay a pre-release.
- Whether the NASA POWER notification email has been sent.
- Approval for any change under `app.py`, `app/` or `.streamlit/`.
- Whether to adopt the v5 draft prospective protocol (`docs/v5/PROSPECTIVE_PROTOCOL_DRAFT.md`) and which candidates (for example HGB v5 and N4) to freeze for prospective scoring.
- A backup of the trusted build machine (section 15).

## 15. Assets that exist only on the trusted build machine

These paths are git-ignored and exist only on the owner's Mac (`~/watershed`):

| Path | Size | Recoverable elsewhere? |
|---|---|---|
| `data/raw/` and `data/raw_manifest.jsonl` (the manifest is tracked) | about 317 MB, 191 files | The data can be downloaded again, but CDEC revises values, so **the exact bytes cannot**. The manifest holds their hashes. |
| `data/processed/`, `data/live/` | — | Rebuildable from `data/raw/` |
| `data/v3/` (GEFS, POWER, private weekly table, caches) | about 30 MB | Mostly re-downloadable; the cleaned tables are in the public release |
| `models/` (7 files) | — | Yes: the release bundle contains all 7 |
| `dist/` (bundles) | — | Yes: the release asset |
| `forecasts/` (issued records, `latest.json`) | — | Yes: both records are in `public_data/…-2026-11-22.2/issued_forecasts/` and hash-verifiable |
| `outputs/`, `results/v1/local_full/` | — | Partly: `ui_outputs/` is in the public release |
| `.venv/` (Python 3.14.5) | — | Rebuild from `requirements-lock.txt` |

**Recommendation:** keep an encrypted off-machine backup of `data/raw/` and `data/v3/`. The project needs no API keys for CDEC, POWER or GEFS. `gh` on this machine is signed in to the owner's account.

## 16. How to re-verify this document

Run these from the repository root with `.venv/bin/python` or an environment built from `requirements-lock.txt`. All are read-only.

```bash
git fetch origin && git rev-parse origin/main && gh pr list --state open
```

```bash
python - <<'PY'
import json
from src.train_v2 import frozen_hashes
rec = json.load(open("results/v2/final/RUN_RECORD.json"))["frozen_hashes"]
assert frozen_hashes() == rec, "v2 frozen hashes differ"
print("v2 frozen hashes OK")
PY
```

```bash
python - <<'PY'
import hashlib, json
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
plan = json.load(open("config/v3_plan.json"))
bad = [f for f, h in plan["code_sha256"].items() if sha(f) != h]
rec = json.load(open("results/v3/final/RUN_RECORD.json"))
assert not bad, bad
assert rec["plan_sha256"] == sha("config/v3_plan.json"), "plan hash differs from the v3 run record"
print("v3 plan and code hashes OK")
PY
```

```bash
python - <<'PY'
import glob, hashlib, json
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for line in open("docs/v3/issued_forecast_ledger.jsonl"):
    e = json.loads(line)
    paths = sorted(glob.glob(f"forecasts/**/{e['forecast_id']}.json", recursive=True)
                   + glob.glob(f"public_data/*/issued_forecasts/{e['forecast_id']}.json"))
    print(e["forecast_id"], "OK" if paths and all(sha(p) == e["sha256"] for p in paths) else "MISMATCH OR MISSING", paths)
PY
```

```bash
python -m src.v3_release --check && python -m pytest -q tests
```

```bash
gh release view models-2026-11-22 --json isPrerelease,assets --jq '.isPrerelease, (.assets[] | "\(.name) \(.size) \(.digest)")'
```

## 17. Glossary

- **Origin:** the Monday a forecast is made from.
- **Target:** origin + 30 days.
- **Capacity in force:** the reservoir's official capacity on the origin date. Oroville's changed on 2024-07-02.
- **Smoothed seasonal (DOY) baseline:** for each reservoir, the mean 30-day change in the training data, weighted by a Gaussian kernel (σ = 15 days) on circular day-of-year distance (`src/models.py`). It is the main baseline.
- **Reforecast:** a forecast produced retrospectively with today's model for a past date. It was never issued at the time.
- **Issued forecast:** a forecast made before its outcome existed, saved write-once and hash-logged in the ledger.
- **Official forecast vs visitor forecast:** an official forecast is issued under the project's protocol. A visitor forecast is generated on demand by a website visitor. They are currently not separated (bug 5).
- **Preregistered vs post-hoc:** preregistered analyses are specified and committed before the outcomes are seen. Everything else is post-hoc and must be labelled so.

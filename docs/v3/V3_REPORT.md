# v3 report: what works, what is unvalidated, what is blocked (2026-11-22)

## Works (verified)

| Deliverable | Evidence |
|---|---|
| Model A kept unchanged, and the 2026-11-22 issued forecast kept immutable | Frozen hashes equal `RUN_RECORD.json`. The record is read-only, and its SHA-256 is in `issued_forecast_ledger.jsonl` (tested). |
| Comparison plan for A, B, B2 and C | `PREREGISTRATION_V3.md` and `config/v3_plan.json`, committed (`1888c1d`) before any v3 model was fitted. The final run checked every code hash. |
| Model B pipeline | Reproduces the v2 HGB + SWE result exactly. Weather lags are tested. Weather-window features match the historical builder. |
| Model B2 experiment | Upstream-box NASA POWER means: 4 to 12 grid cells per reservoir, with the rule fixed in advance. |
| Model C pipeline with provenance | 1,313 GEFSv12 Monday cycles fetched by byte range: 64 messages each, SHA-256 per cycle. Each is labelled reforecast or operational, with S3 publication times. One cycle is malformed (2007-06-04), and 4 backfilled operational cycles are excluded. Days 17-30 are never filled. |
| Scientific comparison | Development folds WY2008-2025, plus one preregistered 2026 test (251 rows): paired MAE, results by station and season, block-bootstrap CIs, availability, and model-specific interval coverage. |
| Model-specific 90% intervals | `sym_season` offsets computed from each model's own out-of-fold residuals. 2026 coverage: A 0.94, B 0.94, B2 0.93, C 0.94. Folsom 0.69-0.74. |
| Interactive forecasting | Generate Forecast was run against live data in the local app for Oroville on 2026-11-22, with all four models available. The record was saved write-once. |
| Shared caching, coalescing, rate limits, failure handling | Tested offline: cache hit, bucket invalidation, 20 concurrent requests causing 1 fetch, token bucket, forced-refresh cooldown, API failure, corrupted cache entry, missing CDEC values, missing snow, stale data, missing or incomplete GEFS cycle, and GEFS not yet published. |
| Publication rights audit and public release | `DATA_PUBLICATION_AUDIT.md`, the release policy, and the write-once release `public_data/watershedshift-public-data-2026-11-22/`, which contains NOAA and NASA-derived data only. |
| Deployment guide | `DEPLOYMENT.md` and the `Dockerfile` |

## Results, stated carefully

- **Verdicts:** H1 (C beats A), H2 (B beats A) and H3 (B2 beats B) are all **not established** on 2026. Every 95% CI includes zero.
- **Development folds (descriptive, not a test):** on the same rows, MAE was 2.72 pp for C against 2.83 for A. B and A were equal, and B2 was slightly higher than B. These folds include 2024-2025, which were seen earlier, and they have no uncertainty interval.
- **What this means:** forecast precipitation is the only added input whose point estimates favour it in both periods, but its value is **not established**. No model is declared the winner, and the website says so.
- **All models vs the seasonal baseline:** on 2026, every model's MAE was 0.53-0.57 pp lower than the smoothed seasonal baseline's. With 42 weekly origins the CIs include zero, so this is not established on 2026 alone. The advantage of the v2 model over that baseline was established on 2024-2025, where every CI was below zero.

## Unvalidated

- **Model C in real time:** live C forecasts use operational GEFS, while about 80% of C's training rows used reforecasts. Its future accuracy is not established.
- **Prospective accuracy:** no forecast has yet been scored against later observations. The first targets fall on 2026-12-09 and 2026-12-16.
- **The upstream boxes** are rectangles, not delineated watersheds.
- **Recent snow values** (sensor 82) and NASA POWER near-real-time values may be revised later.

## Publication and deployment status (updated 2026-11-22, after merging PR #4 and PR #5)

- **CDEC publication:** authorized by the owner on 2026-11-22 (`docs/DATA_PUBLICATION_AUDIT.md`, section 7). The written permission still has to be archived by the owner.
- **Public data:** `public_data/watershedshift-public-data-2026-11-22.2` includes the CDEC-derived data.
- **Model bundle:** built, byte-reproducible and self-checking. It is published as GitHub Release `models-2026-11-22` (pre-release; checked 2026-11-23: its digest equals the pinned SHA-256).
- **The public website is not verified as forecasting.** No public URL was available to the engineering agent, so it is not claimed to be live. A clean, data-free deployment run locally in public mode did install the bundle from a URL and generate real forecasts with all four models (see STATUS.md).

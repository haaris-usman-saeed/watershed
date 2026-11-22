# Deploying the interactive forecast (v3)

## What the app does on "Generate Forecast"

1. **Checks the request:** the reservoir must be one of the six approved IDs, and the release policy must allow the deployment to display the sources (`config/data_release_policy.json`, `WATERSHED_DEPLOYMENT`).
2. **Gets shared source data** for all reservoirs, from the cache when possible:
   - CDEC storage and snow: 6-hour buckets, 14 HTTP requests per refresh;
   - NASA POWER: 12-hour buckets, 47 small requests;
   - the GEFS 00Z cycle of the origin Monday: fetched once per cycle, about 20 MB in byte-range requests, keeping only basin totals.
3. **Builds the inputs** with the frozen v2 rules and runs Models A, B, B2 and C from verified, hash-checked artifacts. Models are never refit per request.
4. **Returns a structured record** with per-model availability, timestamps, intervals and provenance, and saves it write-once to the record store.

**Measured locally on 2026-11-22:** a cold start for Folsom, fetching all three sources, took 29 s. Cached requests are immediate.

## Hosting assessment

Current hosting: the repository has only `.streamlit/config.toml` (theme). The app is presumably deployed on Streamlit Community Cloud from GitHub; the URL is not recorded in the repo.

| Requirement | Streamlit Community Cloud | Private host (Docker on a VM or workstation) |
|---|---|---|
| Outbound requests to CDEC, NASA POWER, AWS (NOAA) | Allowed | Allowed |
| GRIB2 decoding (`eccodes` + `eccodeslib` wheels) | Linux wheels exist; Python 3.14 availability on Cloud must be checked at deploy time | Works (`Dockerfile`) |
| Model artifacts | **Not available:** they are git-ignored, and Cloud deploys from the repository | Mount `models/` read-only |
| Shared cache | In-process plus local disk; the disk is ephemeral, so the cache rebuilds after a restart (acceptable) | Persistent volume `/data/cache` |
| Concurrent visitors | One small shared container: coalescing and the upstream rate limit stop duplicate fetches | Size the VM as needed |
| Request time | First request about 30 s, then cached | Same; add a scheduled `python -m src.v3_service --warm` |
| Persisted forecast records | Local disk is not persistent. Needs `WATERSHED_RECORD_BUCKET` (S3-compatible) with credentials in Streamlit secrets | Volume `/data/records` (write-once) |
| Secrets | `st.secrets` / environment variables | Environment variables |
| **Data rights** | Public. **Blocked:** CDEC public display not cleared | Private: allowed for authorised users |

**Recommendation.** Today the only correct setup is a **private deployment**: Docker on a machine the owner controls, with access limited to authorised users.

The public site will not generate forecasts until both of the following are done:
1. DWR confirms CDEC redistribution and display terms, and the owner sets `public_display: true` in `config/data_release_policy.json` (a reviewed commit).
2. Model artifacts are published, or fetched at startup from private storage with hash verification. That fetch is **not implemented**; it is the next engineering step after clearance.

Until then, a deployment with `WATERSHED_DEPLOYMENT=public` shows an explanation in place of the button result. It never displays CDEC values.

## Private deployment (works now)

```bash
python -m src.forecast_model --build          # Model A artifact (needs the private processed data)
python -m src.v3_experiment --final           # Models B, B2, C (run once, preregistered)
docker build -t watershedshift .
docker run -p 8501:8501 -v "$PWD/models:/app/models:ro" -v "$PWD/forecasts:/app/forecasts:ro" \
  -v watershed-data:/data -e WATERSHED_DEPLOYMENT=private watershedshift
```

- **Optional warm-up**, for example from cron on the host every 6 hours: `docker exec <container> python -m src.v3_service --warm`.
- **Optional S3 record store:** set `WATERSHED_RECORD_BUCKET`, plus AWS credentials through the environment. The bucket should be private.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `WATERSHED_DEPLOYMENT` | `private` or `public`. `public` enforces the release policy before any fetch. | `private` |
| `WATERSHED_CACHE_DIR` | Shared source cache | `data/v3/cache` |
| `WATERSHED_RECORD_DIR` | Write-once local forecast records | `forecasts/records_v3` |
| `WATERSHED_RECORD_BUCKET` | S3-compatible bucket for records (needs `boto3`) | unset |
| `WATERSHED_DATA_DIR` | Backtest UI data (`outputs` or `app/sample_data`) | `outputs` |
| `WATERSHED_FORECAST_FILE` | Issued Model A snapshot shown in the "Issued record" tab | `forecasts/latest.json` |

## Safety properties (tested in `tests/test_v3.py`)

- **Inputs are validated.** Only the six approved reservoir IDs are accepted; no user-supplied URLs.
- **Fetches are bounded.** Every upstream request has a timeout and a retry limit; upstream fetch jobs are rate-limited globally; a forced refresh has a 10-minute cooldown per source bucket.
- **Duplicate fetches are prevented.** Concurrent identical requests are coalesced into one upstream fetch.
- **Failures never fall back.** A failed or incomplete source makes the affected model unavailable and the reason is shown; there is no silent fallback. A failed CDEC fetch yields no forecast at all.
- **Data integrity is checked.** Cache entries carry a SHA-256 hash, and a corrupted entry is discarded and refetched. Records are write-once.

# Deploying WatershedShift with public interactive forecasting

## What a visitor gets

A visitor opens the site, picks a reservoir on **30-Day Forecast** and presses **Generate Forecast**. The app then:

1. **Gets the inputs.** It takes the latest eligible observations from a shared, hash-checked cache, or downloads them:
   - CDEC storage and snow, 14 requests;
   - NASA POWER, 47 small requests;
   - the NOAA GEFS 00Z cycle of the origin Monday, about 20 MB in byte ranges, keeping only basin totals.
2. **Runs Models A, B, B2 and C** from verified artifacts.
3. **Shows the results:** each model's day-30 estimate, 90% range, dates, sources and freshness.

No Python, GitHub or terminal is needed.

**Measured locally on 2026-11-22**, cold start with an empty cache, Shasta, all four models: 21 s and 280 MB peak memory. Later requests within the same refresh window are served from cache.

## Architecture: one Streamlit app, no separate backend

A separate service is not needed at this scale.

- **Forecast service:** one in-process service, shared by all sessions (`st.cache_resource`), in `src/v3_service.py`.
- **Cache buckets:** CDEC 6 hours, POWER 12 hours, GEFS per cycle.
- **Coalescing:** concurrent identical requests share one download. With 20 concurrent requests, there is 1 upstream fetch (tested).
- **Upstream budget:** 6 fetch jobs per minute. Forced refreshes have a 10-minute cooldown.
- **Timeouts:** CDEC 60-90 s per request with retries; NASA POWER 30 s per request, 2 tries, and no new request after 180 s; GEFS file requests time out at 60 s.

**Failure handling:** a failed or stale source marks the affected model unavailable and gives the reason. A failed CDEC fetch means no forecast at all. Nothing is filled in.

**NASA POWER never holds up Model A:**
- POWER is downloaded in the background, at the same time as CDEC.
- A forecast waits for it at most 45 s (`WATERSHED_POWER_WAIT_S`). If it is slower, Models B and B2 are skipped for that forecast with the reason shown, while A and C proceed. The download continues and later forecasts use it from the cache.
- After a POWER failure, new attempts wait 5 minutes instead of every visitor retrying.

**Model artifacts:**
- They are installed on the first forecast request from a hash-pinned bundle (`src/v3_artifacts.py`).
- The download is retried twice on connection errors, timeouts, HTTP 429 and 5xx (waiting 2 s, then 4 s). Other HTTP errors are final.
- If installation fails, the forecast page shows the installer's own message (for example a wrong pin or an unreachable URL), and the server tries again at most once a minute. A restart is no longer needed after a transient failure.
- Installation checks the pinned SHA-256, the archive members (no traversal, no extra files) and the manifest.
- After installing, every model must reproduce reference predictions recorded at build time. A host with a different scikit-learn version fails this self-check and the model is refused, not run.

**Records and cache are not persistent on Streamlit Community Cloud:** its disk is reset on restart or sleep. The cache simply rebuilds. Visitors can always download their forecast as JSON.

**Forecast records:**
- Each forecast is saved once, write-once, with an ID ending in the first 8 hex characters of its SHA-256, so two forecasts issued in the same second cannot collide.
- Below each forecast the page says where the record went. Without a bucket it says "local disk only", which on Community Cloud means the record is lost at the next restart.
- **S3 backend** (`WATERSHED_RECORD_BUCKET`, optionally `WATERSHED_RECORD_ENDPOINT_URL` for an S3-compatible provider):
  - It writes with `If-None-Match: *`, so the bucket itself refuses an overwrite.
  - Only a "not found" answer counts as absent; any other error is raised rather than writing.
  - **It needs `boto3`, which is not installed.** Current `botocore` requires `python-dateutil<3`, while `requirements-lock.txt` pins `python-dateutil==4.0.0`, so adding it means changing a locked dependency. The persistent backend is an owner decision (`docs/PROJECT_HANDOFF.md`, section 14).

**Release policy:** `config/data_release_policy.json` (policy version `2026-11-22.2`, owner-authorized) permits public display of CDEC-derived values. If a source is ever withdrawn, set its flag to `false`. With `WATERSHED_DEPLOYMENT=public`, the gates in the service, the app, the issued-record tab and the artifact installer then refuse again (tested).

## Streamlit Community Cloud (the existing website)

**Requirements**

| Item | Setting or finding |
|---|---|
| Repository and branch | `haaris-usman-saeed/watershed`, `main`, main file `app.py` |
| Python | Choose 3.12 or 3.13 in Advanced settings. Pins in `requirements.txt` are set so `scikit-learn` equals the artifact build (1.9.1); the startup self-check confirms it. |
| Dependencies | `requirements.txt`, including `eccodes` and `eccodeslib`. The binary wheels bundle the ecCodes C library, so no system packages are needed. |
| Internet access | CDEC (cdec.water.ca.gov), NASA POWER (power.larc.nasa.gov) and NOAA on AWS (s3.amazonaws.com). Community Cloud allows outbound HTTPS. |
| CPU and memory | About 280 MB peak measured. Community Cloud gives a small shared container; check its current limits if traffic grows. |
| Data for the backtest pages | No setting needed. When `outputs/` is absent, the app uses the newest `public_data/*/ui_outputs` release. |
| Secrets | See `.streamlit/secrets.toml.example` |

**Steps for the owner**

1. In Streamlit Community Cloud, open the existing app (or create one) for the settings above.
2. In **Secrets**, paste these from `.streamlit/secrets.toml.example`, with real values:
   - `WATERSHED_DEPLOYMENT = "public"`;
   - `WATERSHED_ARTIFACT_URL`;
   - `WATERSHED_ARTIFACT_SHA256`.
3. Reboot the app. On the first Generate Forecast, it downloads and verifies the model bundle once.
4. Run the acceptance check in `docs/v3/LAUNCH_AND_MERGE.md`, section 4.

## Docker on a VM (alternative, persistent)

```bash
docker build -t watershedshift .
docker run -d -p 8501:8501 -v watershed-data:/data \
  -e WATERSHED_DEPLOYMENT=public -e WATERSHED_ARTIFACT_URL=... -e WATERSHED_ARTIFACT_SHA256=... watershedshift
```

- The cache and records persist in the `watershed-data` volume.
- To warm the cache every 6 hours, schedule `docker exec <container> python -m src.v3_service --warm`.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `WATERSHED_DEPLOYMENT` | `public` enforces the release policy before any fetch or display | `private` |
| `WATERSHED_ARTIFACT_URL`, `WATERSHED_ARTIFACT_SHA256` | Model bundle location and pinned hash | unset (no models) |
| `WATERSHED_ARTIFACT_TOKEN` | Bearer token for a private bundle URL | unset |
| `WATERSHED_RECORD_BUCKET` (plus AWS credentials) | Persistent write-once forecast records (needs `boto3`, not yet installed; see "Forecast records") | unset |
| `WATERSHED_RECORD_ENDPOINT_URL` | Endpoint of an S3-compatible provider for the record bucket | unset (AWS) |
| `WATERSHED_POWER_WAIT_S` | Longest a forecast waits for NASA POWER before skipping B and B2 | `45` |
| `WATERSHED_CACHE_DIR` | Shared source cache | `data/v3/cache` |
| `WATERSHED_RECORD_DIR` | Local write-once records | `forecasts/records_v3` |
| `WATERSHED_DATA_DIR` | Backtest UI data | `outputs`, else the newest public release |
| `WATERSHED_FORECAST_FILE` | Issued Model A record shown in the "Issued record" tab | `forecasts/latest.json`, else the release copy |

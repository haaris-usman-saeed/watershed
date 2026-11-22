# Merging PR #4 and PR #5, and launching public interactive forecasting

Checked on 2026-11-22.
- `main` is at `d72cd68`.
- PR #4 (`feature/30-day-forecast` into `main`) is open and CI passes.
- PR #5 (`feature/v3-forecast-platform` into `feature/30-day-forecast`) is stacked on it.

A local simulation merged PR #4 into `main` with a merge commit, then merged PR #5 into the result. There were no conflicts, and the final tree was identical to the v3 branch. Neither PR has been merged.

## 1. Merge sequence (owner)

1. **Merge PR #4** into `main` on GitHub with **"Create a merge commit"**. Do not squash or rebase: PR #5 contains PR #4's commits, and squashing would make them look like conflicts.
2. **Retarget PR #5 to `main`:**
   - If the `feature/30-day-forecast` branch is deleted after the merge, GitHub retargets PR #5 automatically.
   - Otherwise run `gh pr edit 5 --base main`, or use Edit next to the PR title.
   - The base change fires the `edited` event, which `.github/workflows/ci.yml` now listens for, so the checks re-run against `main`.
   - If they do not appear, run `gh pr close 5 && gh pr reopen 5`. Do not use "Re-run jobs": that repeats the old merge ref.
3. **Confirm the new `offline-checks` run is green**, then merge PR #5 with a merge commit.
4. **Optionally, check locally after both merges:**
   - `python -m pytest -q tests`
   - `python -m src.contract app/sample_data`
   - the frozen-hash check:

     ```bash
     python -c "import json; from src.train_v2 import frozen_hashes; assert frozen_hashes() == json.load(open('results/v2/final/RUN_RECORD.json'))['frozen_hashes']"
     ```

## 2. How model artifacts reach a deployment (no private datasets)

**Package contents.** `python -m src.v3_artifacts pack` writes `dist/watershedshift_models_<hash>.tar.gz` and its `.sha256`. It contains exactly 7 files plus `MANIFEST.json`:
- Model A (`models/v2_forecast/model.joblib` and `model_card.json`);
- Models A (v3 refit), B, B2 and C (`models/v3/model_*.joblib` and `model_cards.json`).

It contains no observation table, history, forecast, residual or raw file.
- The model cards hold aggregate CDEC-derived values: snow climatology and interval offsets.
- The models were fitted to CDEC values.

So the package falls under the CDEC entry of the release policy.

**Install checks.** `python -m src.v3_artifacts install --url URL --sha256 HEX` downloads the package, and `ensure_artifacts()` does the same at app start when it is configured. Installation is refused unless:
- the archive's SHA-256 equals the pinned value;
- every member is a regular file at one of the 8 expected paths, so there are no links, no path traversal and no extra files;
- every file matches `MANIFEST.json`;
- on a public deployment, `config/data_release_policy.json` allows public display of CDEC-derived values.

Files are written atomically. After installation, the app still checks each model's hash against its card before loading it.

**Where the package lives** depends on what DWR permits:
- **Publication of derived products permitted:** attach the archive to a GitHub Release, or deposit it in a data archive. Set `WATERSHED_ARTIFACT_URL` to its URL and `WATERSHED_ARTIFACT_SHA256` to the pinned hash.
- **Display permitted but not redistribution of the models:** put the archive in a private bucket. Set `WATERSHED_ARTIFACT_URL` to a private or presigned URL. If the host needs a bearer token, also set `WATERSHED_ARTIFACT_TOKEN`.

**How settings reach the app.** Put these as root-level entries in the host's secret store. Streamlit Community Cloud exposes root-level secrets as environment variables; confirm this on the host at deployment time.

Tests in `tests/test_v3.py` cover:
- the round trip;
- a wrong pin;
- a traversal member;
- an extra private file;
- a refusal on a public deployment without rights;
- the no-configuration case.

## 3. Actions required from the owner to launch public interactive forecasting

1. **Send the DWR permission request** drafted in `docs/DATA_PUBLICATION_AUDIT.md`, section 6, through https://water.ca.gov/Contact, and keep the written reply.
2. **If DWR permits public display (and, if needed, derived models),** commit an edit to `config/data_release_policy.json` that sets `public_display` (and `publish_derived` if applicable) to `true` for `cdec_storage_sensor15` and `cdec_snow_sensor82`. Cite the reply in `docs/DATA_PUBLICATION_AUDIT.md`.
3. **Merge PR #4, then PR #5,** as in section 1.
4. **On the trusted machine, package the artifacts:**
   - run `python -m src.v3_artifacts pack`;
   - upload the archive to the location chosen in section 2;
   - record its SHA-256.
5. **Configure the public app:**
   - set `WATERSHED_DEPLOYMENT=public`, `WATERSHED_ARTIFACT_URL` and `WATERSHED_ARTIFACT_SHA256`;
   - for persistent records, set `WATERSHED_RECORD_BUCKET` plus credentials, and add `boto3` to the requirements;
   - keep `WATERSHED_DATA_DIR=app/sample_data` for the backtest pages unless the real `outputs/` are also cleared and provisioned.
6. **Check the host,** for example Streamlit Community Cloud:
   - that it can install `requirements.txt`, including the `eccodes` and `eccodeslib` wheels, for its Python version;
   - that the app starts.
7. **Acceptance test on the public URL:**
   - select a reservoir and press Generate Forecast;
   - confirm that Models A-C return with source status "ok" and the expected dates;
   - confirm that a second visitor is served from the cache.

   Only then describe the public site as operational.

Until step 2 is done, a public deployment refuses before fetching anything:
- `src/v3_service.py` raises `PolicyBlocked`;
- `app.py` shows no page content when the data are real;
- the "Issued record" tab hides real snapshots;
- `install` refuses the artifacts.

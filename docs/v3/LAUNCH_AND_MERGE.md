# Launching public interactive forecasting

## 1. Merges (done 2026-11-22)

- **PR #4** was merged into `main` with a merge commit: `32d6122`.
- **PR #5** was then retargeted to `main`. Retargeting triggered CI run 44997714780 (97 passed, 11 skipped), and PR #5 was merged with a merge commit: `f972621`.
- **The launch changes** (policy authorization, the CDEC-inclusive public release, the model self-check and the deployment configuration) were reviewed and merged as **PR #6** (`0063cc0`).

## 2. Model bundle: build, upload, configure

**Done:** the bundle was uploaded as GitHub Release `models-2026-11-22` (pre-release), asset `watershedshift_models_7ad581d9f585.tar.gz`. The commands below record how.

The bundle contains exactly 7 model files plus `MANIFEST.json`:
- the manifest lists each file's SHA-256;
- it records reference predictions for the startup self-check;
- it records the library versions the bundle was built with.

It contains no observation table, history, forecast or residual file. It is byte-reproducible: rebuilding from the same artifacts gives the same SHA-256.

**On the trusted machine (the one that holds `models/`):**

```bash
python -m src.v3_artifacts pack --out dist/
```

**Upload it as a public GitHub Release asset** (permitted under the owner-authorized policy):

```bash
gh release create models-2026-11-22 dist/watershedshift_models_<hash>.tar.gz dist/watershedshift_models_<hash>.tar.gz.sha256 \
  --title "WatershedShift model bundle 2026-11-22" \
  --notes "Models A (frozen v2), A/B/B2/C (v3). SHA-256 in the .sha256 file. Built from main at f972621."
```

**Configure the host.** In the host's secrets (`.streamlit/secrets.toml.example`), set:
- `WATERSHED_ARTIFACT_URL=https://github.com/haaris-usman-saeed/watershed/releases/download/models-2026-11-22/watershedshift_models_<hash>.tar.gz`
- `WATERSHED_ARTIFACT_SHA256=<the 64-hex value>`
- `WATERSHED_DEPLOYMENT=public`

**If you prefer a private bucket:** upload the same file there, use a presigned or token-protected URL, and set `WATERSHED_ARTIFACT_TOKEN` if the host needs a bearer token.

**What the app does at startup.** It downloads the bundle once and refuses it if:
- the hash differs from the pinned value;
- it contains an unexpected or path-traversing member;
- a file differs from the manifest;
- the policy forbids public display.

It then refuses any model that fails the self-check. These cases are tested in `tests/test_v3.py`; a clean deployment loaded all four models from the bundle (section 4).

**Raw CDEC responses** (about 29 MB) are permitted for publication. They are not in git. Optionally, attach an archive of `data/raw/cdec`, `data/raw/snow` and `data/raw_manifest.jsonl` to the same release.

## 3. Updating the bundle later

Rebuilding models gives a new bundle and a new SHA-256.
- Publish it under a new release tag; never replace an existing asset.
- Update both secrets together.
- The startup self-check catches mismatched library versions.

## 4. Acceptance check on the public URL

1. Open **30-Day Forecast**, pick each reservoir and press **Generate Forecast**.
2. Expect all four models to be available, with:
   - source status "ok";
   - the origin a Monday;
   - the target exactly 30 days later;
   - days remaining counted from the issue time.
3. Press Generate Forecast again: the sources should be shown as "(cached)".
4. Open the page on a phone: the cards stack and the tables scroll horizontally.
5. **Only then** describe the public site as producing live forecasts.

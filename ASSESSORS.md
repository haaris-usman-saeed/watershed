# WatershedShift v2: guide for assessors

WatershedShift is a historical backtest of 30-day storage change at six California reservoirs. It is not an operational forecast. The results to check are `results/v2/final/metrics.json` (frozen, preregistered in `docs/preregistration.md`).

**Software.** Python 3.14.5 (tested on macOS only), git, and about 1 GB of disk space. Package versions: `requirements.txt` (exact verified versions in `requirements-lock.txt`).

**Getting the data bundle.** The public repository has no CDEC-derived values. Ask the owner for `watershedshift_v2_bundle.tar.gz` and its `.sha256` file, sent privately. It must not be reposted: CDEC redistribution terms are unconfirmed.

**What the bundle contains, and what it leaves out.** `SOURCE_COMMIT.txt` names the git commit the bundle was built from.

| Included | Files |
|---|---|
| Processed observations | `data/processed/daily_joined.csv` (storage plus weather), `snow_daily.csv`, `snow_basin_index.csv`, audits (`coverage.csv`, `station_year_audit.csv`, `discontinuity_scan.csv`) |
| Features | `data/processed/weekly_features.csv` |
| Predictions and metrics | `results/v2/final/` (frozen), `outputs/` (five UI files), `results/v2/development/`, `results/v1/` |
| Metadata | `data/raw_manifest.jsonl` (URL, time and SHA-256 of every download), `data/provenance.md`, `data/*.csv` (capacities, breaks, spike review, snow pillows), `config/v2_frozen.json`, `results/v2/report_facts.*` |
| Code | the full pipeline in `src/`, `tests/`, synthetic UI fixtures in `app/sample_data/` |

The bundle leaves out the raw agency responses, the frontend (`app.py`, `app/` code) and git history. Raw responses are included only if the owner built the bundle with `--include-raw` after the CDEC terms were confirmed. **The bundle cannot rerun the frozen evaluation by itself.** The run guard needs a git checkout (step 2 below).

## 1. Verify and extract

```bash
shasum -a 256 -c watershedshift_v2_bundle.tar.gz.sha256      # archive checksum
tar -tzf watershedshift_v2_bundle.tar.gz | grep -v '^watershedshift_v2_bundle/' # must print nothing
mkdir review && tar -xzf watershedshift_v2_bundle.tar.gz -C review
cd review/watershedshift_v2_bundle && shasum -a 256 -c SHA256SUMS.txt --quiet && cd ../..   # every file
```

Extract into a new, empty directory. Do not run anything from inside the extracted folder.

## 2. Reproduce the frozen configuration (no model changes)

Check out the pinned commit from `SOURCE_COMMIT.txt`, then copy the bundle's data into it:

```bash
git clone https://github.com/haaris-usman-saeed/watershed ws && cd ws && git checkout <source_commit>
python3.14 -m venv .venv && source .venv/bin/activate && pip install -r requirements-lock.txt
cp -R ../review/watershedshift_v2_bundle/data/processed/. data/processed/
cp ../review/watershedshift_v2_bundle/outputs/* outputs/
cp results/v2/final/metrics.json ../archived_metrics.json
python -m src.features            # rebuild the weekly table from the processed daily data
python -m src.experiments && python -m src.dev_report   # rolling development folds
python -m src.train_v2 --final    # refuses unless config and code hashes match the preregistration
```

- **If the raw responses were provided:** copy `data/raw/` too, then run `python -m src.build_dataset` before `src.features` to rebuild every processed file offline.
- **Never pass `--owner-approved-new-config`.** That flag is reserved for a different configuration.

## 3. Compare with the archived results

```bash
python -m src.compare_metrics results/v2/final/metrics.json ../archived_metrics.json   # expect IDENTICAL
git status --short                # expect only results/v2/final/metrics.json (timestamp)
python -m src.report_facts --check
```

The headline should read ML 2.3323 pp, Smoothed DOY seasonal 2.9694 pp and coverage 0.9207 on 593 rows. `src.train_v2` also prints `headline identical to first run: True`.

## 4. Tests and UI contract

```bash
python -m pytest -q tests
python -m src.contract outputs
python -m src.contract app/sample_data
python -m src.make_bundle --verify <path to bundle>
```

## Differences you should expect

**Metadata only:**
- `generated_utc` in `metrics.json` changes on every run. `RUN_RECORD.json` is not rewritten on a rerun.
- A rebuilt bundle therefore has a different archive checksum, with only the `metrics.json` entries in `SHA256SUMS.txt` changing.
- Re-downloaded NASA POWER responses differ in the server `times` field only.

**Real data differences.** Rebuilding from freshly downloaded agency data (`python -m src.fetch` and the other fetchers) can change values:
- CDEC storage is provisional and revised (flag `r`).
- Snow sensor 82 is an adjusted product.
- NASA POWER reprocesses its archive.

Compare such downloads against the SHA-256 values in `data/raw_manifest.jsonl`, which record the responses used on 2026-11-21. A mismatch means the agency changed its data, not that the code is broken.

More detail is in `docs/reproducibility.md` (earlier clean-clone verification), `README.md` (method, results, limitations) and `results/v2/report_facts.md` (source-linked facts).

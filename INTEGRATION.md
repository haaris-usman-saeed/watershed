# WatershedShift v2 frontend integration

This archive contains the corrected Claude Web frontend for the already merged v2 research pipeline. **It does not contain real CDEC-derived outputs or raw data.** It does not change the frozen model, preregistration, or metrics.

## Apply to the repository

From a local clone of `https://github.com/haaris-usman-saeed/watershed` with the latest `main` branch:

```bash
git checkout main && git pull
git checkout -b ui-integration
unzip /path/to/WatershedShift_frontend_v2_integration.zip -d .
python3 -m pip install -r requirements-lock.txt
python3 -m pip install -r app/requirements-ui.txt
```

The ZIP **intentionally does not contain `app/sample_data/`**. Preserve the engineering repo's synthetic fixtures, which comply with UI schema v1.0. Do not copy the old synthetic fixtures from the original Claude Web ZIP.

## Validate

Copy the privately generated five-file `outputs_v2_ui.zip` into the repository and extract its five files to `outputs/` (or use `WATERSHED_DATA_DIR` to point at that folder). Do not commit real outputs to public GitHub until source-data reuse terms have been checked.

```bash
python3 -m src.contract outputs
python3 -m src.contract app/sample_data
python3 app/check_contract.py outputs --require-real
python3 app/check_v2_integration.py outputs
WATERSHED_DATA_DIR=outputs streamlit run app.py
```

Inspect all four views: Overview, Reservoir explorer, Backtesting, and Model and research. Confirm that the selected baseline is `Smoothed DOY seasonal` (2.97 pp on the original frozen v2 test), primary-model MAE is 2.33 pp, the three bootstrap CIs are visible, Folsom's historical 90% interval coverage warning is visible (75.2%), snowpack is a primary input, NASA POWER weather is contextual, and no synthetic warning is shown for real files. Then test synthetic mode with `WATERSHED_DATA_DIR=app/sample_data streamlit run app.py` and confirm its synthetic warning persists on all views.

After verification:

```bash
git add app.py app .streamlit INTEGRATION.md
git commit -m "Integrate WatershedShift v2 Streamlit frontend"
git push -u origin ui-integration
```

Open the UI PR against `main`; after merging, tag the integrated commit `submission-v2` and rebuild the private assessor bundle at that commit.

## What changed compared with the original frontend

- Correct selected smoothed day-of-year baseline and `baseline_doy_pp` support.
- Read the v2 nested moving-block bootstrap structure and all three block lengths, reversing the ML-minus-baseline sign for a positive displayed error reduction.
- Preserve historically normalized `storage_pct` for Oroville rather than dividing older storage measurements by current capacity.
- Correct selected model inputs to storage history, CDEC adjusted SWE, season and reservoir identity; show NASA weather as context or secondary-model input only.
- Updated methodology, a snowpack context chart, and Folsom's 75.2% under-coverage caution.
- Fix the half-width cross-check to compare mean with mean, not mean with median.
- Preserve all synthetic/unverified-data banners, warning watermarks and missing-file states.

Local verification performed before packaging: Python compilation, original synthetic checker, unit-level v2 bootstrap parser check, historical-capacity calculation check, and Plotly figure smoke tests. **A full real-output Streamlit test has not been performed in this environment**, because the private v2 output ZIP and Streamlit installation were not available here.
CI note: GitHub Actions validates backend-only tests and synthetic fixtures. It does not replace the manual Streamlit review against private real v2 outputs.

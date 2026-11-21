# Originality check and relation to prior work (v2, 2026-11-21)

## Our claim, stated narrowly

**We do not claim that reservoir forecasting, reservoir machine learning, or the use of CDEC data is new.** All three have clear precedent (below).

The contribution of WatershedShift is an **independently assembled and quality-audited reservoir-weather-snow dataset for six large California reservoirs, 2001 to 2025, with reproducible 30-day evaluation**. In particular:
1. **Verification of historical measurements.** End-of-day semantics were checked against hourly records in 2003, 2010 and 2024. Spike candidates were adjudicated against the independent elevation sensor.
2. **Correction of an inconsistent date convention.** CDEC `OBS DATE` differs by operator, so we use `DATE TIME`.
3. **Historically appropriate capacity normalization.** This covers Oroville's July 2024 elevation-storage table change, located in both the hourly and daily raw records.
4. **Environmental data integration.** NASA POWER point weather and CDEC snow-pillow SWE are mapped to the upstream basins, with a predetermined core-station rule.
5. **Temporal feature engineering with explicit prediction-time rules,** tested by perturbing future data.
6. **Ground-truth 30-day storage-change labels in pp of the capacity in force,** never interpolated and never spanning a reporting break.
7. **Full provenance:** URLs, retrieval times and SHA-256 hashes for every raw response.
8. **Reproducible evaluation:** rolling water-year folds, a preregistered single held-out run, and block-bootstrap uncertainty.

On the evidence below, we found no public dataset or study that combines exactly these elements for these reservoirs. **That is a statement about a limited search, not proof that no such work exists.**

## Closest prior work

| Work | What it provides | Relation to WatershedShift |
|---|---|---|
| **ResOpsUS** (Steyaert et al., *Scientific Data* 9, 34, 2022; doi:10.1038/s41597-022-01134-7) | Daily historical inflow, outflow and storage for 679 major US reservoirs, compiled from about 40 agencies with direct agency requests. Sources report differing earliest years, with best coverage in recent decades. | The most important precedent for an assembled, documented reservoir-operations dataset. It very likely covers some or all of our reservoirs; we did not verify this per reservoir. Our work is much narrower (6 reservoirs) and differs in focus: joined weather and snow covariates, 30-day forecasting labels, historical capacity normalization and evaluation protocol. **We did not compare our storage values against ResOpsUS. That comparison would be a useful independent check and is listed as future work.** |
| Fill-and-Spill (arXiv 2403.04195, 2024) | Deep reinforcement learning for reservoir operation at Folsom, using CDEC daily storage and release | Uses the same CDEC source for one of our reservoirs, for operations and control rather than forecasting |
| Informal capstone on California reservoir levels (Medium, mlearning-ai) | Time-series ML on CDEC reservoir data, including Folsom | Shows CDEC plus ML forecasting of California reservoirs has been done informally. Results were mixed and attributed to human operations. |
| Decision-tree and regression-tree models of California reservoir operations (eScholarship qt6vh7s39d) | Data-driven release-rule modelling, including Oroville | ML on California reservoir operations; models releases, not 30-day storage change |
| LSTM storage and release models for 17 Texas reservoirs (*Environmental Data Science*, Cambridge) | Deep learning storage prediction from climate, hydrology and storage inputs | Methodological precedent outside California |
| Upper Colorado reservoir release LSTM (Frontiers in Water, doi:10.3389/frwa.2023.1112970) | LSTM release prediction for 30 reservoirs; storage and inflow more influential than precipitation and temperature | Consistent with our finding that weather adds little beyond storage history |
| Conditioned LSTM for daily releases over CONUS (PNNL, 2025) | Large-sample release prediction | Large-sample ML precedent, likely built on ResOpsUS-type data |
| Hybrid rule-based plus LSTM operation model (Gangrade et al.) | Release simulation with storage by water balance | Operations modelling, not storage-change forecasting |
| Weekly storage forecasting with SWE (ESSOAr preprint 10.1002/essoar.10503913.1) | Deep learning weekly storage forecasts using historical SWE in runoff season | Direct precedent for SWE as a storage-forecast input |
| Forecast-informed operations at Don Pedro and New Bullards Bar (eScholarship qt4sm711xr) | Storage initialization and inflow uncertainty by lead time | Supports the role of initial storage at short leads, as we find |
| Bureau of Reclamation CVP operations reports (usbr.gov/mp/cvo) | Official daily storage, inflow and release reports | Official data products, not an ML benchmark |

Note: v1 of this file attributed the Cambridge URL above to a NASA POWER drought-classification study. Later search summaries attributed the same URL to the Texas LSTM storage paper. We could not resolve this without reading the article, so the drought-study attribution has been removed.

## Searches performed

The searches used a general web search tool on 2026-11-21. Results were screened from titles and result summaries; papers were not read in full.
- "machine learning forecast California reservoir storage CDEC NASA POWER"
- "Shasta Oroville Folsom reservoir storage prediction gradient boosting dataset"
- "California reservoir storage forecasting machine learning CDEC daily storage dataset github kaggle"
- "ResOpsUS dataset historical reservoir operations contiguous United States Scientific Data 2022 Steyaert"
- "machine learning reservoir release storage prediction California CDEC daily data study"
- "ResOpsUS machine learning reservoir storage forecasting LSTM inflow snow water equivalent"

## Not checked (would strengthen or weaken the claim)

- Google Scholar, Web of Science and Scopus keyword searches.
- GitHub code search for `CSVDataServlet` together with `power.larc.nasa.gov` or sensor 82.
- Kaggle and Zenodo dataset search for CDEC storage-plus-snow joins.
- Value-level comparison of our storage series with ResOpsUS for the overlapping reservoirs and years.
- CDEC Bulletin 120 forecast products, which are official seasonal runoff forecasts and relevant baselines.

## Sources

- https://www.nature.com/articles/s41597-022-01134-7
- https://www.osti.gov/pages/biblio/1899724-resopsus-dataset-historical-reservoir-operations-contiguous-united-states
- https://arxiv.org/pdf/2403.04195
- https://medium.com/mlearning-ai/can-machine-learning-be-used-in-long-term-reservoir-level-prediction-e088d27e370b
- https://escholarship.org/content/qt6vh7s39d/qt6vh7s39d.pdf
- https://www.cambridge.org/core/product/298F98C659D4E27B49B9616D6593243E
- https://www.doi.org/10.3389/frwa.2023.1112970
- https://www.pnnl.gov/publications/improving-prediction-daily-reservoir-releases-over-conus-using-conditioned-lstm
- https://www.osti.gov/pages/servlets/purl/1908076
- https://essopenarchive.org/doi/full/10.1002/essoar.10503913.1
- https://escholarship.org/content/qt4sm711xr/qt4sm711xr.pdf
- https://www.usbr.gov/mp/cvo/

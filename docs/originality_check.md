# Originality check (modest, 2026-11-21)

**Claim we can make:** WatershedShift assembles a *joint* dataset that combines CDEC daily reservoir storage (sensor 15) for six verified California reservoirs with NASA POWER daily point weather, for 2018–2025. It documents verification (date convention, capacities, Oroville's storage-table break) and evaluates 30-day storage-change forecasts against baselines on held-out dates.
**Claim we cannot make:** that this exact combination has never been used for ML. A short web search does not establish novelty.

## What was searched

The searches were run through the general web search tool on 2026-11-21. Only titles and search-result summaries were screened; the papers were not read in full.

1. "machine learning forecast California reservoir storage CDEC NASA POWER"
2. "Shasta Oroville Folsom reservoir storage prediction gradient boosting dataset"
3. "California reservoir storage forecasting machine learning CDEC daily storage dataset github kaggle"

## What was found (closest related work)

- An informal capstone write-up (Medium) applied time-series ML to California reservoir levels **using CDEC data**. So *CDEC + ML for California reservoirs is not new.* — https://medium.com/mlearning-ai/can-machine-learning-be-used-in-long-term-reservoir-level-prediction-e088d27e370b
- A UCLA drought-classification study used NASA POWER data. It targets drought categories, not reservoir storage. — https://www.cambridge.org/core/product/298F98C659D4E27B49B9616D6593243E
- Tree-based and statistical-learning reservoir-level prediction: Lake Lanier and Lake Travis (Sci. Rep. 2018). — https://par.nsf.gov/servlets/purl/10067871
- Decision-tree modelling of California reservoir operations, including Oroville. — https://escholarship.org/content/qt6vh7s39d/qt6vh7s39d.pdf
- Deep learning for 17 Texas reservoirs. — https://par.nsf.gov/servlets/purl/10530470
- Forecast-informed operations work at Don Pedro and New Bullards Bar. — https://escholarship.org/content/qt4sm711xr/qt4sm711xr.pdf
- Reclamation publishes daily CVP operations reports and modelled storage series. These are official data products, not ML datasets. — https://www.usbr.gov/mp/cvo/

No GitHub, Kaggle or paper result was found that combines **CDEC sensor-15 daily storage for these six reservoirs with NASA POWER T2M/PRECTOTCORR/RH2M into a documented 30-day-change benchmark**. That absence reflects a limited search, not proof.

## Not checked (would strengthen the claim)

Google Scholar or Web of Science keyword searches; GitHub code search for `CSVDataServlet` combined with `power.larc.nasa.gov`; Kaggle dataset search; AGU/EGU abstracts.

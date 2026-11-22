# Data publication audit (v3)

All pages were accessed on 2026-11-22. This is a factual record of the terms each publisher states. It is not legal advice. Where terms are silent or ambiguous, the item is marked **unresolved** and the project does not publish it.

The machine-readable policy that the release pipeline enforces is `config/data_release_policy.json`. Only the owner may change it, after written clarification.

## Summary

| Source | Reuse | Redistribution of raw values | Derived data | Public website display | Status |
|---|---|---|---|---|---|
| CDEC daily reservoir storage (sensor 15) | Appears permitted (no restriction stated) | Not expressly granted | Not expressly granted | Not expressly granted | **Unresolved**: ask DWR |
| CDEC snow water content (sensor 82) | Appears permitted (no restriction stated) | Not expressly granted | Not expressly granted | Not expressly granted | **Unresolved**: ask DWR |
| NASA POWER daily point data | Permitted, no restrictions | Permitted; notification requested | Permitted with citation | Permitted with citation | **Confirmed** |
| NOAA GEFS v12 operational forecasts | Permitted ("can be used as desired") | Permitted; attribution requested | Permitted; must not be called unaltered NOAA data | Permitted | **Confirmed** |
| NOAA GEFS v12 reforecast (retrospective) | Permitted | Permitted | Permitted | Permitted | **Confirmed** |
| Derived tables containing only GEFS and/or POWER values | | | Permitted | Permitted | **Confirmed** |
| Anything containing CDEC values: weekly features, history, forecasts, model artifacts trained on CDEC targets | | | | | **Unresolved**: depends on CDEC |

## 1. CDEC daily reservoir storage (sensor 15) and snow water content (sensor 82)

- **Official publisher:** California Department of Water Resources (DWR), California Data Exchange Center (CDEC). CDEC also relays data that other operators report. Per `stations.csv`:
  - Shasta, Folsom and New Melones are US Bureau of Reclamation;
  - Oroville is DWR;
  - Don Pedro is listed under USGS/USACE in the CDEC operator field;
  - McClure is Merced Irrigation District.
- **Endpoint:** `https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet` (parameters `Stations`, `SensorNums=15` or `82`, `dur_code=D`).
- **Licence or conditions:** No licence is stated.
  - The CDEC footer links to DWR's Conditions of Use (https://water.ca.gov/Conditions-of-Use). That page contains a privacy notice, a security notice and a disclaimer. The disclaimer says information "is made available to provide immediate access for the convenience of interested persons", with no guarantee of accuracy. It contains **no** licence, reuse restriction or redistribution clause.
  - CDEC pages carry "Copyright © State of California".
  - The statewide CA.gov use policy (https://www.ca.gov/use/) says information on that website "unless otherwise indicated, is considered in the public domain" and "may be distributed or copied as permitted by law". It is written for the CA.gov website, and DWR's own page does not repeat it.
- **Attribution:** none is required by any stated term. The project cites CDEC and DWR anyway.
- **Reuse permitted?** Appears so: public, unauthenticated access, and no stated restriction.
- **Redistribution permitted?** Not expressly granted.
  - Third parties do redistribute CDEC values. For example, ResOpsUS (CC BY 4.0, Zenodo 5893641) contains CDEC-derived reservoir records, and DWR's own California Water Watch republishes CDEC storage.
  - That is evidence of practice, not permission for this project.
- **Derived datasets publishable?** Not expressly addressed.
- **Public website display?** Not expressly addressed.
- **Unresolved:**
  - whether "Copyright © State of California" on CDEC pages is meant to restrict reuse of the measurements;
  - whether the CA.gov public-domain statement applies to CDEC data;
  - whether operator-supplied records carry their own terms.
- **Contact:**
  - DWR general contact: https://water.ca.gov/Contact.
  - CDEC contact page: https://cdec.water.ca.gov/CDECContact.html. It lists query tools but no email address, so none is given here.
  - The draft request is in section 6.

## 2. NASA POWER daily point data (Model B weather)

- **Official publisher:** NASA Langley Research Center, POWER project. Data services are provided with the Atmospheric Science Data Center.
- **Endpoint:** `https://power.larc.nasa.gov/api/temporal/daily/point` (parameters T2M, PRECTOTCORR, RH2M; community AG).
- **Licence or conditions:**
  - The POWER referencing guide (https://power.larc.nasa.gov/docs/referencing/) asks users to cite POWER and the data version, and to give the access date.
  - It *requests* notification of publications (larc-power-project@mail.nasa.gov).
  - It *requests* notification "if POWER data is transmitted to other researchers".
  - NASA's Earth science data use policy (https://www.earthdata.nasa.gov/learn/use-data/data-use-policy) states open use. A third-party AWS Marketplace listing gives CC BY 4.0; that was not confirmed on the POWER site.
- **Attribution:** "The data was obtained from the National Aeronautics and Space Administration (NASA) Langley Research Center (LaRC) Prediction of Worldwide Energy Resources (POWER) Project funded through the NASA Earth Science Directorate Applied Science Program", plus version and access date.
- **Reuse, redistribution, derived data, website display:** permitted. Send the notification email when a public dataset is released.

## 3. NOAA GEFS v12 operational forecasts (Model C, live and 2020-09-24 onward)

- **Official publisher:** NOAA National Weather Service, NCEP (Environmental Modeling Center and NCEP Central Operations). Distributed through the NOAA Open Data Dissemination program (NODD) on AWS.
- **Endpoint:** `https://noaa-gefs-pds.s3.amazonaws.com/gefs.YYYYMMDD/00/atmos/pgrb2sp25/` and `.../pgrb2ap5/`. Files are `gec00.t00z.pgrb2s.0p25.fHHH` (control member), each with an `.idx` file.
- **Licence or conditions:** the AWS Open Data Registry entry for NOAA GEFS says:
  - "NOAA data disseminated through NODD are open to the public and can be used as desired";
  - NOAA requests attribution;
  - users must not state or imply NOAA endorsement;
  - modified data must not be presented as original, unaltered NOAA data.

  US Government works are not subject to copyright in the US (17 U.S.C. 105). NWS policy directive 1-12 (https://www.weather.gov/media/directives/001_pdfs/pd00112001curr.pdf) has restricted categories, but they concern specific third-party products. No restriction on GEFS model output was found.
- **Attribution:** "NOAA National Centers for Environmental Prediction (NCEP) Global Ensemble Forecast System version 12 (GEFSv12), accessed via the NOAA Open Data Dissemination program on AWS." Basin aggregates must be described as derived, not unaltered NOAA data.
- **Reuse, redistribution, derived data, website display:** permitted.

## 4. NOAA GEFS v12 reforecast (Model C training, 2000-2019)

- **Official publisher:** NOAA, with the reforecast produced by EMC and the NOAA Physical Sciences Laboratory. The dataset is described in Guan et al. (2022), Monthly Weather Review 150(3), doi:10.1175/MWR-D-21-0245.1.
- **Endpoint:** `https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/YYYY/YYYYMMDD00/c00/Days:1-10/apcp_sfc_YYYYMMDD00_c00.grib2` (and `Days:10-16`).
- **Licence or conditions:** the same NODD statement, on registry entry https://registry.opendata.aws/noaa-gefs-reforecast/.
- **Nature of the product:** retrospective forecasts made around 2019-2020 with the GEFSv12 model. Per the registry:
  - runs are 00 UTC daily with 5 members;
  - an 11-member, 35-day run is made once weekly.

  These forecasts were **never issued in real time** on those dates and are labelled `reforecast` everywhere in this project.
- **Reuse, redistribution, derived data, website display:** permitted, with the same attribution and no-endorsement conditions.

## 5. Derived datasets and trained models

| Item | Contains | Status |
|---|---|---|
| `data/v3/public/gefs_basin_precip.csv` (basin-box GEFS precipitation per cycle) | NOAA-derived only | Publishable |
| `data/v3/public/power_basin_weather.csv` (if built) | NASA POWER-derived only | Publishable |
| Upstream-area box definitions | Coordinates from `stations.csv` and CDEC station metadata | Publishable (station locations are already in the public repo) |
| Weekly v3 feature tables, storage history, forecasts, OOF residuals | CDEC values | Unresolved |
| Model artifacts (A, B, C) | Fitted to CDEC-derived targets | Unresolved. A tree ensemble stores split thresholds and leaf values derived from CDEC records, so it is treated like derived data. |

## 6. Permission clarification request (draft for the owner to send)

> **To:** California Department of Water Resources, California Data Exchange Center (via https://water.ca.gov/Contact, attention: CDEC)
> **Subject:** Permission to republish CDEC daily reservoir storage and snow water content in a research dataset and website
>
> We run WatershedShift, a non-commercial research project that forecasts 30-day storage change at six California reservoirs (Shasta, Oroville, Folsom, New Melones, Don Pedro and McClure). We use CDEC daily data retrieved from `cdec.water.ca.gov/dynamicapp/req/CSVDataServlet`:
> - sensor 15 (reservoir storage) for SHA, ORO, FOL, NML, DNP and EXC, 2001 to present;
> - sensor 82 (snow water content, revised) for 51 snow stations in the contributing river basins.
>
> DWR's Conditions of Use contain a disclaimer but no reuse terms, and CDEC pages show "Copyright © State of California". We would like to confirm whether we may:
> 1. publish a cleaned, versioned copy of these daily values (with flags, our quality-control decisions and checksums) in a public GitHub repository and data archive, credited to CDEC and DWR;
> 2. display recent CDEC storage and snow values, and forecasts computed from them, on a public website, with attribution and a link to CDEC;
> 3. publish derived products, such as weekly feature tables and statistical models fitted to these data;
> 4. and whether records that CDEC relays from other operators (US Bureau of Reclamation, Merced Irrigation District, and the Don Pedro operators) need separate permission.
>
> Please let us know of any required attribution wording or conditions. We will not imply DWR endorsement.

Until a written answer is received, `config/data_release_policy.json` keeps every CDEC-derived item private, and the public website does not display CDEC values. (Superseded by section 7.)

## 7. Authorization record (2026-11-22)

- **Statement.** On 2026-11-22 the project owner stated in writing to the engineering agent that "DWR CDEC and everyone involved has given me permission go public". "Everyone involved" is understood to include the operators whose records CDEC relays.
- **Policy change.** On that basis, the owner authorized setting the following flags to `true` for both CDEC sources (policy version `2026-11-22.2`):
  - `publish_raw`;
  - `publish_derived`;
  - `public_display`.
- **What the engineering agent has not seen.** The agent did not see the permission correspondence itself. The audit trail is therefore the owner's statement, not a reviewed document.
- **Owner action:** archive the written permission (sender, date, scope and any conditions) under `docs/permissions/`, removing personal contact details if preferred. If the permission carries conditions such as attribution wording, scope limits or a non-commercial clause, update the `attribution` text and these flags to match.
- **Unchanged:** the CDEC disclaimer still applies. Values are provisional and may be revised. Public pages and datasets keep the CDEC and operator attribution in `config/data_release_policy.json` and state that no endorsement is implied.

| Item | Now published? | Where |
|---|---|---|
| 1. CDEC daily storage records (cleaned) | Yes | `public_data/watershedshift-public-data-2026-11-22.2/` |
| 2. CDEC snow water equivalent (cleaned pillow values and basin index) | Yes | same |
| 3. Cleaned historical data and audits | Yes | same |
| 4. Derived feature tables (v3 weekly table) | Yes | same |
| 5. Model artifacts | Yes, as a hash-pinned bundle once the owner uploads it (`docs/v3/LAUNCH_AND_MERGE.md`) | GitHub Release asset |
| 6. Generated live forecasts | Yes, on the website and as downloads; issued records are included in the release | website, `public_data/.../issued_forecasts/` |
| Raw CDEC responses (byte-for-byte) | Permitted. Kept out of git for size (about 29 MB); offered as a release asset with checksums | GitHub Release asset |

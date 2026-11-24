# WatershedShift data provenance (v2 dataset)

**What this is:** a newly assembled joint dataset built from three public sources:
- CDEC daily reservoir storage;
- CDEC daily snow-pillow snow water equivalent (SWE);
- NASA POWER daily point meteorology.

It contains **no new physical measurements**. The contribution is the selection, verification, alignment, quality audit, historically appropriate normalization and documentation of these sources.

The v1 dataset (2018 to 2025, storage and weather only) is preserved in `results/v1/provenance.md` and `results/v1/coverage_table.md`.

Raw files stay on the project owner's machine in `data/raw/` and are never committed, mainly for size. Cleaned CDEC data have been published in `public_data/` since the owner authorized CDEC publication on 2026-11-22 (`docs/DATA_PUBLICATION_AUDIT.md`, section 7). `data/raw_manifest.jsonl` (tracked) records every download with URL, UTC retrieval time, HTTP status, byte size and SHA-256. It has 189 entries: 188 for pipeline inputs, matching the count at the frozen commit `e47deab` and in the preregistration, plus 1 for the ResOpsUS archive, which is used only for verification. An earlier version of this line said 187, a stale count.

```bash
python -m src.fetch --start 2001-01-01 --end 2025-12-31        # storage + weather (network)
python -m src.fetch_snow --start 2001-01-01 --end 2025-12-31   # snow pillows (network)
python -m src.fetch_wsihist                                    # DWR water year types (network)
python -m src.build_dataset                                    # offline: raw -> processed + audit
```

Retrieval dates: 2026-11-21 (UTC timestamps in the manifest).

## 1. Reservoir storage: California Data Exchange Center (CDEC), CA Department of Water Resources

| item | value |
|---|---|
| Endpoint | `https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=<ID>&SensorNums=15&dur_code=D&Start=<YYYY-MM-DD>&End=<YYYY-MM-DD>`, one request per station-year |
| Period | 2001-01-01 to 2025-12-31 for all six stations (CDEC metadata lists daily sensor 15 from at least 1985) |
| Sensor | 15, RESERVOIR STORAGE, acre-feet. Every raw row is checked for station, `SENSOR_NUMBER=15`, `DURATION=D`, `UNITS=AF`. |
| **Date convention** | `DATE TIME` = D carries the **end-of-day storage of day D**. This was verified against hourly sensor 15 (daily D equals hourly 00:00 of D+1) for SHA, ORO, FOL, NML and EXC in 2003, 2010 and 2024 (EXC 2010 differs by 0.06 pp). DNP has no hourly series in those years. `OBS DATE` is not used because its offset from `DATE TIME` changes by station and over time. ORO, DNP and EXC are mostly D+1, but EXC is D in 2003 to 2013 and ORO is D on 138 days in 2008. SHA, FOL and NML are D until late 2025, when D+1 values start to appear (first on 2025-09-24). Regression test: `tests/test_build.py::test_date_comes_from_date_time_not_obs_date`. |
| Missing | `---` and blanks stay missing. **Storage is never interpolated.** |
| Flags | `DATA_FLAG` counts per station are in `data/coverage_table.md` (e = estimated, r = revised, N at EXC, 23 days). Flagged values are kept. |
| Publication lag | Observed 2026-11-21: 1 to 2 days |
| Revisions | CDEC data are provisional and revisable. This dataset is a **retrospective snapshot as of the retrieval date**. |

### Station verification

Coordinates and operators come from `https://cdec.water.ca.gov/dynamicapp/staMeta?station_id=<ID>`. Current capacities come from the CDEC Daily Reservoir Storage Summary (`https://cdec.water.ca.gov/reportapp/javareports?name=RES`).

| ID | Reservoir | Operator | Lat | Lon | Capacity AF |
|---|---|---|---|---|---|
| SHA | Shasta Lake | USBR | 40.718 | -122.420 | 4,552,000 |
| ORO | Lake Oroville | CA DWR | 39.540 | -121.493 | 3,537,577 through 2024-07-01; 3,424,753 from 2024-07-02 |
| FOL | Folsom Lake | USBR | 38.683 | -121.183 | 977,000 |
| NML | New Melones | USBR | 37.9481 | -120.525 | 2,400,000 |
| DNP | Don Pedro | USGS/USACE (CDEC field) | 37.702 | -120.421 | 2,030,000 |
| EXC | Lake McClure (Exchequer) | Merced Irrigation District | 37.585 | -120.270 | 1,024,600 |

**Eligibility rule:** a station is `included` only if at most 10% of its storage days are missing over the dataset period. All six qualify (maximum 0.43% missing). Only `included` stations can enter feature construction or training (`tests/test_features.py::test_excluded_station_never_enters_features`).

### Historically appropriate capacity normalization (`data/capacity_history.csv`)

`storage_pct` and every pp target use the capacity in force on that date.

The only verified change in the period is at Lake Oroville:
- DWR adopted a new elevation-storage table from a bathymetric survey. Capacity went from 3,537,577 to 3,424,753 AF, with the loss mainly below 420 ft. The new figure took effect July 1, 2024 ([DWR](https://water.ca.gov/News/Blog/2024/Jun-24/Climate-Readiness---Using-Advanced-Lasers-and-Sonar-to-Determine-if-Lake-Oroville-Has-Lost-Capacity)).
- **Verified against raw records:**
  - CDEC hourly sensor 15 steps by **-110,466 AF at exactly 2024-07-01 00:00**. Hourly and daily values agree on the old basis through 2024-06-25.
  - The **daily** series stays on the old basis through 2024-07-01 (3,352,893 AF) and is first on the new basis on **2024-07-02** (3,229,790 AF). Over the same day, elevation (sensor 6) fell 0.90 ft.
- Values are **not adjusted**. The break is in `data/series_breaks.csv`, and no target or storage-change feature window may span it (`tests/test_features.py::test_no_window_spans_series_break`).

No other capacity or reporting-basis change was found:
- The maximum fill is at most 100% of the capacity in force at every station, except ORO's 101.15% (February 2017, on the old basis, during the spillway emergency).
- A dry-day level-shift scan was run: one-day change > 2 pp with < 5 mm of rain over 4 days. Every hit is part of a multi-day flood-release sequence rather than a one-day step: ORO 2017-02-14/15 (spillway emergency), FOL January 2006, FOL December 2016 to February 2017, SHA 2017-03-01.

### Genuine extremes versus artifacts (`data/spike_review.csv`)

Candidates are one-day jumps > 2 pp that immediately reverse. Each candidate was checked against the independent CDEC daily elevation sensor (sensor 6):
- **Artifacts, set to missing:**
  - DNP 2019-09-26: storage +51,731 AF and back, while elevation fell smoothly.
  - EXC 2022-05-22: +40,873 AF and back, while elevation rose smoothly.
- **Physical, kept:** six FOL candidates in the New Year 2006, December 2016, January 2017 and February 2017 floods, where elevation moves with storage. Example: 2017-02-10, elevation 442.74 to 449.16 to 444.73 ft.

Any future unreviewed candidate is set to missing by default (`tests/test_build.py::test_spike_review_removes_artifacts_and_unreviewed_keeps_physical`). The largest kept daily change is FOL 2022-12-31 (+20.1 pp, about 196,577 AF, during the New Year's Eve 2022 atmospheric river, with 78 mm of rain that day).

### Cross-check against ResOpsUS, partly independent (`results/v2/resopsus_check/report.md`)

ResOpsUS (Steyaert et al. 2022, *Scientific Data*; Zenodo record 5893641, CC-BY 4.0, md5-verified, SHA-256 in the manifest) contains all six reservoirs. Daily storage was compared for 2001 to 2020 after converting million cubic meters to acre-feet. It is used only for verification and never to modify our data. The check is **not fully independent**: four of the six ResOpsUS storage records (SHA, ORO, NML, EXC) come from CDEC, our own upstream source, so for them it tests processing and date alignment rather than the measurement. Only FOL and DNP come from independent sources.

| Reservoir | ResOpsUS storage source | Agreement with our cleaned series |
|---|---|---|
| FOL | Bureau of Reclamation (independent) | 99.6% of 6,574 days identical at the same date; largest gap 10,975 AF (1.1 pp) on 2017-06-19 |
| DNP | USGS gauge 11287500 (independent) | best at the same date: median difference 891 AF (0.04 pp), 99th percentile 6,123 AF, daily-change correlation 0.96 (0.80 and 0.82 at one-day shifts) |
| SHA | CDEC | 100% identical at the same date (differences of 5 to 7 AF come from a slightly different AF to m3 constant) |
| NML | CDEC | 99.99% identical at the same date |
| ORO | CDEC | predominantly offset by one day relative to our end-of-day dates (ResOpsUS labels a value one day later; one-day match share at least 0.91 in every year except 2008, 0.62). 99.5% identical when matched to raw CDEC keyed by `OBS DATE`; 3.4% at the same date |
| EXC | CDEC | the offset changes across periods: one day offset from 2001 through about March 2003; predominantly aligned at zero lag from about April 2003 through May 2014 (2004 to 2013 aligned); mixed during the June 2014 transition; one day offset from July 2014 to the end of the overlap. 98.9% identical keyed by `OBS DATE`; same-date match 55% overall |

**Interpretation:**
- The two independent operator and gauge sources (FOL, DNP) agree with our end-of-day `DATE TIME` dates, with no one-day shift.
- The ResOpsUS ORO record, and its EXC record outside April 2003 to May 2014, match our raw CDEC rows keyed by `OBS DATE`, whose offset varies by station and over time. This is consistent with ResOpsUS dating CDEC values by `OBS DATE`. The exact date convention intended by the ResOpsUS authors is unknown, and this check does not show that their processing is in error. WatershedShift's `DATE TIME` keying is consistent across stations and years.
- The offsets matter mainly for daily-resolution analysis (daily changes, alignment with daily flows or weather, event timing). They have much less effect on monthly or longer aggregates.
- **No WatershedShift value was changed as a result of this check.**
- Our missing days (DNP 35, EXC 37 in the overlap) are reported in ResOpsUS on 35 and 13 of those days respectively. We keep them missing: they are raw CDEC gaps or reviewed artifacts, and filling them from another source would mix sources within one series.

## 2. Weather: NASA POWER daily point API

| item | value |
|---|---|
| Endpoint | `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,PRECTOTCORR,RH2M&community=AG&latitude=<lat>&longitude=<lon>&start=20010101&end=20251231&format=JSON&time-standard=LST` |
| API version | v2.10.5 |
| Units | T2M in deg C; PRECTOTCORR (bias-corrected precipitation) in mm/day; RH2M in %. Units are asserted from each response. |
| Missing | `fill_value = -999`, plus plausibility ranges. **No missing or implausible values for 2001 to 2025.** |
| Quality flags | None per value. The archive source list is MERRA-2 and POWER. |
| Spatial meaning | Reanalysis grid cell containing the dam coordinate (SHA cell elevation 463 m vs dam about 325 m). **This is not watershed-average rainfall.** |
| Latency | About 3 days (observed 2026-11-21). Recent days come from GEOS-IT, while archive years come from MERRA-2, so retrospective values differ from what was available in real time. |
| Terms | NASA data, free to use. Consolidated attribution for all sources: README section "Attribution and terms". Requested acknowledgement: "These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program." |

## 3. Snow water equivalent: CDEC snow pillows (sensor 82)

| item | value |
|---|---|
| Candidate stations | Every CDEC station listed with **daily sensor 82** in the CDEC river basin(s) draining to each reservoir (CDEC `staSearch`, 2026-11-21): SHA: SACRAMENTO R, MC CLOUD R, PIT R; ORO: FEATHER R; FOL: AMERICAN R; NML: STANISLAUS R; DNP: TUOLUMNE R; EXC: MERCED R. 51 stations in total (`data/snow_stations.csv`). The basin assignment relies on CDEC's river-basin field. |
| Endpoint | CSVDataServlet with `SensorNums=82&dur_code=D`, one request per year for all stations (25 raw files) |
| Sensor | 82 = SNOW, WATER CONTENT (adjusted/revised), inches. Units and sensor are asserted. The daily value is a morning reading (`OBS DATE` 04:00 to 07:00). |
| Cleaning | Missing or text stays missing and is **never treated as zero**. Values below -0.5 in or above 250 in are set to missing (652 values). Values from -0.5 to 0 are set to 0 (47 values, pillow noise at bare ground). |
| Flags | r (revised): 341,907; o: 5,327; e: 1,362; c: 14 |
| Core stations | Predetermined rule using availability only (no outcomes) and only WY2002 to WY2023: a pillow is "core" if at least 75% of snow seasons (Dec to May) have at least 80% daily availability. Core counts: SHA 7, ORO 8, FOL 11, NML 5, DNP 3, EXC 3 (counted from `data/processed/snow_basin_index.csv`; an earlier version of this line said FOL 12, a transcription error). |
| Basin index | For each reservoir, mean SWE (inches) over core pillows reporting that day, computed only when at least 2/3 of core pillows report; otherwise missing. This is a point index, not basin-volume SWE. |
| Prediction-time rule | Snow features use SWE dated t-1 or earlier. |
| Revision caveat | Sensor 82 is an **adjusted** product, revised after the fact (most values carry the r flag). Real-time users would have seen raw sensor 3, so the SWE features are retrospective, like the weather. |

## 4. Water year types (descriptive only, never a model input)

DWR Sacramento Valley and San Joaquin Valley water year indices and classifications (W, AN, BN, D, C) come from `https://cdec.water.ca.gov/reportapp/javareports?name=WSIHIST` and are saved raw. They are used only to group evaluation results into wet and dry years: Sacramento Valley for SHA, ORO and FOL; San Joaquin Valley for NML, DNP and EXC.

## 5. Processed dataset

`data/processed/daily_joined.csv` has one row per station and calendar day: 6 x 9,131 = 54,786 rows, 2001-01-01 to 2025-12-31. Columns:
`station_id, reservoir_name, date, storage_af, storage_pct, capacity_af (in force that day), cdec_flag, rain_mm, temp_c, humidity_pct, lat, lon`.

Snow is in `data/processed/snow_daily.csv` (long format). Audits are in `coverage.csv`, `station_year_audit.csv` and `discontinuity_scan.csv`; the summary is in `data/coverage_table.md`.

Transformation steps (`src/build_dataset.py`):
1. Validate the raw header and station, sensor, duration and units columns.
2. Take the date from `DATE TIME`.
3. Convert non-numeric values to missing.
4. Set values <= 0 or > 110% of the capacity in force to missing.
5. Collapse identical duplicate dates; set conflicting duplicates to missing.
6. Apply adjudicated spike reviews.
7. Reindex to the full calendar without filling.
8. Join POWER weather after masking.
9. Attach the capacity in force on each date.

## 6. Forecast-time semantics

- Origin t is the date of the latest end-of-day storage value used. It is published at t+1 or t+2, so the effective lead time from issue is about 28 to 29 days.
- Weather features end at t-7. Snow features use t-1 and earlier.
- No input is ever taken from after t.

## 7. Originality

See `docs/originality_check.md`.

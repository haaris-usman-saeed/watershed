# WatershedShift — data provenance

**What this is:** a newly *assembled joint dataset* built from two public sources: CDEC daily reservoir storage and NASA POWER daily point meteorology. It contains **no new physical measurements**. The contribution is the selection, verification, alignment, quality audit and documentation of these sources.

Raw files stay on the project owner's machine in `data/raw/` and are not committed to git. `data/raw_manifest.jsonl` records every download with its URL, UTC retrieval time, HTTP status, byte size and SHA-256. Anyone can rebuild the processed data with:

```bash
python -m src.fetch --start 2018-01-01 --end 2025-12-31   # network; writes data/raw/ byte-for-byte
python -m src.build_dataset                             # offline; raw -> data/processed/ + audit
```

First retrieval: 2026-11-21 (UTC timestamps in manifest).

## 1. Reservoir storage — California Data Exchange Center (CDEC), CA Dept of Water Resources

| item | value |
|---|---|
| Endpoint | `https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=<ID>&SensorNums=15&dur_code=D&Start=<YYYY-MM-DD>&End=<YYYY-MM-DD>` (one request per station-year) |
| Sensor | 15 = RESERVOIR STORAGE, units **AF** (acre-feet). Every raw row is checked for `STATION_ID`, `SENSOR_NUMBER=15`, `DURATION=D`, `UNITS=AF` |
| Fields | `STATION_ID, DURATION, SENSOR_NUMBER, SENSOR_TYPE, DATE TIME, OBS DATE, VALUE, DATA_FLAG, UNITS` |
| **Date convention** | `DATE TIME` = D carries the **end-of-day (midnight) storage of day D**. This was verified against hourly sensor 15: SHA daily 2024-01-03 = hourly 2024-01-04 00:00 = 3,136,198 AF, and ORO/NML show the same pattern. The CDEC RES summary report is also labelled "Ending at midnight". `OBS DATE` is **inconsistent across operators**: it equals D for USBR stations (SHA, FOL, NML) and D+1 for ORO, DNP and EXC. It is not used. *(The starter code used `OBS DATE`, which would have shifted 3 of 6 stations by one day.)* |
| Missing values | `---` / blank → missing. They are **never interpolated**. |
| Flags | `DATA_FLAG` values are counted and kept (`e` = estimated, `r` = revised), as shown in `data/coverage_table.md` |
| Publication lag (observed 2026-11-21) | 1–2 days (latest daily value on 11-21: SHA 11-20, ORO 11-19) |
| Revisions | CDEC data are provisional and can be revised later (`r` flags). The dataset is a **retrospective snapshot** as of the retrieval date, not what was visible in real time. |
| Terms | Public data from a California state agency. No explicit open-data licence was found during this check. Cite CDEC/DWR as the source and treat values as provisional. Raw records are not redistributed in this repo. **The project owner should confirm the reuse wording before publication.** |

### Station verification (2026-11-21)

Coordinates and operator come from `https://cdec.water.ca.gov/dynamicapp/staMeta?station_id=<ID>`. Each metadata page lists RESERVOIR STORAGE sensor 15 as *daily* with data available since at least 1985 (SHA: 01/01/1985–present). Capacities come from the CDEC *Daily Reservoir Storage Summary* (`https://cdec.water.ca.gov/reportapp/javareports?name=RES`, report ending 11/19/2026).

| ID | Reservoir | Operator (CDEC) | Lat | Lon | Capacity AF (verified) | Starter value |
|---|---|---|---|---|---|---|
| SHA | Shasta Lake | USBR | 40.718 | −122.420 | 4,552,000 | 4,552,000 ✓ |
| ORO | Lake Oroville | CA DWR | 39.540 | −121.493 | **3,424,753** | 3,537,577 ✗ (pre-2024 figure) |
| FOL | Folsom Lake | USBR | 38.683 | −121.183 | 977,000 | 977,000 ✓ (coords corrected) |
| NML | New Melones | USBR | 37.9481 | −120.525 | 2,400,000 | 2,400,000 ✓ (coords corrected) |
| DNP | Don Pedro | USGS/USACE | 37.702 | −120.421 | 2,030,000 | 2,030,000 ✓ |
| EXC | Lake McClure (Exchequer) | Merced ID | 37.585 | −120.270 | **1,024,600** | 1,025,000 ✗ |

All six stations passed verification and are included for 2018-01-01 → 2025-12-31. See `data/coverage_table.md`.

### Series break: Lake Oroville, 2024-07-02

DWR adopted a new elevation–storage table for Lake Oroville based on a bathymetric survey. Capacity changed from 3,537,577 to 3,424,753 AF, with the loss mainly below 420 ft, and the new figure was used from **July 1, 2024** ([DWR](https://water.ca.gov/News/Blog/2024/Jun-24/Climate-Readiness---Using-Advanced-Lasers-and-Sonar-to-Determine-if-Lake-Oroville-Has-Lost-Capacity)). In the CDEC daily series, values through 2024-07-01 are on the old basis (June 2024 reached about 3.53M AF, or 103% of the new capacity). On 2024-07-02 storage drops by 123,103 AF on a dry day while elevation falls only 0.90 ft. This is a reporting-basis change, not a loss of water. **Values are not adjusted.** The break is recorded in `data/series_breaks.csv`, and no target or storage-change feature may span it.

### Other audit notes

- Large single-day increases coincide with same-day heavy NASA POWER precipitation during atmospheric-river storms. For example, FOL rose by 196,577 AF (20.1 pp) on 2022-12-31 with 78 mm of rain. These are kept as physical. The same-day alignment also supports the end-of-day date convention.
- Two isolated one-day spikes (> 2 pp that immediately reverse) were set to missing: DNP 2019-09-26 and EXC 2022-05-22.
- DNP daily values differ from DNP hourly (USGS) values by about 5,000 AF (about 0.25 pp). The daily series comes from a different reporting source; daily is used consistently.

## 2. Weather — NASA POWER daily point API

| item | value |
|---|---|
| Endpoint | `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,PRECTOTCORR,RH2M&community=AG&latitude=<lat>&longitude=<lon>&start=YYYYMMDD&end=YYYYMMDD&format=JSON&time-standard=LST` |
| API version | v2.10.5 (from the response header) |
| Parameters / units | `T2M` 2 m air temperature °C; `PRECTOTCORR` bias-corrected precipitation mm/day; `RH2M` 2 m relative humidity % (units asserted from the response) |
| Time | Daily values in **local solar time (LST)**, keyed `YYYYMMDD` |
| Missing | `fill_value = -999` → missing. Plausibility ranges are also applied (T −40..55 °C, P 0..500 mm, RH 0..100 %). No values were missing or implausible for 2018–2025. |
| Quality flags | **None per value.** The API returns only a source list (`MERRA2`, `POWER` for the archive) |
| Spatial meaning | Value for the reanalysis grid cell containing the dam coordinate. The returned cell elevation is 463 m, versus about 325 m at Shasta Dam. **This is a coarse point/grid-cell approximation, not watershed-average rainfall**, and it misses orographic snowfall and snowmelt timing in the contributing basins. |
| Publication lag (observed 2026-11-21) | About 3 days (last non-fill date was 2026-11-18). Recent days come from **GEOS-IT**, while archive years come from **MERRA-2**. A retrospective series therefore differs from what was available in real time. The 7-day feature lag is a conservative proxy, **not** a reconstruction of as-of-time data. |
| Terms | NASA data, free to use. Requested acknowledgement: *"These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program."* |

## 3. Processed dataset

`data/processed/daily_joined.csv` has one row per station × calendar day, for 6 × 2,922 = 17,532 rows. Columns: `station_id, reservoir_name, date, storage_af, storage_pct, capacity_af, rain_mm, temp_c, humidity_pct, lat, lon`. Storage rows remain blank where missing. Weather is joined on the calendar date (CDEC Pacific day ↔ POWER LST day).

Transformation steps (`src/build_dataset.py`):
1. Validate the raw header and station, sensor, duration and units columns.
2. Parse the date from `DATE TIME`.
3. Convert non-numeric values to missing.
4. Set values ≤ 0 or > 110% of capacity to missing.
5. Collapse exact duplicate dates; set conflicting duplicates to missing.
6. Remove isolated one-day spikes.
7. Reindex to the full calendar.
8. Join POWER weather after fill-value and plausibility masking.

## 4. Forecast-time semantics

- Forecast origin *t* is the date of the latest end-of-day storage value used. In real time this value is published at *t*+1 or *t*+2, so the effective lead time from issue is about 28–29 days for a 30-day target.
- Weather features end at *t*−7.

## 5. Originality check

See `docs/originality_check.md` (added at checkpoint C3).

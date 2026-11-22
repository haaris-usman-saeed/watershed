# Model C feasibility: weather-forecast archives (audited 2026-11-22)

Model C must use weather **forecasts** that existed, or would have existed, at the forecast issue time. It must never use observed future weather. This audit was done before Model C's features were fixed. Every statement below was checked by listing the public AWS buckets and decoding sample files. Where a statement comes from documentation instead, it is marked so.

## Products checked

| Product | Where | Period | Members, cycles | Leads | Verified how |
|---|---|---|---|---|---|
| **GEFSv12 reforecast** | `s3://noaa-gefs-retrospective/GEFSv12/reforecast/` | 2000-2019 (folders 2000 to 2020) | 00Z daily: c00 + p01-p04 (5). Wednesdays: c00 + p01-p10 (11), extending to 35 days per the registry | Days 1-10 (3-hourly, 0.25°) and Days 10-16 (6-hourly) | Listed 2010-01-04 and 2010-01-06. Decoded `apcp_sfc` messages: 1440x721 grid, "0-6 hour acc", units kg m-2 |
| **GEFSv12 operational** | `s3://noaa-gefs-pds/gefs.YYYYMMDD/00/atmos/` | v12 since 2020-09-23 (NWS SCN 20-75); the bucket starts 2017-01-01 (v11 before) | 31 members, 4 cycles/day; control `gec00` used | `pgrb2sp25` 0.25° to 240 h; `pgrb2ap5` 0.5° to 384 h (35 days at 00Z per NOAA) | Listed 2026-01-05, 2026-11-16, 2026-11-22. `f012.idx` has "APCP:surface:6-12 hour acc fcst:ENS=low-res ctl", the same encoding as the reforecast |
| GEFSv13 | not operational | | | | NOAA EMC (September 2025): v13 deferred until after GFSv17; no implementation date found. The 2026 cycles above still have the v12 layout. |

## What is possible

1. **Training (2001-2019): reforecast control member, 00Z on the origin Monday.**
   - These are retrospective forecasts made with the same GEFSv12 model that ran operationally from 2020-09-23. They are labelled `reforecast` and are never described as issued on those dates.
2. **2020-01-01 to 2020-09-23: no GEFSv12 product.** Model C is unavailable for those origins. GEFSv11 is not mixed in.
3. **2020-09-24 onward: operational GEFSv12 control member.**
   - Each file's S3 `Last-Modified` time is recorded.
   - Example: all 64 files of the 2026-11-16 00Z cycle were published by 2026-11-16 06:02 UTC, which is Sunday 22:02 Pacific time.
   - A Monday-origin storage forecast needs Monday's end-of-day storage, which CDEC reports on Tuesday at the earliest. So the 00Z Monday GEFS cycle is always available before such a forecast can be issued.
   - The availability check is enforced per cycle by comparing the recorded publication time with the forecast issue time.
   - **Archive backfills:** 4 early operational cycles (2020-10-19, 2020-12-14, 2021-02-22, 2021-03-01) were uploaded to AWS weeks later. With no evidence of real-time availability, they are excluded.
4. **Live inference uses the same rule:** the 00Z cycle on the origin Monday, with every message present.

## Feature definition (fixed before any Model C training)

- `gefs_p_d1_7_mm`: forecast precipitation summed over lead hours (0, 168], averaged over the reservoir's upstream-area box (`src/v3_basins.py`).
- `gefs_p_d8_16_mm`: the same over (168, 384].
- Built from 64 six-hour APCP accumulation messages. If any message is missing, the cycle is incomplete and Model C gives no forecast for that origin.

## What is not done, and why

- **Days 17-30:** no precipitation forecast is used.
  - The daily reforecast ends at day 16. The 35-day runs exist only for Wednesday 00Z cycles. For a Monday origin, the latest Wednesday cycle would be 5 days old, and using it would add a second issue-time rule.
  - Days 17-30 are left to the model's seasonal inputs. They are never filled with observed precipitation.
  - A Wednesday-based extension is documented as a possible C2 experiment.
- **Ensemble mean and spread:** not used. Only the control member is downloaded (about 20 MB per cycle through byte-range requests). All 5 reforecast members would multiply the download about fivefold, to roughly 100 GB.
- **Forecast temperature:** not used in this first version. It would need a second variable download, and adding it is a separate decision.

## Known limitations

- **Reforecast and operational are not identical.** The reforecast used different initial conditions and only 5 members, while the operational system has 31. Using the control member in both keeps them as close as the archive allows.
- **The upstream-area boxes are coarse rectangles,** not delineated watersheds.
- **This is a working pipeline, not a validated model.** Model C is validated only by the preregistered v3 evaluation (`docs/v3/PREREGISTRATION_V3.md`).

"""v3 upstream-area boxes used to aggregate gridded weather (GEFS forecasts and NASA POWER).

Rule fixed in advance, without looking at any outcome: for each reservoir, the latitude and
longitude bounding box of the reservoir location and every candidate snow station that CDEC
lists for its river basin(s) (data/snow_stations.csv), padded by 0.25 degrees on every side.
It is a coarse proxy for the contributing watershed, not a delineated basin boundary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAD_DEG = 0.25
STATIONS = ["DNP", "EXC", "FOL", "NML", "ORO", "SHA"]


def basin_boxes():
    snow = pd.read_csv(ROOT / "data" / "snow_stations.csv")
    st = pd.read_csv(ROOT / "stations.csv").set_index("station_id")
    out = {}
    for r in STATIONS:
        g = snow[snow.reservoir_id == r]
        lat = list(g.lat) + [float(st.loc[r, "lat"])]
        lon = list(g.lon) + [float(st.loc[r, "lon"])]
        out[r] = {"lat_min": round(min(lat) - PAD_DEG, 4), "lat_max": round(max(lat) + PAD_DEG, 4),
                  "lon_min": round(min(lon) - PAD_DEG, 4), "lon_max": round(max(lon) + PAD_DEG, 4),
                  "n_points_defining": len(lat)}
    return out


def box_mask(lats, lons, box):
    """Boolean mask over a regular lat/lon grid (lons may be 0..360). lats, lons are 1-D axes."""
    lon180 = np.where(lons > 180, lons - 360, lons)
    la = (lats >= box["lat_min"]) & (lats <= box["lat_max"])
    lo = (lon180 >= box["lon_min"]) & (lon180 <= box["lon_max"])
    return la[:, None] & lo[None, :]

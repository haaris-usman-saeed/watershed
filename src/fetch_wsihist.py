"""Download DWR water year hydrologic classification indices (CDEC WSIHIST) and parse them.

Run: python -m src.fetch_wsihist
Raw HTML is saved unchanged to data/raw/wsihist/; parsed classifications go to
data/water_year_types.csv (Sacramento Valley and San Joaquin Valley, W/AN/BN/D/C).
Used ONLY to group evaluation results by wet and dry years, never as a model input.
"""
import re
from io import StringIO

import pandas as pd

from src.fetch import RAW, ROOT, get, save

URL = "https://cdec.water.ca.gov/reportapp/javareports"


def parse(html):
    """Rows of the preformatted WSIHIST report: WY, Sacramento Valley (Oct-Mar, Apr-Jul, WY sum,
    index, type), San Joaquin Valley (Oct-Mar, Apr-Jul, WY sum, index, type)."""
    text = re.sub(r"<[^>]+>", " ", html)
    rows = []
    pat = re.compile(r"^\s*(\d{4})\s+" + r"([\d.]+)\s+" * 4 + r"(W|AN|BN|D|C)\s+" + r"([\d.]+)\s+" * 4 + r"(W|AN|BN|D|C)\s*$")
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            g = m.groups()
            rows.append({"water_year": int(g[0]), "sac_index": float(g[4]), "sac_type": g[5],
                         "sjv_index": float(g[9]), "sjv_type": g[10]})
    if len(rows) < 50:
        raise ValueError("WSIHIST rows not found")
    return pd.DataFrame(rows)


def main():
    path = RAW / "wsihist" / "wsihist.html"
    if not path.exists():
        res = get(URL, {"name": "WSIHIST"}, lambda r: "Sacramento" in r.text or "SACRAMENTO" in r.text)
        save(path, res)
    html = path.read_text(errors="ignore")
    t = parse(html)
    t = t[t.water_year >= 2001]
    t.to_csv(ROOT / "data" / "water_year_types.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()

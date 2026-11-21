"""Independent cross-check of our CDEC daily storage against ResOpsUS (Steyaert et al. 2022).

Run: python -m src.compare_resopsus   (after src.fetch_resopsus and src.build_dataset)
Reads only CSV members of data/raw/resopsus/ResOpsUS.zip (in memory; nothing else in the
archive is loaded or executed). Writes results/v2/resopsus_check/report.md and summary.csv.

ResOpsUS storage sources (attributes/time_series_inventory.csv): FOL = Bureau of Reclamation and
DNP = USGS 11287500 (independent of CDEC); SHA, ORO, NML, EXC = CDEC (same upstream source as
ours, so agreement there checks our processing and date alignment, not the measurement).
Used only for verification; never a model input and never used to modify our data.
"""
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "raw" / "resopsus" / "ResOpsUS.zip"
OUT = ROOT / "results" / "v2" / "resopsus_check"
DAM_IDS = {"SHA": 132, "ORO": 148, "FOL": 182, "NML": 198, "DNP": 210, "EXC": 214}
AF_PER_MCM = 1e6 / 1233.48183754752     # 1 acre-foot = 1233.48183754752 m3
ROUND_TOL_AF = 2.0                      # ResOpsUS MCM has 3 decimals = 1,000 m3 = 0.81 AF
REL_TOL = 5e-6                          # ResOpsUS used a slightly different AF to m3 constant (about 1.5e-6 relative)
OVERLAP = ("2001-01-01", "2020-12-31")


def load_resopsus(z, dam_id):
    df = pd.read_csv(BytesIO(z.read(f"ResOpsUS/time_series_all/ResOpsUS_{dam_id}.csv")), parse_dates=["date"])
    return df.set_index("date").storage * AF_PER_MCM


def match(a, b):
    """Agreement within rounding (2 AF) plus the conversion-constant difference (5e-6 relative)."""
    return (a - b).abs() <= ROUND_TOL_AF + REL_TOL * b.abs()


def yearly_lag_match(ours, theirs):
    out = {}
    for lag in (-1, 0, 1):
        t = theirs.copy()
        t.index = t.index.shift(lag, freq="D")
        j = pd.concat([ours.rename("ours"), t.rename("theirs")], axis=1).dropna()
        out[f"lag{lag:+d}"] = match(j.ours, j.theirs).groupby(j.index.year).mean()
    return pd.DataFrame(out)


def compare(ours, theirs, cap):
    """Statistics on the common days at lags -1, 0, +1 (their date shifted relative to ours)."""
    res = {}
    for lag in (-1, 0, 1):
        t = theirs.copy()
        t.index = t.index.shift(lag, freq="D")
        j = pd.concat([ours.rename("ours"), t.rename("theirs")], axis=1).dropna()
        d = (j.ours - j.theirs).abs()
        res[lag] = {"n": len(j), "match_share": float(match(j.ours, j.theirs).mean()),
                    "median_abs_diff_af": float(d.median()), "p99_abs_diff_af": float(d.quantile(0.99)),
                    "max_abs_diff_af": float(d.max()), "share_gt_0p5pp": float((d > 0.005 * cap).mean()),
                    "median_abs_diff_pp": float(100 * d.median() / cap)}
    return res


def raw_cdec_keyed(sid):
    """Our RAW CDEC rows (before cleaning) as two series: keyed by DATE TIME and keyed by OBS DATE."""
    files = sorted((ROOT / "data" / "raw" / "cdec").glob(f"{sid}_sensor15_D_*.csv"))
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    d["dt"] = pd.to_datetime(d["DATE TIME"].str.strip(), format="%Y%m%d %H%M")
    d["obs"] = pd.to_datetime(d["OBS DATE"].str.strip(), format="%Y%m%d %H%M").dt.normalize()
    d["v"] = pd.to_numeric(d.VALUE.str.strip(), errors="coerce")
    d = d.dropna(subset=["v"])
    return {k: d.drop_duplicates(k).set_index(k).v for k in ("dt", "obs")}


def keying_test(sid, theirs):
    """Share of days on which ResOpsUS equals raw CDEC keyed by DATE TIME vs keyed by OBS DATE."""
    out = {}
    for key, s in raw_cdec_keyed(sid).items():
        j = pd.concat([s.rename("o"), theirs.rename("t")], axis=1).dropna()
        out[key] = float(match(j.o, j.t).mean())
    return out


def main():
    z = zipfile.ZipFile(ZIP)
    daily = pd.read_csv(ROOT / "data/processed/daily_joined.csv", parse_dates=["date"])
    inv = pd.read_csv(BytesIO(z.read("ResOpsUS/attributes/time_series_inventory.csv")))
    src = inv.set_index("DAM_ID").DATA_SOURCE
    rows, worst, lines, yearly = [], [], [], []
    for sid, dam in DAM_IDS.items():
        g = daily[daily.station_id == sid].set_index("date")
        g = g.loc[OVERLAP[0]:OVERLAP[1]]
        cap = float(g.capacity_af.iloc[0])
        theirs = load_resopsus(z, dam).loc[OVERLAP[0]:OVERLAP[1]]
        st = compare(g.storage_af.dropna(), theirs.dropna(), cap)
        best = max(st, key=lambda k: (st[k]["match_share"], -st[k]["median_abs_diff_af"]))
        kt = keying_test(sid, theirs.dropna()) if src.get(dam) == "CDEC" else {"dt": np.nan, "obs": np.nan}
        rows.append({"station_id": sid, "resopsus_dam_id": dam, "resopsus_storage_source": src.get(dam),
                     "independent_of_cdec": src.get(dam) not in ("CDEC",), "best_lag_days": best,
                     "raw_cdec_keyed_by_date_time_match": kt["dt"], "raw_cdec_keyed_by_obs_date_match": kt["obs"],
                     **{f"lag{k:+d}_{m}": v for k, s in st.items() for m, v in s.items()
                        if m in ("n", "match_share", "median_abs_diff_af", "median_abs_diff_pp",
                                 "p99_abs_diff_af", "max_abs_diff_af", "share_gt_0p5pp")}})
        y = yearly_lag_match(g.storage_af.dropna(), theirs.dropna()).round(3)
        y.insert(0, "station_id", sid)
        yearly.append(y.reset_index(names="year"))
        j = pd.concat([g.storage_af.rename("ours_af"), theirs.rename("resopsus_af")], axis=1).dropna()
        j["diff_af"] = j.ours_af - j.resopsus_af
        j["diff_pp"] = 100 * j.diff_af / cap
        top = j.reindex(j.diff_af.abs().sort_values(ascending=False).index).head(5)
        for d, r in top.iterrows():
            worst.append({"station_id": sid, "date": d.strftime("%Y-%m-%d"), "ours_af": round(r.ours_af),
                          "resopsus_af": round(r.resopsus_af), "diff_af": round(r.diff_af), "diff_pp": round(r.diff_pp, 3)})
        # days we set to missing (artifact spikes / unparseable) that ResOpsUS still reports
        ours_missing = g.storage_af.isna()
        theirs_on_missing = theirs.reindex(g.index[ours_missing]).dropna()
        lines.append(f"- {sid}: our missing days in overlap = {int(ours_missing.sum())}; ResOpsUS has a value on "
                     f"{len(theirs_on_missing)} of them" + (f" ({', '.join(d.strftime('%Y-%m-%d') for d in theirs_on_missing.index[:6])})"
                                                           if len(theirs_on_missing) else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    yl = pd.concat(yearly, ignore_index=True)
    yl.to_csv(OUT / "match_share_by_year_and_lag.csv", index=False)
    w = pd.DataFrame(worst)
    w.to_csv(OUT / "largest_differences.csv", index=False)
    view = s[["station_id", "resopsus_storage_source", "independent_of_cdec", "best_lag_days", "lag+0_n",
              "lag+0_match_share", "lag-1_match_share", "lag+1_match_share", "lag+0_median_abs_diff_af",
              "lag+0_median_abs_diff_pp", "lag+0_p99_abs_diff_af", "lag+0_share_gt_0p5pp",
              "raw_cdec_keyed_by_date_time_match", "raw_cdec_keyed_by_obs_date_match"]]
    md = ["# Cross-check of WatershedShift storage against ResOpsUS", "",
          "ResOpsUS: Steyaert, Condon, Turner and Voisin (2022), *Scientific Data* 9, 34, doi:10.1038/s41597-022-01134-7; "
          "Zenodo record 5893641 (ResOpsUS.zip, CC-BY 4.0; SHA-256 in data/raw_manifest.jsonl). Storage converted from "
          "million cubic meters to acre-feet (1 MCM = 810.7132 AF). Comparison window "
          f"{OVERLAP[0]} to {OVERLAP[1]} (ResOpsUS ends about 2021-01-01). Our side: data/processed/daily_joined.csv "
          "after cleaning. Generated by `python -m src.compare_resopsus`.", "",
          "Lag k means the ResOpsUS date is shifted by k days before matching (lag 0 = same calendar date). "
          "`match_share` is the share of common days that agree within 2 AF (rounding of the 3-decimal MCM values) plus "
          "5e-6 x storage (ResOpsUS used a slightly different acre-foot to cubic-meter constant; SHA and NML differ by a "
          "near-constant 1.5e-6 relative).", "",
          view.round(4).to_markdown(index=False), "",
          "The last two columns (CDEC-sourced ResOpsUS records only) compare ResOpsUS with our RAW CDEC rows keyed by "
          "`DATE TIME` (what WatershedShift uses) and by `OBS DATE`.", "",
          "## Largest differences at lag 0 (top 5 per reservoir)", "", w.to_markdown(index=False), "",
          "## Days our pipeline treats as missing", "", *lines, "",
          "## Share of days matching, by year and lag (ORO, DNP, EXC)", "",
          yl[yl.station_id.isin(["ORO", "DNP", "EXC"])].pivot(index="year", columns="station_id",
                                                               values=["lag-1", "lag+0"]).to_markdown(), ""]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

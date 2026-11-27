"""v5 dataset and training-row accounting (aggregates only; nothing here contains a CDEC value).

Run: python -m src.v5.accounting    # -> results/v5/dataset/{accounting.json, missing_by_variable.csv,
                                    #    gefs_availability.csv, flow_availability.csv, fold_rows.csv}

Terms (never interchanged)
- reservoir-day: one calendar row of the daily panel (observed or not);
- Monday row: a v2 evaluation/training row (Monday origin with storage, dam weather and target present);
- labelled daily origin: a reservoir-day whose 30-day target exists (same series segment, both storage values observed);
- sampled training sequence: a labelled daily origin in a network's fitting set (every 2nd day) or early-stopping set;
- independent outcomes: 30-day targets overlap (Monday rows share 23 of 30 days with the next week's), so the number of
  non-overlapping 30-day windows per reservoir is reported as a rough upper bound on independent information.
"""
import json

import numpy as np
import pandas as pd

from src.v5 import DEV_END, RESULTS, RETRO, STATIONS
from src.v5.data import FLOW, GEFS_COLS, GROUPS, load_features, load_panel
from src.v5.tabular import folds

OUT = RESULTS / "dataset"
RAW_VARS = ["storage_af", "swe_in", "rain_mm", "temp_c", "rh_pct", "rain_box_mm", "temp_box_c", "rh_box_pct",
            "inflow_cfs", "outflow_cfs", "gefs_p_d1_7_mm"]
TAB_INPUTS = [c for g in ("S", "N", "Wd", "Wu", "G", "Q", "C") for c in GROUPS[g] if c != "swe_anom_in"]


def population_rows(feat):
    v2 = feat[feat.v2_row]
    pops = {}
    for fid, tr_end, ev_s, ev_e, tg_e in folds("dev") + folds("retro"):
        pops[fid] = v2[(v2.date >= ev_s) & (v2.date <= ev_e) & (v2.target_date <= tg_e)]
    return pops


def main():
    from src.v5.sequences import SeqData, split_at
    OUT.mkdir(parents=True, exist_ok=True)
    full = load_panel(pd.Timestamp("2026-11-21"))
    dev = full[full.date < DEV_END]
    feat = load_features(None)
    acc = {"terms": __doc__.split("Terms (never interchanged)")[1].strip()}
    acc["reservoir_days"] = {"full_panel": int(len(full)), "development_panel": int(len(dev)),
                             "period_full": [str(full.date.min().date()), str(full.date.max().date())],
                             "period_development": [str(dev.date.min().date()), str(dev.date.max().date())]}
    # missing by variable (daily)
    rows = []
    for name, p in (("full panel 2001-01-01..2026-11-20", full), ("development panel 2001-01-01..2023-12-31", dev)):
        for v in RAW_VARS:
            rows.append({"panel": name, "variable": v, "reservoir_days": len(p), "observed": int(p[v].notna().sum()),
                         "missing": int(p[v].isna().sum()), "observed_share": round(float(p[v].notna().mean()), 4)})
    # missing by tabular input on Monday rows, per population
    pops = population_rows(feat)
    groups = {"tuning WY2012-2015": [f"WY{w}" for w in range(2012, 2016)],
              "confirmation WY2016-2023": [f"WY{w}" for w in range(2016, 2024)], "HELD": ["HELD"], "T26": ["T26"]}
    for gname, fids in groups.items():
        r = pd.concat([pops[f] for f in fids])
        for v in TAB_INPUTS:
            rows.append({"panel": f"Monday rows, {gname}", "variable": v, "reservoir_days": len(r),
                         "observed": int(r[v].notna().sum()), "missing": int(r[v].isna().sum()),
                         "observed_share": round(float(r[v].notna().mean()), 4)})
    pd.DataFrame(rows).to_csv(OUT / "missing_by_variable.csv", index=False)
    v2 = feat[feat.v2_row]
    labelled_full = feat.target_delta_pp.notna()
    acc["monday_rows"] = {
        "all_v2_rows": int(len(v2)), "development_target_before_2024": int(((v2.target_date < DEV_END)).sum()),
        "by_population": {k: int(len(v)) for k, v in pops.items()},
        "unused_gap_rows": int(len(v2) - (v2.target_date < DEV_END).sum() - len(pops["HELD"]) - len(pops["T26"])),
        "gap_rule": "Monday rows whose 30-day target crosses a population boundary (Dec 2023 -> 2024, Dec 2025 -> 2026) "
                    "belong to no population",
        "all_mondays_in_panel": int((full.date.dt.dayofweek == 0).sum())}
    acc["labelled_daily_origins"] = {
        "full_panel": int(labelled_full.sum()),
        "development_target_before_2024": int((labelled_full & (feat.target_date < DEV_END)).sum())}
    # GEFS availability
    g = []
    for wy, r in v2.assign(wy=v2.date.dt.year + (v2.date.dt.month >= 10)).groupby("wy"):
        g.append({"water_year": int(wy), "monday_rows": len(r), "gefs_available": int(r.gefs_p_d1_7_mm.notna().sum()),
                  "reforecast": int((r.gefs_source == "reforecast").sum()), "operational": int((r.gefs_source == "operational").sum())})
    pd.DataFrame(g).to_csv(OUT / "gefs_availability.csv", index=False)
    lab = feat[labelled_full]
    acc["gefs"] = {"monday_rows_available_share": round(float(v2.gefs_p_d1_7_mm.notna().mean()), 4),
                   "labelled_daily_origins_available_share": round(float(lab.gefs_p_d1_7_mm.notna().mean()), 4),
                   "daily_age_distribution": {str(int(k)): int(v) for k, v in lab.gefs_age_days.value_counts().sort_index().items()},
                   "gap": "no GEFSv12 product for 2020-01-06..2020-09-21; 4 late-published operational cycles excluded (v3 rule)",
                   "source_rule": "reforecast 2001-2019 (retrospective), operational from 2020-09-28"}
    # flow availability
    f = []
    for (st, yr), r in full.groupby([full.station_id, full.date.dt.year]):
        f.append({"station_id": st, "year": int(yr), "days": len(r), "inflow_observed": int(r.inflow_cfs.notna().sum()),
                  "outflow_observed": int(r.outflow_cfs.notna().sum())})
    pd.DataFrame(f).to_csv(OUT / "flow_availability.csv", index=False)
    acc["flows"] = {k: {"inflow_share": round(float(v.inflow_cfs.notna().mean()), 4),
                        "outflow_share": round(float(v.outflow_cfs.notna().mean()), 4)}
                    for k, v in (("2001-2023", dev), ("2024-2026", full[full.date >= DEV_END]))}
    acc["flows"]["source"] = "CDEC sensors 76/23: 2001-2023 v4 download (2026-11-24), 2024-01-01..2026-11-20 v5 download (2026-11-27)"
    # common availability (all 28 non-one-hot inputs observed; swe_anom observed iff swe observed)
    comp = {}
    for gname, fids in groups.items():
        r = pd.concat([pops[x] for x in fids])
        comp[gname] = {"rows": int(len(r)), "gefs_available": int(r.gefs_p_d1_7_mm.notna().sum()),
                       "flows_complete": int(r[FLOW].notna().all(axis=1).sum()),
                       "all_inputs_observed": int(r[TAB_INPUTS].notna().all(axis=1).sum())}
    acc["common_availability"] = comp
    # variables and features
    acc["variables"] = {"raw_daily": ["storage_af", "capacity_af", "swe_in", "rain_mm", "temp_c", "rh_pct", "rain_box_mm",
                                      "temp_box_c", "rh_box_pct", "inflow_cfs", "outflow_cfs"],
                        "raw_weekly_forecast": GEFS_COLS, "n_raw": 13,
                        "tabular_engineered": [c for g in ("S", "N", "Wd", "Wu", "G", "Q", "C") for c in GROUPS[g]] +
                                              [f"station_{s}" for s in sorted(STATIONS)],
                        "n_tabular_engineered": 29, "daily_tabular_extra": ["gefs_age_days"],
                        "network_per_step": "10 scaled channels x value*mask, 10 masks, doy sin/cos = 22",
                        "network_static": "log capacity (+ 4 GEFS static inputs in Experiment A)",
                        "network_decoder_n4": "doy sin/cos of each future day + lead k/30 = 3"}
    # chronological coverage and independence
    cov = {}
    for gname, fids in groups.items():
        r = pd.concat([pops[x] for x in fids])
        span = (r.date.max() - r.date.min()).days + 1
        cov[gname] = {"origins": [str(r.date.min().date()), str(r.date.max().date())],
                      "targets": [str(r.target_date.min().date()), str(r.target_date.max().date())],
                      "monday_rows": int(len(r)), "distinct_origin_dates": int(r.date.nunique()),
                      "non_overlapping_30d_windows_per_reservoir_approx": round(span / 30, 1),
                      "non_overlapping_30d_windows_total_approx": int(round(span / 30 * r.station_id.nunique()))}
    acc["coverage"] = cov
    # per-fold rows (tabular weekly, tabular daily, network fit / early-stopping, evaluation)
    seq_dev = SeqData(dev, DEV_END)
    seq_full = SeqData(full, pd.Timestamp("2026-11-21"))
    fr = []
    for fid, tr_end, ev_s, ev_e, tg_e in folds("dev") + folds("retro"):
        sub = feat if fid in RETRO else feat[(feat.date < DEV_END) & (feat.target_date < DEV_END)]
        tr, te, daily = _rows(sub, tr_end, ev_s, ev_e, tg_e)
        s = seq_full if fid in RETRO else seq_dev
        fit, val = split_at(s.samples, tr_end, stride=2)
        fr.append({"fold": fid, "train_end_exclusive": str(tr_end.date()), "eval_origins": f"{ev_s.date()}..{ev_e.date()}",
                   "tabular_weekly_train": int(len(tr)), "tabular_daily_train": int(len(daily)),
                   "network_fit_sequences": int(len(fit)), "network_early_stopping_sequences": int(len(val)),
                   "network_fit_period": f"{s.samples.date.loc[fit].min().date()}..{s.samples.target_date.loc[fit].max().date()}",
                   "network_early_stopping_period": f"{s.samples.date.loc[val].min().date()}..{s.samples.target_date.loc[val].max().date()}",
                   "evaluation_rows": int(len(te))})
    pd.DataFrame(fr).to_csv(OUT / "fold_rows.csv", index=False)
    acc["labelled_daily_origins"]["sequence_samples_development"] = int(len(seq_dev.samples))
    acc["labelled_daily_origins"]["sequence_samples_full"] = int(len(seq_full.samples))
    (OUT / "accounting.json").write_text(json.dumps(acc, indent=1))
    print(json.dumps({k: v for k, v in acc.items() if k not in ("terms",)}, indent=1)[:6000])


def _rows(feat, tr_end, ev_s, ev_e, tg_e):
    v2 = feat[feat.v2_row]
    tr = v2[v2.target_date < tr_end]
    te = v2[(v2.date >= ev_s) & (v2.date <= ev_e) & (v2.target_date <= tg_e)]
    daily = feat[feat.target_delta_pp.notna() & (feat.target_date < tr_end)]
    daily = daily[((daily.date - pd.Timestamp("2001-01-01")).dt.days % 2) == 0]
    return tr, te, daily


if __name__ == "__main__":
    main()

"""Descriptive diagnostics of the frozen v1 held-out predictions.

Run: python -m src.v1_diagnostics
Reads results/v1/local_full/test_predictions.csv (local copy of v1 outputs) and
writes results/v1/diagnostics.md and results/v1/diagnostics.json.

These diagnostics DESCRIBE v1 only. They are not used to choose any v2 setting.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.evaluation import block_bootstrap_diff, mae

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results" / "v1"
MODELS = {"GBM": "pred_delta_pp", "Ridge": "pred_ridge_pp", "Persistence": "baseline_persistence_pp",
          "30-day trend": "baseline_trend_pp", "Monthly seasonal": "baseline_seasonal_pp"}
ORO_BREAK = pd.Timestamp("2024-07-02")


def season(month):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA"}.get(month, "SON")


def table(df, key):
    rows = []
    for k, g in df.groupby(key, observed=True):
        inside = ((g.target_delta_pp >= g.lower_90_pp) & (g.target_delta_pp <= g.upper_90_pp)).mean()
        rows.append({key: k, "n": len(g), **{m: round(mae(g.target_delta_pp, g[c]), 3) for m, c in MODELS.items()},
                     "interval_90_coverage": round(float(inside), 3)})
    return pd.DataFrame(rows)


def main():
    p = pd.read_csv(V1 / "local_full" / "test_predictions.csv", parse_dates=["date", "target_date"])
    p["season"] = p.date.dt.month.map(season)
    p["fullness"] = pd.cut(p.storage_pct, [0, 40, 60, 80, 200], labels=["<40%", "40-60%", "60-80%", ">=80%"])
    p["regime"] = np.where(p.station_id != "ORO", "other stations",
                           np.where(p.target_date < ORO_BREAK, "ORO old storage table", "ORO new storage table"))
    out = {k: table(p, k) for k in ["station_id", "season", "fullness", "regime"]}
    boots = {}
    for b in (5, 9, 13):
        r = block_bootstrap_diff(p, "pred_delta_pp", "baseline_seasonal_pp", block=b)
        boots[f"{b}_weeks"] = r
    md = ["# v1 held-out diagnostics (descriptive only)", "",
          "Source: frozen v1 predictions (`git tag v1-results`, commit 01e43cb), test origins 2024-01-01 to 2025-12-01, n = "
          f"{len(p)}. MAE in percentage points (pp) of capacity. These tables describe v1 and were not used to configure v2.",
          "", "Season is by forecast origin month. Fullness is storage at origin as % of capacity. "
          "For ORO, v1 used the post-2024 capacity (3,424,753 AF) as the denominator for all dates.", ""]
    for k, t in out.items():
        md += [f"## By {k}", "", t.to_markdown(index=False), ""]
    md += ["## v1 GBM minus monthly seasonal baseline: moving-block bootstrap (2,000 replicates)", "",
           "| block length | estimate | 95% CI low | 95% CI high | share of replicates GBM better |", "|---|---|---|---|---|"]
    for b, r in boots.items():
        md.append(f"| {b.replace('_', ' ')} | {r['estimate']:.3f} | {r['ci95_low']:.3f} | {r['ci95_high']:.3f} | "
                  f"{r['share_better']:.3f} |")
    md += ["", "## Observations", "",
           "- See the tables above. Interpretation is in STATUS.md (Checkpoint A)."]
    (V1 / "diagnostics.md").write_text("\n".join(md) + "\n")
    (V1 / "diagnostics.json").write_text(json.dumps(
        {**{k: t.to_dict(orient="records") for k, t in out.items()}, "bootstrap_gbm_minus_seasonal": boots}, indent=2,
        default=str))
    for k, t in out.items():
        print(t.to_string(index=False))
    print(json.dumps(boots, indent=1))


if __name__ == "__main__":
    main()

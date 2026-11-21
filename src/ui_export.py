"""Additive UI handoff metadata and a read-only UI export package (contract schema v1.0, unchanged).

Run: python -m src.ui_export            enrich outputs/metrics.json and outputs/feature_catalog.json in place
                                        (additive keys only), check them against the frozen results, and write
                                        dist/outputs_v2_ui.zip (+ .sha256)
     python -m src.ui_export --verify   re-open the ZIP and check its contents, contract and consistency

Nothing is retrained or re-evaluated. The added values are derived from the frozen configuration
(config/v2_frozen.json, src/models.py FEATURE_SETS) and from the frozen test results
(results/v2/final/metrics.json). Re-run this after any identical-configuration rerun of
src.train_v2 --final, because that rewrites outputs/ without these additive keys.
The ZIP holds CDEC-derived values: it stays local (dist/ is git-ignored) and must not be shared
publicly until the owner has confirmed the CDEC reuse terms.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import pandas as pd

from src.compare_metrics import walk
from src.contract import validate_dir
from src.features import SEASON, SNOW, STORAGE, WEATHER
from src.models import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FROZEN = ROOT / "results" / "v2" / "final" / "metrics.json"
CONFIG = ROOT / "config" / "v2_frozen.json"
ZIP = ROOT / "dist" / "outputs_v2_ui.zip"
FIVE = ["stations.csv", "history_daily.csv", "test_predictions.csv", "metrics.json", "feature_catalog.json"]
RELIABLE_MIN_COVERAGE = 0.85
ADDED_KEYS = {"primary_model_features", "primary_model_feature_groups", "backtest", "interval_reliability"}
CAVEAT = ("Retrospective UI warning threshold only: interval_reliable is true when the frozen 2024-2025 test coverage "
          "of the nominal 90% interval was at least 0.85 for that reservoir. It is not proof of future reliability, "
          "not a new calibration method and not a scientific certification; the intervals themselves are unchanged.")


def primary_features(stations):
    cfg = json.loads(CONFIG.read_text())
    cols = FEATURE_SETS["storage_snow"]
    assert cols == [c for c in cfg["primary_model"]["features"] if c != "station one-hot"]
    assert not set(cols) & set(WEATHER)
    one_hot = [f"station_{s}" for s in sorted(stations)]
    groups = {"storage": [c for c in cols if c in STORAGE],
              "snow_water_equivalent": [c for c in cols if c in SNOW or c == "swe_anom_in"],
              "seasonality": [c for c in cols if c in SEASON], "reservoir_identity": one_hot}
    assert sum(map(len, groups.values())) == len(cols) + len(one_hot)
    return cols + one_hot, groups


def enrich(metrics, catalog, preds):
    frozen = json.loads(FROZEN.read_text())
    base = copy.deepcopy({k: v for k, v in metrics.items() if k not in ADDED_KEYS})
    for s in base.get("per_station", {}).values():
        s.pop("interval_reliable", None)
    diffs = walk(base, {k: v for k, v in frozen.items() if k != "frozen_hashes"})
    if diffs:
        sys.exit("outputs/metrics.json differs from the frozen results/v2/final/metrics.json:\n" + "\n".join(diffs))
    # recompute from predictions as an independent check of the exported file
    assert len(preds) == frozen["test_count"] == 593
    assert abs((preds.target_delta_pp - preds.pred_delta_pp).abs().mean() - frozen["ML"]) < 1e-9
    inside = (preds.target_delta_pp >= preds.lower_90_pp) & (preds.target_delta_pp <= preds.upper_90_pp)
    cov = inside.groupby(preds.station_id).mean()
    for s, c in cov.items():
        assert abs(c - frozen["per_station"][s]["interval_90_coverage"]) < 1e-12

    cols, groups = primary_features(frozen["per_station"].keys())
    used = {f["name"] for f in catalog["features"] if f["used_by_primary_model"]}
    assert used == set(FEATURE_SETS["storage_snow"]), used
    assert all(not f["used_by_primary_model"] for f in catalog["features"] if f["name"] in WEATHER)
    reliable = {s: bool(v["interval_90_coverage"] >= RELIABLE_MIN_COVERAGE) for s, v in sorted(frozen["per_station"].items())}
    assert reliable["FOL"] is False

    m = dict(base)
    for s, flag in reliable.items():
        m["per_station"][s]["interval_reliable"] = flag
    m["primary_model_features"] = cols
    m["primary_model_feature_groups"] = groups
    m["backtest"] = {
        "is_live_forecast": False, "evaluation_type": "historical backtest",
        "evaluation_period_start": frozen["split"]["test_start"],
        "evaluation_period_end": frozen["split"]["test_target_end_inclusive"],
        "first_origin_date": str(pd.to_datetime(preds.date).min().date()),
        "last_origin_date": str(pd.to_datetime(preds.date).max().date()),
        "description": ("2024-2025 is a historical evaluation period: every prediction was made retrospectively for past "
                        "dates whose outcomes are known. This is not a live or operational forecast."),
    }
    m["interval_reliability"] = {"threshold_min_coverage": RELIABLE_MIN_COVERAGE,
                                 "rule": "interval_reliable = per_station[s].interval_90_coverage >= 0.85",
                                 "by_station": reliable, "caveat": CAVEAT}
    c = dict(catalog)
    c["primary_model_features"] = cols
    return m, c


def readme(m):
    cov = {s: v["interval_90_coverage"] for s, v in m["per_station"].items()}
    best = m["bootstrap"]["best_baseline"]
    unreliable = [s for s, ok in m["interval_reliability"]["by_station"].items() if not ok]
    d = m["bootstrap"]["mae_diff_ml_minus_best_baseline"]
    return f"""# WatershedShift v2 UI export (read-only)

These five files are **real historical backtest outputs** of the frozen, preregistered v2 run
(`data_mode: "real"`, `model_version: "v2"`, contract schema {m['schema_version']}). They are not synthetic and not a
live forecast: 2024-2025 is a historical evaluation period ({m['backtest']['first_origin_date']} to
{m['backtest']['last_origin_date']} origins, `metrics.json` key `backtest`).

Do not edit these files. They contain values derived from CDEC records, so do not publish them or commit them
to a public repository until the owner confirms the CDEC reuse terms.

## Point Streamlit at them

```bash
unzip outputs_v2_ui.zip -d /path/to/watershed
WATERSHED_DATA_DIR=outputs_v2_ui streamlit run app.py
```

Alternatively, copy the five files into `outputs/` and run `streamlit run app.py`.

## Primary model inputs

The primary model is HistGradientBoosting "HGB SWE no weather". Its inputs are listed in `metrics.json` and
`feature_catalog.json` under `primary_model_features`, with the same columns grouped under
`metrics.json` `primary_model_feature_groups`:
{chr(10).join(f'- {g}: ' + ', '.join(v) for g, v in m['primary_model_feature_groups'].items())}

## NASA POWER weather is context only

NASA POWER weather (`rain_mm`, `temp_c` and `humidity_pct` in `history_daily.csv`; the `rain_*`, `temp_*` and
`humidity_*` catalog entries) is historical context. It is **not an input to the primary model**. The catalog
marks it `used_by_primary_model: false`. Weather is used only by the secondary models ("HGB (v1 features)",
"HGB + SWE", "Ridge").

## Strongest baseline

Read `metrics.json["bootstrap"]["best_baseline"]`, which is "{best}". Its test MAE is
`metrics.json["{best}"]` ({m[best]:.2f} pp), and its predictions are column `baseline_doy_pp` in
`test_predictions.csv`. The monthly seasonal baseline ("Historical seasonal", `baseline_seasonal_pp`,
{m['Historical seasonal']:.2f} pp) is kept for v1 compatibility, but it is not the strongest baseline.

Sign convention: the differences in `bootstrap.by_comparison_and_block` are **ML minus baseline**, so a negative
value means ML has lower error. The headline is ML {m['ML']:.2f} pp against {best} {m[best]:.2f} pp, a difference of
{d['estimate']:.2f} pp with a 95% CI of [{d['ci95_low']:.2f}, {d['ci95_high']:.2f}] (5-week blocks).

## Interval reliability warning (must be shown)

`metrics.json["per_station"][s]["interval_reliable"]` is false for: {', '.join(unreliable)}. For Folsom (FOL), only
{cov['FOL']:.0%} of 2024-2025 outcomes fell inside the nominal 90% interval. Show a visible warning wherever
Folsom's interval band appears. {CAVEAT}

Per-reservoir coverage: {', '.join(f'{s} {v:.2f}' for s, v in sorted(cov.items()))}.

## Notes from running the frontend checker (`app/check_contract.py`) on this export

- **Seasonal comparison.** The checker's verdict compares ML with "Historical seasonal", the monthly baseline at
  {m['Historical seasonal']:.2f} pp. Use the strongest baseline above instead.
- **Bootstrap block.** The checker reported "bootstrap key 'bootstrap' present but not parseable", which led it to
  say that uncertainty was not reported. The CI is present:
  - headline: `bootstrap.mae_diff_ml_minus_best_baseline` = `{{estimate, ci95_low, ci95_high}}`, ML minus baseline;
  - all block lengths: `bootstrap.by_comparison_and_block["ML minus {best}"]["5_weeks" | "9_weeks" | "13_weeks"]`.
- **Percent of capacity.** For the percent-full display, use `storage_pct` from `history_daily.csv`, which is based on
  the capacity in force on each date. `stations.csv` `capacity_af` is the current capacity. Oroville's capacity
  changed from 3,537,577 to 3,424,753 AF on 2024-07-02, so earlier Oroville storage divided by the current capacity
  can exceed 100%. This is the source of the checker's "storage above 102% of capacity_af" note (239 Oroville days
  before 2024-07-02).
- **Interval half-width.** `interval_90_halfwidth_pp` is the mean half-width over test rows. The half-width varies
  by season (`interval_90_halfwidth_by_season_pp`), so it differs from the median of the predictions.

## Disclosure to display

Display `metrics.json["note"]` verbatim. In particular, the 2024-2025 period was also used for v1, so the v2 evaluation
is additional evidence rather than a fully independent replication.
"""


def build_zip(files, out):
    """Deterministic ZIP: sorted entries, fixed timestamp, fixed permissions."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(f"outputs_v2_ui/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[name])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf.getvalue())
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    out.with_suffix(".zip.sha256").write_text(f"{digest}  {out.name}\n")
    return digest


def verify(path=ZIP):
    side = Path(str(path) + ".sha256")
    problems = []
    if side.exists() and side.read_text().split()[0] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
        problems.append("archive SHA-256 does not match .sha256 file")
    with zipfile.ZipFile(path) as z:
        names = sorted(n.split("/", 1)[1] for n in z.namelist())
        if names != sorted(FIVE + ["README.md"]) or any(not n.startswith("outputs_v2_ui/") for n in z.namelist()):
            problems.append(f"unexpected contents {names}")
        with tempfile.TemporaryDirectory() as d:
            for n in z.namelist():
                (Path(d) / n.split("/", 1)[1]).write_bytes(z.read(n))
            problems += validate_dir(d)
            m = json.loads((Path(d) / "metrics.json").read_text())
            frozen = json.loads(FROZEN.read_text())
            base = copy.deepcopy({k: v for k, v in m.items() if k not in ADDED_KEYS})
            for s in base["per_station"].values():
                s.pop("interval_reliable", None)
            problems += walk(base, {k: v for k, v in frozen.items() if k != "frozen_hashes"})
            if m["per_station"]["FOL"].get("interval_reliable") is not False:
                problems.append("FOL interval_reliable must be false")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        problems = verify()
        print("\n".join(problems) if problems else f"{ZIP.relative_to(ROOT)}: OK (6 files, contract valid, matches frozen metrics)")
        sys.exit(1 if problems else 0)
    metrics = json.loads((OUT / "metrics.json").read_text())
    catalog = json.loads((OUT / "feature_catalog.json").read_text())
    preds = pd.read_csv(OUT / "test_predictions.csv")
    m, c = enrich(metrics, catalog, preds)
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2))
    (OUT / "feature_catalog.json").write_text(json.dumps(c, indent=2))
    errs = validate_dir(OUT)
    if errs:
        sys.exit("contract violated: " + "; ".join(errs))
    files = {n: (OUT / n).read_bytes() for n in FIVE}
    files["README.md"] = readme(m).encode()
    digest = build_zip(files, ZIP)
    problems = verify()
    if problems:
        sys.exit("verification failed: " + "; ".join(problems))
    print(f"{ZIP.relative_to(ROOT)} sha256={digest} ({ZIP.stat().st_size / 1e6:.2f} MB); outputs/ enriched; contract OK")


if __name__ == "__main__":
    main()

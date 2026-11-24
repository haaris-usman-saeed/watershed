"""Render the v4 development result files as Markdown tables (no numbers are typed by hand).

Run: python -m src.v4.report > docs/v4/RESULTS_TABLES.md
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "results" / "v4" / "development"
NAMES = {"persistence": "Persistence", "trend_30d": "30-day trend", "monthly_seasonal": "Monthly seasonal",
         "doy_seasonal": "Smoothed DOY seasonal", "ridge_ar": "Ridge autoregression", "hgb_v2": "HGB (frozen v2 config)",
         "lstm_ens": "N1 shared LSTM (3-seed ensemble)", "gru_ens": "N2 shared GRU (3-seed ensemble)",
         "tcn_ens": "N3 TCN (3-seed ensemble)"}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def context_section(s):
    rows = []
    for L, r in s["results"].items():
        fi = r["fold_info"]
        rows.append([L, f"{r['pooled_mae_pp']:.3f}", round(np.mean([v["best_epoch"] for v in fi.values()]), 1),
                     round(sum(v["train_seconds"] for v in fi.values())), next(iter(fi.values()))["n_params"]])
    return ("## History length (tuning folds WY2012-2015, N1 LSTM, seed 0)\n\n"
            + table(["History (days)", "Pooled MAE (pp)", "Mean best epoch", "Train seconds (4 folds)", "Parameters"], rows)
            + f"\n\nSelected by the predetermined rule ({s['rule']}): **{s['selected_length']} days**.\n")


def compare_section(s):
    mae = s["pooled_mae_pp"]
    order = sorted(mae, key=mae.get)
    out = [f"## Confirmation folds WY2016-2023 ({s['n_rows']} identical Monday rows, history {s['history_length']} days)\n",
           table(["Model", "Pooled MAE (pp)"], [[NAMES.get(k, k), f"{mae[k]:.3f}"] for k in order])]
    seeds = s["seed_mae_pp"]
    out += ["\n### Seed variability\n", table(["Model", "Per-seed MAE (pp)", "Mean", "SD"],
            [[NAMES.get(f"{k}_ens", k).replace(" (3-seed ensemble)", ""), ", ".join(f"{v:.3f}" for v in d["per_seed"]),
              f"{d['mean']:.3f}", f"{d['sd']:.3f}" if d["sd"] is not None else "n/a"] for k, d in seeds.items()])]
    out.append("\n### Paired differences (moving-block bootstrap over origin dates; negative = neural model better)\n")
    rows = []
    for k, d in s["block_bootstrap"].items():
        cis = [f"[{d[b]['ci95_low']:+.3f}, {d[b]['ci95_high']:+.3f}]" for b in ("5_weeks", "9_weeks", "13_weeks")]
        supported = all(d[b]["ci95_high"] < 0 for b in d)
        rows.append([k.replace("_ens", " ensemble"), f"{d['5_weeks']['estimate']:+.3f}"] + cis + ["yes" if supported else "no"])
    out.append(table(["Comparison", "MAE difference (pp)", "95% CI, 5 wk", "95% CI, 9 wk", "95% CI, 13 wk",
                      "Supported (all CIs < 0)"], rows))
    cols = [c for c in order if c in ("hgb_v2", "doy_seasonal", "ridge_ar") or c.endswith("_ens")]
    for title, key in (("reservoir", "mae_by_station"), ("water year", "mae_by_water_year"), ("season", "mae_by_season"),
                       ("water-year type (Sacramento or San Joaquin index, by reservoir)", "mae_by_water_year_type"),
                       ("regime (observed 30-day change)", "mae_by_regime")):
        d = s[key]
        out.append(f"\n### MAE by {title} (pp)\n")
        out.append(table([title.split(" (")[0].capitalize()] + [NAMES.get(c, c) for c in cols],
                         [[g] + [f"{d[g][c]:.3f}" for c in cols] for g in d]))
    out.append(f"\nRegime counts: {s['regime_counts']}\n")
    for k, iv in s["intervals"].items():
        a = iv["all"]
        out.append(f"\n### Probabilistic output, {NAMES.get(k + '_ens', k)}\n")
        out.append(table(["Nominal", "Coverage", "Mean width (pp)"],
                         [[f"{n}%", f"{a[n]['coverage']:.3f}", f"{a[n]['mean_width_pp']:.2f}"] for n in ("50", "80", "90", "95")]))
        out.append(f"\nMean pinball loss {a['mean_pinball_pp']:.3f} pp; approximate CRPS {a['approx_crps_pp']:.3f} pp.\n")
        out.append(table(["Reservoir", "90% coverage", "90% mean width (pp)", "Pinball (pp)"],
                         [[st, f"{v['90']['coverage']:.3f}", f"{v['90']['mean_width_pp']:.2f}", f"{v['mean_pinball_pp']:.3f}"]
                          for st, v in iv["by_station"].items()]))
    out.append("\n### Robustness: source removed at inference only (no source-dropout training), ensemble MAE (pp)\n")
    out.append(table(["Model", "All inputs", "No flows", "No snow", "No weather"],
                     [[NAMES.get(f"{k}_ens", k), f"{mae[k + '_ens']:.3f}", f"{v['no_flow']:.3f}", f"{v['no_snow']:.3f}",
                       f"{v['no_weather']:.3f}"] for k, v in s["robustness_missing_source_at_inference_mae_pp"].items()]))
    runs = s["compute"]["fold_seed_runs"]
    out.append("\n### Compute (CPU, 8 threads)\n")
    rows = []
    for kind in sorted({k.split()[0] for k in runs}):
        r = [v for k, v in runs.items() if k.startswith(kind + " ")]
        rows.append([kind, r[0]["n_params"], round(np.mean([v["train_seconds"] for v in r]), 1),
                     round(np.mean([v["best_epoch"] for v in r]), 1), round(np.mean([v["n_fit"] for v in r]))])
    out.append(table(["Model", "Parameters", "Mean train seconds per fold-seed", "Mean best epoch", "Mean fitting samples"], rows))
    out.append(f"\nTotal wall time for the stage: {s['compute']['total_seconds'] / 60:.1f} minutes.\n")
    return "\n".join(out)


def ablation_section(s):
    names = dict(NAMES, hgb_v2_flows="HGB (frozen v2 config) + recent-flow features",
                 lstm_noflow_ens="N1 shared LSTM without flow inputs (3-seed ensemble)")
    mae = s["pooled_mae_pp"]
    out = [f"## Flow ablation (confirmation folds WY2016-2023, {s['n_rows']} identical rows)\n",
           table(["Model", "Pooled MAE (pp)"], [[names.get(k, k), f"{mae[k]:.3f}"] for k in sorted(mae, key=mae.get)]),
           f"\nPer-seed MAE, LSTM without flows: {', '.join(f'{v:.3f}' for v in s['seed_mae_pp_lstm_noflow'])}. "
           f"Flow features added to HGB: {', '.join(s['flow_features'])}.\n"]
    rows = []
    for k, d in s["block_bootstrap"].items():
        cis = [f"[{d[b]['ci95_low']:+.3f}, {d[b]['ci95_high']:+.3f}]" for b in ("5_weeks", "9_weeks", "13_weeks")]
        verdict = ("first better" if all(d[b]["ci95_high"] < 0 for b in d) else
                   "second better" if all(d[b]["ci95_low"] > 0 for b in d) else "not established")
        rows.append([k, f"{d['5_weeks']['estimate']:+.3f}"] + cis + [verdict])
    out.append(table(["Comparison (first vs second)", "MAE difference (pp)", "95% CI, 5 wk", "95% CI, 9 wk",
                      "95% CI, 13 wk", "Verdict"], rows))
    return "\n".join(out)


def main():
    parts = ["# v4 development result tables (generated by `python -m src.v4.report`; do not edit by hand)\n"]
    for name, fn in (("context", context_section), ("compare", compare_section), ("ablation", ablation_section)):
        p = DEV / name / "summary.json"
        if p.exists():
            parts.append(fn(json.loads(p.read_text())))
    print("\n".join(parts))


if __name__ == "__main__":
    main()

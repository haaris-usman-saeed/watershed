"""Render the v4 development result files as Markdown tables (no numbers are typed by hand).

Run: python -m src.v4.report > docs/v4/RESULTS_TABLES.md
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "results" / "v4" / "development"
NAMES = {"persistence": "Persistence", "trend_30d": "30-day trend", "monthly_seasonal": "Monthly seasonal",
         "doy_seasonal": "Smoothed DOY seasonal", "ridge_ar": "Ridge autoregression", "hgb_v2": "HGB SWE no weather (per-fold refit, frozen v2 config)",
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
    names = dict(NAMES, hgb_v2_flows="HGB SWE no weather refit + recent-flow features",
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


def n4_name(col):
    """Readable name for an N4 result column such as n4_wb_balanced_L30_ens."""
    c = col[:-4] if col.endswith("_ens") else col
    c = c[3:] if c.startswith("n4_") else c
    length = ""
    if "_L" in c:
        c, length = c.rsplit("_L", 1)
        length = f", {length} days"
    label = "direct-head ablation" if c == "direct" else c
    return f"N4 {label}{length} (3-seed ensemble)"


def _verdict(d):
    if all(d[b]["ci95_high"] < 0 for b in d):
        return "supported (first better)"
    if all(d[b]["ci95_low"] > 0 for b in d):
        return "second better"
    return "not established"


def _name_any(c):
    return n4_name(c) if c.startswith("n4_") else NAMES.get(c, NAMES.get(c.replace("_L30", ""), c))


def n4_tune_section(s):
    names = dict(NAMES, lstm_L30_ens="N1 shared LSTM, 30 days (3-seed ensemble)")
    mae = s["pooled_mae_pp"]
    out = [f"## N4 tuning (tuning folds WY{s['folds'][0]}-WY{s['folds'][-1]}, {s['n_rows']} identical Monday rows, seeds "
           f"{', '.join(map(str, s['seeds']))})\n",
           f"Rule ({s['rule']}). Selected: **{s['selected']['candidate']}, {s['selected']['history_length']} days**.\n",
           table(["Model", "Pooled MAE (pp)"],
                 [[names.get(k) or n4_name(k), f"{mae[k]:.3f}"] for k in sorted(mae, key=mae.get)]),
           "\n### Step 1, loss-weight candidates at 30 days (3-seed-ensemble MAE, pp)\n",
           table(["Candidate", "MAE (pp)", "Eligible as N4"],
                 [[k, f"{v:.3f}", "yes" if k in s["eligible"] else "no (ablation)"]
                  for k, v in sorted(s["step1_ensemble_mae_pp"].items(), key=lambda kv: kv[1])]),
           f"\n### Step 2, history length for {s['step1_winner']} (3-seed-ensemble MAE, pp)\n",
           table(["History (days)", "MAE (pp)"], [[k, f"{v:.3f}"] for k, v in s["step2_ensemble_mae_pp"].items()]),
           "\n### Seed variability (per-seed MAE, pp)\n",
           table(["Model", "Per-seed", "Mean", "SD"],
                 [[names.get(k + "_ens") or n4_name(k), ", ".join(f"{v:.3f}" for v in d["per_seed"]), f"{d['mean']:.3f}",
                   f"{d['sd']:.3f}"] for k, d in s["seed_mae_pp"].items()])]
    out.append("\n### MAE by tuning fold (pp)\n")
    cols = ["hgb_v2", "lstm_L30_ens"] + [k for k in mae if k.startswith("n4_")]
    out.append(table(["Water year"] + [names.get(c) or n4_name(c) for c in cols],
                     [[wy] + [f"{d[c]:.3f}" for c in cols] for wy, d in s["mae_by_fold"].items()]))
    out.append(_diagnostics_table(s["diagnostics"]))
    return "\n".join(out)


def _diagnostics_table(diag):
    rows = [[n4_name(k) if k.startswith("n4_") else n4_name("n4_" + k), f"{d['mean_abs_residual_30_pp']:.3f}",
             f"{d['p95_abs_residual_30_pp']:.3f}", f"{d['mean_cum_loss_pp']:.3f}", f"{d['cum_inflow_mae_pp']:.2f}",
             f"{d['cum_release_mae_pp']:.2f}"] for k, d in diag.items() if d]
    if not rows:
        return ""
    return ("\n### Water-balance components on the evaluation rows (pp, seed ensemble)\n\n"
            "The inflow and release labels are operator-computed CDEC sensors 76 and 23 (private; aggregates only).\n\n"
            + table(["Model", "Mean abs r(30)", "p95 abs r(30)", "Mean 30-day loss", "Cumulative inflow MAE",
                     "Cumulative release MAE"], rows))


def n4_confirm_section(s):
    mae = s["pooled_mae_pp"]
    sel = s["selected"]
    out = [f"## N4 confirmation (folds WY{s['folds'][0]}-WY{s['folds'][-1]}, {s['n_rows']} identical Monday rows, history "
           f"{s['history_length']} days, seeds {', '.join(map(str, s['seeds']))})\n",
           f"Selected on tuning folds: **{sel['candidate']}, {sel['history_length']} days**. HGB is the per-fold refit of "
           "the frozen v2 \"HGB SWE no weather\" configuration, not the frozen artifact.\n",
           table(["Model", "Pooled MAE (pp)"], [[_name_any(k), f"{mae[k]:.3f}"] for k in sorted(mae, key=mae.get)]),
           "\n### Seed variability (per-seed MAE, pp)\n",
           table(["Model", "Per-seed", "Mean", "SD"],
                 [[n4_name("n4_" + k), ", ".join(f"{v:.3f}" for v in d["per_seed"]), f"{d['mean']:.3f}", f"{d['sd']:.3f}"]
                  for k, d in s["seed_mae_pp"].items()]),
           "\n### Paired differences (moving-block bootstrap over origin dates; negative = first model better)\n"]
    rows = []
    for k, d in s["block_bootstrap"].items():
        a, b = k.split(" vs ")
        cis = [f"[{d[x]['ci95_low']:+.3f}, {d[x]['ci95_high']:+.3f}]" for x in ("5_weeks", "9_weeks", "13_weeks")]
        rows.append([f"{_name_any(a)} vs {_name_any(b)}", f"{d['5_weeks']['estimate']:+.3f}"] + cis + [_verdict(d)])
    out.append(table(["Comparison (first vs second)", "MAE difference (pp)", "95% CI, 5 wk", "95% CI, 9 wk",
                      "95% CI, 13 wk", "Verdict"], rows))
    cols = ["hgb_v2", "doy_seasonal", "gru_ens", "lstm_ens"] + [k for k in mae if k.startswith("n4_")]
    for title, key in (("reservoir", "mae_by_station"), ("water year", "mae_by_water_year"), ("season", "mae_by_season"),
                       ("water-year type (Sacramento or San Joaquin index, by reservoir)", "mae_by_water_year_type"),
                       ("regime (observed 30-day change)", "mae_by_regime")):
        d = s[key]
        out.append(f"\n### MAE by {title} (pp)\n")
        out.append(table([title.split(" (")[0].capitalize()] + [_name_any(c) for c in cols],
                         [[g] + [f"{d[g][c]:.3f}" for c in cols] for g in d]))
    out.append(f"\nRegime counts: {s['regime_counts']}\n")
    for k, iv in s["intervals"].items():
        a = iv["all"]
        out.append(f"\n### Probabilistic output, {n4_name('n4_' + k)}\n")
        out.append(table(["Nominal", "Coverage", "Mean width (pp)"],
                         [[f"{n}%", f"{a[n]['coverage']:.3f}", f"{a[n]['mean_width_pp']:.2f}"] for n in ("50", "80", "90", "95")]))
        out.append(f"\nMean pinball loss {a['mean_pinball_pp']:.3f} pp; approximate CRPS {a['approx_crps_pp']:.3f} pp.\n")
        out.append(table(["Reservoir", "90% coverage", "90% mean width (pp)", "Pinball (pp)"],
                         [[st, f"{v['90']['coverage']:.3f}", f"{v['90']['mean_width_pp']:.2f}", f"{v['mean_pinball_pp']:.3f}"]
                          for st, v in iv["by_station"].items()]))
    out.append("\n### Robustness: source removed at inference only (no source-dropout training), ensemble MAE (pp)\n")
    out.append(table(["Model", "All inputs", "No flows", "No snow", "No weather"],
                     [[n4_name("n4_" + k), f"{mae['n4_' + k + '_ens']:.3f}", f"{v['no_flow']:.3f}", f"{v['no_snow']:.3f}",
                       f"{v['no_weather']:.3f}"] for k, v in s["robustness_missing_source_at_inference_mae_pp"].items()]))
    out.append(_diagnostics_table(s["diagnostics"]))
    runs = s["compute"]["fold_seed_runs"]
    rows = []
    for k in s["seed_mae_pp"]:
        r = [v for name, v in runs.items() if name.startswith(f"n4_{k} ")]
        rows.append([n4_name("n4_" + k), r[0]["n_params"], round(np.mean([v["train_seconds"] for v in r]), 1),
                     round(np.mean([v["best_epoch"] for v in r]), 1)])
    out.append("\n### Compute (CPU, 8 threads)\n")
    out.append(table(["Model", "Parameters", "Mean train seconds per fold-seed", "Mean best epoch"], rows))
    out.append(f"\nTotal wall time for the stage: {s['compute']['total_seconds'] / 60:.1f} minutes.\n")
    return "\n".join(out)


def main():
    parts = ["# v4 development result tables (generated by `python -m src.v4.report`; do not edit by hand)\n"]
    for name, fn in (("context", context_section), ("compare", compare_section), ("ablation", ablation_section),
                     ("n4_tune", n4_tune_section), ("n4_confirm", n4_confirm_section)):
        p = DEV / name / "summary.json"
        if p.exists():
            parts.append(fn(json.loads(p.read_text())))
    print("\n".join(parts))


if __name__ == "__main__":
    main()

"""v5 figures from real saved results only (src.v5.report calls make_figures). Light theme, reference palette slots;
categorical hues follow a fixed order by model family, never by rank. Every figure states its population."""
import json

import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                             "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FAMILY_COLOR = {"baseline": MUTED, "HGB": BLUE, "Ridge": ORANGE, "network": AQUA}
LABEL = {
    "BL-PERS": "Persistence", "BL-TREND": "30-day trend", "BL-MONTH": "Monthly climatology", "BL-DOY": "DOY climatology",
    "B-HGB-SWdCR": "HGB[S,Wd,C,R]", "B-HGB-SCR": "HGB[S,C,R]", "B-HGB-SNCR": "HGB[S,N,C,R] (reference)",
    "B-HGB-SNWdCR": "HGB[S,N,Wd,C,R]", "B-HGB-SNWuCR": "HGB[S,N,Wu,C,R]", "B-HGB-SNGCR": "HGB[S,N,G,C,R]",
    "B-HGB-SNQCR": "HGB[S,N,Q,C,R]", "HGB-V5": "HGB v5 [all]", "A-HGB-DAILY": "HGB v5 [all] · daily origins",
    "B-RIDGE-SWdCR": "Ridge[S,Wd,C,R] (v1)", "B-RIDGE-SCR": "Ridge[S,C,R] (AR)", "A-RIDGE": "Ridge [all]",
    "B-N1": "N1 LSTM", "B-N1-NOFLOW": "N1 LSTM, no flows", "B-N2": "N2 GRU", "B-N3": "N3 TCN",
    "B-N4-NOAUX": "N4 wb_noaux", "B-N4-BAL": "N4 wb_balanced", "B-N4-PRIM": "N4 wb_primary", "B-N4-STRICT": "N4 wb_strict",
    "B-N4-DIRECT": "N4 direct", "A-N1": "N1 LSTM [all]", "A-N2": "N2 GRU [all]", "A-N3": "N3 TCN [all]",
    "A-N4": "N4 wb_noaux [all]", "A-N4-DIRECT": "N4 direct [all]",
    "T-RIDGE": "Ridge [all] tuned", "T-HGB-V5": "HGB v5 tuned", "T-N1": "N1 LSTM [all] tuned", "T-N2": "N2 GRU [all] tuned",
    "T-N3": "N3 TCN [all] tuned", "T-N4": "N4 [all] tuned", "REF-V2-ARTIFACT": "frozen v2 artifact HGB[S,N,C,R]",
}
GROUPS = [("Baselines", ["BL-PERS", "BL-TREND", "BL-MONTH", "BL-DOY"]),
          ("Exp. B · HGB, original feature sets", ["B-HGB-SCR", "B-HGB-SWdCR", "B-HGB-SNCR", "B-HGB-SNWdCR", "B-HGB-SNWuCR",
                                                   "B-HGB-SNQCR", "B-HGB-SNGCR"]),
          ("Exp. B · Ridge, original", ["B-RIDGE-SWdCR", "B-RIDGE-SCR"]),
          ("Exp. B · networks, original inputs (no GEFS)", ["B-N1", "B-N1-NOFLOW", "B-N2", "B-N3", "B-N4-NOAUX", "B-N4-BAL",
                                                            "B-N4-PRIM", "B-N4-STRICT", "B-N4-DIRECT"]),
          ("Exp. A · full information, frozen", ["A-RIDGE", "HGB-V5", "A-HGB-DAILY", "A-N1", "A-N2", "A-N3", "A-N4", "A-N4-DIRECT"]),
          ("Exp. A · full information, tuned", ["T-RIDGE", "T-HGB-V5", "T-N1", "T-N2", "T-N3", "T-N4"])]


def fam(m):
    if m.startswith("BL"):
        return "baseline"
    if "RIDGE" in m:
        return "Ridge"
    if "HGB" in m or m == "REF-V2-ARTIFACT":
        return "HGB"
    return "network"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7.5, length=2.5, width=0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")


def fig_mae_dots(t, out, name, title, plt):
    t = t.set_index("model")
    rows, ys, labels, heads = [], [], [], []
    y = 0
    for gname, ms in GROUPS:
        ms = [m for m in ms if m in t.index]
        if not ms:
            continue
        heads.append((y, gname))
        y += 1
        for m in ms:
            rows.append((y, m))
            y += 1
        y += 0.4
    fig, ax = plt.subplots(figsize=(7.2, 0.19 * y + 0.9))
    _style(ax)
    ref = t.loc["B-HGB-SNCR", "mae"] if "B-HGB-SNCR" in t.index else None
    for yy, m in rows:
        r = t.loc[m]
        c = FAMILY_COLOR[fam(m)]
        right = r["mae"]
        if "seed_mae_mean" in t.columns and pd.notna(r.get("seed_mae_sd")):
            sm = [r["seed_mae_mean"] - r["seed_mae_sd"], r["seed_mae_mean"] + r["seed_mae_sd"]]
            ax.plot(sm, [yy, yy], color=c, lw=1.2, alpha=0.55, solid_capstyle="round")
            right = max(right, sm[1])
        ax.plot(r["mae"], yy, "o", ms=4.6, color=c, mec="white", mew=0.8, zorder=3)
        ax.text(right + 0.03, yy, f"{r['mae']:.3f}", va="center", fontsize=6.6, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=BLUE, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(ref, -0.9, "HGB[S,N,C,R] reference", fontsize=6.6, color=BLUE, ha="center")
    ax.set_yticks([yy for yy, _ in rows])
    ax.set_yticklabels([LABEL.get(m, m) for _, m in rows], fontsize=7.2, color=INK)
    for yy, g in heads:
        ax.text(0.005, yy, g, transform=ax.get_yaxis_transform(), fontsize=7.4, color=INK,
                fontweight="bold", va="center", ha="left")
    ax.set_ylim(y, -1.6)
    ax.set_xlabel("MAE (percentage points of capacity)", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.tick_params(axis="y", length=0)
    fig.text(0.01, -0.01, "Dots: MAE of the point forecast (networks: mean of 3 seed medians). Bars: seed mean ± SD. "
             "Same rows for every model.", fontsize=6.5, color=MUTED)
    _save(fig, out, name)
    plt.close(fig)


def fig_ladder(tables, out, plt):
    ms = ["B-HGB-SCR", "B-HGB-SWdCR", "B-HGB-SNCR", "B-HGB-SNWdCR", "B-HGB-SNWuCR", "B-HGB-SNQCR", "B-HGB-SNGCR", "HGB-V5",
          "T-HGB-V5"]
    pops = [("tuning_WY2012-2015", "Tuning WY2012–2015\n(1,248 rows)"), ("confirmation_WY2016-2023", "Confirmation WY2016–2023\n(2,500 rows)"),
            ("HELD_posthoc", "HELD 2024–2025, post-hoc\n(593 rows)"), ("T26_posthoc", "T26 2026, post-hoc\n(251 rows)")]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.3), sharey=True)
    for ax, (k, lab) in zip(axes, pops):
        _style(ax)
        t = tables[k].set_index("model")
        ref = t.loc["B-HGB-SNCR", "mae"]
        for i, m in enumerate(ms):
            if m not in t.index:
                continue
            c = ORANGE if m in ("HGB-V5", "T-HGB-V5") else BLUE
            ax.plot(t.loc[m, "mae"], i, "o", ms=4.6, color=c, mec="white", mew=0.8, zorder=3)
            ax.text(t.loc[m, "mae"], i - 0.33, f"{t.loc[m, 'mae']:.2f}", fontsize=6.2, color=INK2, ha="center")
        ax.axvline(ref, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_title(lab, fontsize=7.6, color=INK)
        ax.margins(x=0.14)
        ax.set_xlabel("MAE (pp)", fontsize=7.4, color=INK2)
    axes[0].set_yticks(range(len(ms)))
    axes[0].set_yticklabels([LABEL[m] for m in ms], fontsize=7.2)
    axes[0].set_ylim(len(ms) - 0.4, -0.8)
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle("Experiment B: what each information source adds to HGB (frozen v2 hyperparameters; per-fold refit)",
                 fontsize=9.3, color=INK, x=0.01, ha="left")
    fig.text(0.01, -0.03, "Each panel is a separate population; values are never pooled across panels. Dashed line: HGB[S,N,C,R]. "
             "HELD and T26 were viewed in v2/v3: post-hoc, not independent tests.", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig02_information_ladder")
    plt.close(fig)


def fig_forest(stats, out, plt):
    items = [(k, v) for k, v in stats["family_F"].items()] + [(k, v) for k, v in stats["family_T"].items()]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(items) + 0.9))
    _style(ax)
    for i, (k, v) in enumerate(items):
        for j, (b, off) in enumerate((("5w", -0.18), ("9w", 0.0), ("13w", 0.18))):
            r = v["blocks"][b]
            ax.plot([r["ci95_low"], r["ci95_high"]], [i + off, i + off], color=[MUTED, INK2, INK][j], lw=1.0)
        sup = v["verdict"].startswith("supported")
        ax.plot(v["estimate"], i, "D" if sup else "o", ms=5, color=ORANGE if sup else BLUE, mec="white", mew=0.7, zorder=3)
        ax.text(1.02, i, f"{v['verdict']}  (Holm p={v['p_holm']:.3f})", transform=ax.get_yaxis_transform(), fontsize=6.6,
                color=INK if sup else INK2, va="center")
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(range(len(items)))
    short = {"P1": "HGB v5 − HGB[S,N,C,R]", "P2": "Ridge gap change: [Ridge[all] − HGB v5] − [Ridge[S,C,R] − HGB[S,C,R]]",
             "P3": "LSTM [all] − HGB v5", "P4": "GRU [all] − HGB v5", "P5": "TCN [all] − HGB v5", "P6": "N4 wb_noaux [all] − HGB v5",
             "P7": "HGB v5 − HGB[S,C,R] (all variables vs storage only)", "P8": "N4 wb_noaux [all] − N4 direct [all]",
             "T1": "tuned HGB v5 − HGB[S,N,C,R]", "T2": "tuned Ridge [all] − tuned HGB v5", "T3": "tuned LSTM − tuned HGB v5",
             "T4": "tuned GRU − tuned HGB v5", "T5": "tuned TCN − tuned HGB v5", "T6": "tuned N4 − tuned HGB v5"}
    ax.set_yticklabels([f"{k}: {short.get(k, v['description'][:60])}" for k, v in items], fontsize=6.8)
    ax.set_ylim(len(items) - 0.4, -0.7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Paired MAE difference, first − second (pp); negative favours the first model", fontsize=7.6, color=INK2)
    ax.set_title("Planned comparisons, confirmation folds WY2016–2023 (2,500 rows): estimates and 95% block-bootstrap CIs",
                 fontsize=8.8, color=INK, loc="left")
    fig.text(0.01, -0.02, "CI lines: 5-, 9- and 13-week blocks (top to bottom), 2,000 replicates. Families F (frozen) and T (tuned) "
             "are Holm-adjusted separately on the largest p across block lengths. Exploratory: these folds were used in v2–v4.",
             fontsize=6.4, color=MUTED)
    _save(fig, out, "fig03_planned_comparisons")
    plt.close(fig)


def fig_heatmap(dev, out, plt):
    ms = ["BL-DOY", "B-RIDGE-SCR", "A-RIDGE", "B-HGB-SNCR", "HGB-V5", "T-HGB-V5", "A-HGB-DAILY", "B-N1", "A-N1", "A-N2", "A-N3",
          "A-N4", "T-N1", "T-N2", "T-N3", "T-N4"]
    ms = [m for m in ms if m in dev and dev[m].notna().all()]
    st = sorted(dev.station_id.unique())
    M = np.array([[np.mean(np.abs(dev[dev.station_id == s][m] - dev[dev.station_id == s].target_delta_pp)) for s in st] for m in ms])
    fig, ax = plt.subplots(figsize=(5.6, 0.27 * len(ms) + 1.1))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#9cc0ee", "#2a78d6", "#123d73"])
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    for i in range(len(ms)):
        for j in range(len(st)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > np.percentile(M, 60) else INK)
    ax.set_xticks(range(len(st)))
    ax.set_xticklabels(st, fontsize=7.5)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([LABEL[m] for m in ms], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("MAE (pp)", fontsize=7)
    ax.set_title("MAE by reservoir, confirmation folds WY2016–2023", fontsize=9, color=INK, loc="left")
    _save(fig, out, "fig04_mae_by_reservoir")
    plt.close(fig)


def fig_search(out, plt, results_dir):
    log = json.loads((results_dir / "tuning" / "search_log.json").read_text())
    sel = json.loads((results_dir / "selection.json").read_text())
    names = [("T-RIDGE", "Ridge [all]"), ("T-HGB-V5", "HGB v5"), ("T-N1", "N1 LSTM"), ("T-N2", "N2 GRU"), ("T-N3", "N3 TCN"), ("T-N4", "N4")]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.grid(axis="x", visible=False)
    for i, (k, lab) in enumerate(names):
        res = log[k]
        maes = np.array([r["tuning_mae"] for r in res])
        jitter = np.linspace(-0.22, 0.22, len(maes))
        c = FAMILY_COLOR[fam(k)]
        ax.scatter(i + jitter, maes, s=10, color=c, alpha=0.55, linewidths=0)
        ax.plot(i + jitter[0], maes[0], "s", ms=5, mfc="white", mec=INK, mew=0.9, zorder=3)
        b = sel[k]["config"]
        ax.plot(i + jitter[b], maes[b], "D", ms=5.5, color=ORANGE, mec="white", mew=0.7, zorder=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([lab for _, lab in names], fontsize=7.5)
    ax.set_ylabel("Tuning-fold MAE (pp), seed 0", fontsize=7.6, color=INK2)
    lo = min(min(r["tuning_mae"] for r in log[k]) for k, _ in names)
    ax.set_ylim(lo - 0.05, min(lo + 1.2, max(max(r["tuning_mae"] for r in log[k]) for k, _ in names) + 0.05))
    ax.set_title("Equal-budget search: 20 configurations per algorithm, tuning folds WY2012–2015 (1,248 rows)",
                 fontsize=8.6, color=INK, loc="left")
    fig.text(0.01, -0.03, "Square: configuration 0 (frozen). Diamond: selected (lowest tuning MAE). Points above the axis limit are clipped.",
             fontsize=6.4, color=MUTED)
    _save(fig, out, "fig05_search")
    plt.close(fig)


def fig_complexity(t, out, plt):
    """Confirmation MAE against the controlled single-thread training time (results/v5/compute_benchmark.csv)."""
    cb = pd.read_csv(out.parent / "compute_benchmark.csv").set_index("model")
    t = t.set_index("model")
    pts = [m for m in cb.index if m in t.index and m != "T-N4"]          # T-N4 is identical to A-N4 (config 0 selected)
    off = {"B-RIDGE-SCR": (6, 0, "left"), "A-RIDGE": (6, -4, "left"), "T-RIDGE": (-6, 0, "right"), "B-HGB-SNCR": (6, 3, "left"),
           "HGB-V5": (-6, -2, "right"), "A-HGB-DAILY": (-4, -11, "right"), "T-HGB-V5": (6, 7, "left"), "B-N1": (6, 2, "left"),
           "A-N1": (0, 8, "center"), "A-N2": (0, -10, "center"), "A-N3": (6, -5, "left"), "A-N4": (-6, 0, "right"),
           "A-N4-DIRECT": (6, -4, "left"), "T-N1": (6, 8, "left"), "T-N2": (0, -10, "center"), "T-N3": (6, 3, "left")}
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    for m in pts:
        c = FAMILY_COLOR[fam(m)]
        ax.plot(cb.loc[m, "train_seconds"], t.loc[m, "mae"], "o", ms=4.8, color=c, mec="white", mew=0.7, zorder=3)
        lab = LABEL[m] + (" (= tuned N4)" if m == "A-N4" else "")
        dx, dy, ha = off.get(m, (5, 2, "left"))
        ax.annotate(lab, (cb.loc[m, "train_seconds"], t.loc[m, "mae"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.2, color=INK2, va="center", ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 300)
    ax.set_xlabel("Training time for one fit, WY2023 training set, one CPU thread (s, log scale)", fontsize=7.4, color=INK2)
    ax.set_ylabel("Confirmation MAE (pp)", fontsize=7.6, color=INK2)
    for k, c in (("HGB", BLUE), ("Ridge", ORANGE), ("network", AQUA)):
        ax.plot([], [], "o", color=c, label=k, ms=4.5)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Accuracy against training cost (MAE: confirmation folds WY2016–2023)", fontsize=8.8, color=INK, loc="left")
    fig.text(0.01, -0.03, "Timing: one seed, one fit, sequential, nothing else running (src/v5/compute.py). Network MAE is the 3-seed "
             "ensemble; the ensemble costs 3 fits.", fontsize=6.3, color=MUTED)
    _save(fig, out, "fig06_complexity")
    plt.close(fig)


def fig_intervals(t, out, plt):
    t = t[t.get("conformal90_coverage").notna()] if "conformal90_coverage" in t else t.iloc[:0]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    for _, r in t.iterrows():
        c = FAMILY_COLOR[fam(r.model)]
        ax.plot(r.conformal90_width, r.conformal90_coverage, "o", ms=4.3, color=c, mec="white", mew=0.7)
        if "native_cov90" in t and pd.notna(r.get("native_cov90")):
            ax.plot(r.native_width90, r.native_cov90, "^", ms=4.5, mfc="white", mec=c, mew=1.0)
    for m, (dx, dy, ha) in {"BL-DOY": (4, -8, "left"), "B-HGB-SNCR": (8, 6, "left"), "HGB-V5": (-8, 6, "right"),
                            "A-N4": (-8, -6, "right"), "A-RIDGE": (8, -6, "left")}.items():
        if m in set(t.model):
            r = t[t.model == m].iloc[0]
            ax.annotate(LABEL[m], (r.conformal90_width, r.conformal90_coverage), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.3, color=INK2, ha=ha, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.axhline(0.9, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("Mean width of the 90% interval (pp)", fontsize=7.6, color=INK2)
    ax.set_ylabel("Empirical coverage", fontsize=7.6, color=INK2)
    ax.plot([], [], "o", color=INK2, label="chronological conformal 90% (every model)", ms=4)
    ax.plot([], [], "^", mfc="white", mec=INK2, label="native 5–95% quantiles (networks)", ms=4.5)
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.set_title("90% intervals: coverage against width, confirmation folds WY2016–2023", fontsize=8.8, color=INK, loc="left")
    _save(fig, out, "fig07_intervals")
    plt.close(fig)


def fig_conditions(dev, out, plt):
    ms = ["BL-DOY", "B-HGB-SNCR", "HGB-V5", "T-HGB-V5", "A-RIDGE", "A-N1", "A-N4", "T-N4"]
    ms = [m for m in ms if m in dev and dev[m].notna().all()]
    conds = [("wy_class", "wet (W/AN)"), ("wy_class", "dry (BN/D/C)"), ("regime", "rapid fill (>= +10 pp)"),
             ("regime", "rapid drawdown (<= -10 pp)"), ("regime", "other")]
    fig, axes = plt.subplots(1, len(conds), figsize=(10.2, 2.9), sharey=True)
    for ax, (col, val) in zip(axes, conds):
        _style(ax)
        g = dev[dev[col] == val]
        for i, m in enumerate(ms):
            v = np.mean(np.abs(g[m] - g.target_delta_pp))
            ax.plot(v, i, "o", ms=4.4, color=FAMILY_COLOR[fam(m)], mec="white", mew=0.7)
            ax.text(v, i - 0.32, f"{v:.2f}", fontsize=6, color=INK2, ha="center")
        ax.set_title(f"{val}\n(n = {len(g)})", fontsize=7.3, color=INK)
        ax.margins(x=0.15)
        ax.set_xlabel("MAE (pp)", fontsize=7, color=INK2)
    axes[0].set_yticks(range(len(ms)))
    axes[0].set_yticklabels([LABEL[m] for m in ms], fontsize=7)
    axes[0].set_ylim(len(ms) - 0.4, -0.8)
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle("Error by hydrological condition, confirmation folds WY2016–2023", fontsize=9, color=INK, x=0.01, ha="left")
    fig.text(0.01, -0.04, "Water-year class from the Sacramento (SHA, ORO, FOL) or San Joaquin (NML, DNP, EXC) index. Regimes use the "
             "observed 30-day change.", fontsize=6.4, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig08_conditions")
    plt.close(fig)


def make_figures(dev, retro, tables, stats, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Helvetica Neue", "svg.fonttype": "none"})
    out.mkdir(parents=True, exist_ok=True)
    conf = dev[dev.fold.str.startswith("WY") & (dev.wy >= 2016)]
    fig_mae_dots(tables["confirmation_WY2016-2023"], out, "fig01_confirmation_mae",
                 "All models, confirmation folds WY2016–2023 (2,500 Monday rows)", plt)
    fig_ladder(tables, out, plt)
    fig_forest(stats, out, plt)
    fig_heatmap(conf, out, plt)
    fig_search(out, plt, out.parent)
    fig_complexity(tables["confirmation_WY2016-2023"], out, plt)
    fig_intervals(tables["confirmation_WY2016-2023"], out, plt)
    fig_conditions(conf, out, plt)
    for w, lab in (("HELD_posthoc", "HELD 2024-01-01..2025-12-01 (593 rows), POST-HOC retrospective"),
                   ("T26_posthoc", "T26 2026-01-05..2026-10-19 (251 rows), POST-HOC retrospective")):
        fig_mae_dots(tables[w], out, f"fig09_{w}", lab, plt)

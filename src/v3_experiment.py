"""v3 experiment: Models A, B, B2 and C (docs/v3/PREREGISTRATION_V3.md).

Run: python -m src.v3_experiment --reproduce-v2   # read-only check of the v2 HGB + SWE comparison
     python -m src.v3_experiment --dev            # development folds WY2008-2025 (no 2026 rows)
     python -m src.v3_experiment --final          # the one preregistered 2026 test (guarded)

Outputs: results/v3/development/ (aggregate metrics only, publishable) and
         results/v3/final/ (aggregate metrics; per-row predictions go to data/v3/private/).
Model artifacts for the website: models/v3/ (git-ignored), written by --final only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd

from src.evaluation import BLOCK_LENGTHS, block_bootstrap_diff, coverage, mae, water_year
from src.models import add_snow_anomaly, baselines, design, hgb, interval_offsets, season_of, snow_climatology

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config" / "v3_plan.json"
PREREG = ROOT / "docs" / "v3" / "PREREGISTRATION_V3.md"
TABLE = ROOT / "data" / "v3" / "private" / "weekly_v3.csv"
SNOW = ROOT / "data" / "v3" / "private" / "snow_index_v3.csv"
DEV_OUT = ROOT / "results" / "v3" / "development"
FINAL_OUT = ROOT / "results" / "v3" / "final"
PRIVATE = ROOT / "data" / "v3" / "private"
ART = ROOT / "models" / "v3"
HP = dict(learning_rate=0.03, max_iter=150, max_depth=None, min_samples_leaf=30)   # frozen v2 (l2=1.0, seed 42 in hgb())
TEST_START, TEST_TARGET_END, TRAIN_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-20"), pd.Timestamp("2026-01-01")
DEV_WYS = list(range(2008, 2026))
BASE_A = ["storage_pct", "prev7_pct", "prev30_pct", "prev60_pct", "swe_basin_in", "swe_change_14d_in", "swe_anom_in",
          "season_sin", "season_cos"]
MODELS = {
    "A": BASE_A,
    "B": BASE_A + ["rain_7d_mm", "rain_30d_mm", "temp_30d_c", "humidity_30d_pct"],
    "B2": BASE_A + ["rain_7d_mm_box", "rain_30d_mm_box", "temp_30d_c_box", "humidity_30d_pct_box"],
    "C": BASE_A + ["gefs_p_d1_7_mm", "gefs_p_d8_16_mm"],
}
NEEDS = {"A": [], "B": [], "B2": ["rain_7d_mm_box"], "C": ["gefs_p_d1_7_mm", "gefs_p_d8_16_mm"]}
PAIRS = [("C", "A"), ("B", "A"), ("B2", "B"), ("B2", "A")]
STATIONS = ["DNP", "EXC", "FOL", "NML", "ORO", "SHA"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    df = pd.read_csv(TABLE, parse_dates=["date", "target_date"])
    df["wy"], df["season"] = water_year(df.date), season_of(df.date)
    snow = pd.read_csv(SNOW, parse_dates=["date"])
    return df, snow


def available(df, m):
    return df[NEEDS[m]].notna().all(axis=1).to_numpy() if NEEDS[m] else np.ones(len(df), bool)


def fit_predict(train, test, m):
    tr = train[available(train, m)]
    model = hgb(HP).fit(design(tr, MODELS[m], STATIONS), tr.target_delta_pp)
    ok = available(test, m)
    pred = np.full(len(test), np.nan)
    if ok.any():
        pred[ok] = model.predict(design(test[ok], MODELS[m], STATIONS))
    return model, pred, int(len(tr))


def split(df, snow, cutoff, test_mask):
    train = df[df.target_date < cutoff]
    test = df[test_mask]
    assert train.target_date.max() < test.date.min()
    clim = snow_climatology(snow[snow.date < cutoff], cutoff)
    return add_snow_anomaly(train, clim), add_snow_anomaly(test, clim), clim


def score(test, train):
    out = test[["station_id", "date", "target_date", "wy", "season", "target_delta_pp", "storage_pct"]].copy()
    n_train = {}
    models = {}
    for m in MODELS:
        models[m], out[f"pred_{m}"], n_train[m] = fit_predict(train, test, m)
    b = baselines(train, test)
    out["base_persistence"], out["base_monthly"], out["base_doy"] = 0.0, b.baseline_seasonal_pp.to_numpy(), b.baseline_doy_pp.to_numpy()
    return out, models, n_train


def summarize(p, cols):
    res = {"n_rows": int(len(p)), "mae": {}, "availability": {}, "paired": {}}
    for c in cols:
        ok = p[c].notna()
        res["availability"][c] = float(ok.mean())
        res["mae"][c] = {"n": int(ok.sum()), "mae": mae(p.target_delta_pp[ok], p[c][ok])}
    for a, b in PAIRS:
        ca, cb = f"pred_{a}", f"pred_{b}"
        q = p[p[ca].notna() & p[cb].notna()]
        res["paired"][f"{a} vs {b}"] = {"n": int(len(q)), f"mae_{a}": mae(q.target_delta_pp, q[ca]),
                                        f"mae_{b}": mae(q.target_delta_pp, q[cb]),
                                        "diff": mae(q.target_delta_pp, q[ca]) - mae(q.target_delta_pp, q[cb])}
    for key in ("station_id", "season"):
        res[f"by_{key}"] = {str(k): {c: mae(g.target_delta_pp[g[c].notna()], g[c][g[c].notna()]) if g[c].notna().any() else None
                                     for c in cols} | {"n": int(len(g))} for k, g in p.groupby(key)}
    return res


# ------------------------------------------------------------------ v2 reproduction (read-only)

def reproduce_v2():
    from src.train_v2 import load as v2load
    from src.models import FEATURE_SETS
    train, test, _ = v2load(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"))
    st = sorted(train.station_id.unique())
    hp = HP
    res = {}
    for name, fs in (("HGB SWE no weather (Model A)", "storage_snow"), ("HGB + SWE (Model B)", "storage_weather_snow")):
        p = hgb(hp).fit(design(train, FEATURE_SETS[fs], st), train.target_delta_pp).predict(design(test, FEATURE_SETS[fs], st))
        res[name] = mae(test.target_delta_pp, p)
    arc = json.loads((ROOT / "results/v2/final/metrics.json").read_text())
    out = {"recomputed": res, "archived": {"HGB SWE no weather (Model A)": arc["ML"], "HGB + SWE (Model B)": arc["HGB + SWE"]},
           "identical": all(abs(res[k] - v) < 1e-9 for k, v in {"HGB SWE no weather (Model A)": arc["ML"],
                                                                 "HGB + SWE (Model B)": arc["HGB + SWE"]}.items()),
           "note": "2024-2025 v2 held-out comparison; already published, not a v3 test."}
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ development folds

def dev():
    df, snow = load()
    df = df[df.target_date < TEST_START]
    parts = []
    for w in DEV_WYS:
        start, end = pd.Timestamp(f"{w - 1}-10-01"), pd.Timestamp(f"{w}-10-01")
        train, test, _ = split(df, snow, start, ((df.date >= start) & (df.date < end)).to_numpy())
        out, _, _ = score(test, train)
        parts.append(out)
        print(f"fold WY{w}: train {len(train)} eval {len(test)}", flush=True)
    p = pd.concat(parts, ignore_index=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    p.to_csv(PRIVATE / "oof_v3.csv", index=False, date_format="%Y-%m-%d")
    cols = [f"pred_{m}" for m in MODELS] + ["base_persistence", "base_monthly", "base_doy"]
    summ = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "folds": DEV_WYS,
            "note": "Development folds; WY2024-2025 were seen in v2 and are not independent. No selection was made.",
            "all_folds": summarize(p, cols), "wy2008_2023": summarize(p[p.wy <= 2023], cols),
            "interval_offsets_pp": intervals(p), "plan_sha256": sha(PLAN) if PLAN.exists() else None}
    DEV_OUT.mkdir(parents=True, exist_ok=True)
    (DEV_OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["all_folds"]["paired"], indent=1))
    return summ


def intervals(oof):
    """sym_season offsets per model from that model's own OOF residuals (all dev folds)."""
    out = {}
    for m in MODELS:
        q = oof[oof[f"pred_{m}"].notna()]
        cal = pd.DataFrame({"r": q.target_delta_pp - q[f"pred_{m}"], "season": q.season, "station_id": q.station_id})
        apply = interval_offsets(cal, "sym_season")
        lo, hi = apply(pd.DataFrame({"season": ["DJF", "MAM", "JJA", "SON"]}))
        out[m] = {s: [float(a), float(b)] for s, a, b in zip(["DJF", "MAM", "JJA", "SON"], lo, hi)} | \
                 {"n_residuals": int(len(q))}
    return out


# ------------------------------------------------------------------ final (guarded)

def check_guard():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    for f in (PLAN, PREREG):
        if git("ls-files", "--error-unmatch", str(f)).returncode or git("diff", "--quiet", "HEAD", "--", str(f)).returncode:
            raise SystemExit(f"REFUSED: {f.relative_to(ROOT)} must be committed and unmodified.")
    plan = json.loads(PLAN.read_text())
    for f, h in plan["code_sha256"].items():
        if sha(ROOT / f) != h:
            raise SystemExit(f"REFUSED: {f} differs from the preregistered hash.")
    if (FINAL_OUT / "RUN_RECORD.json").exists():
        raise SystemExit("REFUSED: the v3 final test has already been run.")
    return plan, git("rev-parse", "HEAD").stdout.strip()


def frozen_a_predictions(test):
    """Unchanged v2 artifact, with its own pre-2024 snow climatology (from its model card)."""
    from src.forecast_model import load_artifact
    fa, card = load_artifact()
    t = test.reset_index(drop=True).copy()
    doy = (t.date - pd.Timedelta(days=1)).dt.dayofyear
    t["swe_anom_in"] = t.swe_basin_in.to_numpy() - np.array([card["snow_climatology_in"][s][d - 1] for s, d in zip(t.station_id, doy)])
    return fa.predict(design(t, BASE_A, STATIONS))


def interval_report(out, m):
    q = out[out[f"pred_{m}"].notna()]
    return {"coverage": coverage(q.target_delta_pp, q[f"lo_{m}"], q[f"hi_{m}"]),
            "mean_width_pp": float((q[f"hi_{m}"] - q[f"lo_{m}"]).mean()),
            "by_station": {s: coverage(g.target_delta_pp, g[f"lo_{m}"], g[f"hi_{m}"]) for s, g in q.groupby("station_id")}}


def final():
    plan, commit = check_guard()
    df, snow = load()
    oof = pd.read_csv(PRIVATE / "oof_v3.csv", parse_dates=["date"])
    offs = intervals(oof)
    test_mask = ((df.date >= TEST_START) & (df.target_date <= TEST_TARGET_END)).to_numpy()
    train, test, clim = split(df, snow, TRAIN_END, test_mask)
    out, models, n_train = score(test, train)
    out["wy"] = water_year(out.date)
    for m in MODELS:
        lo = np.array([offs[m][s][0] for s in out.season]); hi = np.array([offs[m][s][1] for s in out.season])
        out[f"lo_{m}"], out[f"hi_{m}"] = out[f"pred_{m}"] + lo, out[f"pred_{m}"] + hi
    out["pred_A_frozen"] = frozen_a_predictions(test)
    cols = [f"pred_{m}" for m in MODELS] + ["pred_A_frozen", "base_persistence", "base_monthly", "base_doy"]
    res = summarize(out, cols)
    res["hypotheses"] = {}
    for a, b, name in (("C", "A", "H1"), ("B", "A", "H2"), ("B2", "B", "H3")):
        q = out[out[f"pred_{a}"].notna() & out[f"pred_{b}"].notna()].copy()
        boots = {f"{k}_weeks": block_bootstrap_diff(q, f"pred_{a}", f"pred_{b}", block=k) for k in BLOCK_LENGTHS}
        supported = all(v["ci95_high"] < 0 for v in boots.values())
        res["hypotheses"][name] = {"claim": f"{a} has lower MAE than {b}", "n": int(len(q)), "bootstrap": boots,
                                   "verdict": "supported" if supported else "not established"}
    res["vs_doy_baseline"] = {m: {f"{k}_weeks": block_bootstrap_diff(out[out[f"pred_{m}"].notna()], f"pred_{m}", "base_doy", block=k)
                                  for k in BLOCK_LENGTHS} for m in MODELS}
    res["intervals"] = {m: interval_report(out, m) | {"offsets_pp": offs[m]} for m in MODELS}
    res["train_rows"] = n_train
    res["test_window"] = [str(out.date.min().date()), str(out.target_date.max().date())]
    res["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    (FINAL_OUT / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    out.to_csv(PRIVATE / "test_predictions_v3.csv", index=False, date_format="%Y-%m-%d")
    ART.mkdir(parents=True, exist_ok=True)
    cards = {}
    for m, model in models.items():
        p = ART / f"model_{m}.joblib"
        joblib.dump(model, p)
        cards[m] = {"model": m, "features": MODELS[m] + [f"station_{s}" for s in STATIONS],
                    "feature_order": list(model.feature_names_in_), "hyperparameters": HP | {"l2_regularization": 1.0, "random_state": 42},
                    "training_cutoff_exclusive": str(TRAIN_END.date()), "training_rows": n_train[m],
                    "interval_offsets_pp": offs[m], "artifact_sha256": sha(p),
                    "historical_interval_coverage_2026_test": res["intervals"][m]["by_station"],
                    "snow_climatology_in": {s: [float(v) for v in clim[s].reindex(range(1, 367)).to_numpy()] for s in STATIONS}}
    (ART / "model_cards.json").write_text(json.dumps({"source_commit": commit, "created_utc": res["generated_utc"],
                                                      "plan_sha256": sha(PLAN), "models": cards}, indent=1))
    (FINAL_OUT / "RUN_RECORD.json").write_text(json.dumps({"run_utc": res["generated_utc"], "git_commit_at_run": commit,
                                                           "plan_sha256": sha(PLAN), "code_sha256": plan["code_sha256"],
                                                           "table_sha256": sha(TABLE),
                                                           "verdicts": {k: v["verdict"] for k, v in res["hypotheses"].items()}}, indent=1))
    print(json.dumps({"paired": res["paired"], "verdicts": {k: v["verdict"] for k, v in res["hypotheses"].items()},
                      "coverage": {m: res["intervals"][m]["coverage"] for m in MODELS}}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reproduce-v2", action="store_true")
    g.add_argument("--dev", action="store_true")
    g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.reproduce_v2:
        reproduce_v2()
    elif a.dev:
        dev()
    else:
        final()


if __name__ == "__main__":
    main()

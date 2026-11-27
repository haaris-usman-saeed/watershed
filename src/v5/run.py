"""v5 benchmark runner (BENCHMARK_PLAN.md). Run in the v4 environment (.venv-v4).

  python -m src.v5.run tabular --which dev            # frozen tabular models + baselines, tuning and confirmation folds
  python -m src.v5.run nn --stage frozen --which dev  # every neural configuration, 3 seeds, tuning + confirmation folds
  python -m src.v5.run tune                           # equal-budget search, tuning folds WY2012-2015 (seed 0)
  python -m src.v5.run select                         # results/v5/selection.json from the tuning-fold MAE only
  python -m src.v5.run tuned --which confirm          # selected configurations, 3 seeds, confirmation folds
  python -m src.v5.run retro                          # post-hoc retrospective windows HELD and T26 (after select)

Workers: --workers N processes with --threads T torch threads each. Every stage writes a manifest with the plan SHA-256,
code SHA-256, start/end times (UTC) and the dataset hashes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time

import numpy as np
import pandas as pd

from src.v5 import CONFIRM_WYS, DATA, DEV_END, PLAN, RESULTS, RETRO, ROOT, SEEDS, TUNE_WYS
from src.v5.registry import NEURAL, NN_DEFAULT, TABULAR, TUNED, search_space

PREDS = DATA / "preds"
MANIFESTS = RESULTS / "manifests"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def provenance():
    files = [PLAN] + sorted((ROOT / "src" / "v5").glob("*.py")) + sorted((ROOT / "src" / "v4").glob("*.py"))
    out = {str(f.relative_to(ROOT)): sha(f) for f in files}
    for f in (DATA / "daily_panel.parquet", DATA / "features_daily.parquet"):
        out[str(f.relative_to(ROOT))] = sha(f)
    return out


def write_manifest(name, started, extra):
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    m = {"stage": name, "started_utc": started, "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
         "plan_sha256": sha(PLAN), "provenance_sha256": provenance(), **extra}
    (MANIFESTS / f"{name}.json").write_text(json.dumps(m, indent=1, default=str))


def now():
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


def wy_list(which):
    return {"tune": TUNE_WYS, "confirm": CONFIRM_WYS, "dev": TUNE_WYS + CONFIRM_WYS}[which]


# ------------------------------------------------------------------ tabular

def stage_tabular(a):
    from src.v5.data import load_features, load_snow
    from src.v5.tabular import run_folds
    started = now()
    end = DEV_END if a.which != "retro" else None
    feat, snow = load_features(end), load_snow(end if end is not None else pd.Timestamp("2026-11-21"))
    ids = list(TABULAR)
    p, infos = run_folds(feat, snow, a.which, ids)
    hp = {}
    if a.which == "retro":                        # tuned tabular models need results/v5/selection.json
        sel = json.loads((RESULTS / "selection.json").read_text())
        for tid in ("T-RIDGE", "T-HGB-V5"):
            base = TUNED[tid]
            hp[tid] = sel[tid]["hp"]
            q, inf = run_folds(feat, snow, "retro", [base], {base: hp[tid]}, stage=tid)
            p[tid] = q[tid].to_numpy()
            infos.update({k: v for k, v in inf.items() if k.startswith(tid)})
    PREDS.mkdir(parents=True, exist_ok=True)
    p.to_parquet(PREDS / f"tabular_{a.which}.parquet", index=False)
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS / "runs" / f"tabular_{a.which}_info.json").write_text(json.dumps(infos, indent=1, default=float))
    write_manifest(f"tabular_{a.which}", started, {"models": ids, "rows": int(len(p)), "tuned_hp": hp})
    cols = ["BL-PERS", "BL-TREND", "BL-MONTH", "BL-DOY"] + ids + list(hp)
    for fold_set, g in (("all", p),) + ((("tune", p[p.fold.isin([f"WY{w}" for w in TUNE_WYS])]),
                                         ("confirm", p[p.fold.isin([f"WY{w}" for w in CONFIRM_WYS])])) if a.which == "dev" else
                                        tuple((k, p[p.fold == k]) for k in RETRO)):
        print(fold_set, len(g), {c: round(float(np.mean(np.abs(g[c] - g.target_delta_pp))), 4) for c in cols})


# ------------------------------------------------------------------ neural

def run_pool(jobs, workers, threads, log_path):
    from src.v5.neural import done, init_worker, run_job
    todo = [j for j in jobs if not done(j)]
    print(f"{len(jobs)} jobs, {len(todo)} to run, {workers} workers x {threads} threads", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "a") as log, ProcessPoolExecutor(workers, initializer=init_worker, initargs=(threads,)) as ex:
        futs = {ex.submit(run_job, j): j for j in todo}
        for i, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                _, msg = f.result()
            except Exception as e:                       # noqa: BLE001 - report and continue; reruns pick it up
                msg = f"FAILED {e!r}"
            line = f"{now()} [{i}/{len(todo)}] {j['stage']} {j['model']} {j['fold']} s{j['seed']}: {msg} " \
                   f"(elapsed {time.perf_counter() - t0:.0f} s)"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


def frozen_jobs(which, models=None):
    folds = [f"WY{w}" for w in wy_list(which)] if which != "retro" else list(RETRO)
    jobs = []
    for m in (models or list(NEURAL)):
        hp = dict(NN_DEFAULT, cand=NEURAL[m].get("cand")) if NEURAL[m]["kind"] == "n4" else dict(NN_DEFAULT)
        jobs += [{"stage": "frozen", "model": m, "base": m, "fold": f, "seed": s, "hp": hp} for f in folds for s in SEEDS]
    return jobs


def stage_nn(a):
    started = now()
    jobs = frozen_jobs(a.which, a.models)
    # longest folds first keeps the pool busy at the end
    jobs.sort(key=lambda j: (-int(j["fold"][2:]) if j["fold"].startswith("WY") else -3000))
    DATA.mkdir(parents=True, exist_ok=True)
    run_pool(jobs, a.workers, a.threads, DATA / f"nn_frozen_{a.which}.log")
    write_manifest(f"nn_frozen_{a.which}", started, {"jobs": len(jobs), "workers": a.workers, "threads": a.threads})


def tune_jobs():
    jobs = []
    for tid, base in TUNED.items():
        if base not in NEURAL:
            continue
        for c, hp in enumerate(search_space(tid)):
            if c == 0:
                continue                                 # configuration 0 = the frozen run (stage "frozen", seed 0)
            jobs += [{"stage": "tune", "model": f"{tid}-c{c:02d}", "base": base, "fold": f"WY{w}", "seed": 0, "hp": hp}
                     for w in TUNE_WYS]
    return jobs


def stage_tune(a):
    """Equal-budget search: tabular (Ridge[all], HGB v5) in-process, networks through the pool. Tuning folds only."""
    from src.v5.data import load_features, load_snow
    from src.v5.tabular import run_folds
    started = now()
    feat, snow = load_features(DEV_END), load_snow(DEV_END)
    out = {}
    for tid in ("T-RIDGE", "T-HGB-V5"):
        base = TUNED[tid]
        res = []
        for c, hp in enumerate(search_space(tid)):
            p, inf = run_folds(feat, snow, "tune", [base], {base: hp}, stage=f"{tid}-c{c:02d}", save=False, log=lambda s: None)
            col = f"{tid}-c{c:02d}"
            res.append({"config": c, "hp": hp, "tuning_mae": float(np.mean(np.abs(p[col] - p.target_delta_pp))),
                        "fit_seconds": float(sum(v["fit_seconds"] for k, v in inf.items() if k.startswith(col)))})
            print(tid, c, round(res[-1]["tuning_mae"], 4), flush=True)
        out[tid] = res
    (RESULTS / "tuning").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tuning" / "tabular_search.json").write_text(json.dumps(out, indent=1, default=str))
    if not a.tabular_only:
        run_pool(tune_jobs(), a.workers, a.threads, DATA / "nn_tune.log")
    write_manifest("tune", started, {"workers": a.workers, "threads": a.threads})


def stage_select(a):
    """Selection from tuning-fold MAE only (pooled over the Monday rows of WY2012-2015)."""
    from src.v5.neural import job_paths
    started = now()
    tab = json.loads((RESULTS / "tuning" / "tabular_search.json").read_text())
    sel, log = {}, {}
    for tid, res in tab.items():
        best = min(res, key=lambda r: r["tuning_mae"])
        sel[tid] = {"config": best["config"], "hp": best["hp"], "tuning_mae": best["tuning_mae"]}
        log[tid] = [{"config": r["config"], "tuning_mae": r["tuning_mae"], "hp": r["hp"]} for r in res]
    for tid, base in TUNED.items():
        if base not in NEURAL:
            continue
        res = []
        for c, hp in enumerate(search_space(tid)):
            stage, model = ("frozen", base) if c == 0 else ("tune", f"{tid}-c{c:02d}")
            errs, n, secs, ok = 0.0, 0, 0.0, True
            for w in TUNE_WYS:
                npz, js, _ = job_paths(stage, model, f"WY{w}", 0)
                if not npz.exists():
                    ok = False
                    break
                z = np.load(npz, allow_pickle=True)
                errs += float(np.abs(z["q"][:, 4] - z["y"]).sum())
                n += len(z["y"])
                secs += json.loads(js.read_text())["train_seconds"]
            if ok:
                res.append({"config": c, "hp": hp, "tuning_mae": errs / n, "n_rows": n, "train_seconds": secs})
        assert len(res) == len(search_space(tid)), f"{tid}: {len(res)} of {len(search_space(tid))} configurations finished"
        best = min(res, key=lambda r: r["tuning_mae"])
        sel[tid] = {"config": best["config"], "hp": best["hp"], "tuning_mae": best["tuning_mae"]}
        log[tid] = res
    sel["_rule"] = "lowest pooled MAE on the 1,248 Monday rows of tuning folds WY2012-2015, seed 0; 20 configurations each"
    (RESULTS / "selection.json").write_text(json.dumps(sel, indent=1, default=str))
    (RESULTS / "tuning" / "search_log.json").write_text(json.dumps(log, indent=1, default=str))
    write_manifest("select", started, {"selection_sha256": sha(RESULTS / "selection.json")})
    print(json.dumps({k: v for k, v in sel.items() if not k.startswith("_")}, indent=1, default=str))


def tuned_jobs(which):
    sel = json.loads((RESULTS / "selection.json").read_text())
    folds = [f"WY{w}" for w in wy_list(which)] if which != "retro" else list(RETRO)
    jobs = []
    for tid, base in TUNED.items():
        if base in NEURAL:
            jobs += [{"stage": "tuned", "model": tid, "base": base, "fold": f, "seed": s, "hp": sel[tid]["hp"]}
                     for f in folds for s in SEEDS]
    return jobs


def stage_tuned(a):
    from src.v5.data import load_features, load_snow
    from src.v5.tabular import run_folds
    started = now()
    sel = json.loads((RESULTS / "selection.json").read_text())
    if a.which != "retro":
        feat, snow = load_features(DEV_END), load_snow(DEV_END)
        frames = []
        for tid in ("T-RIDGE", "T-HGB-V5"):
            base = TUNED[tid]
            p, inf = run_folds(feat, snow, a.which, [base], {base: sel[tid]["hp"]}, stage=tid)
            frames.append((tid, p, inf))
        p = frames[0][1].copy()
        p["T-HGB-V5"] = frames[1][1]["T-HGB-V5"].to_numpy()
        p.to_parquet(PREDS / f"tabular_tuned_{a.which}.parquet", index=False)
        infos = {**frames[0][2], **frames[1][2]}
        (RESULTS / "runs" / f"tabular_tuned_{a.which}_info.json").write_text(json.dumps(infos, indent=1, default=float))
    run_pool(tuned_jobs(a.which), a.workers, a.threads, DATA / f"nn_tuned_{a.which}.log")
    write_manifest(f"tuned_{a.which}", started, {"selection_sha256": sha(RESULTS / "selection.json")})


def stage_retro(a):
    """Post-hoc retrospective windows. Refuses to run before selection.json exists."""
    assert (RESULTS / "selection.json").exists(), "run `select` first: retrospective windows are scored after selection"
    started = now()
    a.which = "retro"
    stage_tabular(a)
    jobs = frozen_jobs("retro") + tuned_jobs("retro")
    run_pool(jobs, a.workers, a.threads, DATA / "nn_retro.log")
    write_manifest("retro", started, {"selection_sha256": sha(RESULTS / "selection.json"),
                                      "status": "post-hoc retrospective: periods viewed in v2/v3; no selection used them"})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("tabular")
    t.add_argument("--which", default="dev", choices=["dev", "tune", "confirm", "retro"])
    n = sub.add_parser("nn")
    n.add_argument("--which", default="dev", choices=["dev", "tune", "confirm"])
    n.add_argument("--models", nargs="*")
    for p in (n, sub.add_parser("tune"), sub.add_parser("tuned"), sub.add_parser("retro")):
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--threads", type=int, default=2)
    sub.choices["tune"].add_argument("--tabular-only", action="store_true")
    sub.choices["tuned"].add_argument("--which", default="confirm", choices=["confirm", "tune", "retro"])
    sub.add_parser("select")
    a = ap.parse_args()
    {"tabular": stage_tabular, "nn": stage_nn, "tune": stage_tune, "select": stage_select, "tuned": stage_tuned,
     "retro": stage_retro}[a.stage](a)


if __name__ == "__main__":
    main()

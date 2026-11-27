"""v5 neural jobs: one (model, fold, seed, hyperparameters) training run, using the v4 networks and training loops
unchanged (src.v4.nets.build, src.v4.train.fit / fit_n4 / predict / predict_n4) on src.v5.sequences.SeqData.

A job writes predictions and run information to data/v5/runs/<stage>/<model>/<fold>_s<seed>.npz/.json and the trained
weights to models/v5/nn/<stage>/<model>/<fold>_s<seed>.pt. Jobs that already have outputs are skipped (resumable).
"""
import json
import os
import time

import numpy as np
import pandas as pd

from src.v5 import DATA, DEV_END, MODELS, PANEL_END, STATIONS
from src.v5.registry import N4_CANDIDATES, NEURAL

RUNS = DATA / "runs"
_CACHE = {}


def job_paths(stage, model, fold, seed):
    base = RUNS / stage / model / f"{fold}_s{seed}"
    return base.with_suffix(".npz"), base.with_suffix(".json"), MODELS / "nn" / stage / model / f"{fold}_s{seed}.pt"


def done(job):
    npz, js, _ = job_paths(job["stage"], job["model"], job["fold"], job["seed"])
    return npz.exists() and js.exists()


def _data(end, gefs):
    key = (str(end.date()), gefs)
    if key not in _CACHE:
        from src.v5.data import load_features, load_panel
        from src.v5.sequences import SeqData
        _CACHE[key] = SeqData(load_panel(end), end, gefs=gefs)
        if ("feat", str(end.date())) not in _CACHE:
            _CACHE[("feat", str(end.date()))] = load_features(end if end <= DEV_END else None)
    return _CACHE[key], _CACHE[("feat", str(end.date()))]


def init_worker(threads):
    import torch
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)


def eval_rows(feat, fold):
    from src.v5.tabular import folds
    for fid, tr_end, ev_s, ev_e, tg_e in folds("dev") + folds("retro"):
        if fid == fold:
            v2 = feat[feat.v2_row]
            rows = v2[(v2.date >= ev_s) & (v2.date <= ev_e) & (v2.target_date <= tg_e)].reset_index(drop=True)
            return tr_end, rows
    raise KeyError(fold)


def run_job(job):
    """job: {stage, model (output name), base (NEURAL key), fold, seed, hp}"""
    import torch
    from src.v4.nets import QUANTILES, build
    from src.v4.train import fit, fit_n4, predict, predict_n4, set_seed
    from src.v5.sequences import eval_index, split_at
    if done(job):
        return job, "skipped"
    cfg = NEURAL[job["base"]]
    hp = job["hp"]
    end = DEV_END if job["fold"].startswith("WY") else PANEL_END
    data, feat = _data(end, cfg["gefs"])
    tr_end, rows = eval_rows(feat, job["fold"])
    fit_idx, val_idx = split_at(data.samples, tr_end, stride=2)
    ev_idx = eval_index(data.samples, rows)
    L, seed = int(hp["length"]), int(job["seed"])
    arch = {k: hp[k] for k in ("hidden", "layers", "emb_dim", "dropout", "kernel") if k in hp}
    train = {"max_epochs": hp["max_epochs"], "patience": hp["patience"], "batch": hp["batch"], "lr": hp["lr"],
             "weight_decay": hp["weight_decay"]}
    exclude = tuple(cfg["exclude"])
    set_seed(seed)
    if cfg["kind"] == "n4":
        cand = N4_CANDIDATES[hp.get("cand", cfg["cand"])]
        model = build("n4", data.n_inputs, data.n_static, len(STATIONS), L, n_future=data.n_future,
                      r_max=cand["r_max"], mode=cand["mode"], **arch)
        info = fit_n4(model, data, fit_idx, val_idx, L, cand["weights"], seed=seed, drop_sources=exclude, **train)
        pred_fn = lambda idx, drop=(): predict_n4(model, data, idx, L, drop_sources=tuple(drop) + exclude)  # noqa: E731
    else:
        model = build(cfg["kind"], data.n_inputs, data.n_static, len(STATIONS), L, **arch)
        info = fit(model, data, fit_idx, val_idx, L, seed=seed, drop_sources=exclude, **train)
        pred_fn = lambda idx, drop=(): predict(model, data, idx, L, drop_sources=tuple(drop) + exclude)  # noqa: E731
    t0 = time.perf_counter()
    q = pred_fn(ev_idx)
    infer_s = time.perf_counter() - t0
    med = QUANTILES.index(0.5)
    robust = {"no_flow": pred_fn(ev_idx, ("flow",))[:, med], "no_snow": pred_fn(ev_idx, ("snow",))[:, med],
              "no_weather": pred_fn(ev_idx, ("weather_dam", "weather_box"))[:, med]}
    if cfg["gefs"]:
        robust["no_gefs"] = pred_fn(ev_idx, ("gefs",))[:, med]
    fit_pred = pred_fn(fit_idx)[:, med]
    y_fit = data.samples.y30.to_numpy()[fit_idx]
    npz, js, pt = job_paths(job["stage"], job["model"], job["fold"], seed)
    npz.parent.mkdir(parents=True, exist_ok=True)
    pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "job": job, "n_static": data.n_static, "n_inputs": data.n_inputs}, pt)
    np.savez_compressed(npz, q=q, station=rows.station_id.to_numpy(), date=rows.date.dt.strftime("%Y-%m-%d").to_numpy(),
                        y=rows.target_delta_pp.to_numpy(), **robust)
    out = {k: info[k] for k in ("best_epoch", "epochs_run", "best_val_pinball", "train_seconds", "n_params", "n_fit", "n_val")}
    out.update(job=job, n_test=int(len(rows)), infer_seconds=infer_s, train_mae=float(np.mean(np.abs(fit_pred - y_fit))),
               val_mae_best=float(info["curves"]["val_mae_pp"][info["best_epoch"] - 1]),
               test_mae=float(np.mean(np.abs(q[:, med] - rows.target_delta_pp.to_numpy()))),
               fit_period=[str(data.samples.date.loc[fit_idx].min().date()), str(data.samples.target_date.loc[fit_idx].max().date())],
               val_period=[str(data.samples.date.loc[val_idx].min().date()), str(data.samples.target_date.loc[val_idx].max().date())],
               curves={k: info["curves"][k] for k in ("train_pinball", "val_pinball", "val_mae_pp")},
               threads=int(torch.get_num_threads()), finished_utc=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
    js.write_text(json.dumps(out, default=float))
    return job, f"{out['test_mae']:.3f} ({out['train_seconds']:.0f} s)"


def load_runs(stage, model, folds, seeds):
    """Per-row seed medians, ensemble median and ensemble quantiles for one model over folds."""
    from src.v4.nets import QUANTILES
    frames, infos = [], {}
    med = QUANTILES.index(0.5)
    for fold in folds:
        qs, rob = [], []
        for s in seeds:
            npz, js, _ = job_paths(stage, model, fold, s)
            z = np.load(npz, allow_pickle=True)
            qs.append(z["q"])
            rob.append({k: z[k] for k in z.files if k.startswith("no_")})
            infos[f"{model} {fold} s{s}"] = json.loads(js.read_text())
            if s == seeds[0]:
                f = pd.DataFrame({"station_id": z["station"], "date": pd.to_datetime(z["date"]), "fold": fold})
        for s, q in zip(seeds, qs):
            f[f"{model}_s{s}"] = q[:, med]
        qe = np.mean(qs, axis=0)
        f[model] = np.mean([q[:, med] for q in qs], axis=0)
        for i, tau in enumerate(QUANTILES):
            f[f"{model}_q{tau}"] = qe[:, i]
        for k in rob[0]:
            f[f"{model}_{k}"] = np.mean([r[k] for r in rob], axis=0)
        frames.append(f)
    return pd.concat(frames, ignore_index=True), infos

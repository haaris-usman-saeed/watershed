"""v5 controlled compute benchmark: one training run per model family on the largest development fold (WY2023 training
set: targets before 2022-10-01), sequentially, one CPU thread, nothing else running. Timing only; nothing is saved
except results/v5/compute_benchmark.csv. Run after every other stage:

  OMP_NUM_THREADS=1 .venv-v4/bin/python -m src.v5.compute
"""
import json
import os
import time

import numpy as np
import pandas as pd

from src.v5 import DEV_END, RESULTS, STATIONS
from src.v5.registry import N4_CANDIDATES, NEURAL, NN_DEFAULT, TABULAR, TUNED

FOLD = ("WY2023", pd.Timestamp("2022-10-01"), pd.Timestamp("2022-10-01"), pd.Timestamp("2023-09-30"), pd.Timestamp("2023-12-31"))


def main():
    assert os.environ.get("OMP_NUM_THREADS") == "1", "run with OMP_NUM_THREADS=1"
    import torch
    torch.set_num_threads(1)
    from src.v4.nets import build
    from src.v4.train import fit, fit_n4, predict, predict_n4, set_seed
    from src.v5.data import load_features, load_panel, load_snow
    from src.v5.sequences import SeqData, eval_index, split_at
    from src.v5.tabular import fit_predict, fold_frames
    sel = json.loads((RESULTS / "selection.json").read_text())
    feat, snow = load_features(DEV_END), load_snow(DEV_END)
    train, test, daily = fold_frames(feat, snow, *FOLD[1:])
    rows = []
    tab = [(m, None) for m in ("B-RIDGE-SCR", "A-RIDGE", "B-HGB-SNCR", "HGB-V5", "A-HGB-DAILY")] + \
          [(TUNED[t], sel[t]["hp"]) for t in ("T-RIDGE", "T-HGB-V5")]
    for mid, hp in tab:
        _, info = fit_predict(mid, train, test, daily, hp)
        rows.append({"model": mid if hp is None else [k for k, v in TUNED.items() if v == mid][0], "fold": FOLD[0],
                     "train_rows": info["n_train"], "train_seconds": info["fit_seconds"],
                     "infer_ms_per_1000_rows": 1e6 * info["predict_seconds"] / info["n_test"], "threads": 1})
        print(rows[-1], flush=True)
    data = {g: SeqData(load_panel(DEV_END), DEV_END, gefs=g) for g in (False, True)}
    v2 = feat[feat.v2_row]
    ev = v2[(v2.date >= FOLD[2]) & (v2.date <= FOLD[3])].reset_index(drop=True)
    nets = [(m, dict(NN_DEFAULT, cand=NEURAL[m].get("cand"))) for m in ("B-N1", "A-N1", "A-N2", "A-N3", "A-N4", "A-N4-DIRECT")] + \
           [(t, sel[t]["hp"]) for t in ("T-N1", "T-N2", "T-N3", "T-N4")]
    for name, hp in nets:
        cfg = NEURAL[TUNED.get(name, name)]
        d = data[cfg["gefs"]]
        fi, vi = split_at(d.samples, FOLD[1], stride=2)
        ei = eval_index(d.samples, ev)
        L = int(hp["length"])
        arch = {k: hp[k] for k in ("hidden", "layers", "emb_dim", "dropout", "kernel") if k in hp}
        tr = {k: hp[k] for k in ("max_epochs", "patience", "batch", "lr", "weight_decay")}
        set_seed(0)
        t0 = time.perf_counter()
        if cfg["kind"] == "n4":
            cand = N4_CANDIDATES[hp.get("cand") or cfg["cand"]]
            model = build("n4", d.n_inputs, d.n_static, len(STATIONS), L, n_future=d.n_future, r_max=cand["r_max"],
                          mode=cand["mode"], **arch)
            info = fit_n4(model, d, fi, vi, L, cand["weights"], seed=0, **tr)
            t1 = time.perf_counter()
            predict_n4(model, d, ei, L)
        else:
            model = build(cfg["kind"], d.n_inputs, d.n_static, len(STATIONS), L, **arch)
            info = fit(model, d, fi, vi, L, seed=0, **tr)
            t1 = time.perf_counter()
            predict(model, d, ei, L)
        t2 = time.perf_counter()
        rows.append({"model": name, "fold": FOLD[0], "train_rows": int(len(fi)), "validation_rows": int(len(vi)),
                     "train_seconds": t1 - t0, "epochs_run": info["epochs_run"], "seconds_per_epoch": (t1 - t0) / info["epochs_run"],
                     "infer_ms_per_1000_rows": 1e6 * (t2 - t1) / len(ei), "n_params": info["n_params"], "threads": 1})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "compute_benchmark.csv", index=False)


if __name__ == "__main__":
    main()

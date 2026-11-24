"""v4 training: seeded, early-stopped on a chronological inner set, with loss curves and timings.

The loss is the pinball loss over QUANTILES (its 0.5 term is the L1 loss of the median). Targets are divided
by Y_SCALE for numerical conditioning and multiplied back in predictions.
"""
import random
import time

import numpy as np
import torch

from src.v4.nets import QUANTILES, n_params, pinball

Y_SCALE = 10.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _tensors(data, idx, length, device, **kw):
    x, static, station = data.batch(idx, length, **kw)
    return torch.from_numpy(x).to(device), torch.from_numpy(static).to(device), torch.from_numpy(station).to(device)


@torch.no_grad()
def predict(model, data, idx, length, batch=1024, device="cpu", drop_sources=()):
    """Quantile predictions in percentage points, shape [len(idx), len(QUANTILES)]."""
    model.eval()
    out = []
    idx = np.asarray(idx)
    for i in range(0, len(idx), batch):
        x, s, st = _tensors(data, idx[i:i + batch], length, device, drop_sources=drop_sources)
        out.append(model(x, s, st).cpu().numpy() * Y_SCALE)
    return np.vstack(out) if out else np.empty((0, len(QUANTILES)))


def _val_loss(model, data, idx, length, device, batch=1024, drop_sources=()):
    q = predict(model, data, idx, length, batch=batch, device=device, drop_sources=drop_sources) / Y_SCALE
    y = torch.from_numpy(data.samples.y30.to_numpy()[idx] / Y_SCALE).float()
    pin = float(pinball(torch.from_numpy(q).float(), y))
    mae = float(np.mean(np.abs(q[:, QUANTILES.index(0.5)] * Y_SCALE - y.numpy() * Y_SCALE)))
    return pin, mae


def fit(model, data, fit_idx, val_idx, length, seed=0, max_epochs=40, patience=5, batch=256, lr=1e-3,
        weight_decay=1e-4, source_dropout=0.0, drop_sources=(), device="cpu", log=None):
    """Train until the early-stopping pinball loss has not improved for `patience` epochs; restore the best
    weights. `drop_sources` removes source groups from every training and early-stopping input (ablations).
    Returns a summary with per-epoch curves, the best epoch, timings and the parameter count."""
    set_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    y_all = data.samples.y30.to_numpy(np.float32) / Y_SCALE
    curves = {"train_pinball": [], "val_pinball": [], "val_mae_pp": [], "epoch_seconds": []}
    best, best_state, bad = np.inf, None, 0
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        te = time.perf_counter()
        model.train()
        order = rng.permutation(fit_idx)
        tot, n = 0.0, 0
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            x, s, st = _tensors(data, b, length, device, rng=rng, source_dropout=source_dropout,
                                drop_sources=drop_sources)
            y = torch.from_numpy(y_all[b]).to(device)
            loss = pinball(model(x, s, st), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * len(b)
            n += len(b)
        vp, vm = _val_loss(model, data, val_idx, length, device, drop_sources=drop_sources)
        curves["train_pinball"].append(tot / max(n, 1))
        curves["val_pinball"].append(vp)
        curves["val_mae_pp"].append(vm)
        curves["epoch_seconds"].append(time.perf_counter() - te)
        if log:
            log(f"    epoch {epoch + 1}: train {tot / max(n, 1):.4f} val {vp:.4f} (MAE {vm:.2f} pp)")
        if vp < best - 1e-5:
            best, bad = vp, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_epoch": int(np.argmin(curves["val_pinball"]) + 1), "epochs_run": len(curves["val_pinball"]),
            "best_val_pinball": float(best), "train_seconds": time.perf_counter() - t0, "n_params": n_params(model),
            "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)), "curves": curves}

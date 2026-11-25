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


# ---------------------------------------------------------------- N4 (water-balance encoder-decoder)

N4_TERMS = ("quantile", "storage", "inflow", "outflow", "feasible", "residual")


def _n4_tensors(data, idx, length, device, **kw):
    x, s, st = _tensors(data, idx, length, device, **kw)
    return x, s, st, torch.from_numpy(data.future(idx)).to(device)


def _masked_l1(pred, target, mask):
    m = mask.to(pred.dtype)
    return (torch.abs(pred - target) * m).sum() / m.sum().clamp(min=1.0)


def n4_loss(out, tgt, weights):
    """Composite N4 objective (docs/v4/ARCHITECTURE.md), every term in Y_SCALE units of pp:
    quantile  pinball loss of the 30-day change quantiles (the primary target);
    storage   L1 on the daily trajectory dS(k), where that day's storage label exists in the same segment;
    inflow, outflow  L1 on cumulative 30-day inflow and outflow, where the full 30-day label window is observed;
    feasible  hinge on the projected level S(t) + dS(k) below 0% or above 100% of capacity;
    residual  mean |r(k)|. The design's separate "balance" and "residual" penalties are the same quantity here
              (dS(k) minus the water-balance sum is exactly r(k)), so they are one term.
    Terms the model does not produce (flows and residual in the direct ablation) are skipped.
    Returns (total, {term: value})."""
    terms = {"quantile": pinball(out["q"], tgt["y30"] / Y_SCALE),
             "storage": _masked_l1(out["traj"], tgt["traj"] / Y_SCALE, tgt["traj_mask"])}
    level = tgt["s0_pct"][:, None] + out["traj"] * Y_SCALE
    terms["feasible"] = (torch.relu(-level) + torch.relu(level - 100.0)).mean() / Y_SCALE
    if "cum_in" in out:
        terms["inflow"] = _masked_l1(out["cum_in"], tgt["cum_in"] / Y_SCALE, tgt["cum_in_ok"])
        terms["outflow"] = _masked_l1(out["cum_out"], tgt["cum_out"] / Y_SCALE, tgt["cum_out_ok"])
        terms["residual"] = out["residual"].abs().mean()
    total = sum(weights.get(k, 0.0) * v for k, v in terms.items())
    return total, terms


def _target_tensors(data, idx, device):
    return {k: torch.from_numpy(v).to(device) for k, v in data.targets(idx).items()}


@torch.no_grad()
def predict_n4(model, data, idx, length, batch=1024, device="cpu", drop_sources=(), parts=False):
    """Quantiles of the 30-day change in pp, [len(idx), len(QUANTILES)]. With parts=True also returns a dict of
    per-row components in pp: cumulative predicted inflow, release and loss over 30 days, and the residual r(30)
    (water-balance mode only)."""
    model.eval()
    idx = np.asarray(idx)
    qs, comp = [], {"cum_in": [], "cum_out": [], "cum_loss": [], "residual_30": []}
    for i in range(0, len(idx), batch):
        x, s, st, fut = _n4_tensors(data, idx[i:i + batch], length, device, drop_sources=drop_sources)
        out = model(x, s, st, fut)
        qs.append(out["q"].cpu().numpy() * Y_SCALE)
        if parts and "cum_in" in out:
            comp["cum_in"].append(out["cum_in"].cpu().numpy() * Y_SCALE)
            comp["cum_out"].append(out["cum_out"].cpu().numpy() * Y_SCALE)
            comp["cum_loss"].append(out["loss"].sum(1).cpu().numpy() * Y_SCALE)
            comp["residual_30"].append(out["residual"][:, -1].cpu().numpy() * Y_SCALE)
    q = np.vstack(qs) if qs else np.empty((0, len(QUANTILES)))
    if not parts:
        return q
    return q, {k: np.concatenate(v) for k, v in comp.items() if v}


def fit_n4(model, data, fit_idx, val_idx, length, weights, seed=0, max_epochs=40, patience=5, batch=256, lr=1e-3,
           weight_decay=1e-4, source_dropout=0.0, drop_sources=(), device="cpu", log=None):
    """Train N4 on the composite loss. Early stopping uses the same criterion as N1-N3 (pinball loss of the 30-day
    change quantiles on the early-stopping set), so the stopping rule does not depend on the loss weights.
    Returns the same summary as fit(), plus the mean of each loss term per epoch."""
    set_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    y_val = data.samples.y30.to_numpy()[np.asarray(val_idx)]
    curves = {"train_loss": [], "train_pinball": [], "val_pinball": [], "val_mae_pp": [], "epoch_seconds": [],
              "train_terms": []}
    best, best_state, bad = np.inf, None, 0
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        te = time.perf_counter()
        model.train()
        order = rng.permutation(fit_idx)
        tot, pin, n, term_sum = 0.0, 0.0, 0, {}
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            x, s, st, fut = _n4_tensors(data, b, length, device, rng=rng, source_dropout=source_dropout,
                                        drop_sources=drop_sources)
            loss, terms = n4_loss(model(x, s, st, fut), _target_tensors(data, b, device), weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * len(b)
            pin += terms["quantile"].detach().item() * len(b)
            for k, v in terms.items():
                term_sum[k] = term_sum.get(k, 0.0) + v.detach().item() * len(b)
            n += len(b)
        q = predict_n4(model, data, val_idx, length, device=device, drop_sources=drop_sources)
        vp = float(pinball(torch.from_numpy(q / Y_SCALE).float(), torch.from_numpy(y_val / Y_SCALE).float()))
        vm = float(np.mean(np.abs(q[:, QUANTILES.index(0.5)] - y_val)))
        curves["train_loss"].append(tot / max(n, 1))
        curves["train_pinball"].append(pin / max(n, 1))
        curves["val_pinball"].append(vp)
        curves["val_mae_pp"].append(vm)
        curves["epoch_seconds"].append(time.perf_counter() - te)
        curves["train_terms"].append({k: v / max(n, 1) for k, v in term_sum.items()})
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

"""v4 PyTorch sequence models (requirements-v4.txt). Small by design; complexity is not assumed to help.

- N1 SharedRNN(cell="lstm"): one LSTM for all six reservoirs, with a reservoir embedding and static inputs
  concatenated to every time step and to the head (Kratzert et al. 2019 style "regional" model).
- N2 SharedRNN(cell="gru").
- N3 TCN: causal dilated 1-D convolutions over the same inputs.
- N4 WaterBalanceED: shared encoder-decoder LSTM with inflow, release and loss heads and a water-balance
  integration layer (docs/v4/ARCHITECTURE.md). Its forward pass also takes the known-future decoder inputs and
  returns a dict (quantiles, daily trajectory, flow components); see the class docstring.

All heads output non-crossing quantiles of the 30-day storage change (in units of 10 pp, see train.Y_SCALE):
the median plus cumulative softplus gaps on each side. The point forecast is the median.
"""
import torch
from torch import nn
import torch.nn.functional as F

QUANTILES = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
MEDIAN = QUANTILES.index(0.50)
INTERVALS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975)}


class QuantileHead(nn.Module):
    def __init__(self, d_in, n_q=len(QUANTILES), mid=MEDIAN):
        super().__init__()
        self.lin, self.mid = nn.Linear(d_in, n_q), mid

    def forward(self, h):
        z = self.lin(h)
        m = z[:, self.mid:self.mid + 1]
        up = m + torch.cumsum(F.softplus(z[:, self.mid + 1:]), dim=1)
        lo = m - torch.cumsum(F.softplus(z[:, :self.mid].flip(1)), dim=1).flip(1)
        return torch.cat([lo, m, up], dim=1)


class SharedRNN(nn.Module):
    def __init__(self, n_in, n_static, n_stations, cell="lstm", hidden=64, layers=1, emb_dim=4, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_stations, emb_dim)
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        d_ctx = emb_dim + n_static
        self.rnn = rnn(n_in + d_ctx, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden + d_ctx, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = QuantileHead(hidden)

    def forward(self, x, static, station):
        ctx = torch.cat([self.emb(station), static], dim=1)
        out, _ = self.rnn(torch.cat([x, ctx[:, None, :].expand(-1, x.shape[1], -1)], dim=2))
        return self.head(self.mlp(torch.cat([self.drop(out[:, -1]), ctx], dim=1)))


class TCN(nn.Module):
    """Causal dilated convolutions (dilations 1, 2, 4, ... until the receptive field covers the window)."""

    def __init__(self, n_in, n_static, n_stations, hidden=64, kernel=3, length=180, emb_dim=4, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_stations, emb_dim)
        d_ctx = emb_dim + n_static
        layers, d, field, c_in = [], 1, 1, n_in + d_ctx
        while field < length:
            layers.append(nn.Conv1d(c_in, hidden, kernel, dilation=d))
            field += (kernel - 1) * d
            c_in, d = hidden, d * 2
        self.convs = nn.ModuleList(layers)
        self.pads = [(kernel - 1) * c.dilation[0] for c in self.convs]
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden + d_ctx, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = QuantileHead(hidden)

    def forward(self, x, static, station):
        ctx = torch.cat([self.emb(station), static], dim=1)
        h = torch.cat([x, ctx[:, None, :].expand(-1, x.shape[1], -1)], dim=2).transpose(1, 2)
        for conv, pad in zip(self.convs, self.pads):
            h = self.drop(F.relu(conv(F.pad(h, (pad, 0)))))
        return self.head(self.mlp(torch.cat([h[:, :, -1], ctx], dim=1)))


FLOW_SCALE = 0.5       # pp of capacity per day per unit of flow-head output (daily flows are mostly 0-2 pp/day)
LOSS_SCALE = 0.02      # pp per day per unit; evaporation and seepage are about 0.005-0.01 pp/day (FLOW_DATA.md)


class WaterBalanceED(nn.Module):
    """N4: shared encoder-decoder LSTM with a water-balance integration layer.

    Encoder: LSTM over the past window (the N1 inputs), with the reservoir embedding and static inputs at every step.
    Decoder: LSTM over days k = 1..30 after the origin, initialised from the encoder's final state; its inputs are
    known-future values only (calendar and lead time, SequenceData.future) plus the same context.

    mode="water_balance" (N4). Per decoder day k, in percentage points of capacity per day:
      inflow  q_in(k)  unbounded (operator-computed inflow can be slightly negative)
      release q_out(k) >= 0 (softplus)
      loss    l(k)     >= 0 (softplus; evaporation and seepage)
    and a bounded residual r(k) = r_max * tanh(.) in pp. The predicted change is
      dS(k) = sum over j <= k of [q_in(j) - q_out(j) - l(j)] + r(k).
    With r_max = 0 the predicted change equals predicted inflow minus predicted release and loss exactly. That is
    an accounting identity between predicted terms, not a guarantee that real water is conserved: the inflow
    labels are operator-computed and the losses are learned. Describe it as "water-balance-structured".

    mode="direct" (ablation): the same encoder-decoder, but dS(k) comes from a direct per-day head, with no flow
    decomposition and no integration.

    Quantiles of the 30-day change: the median is dS(30) itself; the other levels add cumulative softplus gaps
    computed from the final decoder state, so they never cross. All outputs are in units of y_scale pp
    (train.Y_SCALE), like N1-N3.
    """

    def __init__(self, n_in, n_static, n_stations, n_future=3, hidden=64, layers=1, emb_dim=4, dropout=0.2,
                 r_max=1.0, mode="water_balance", y_scale=10.0):
        super().__init__()
        if mode not in ("water_balance", "direct"):
            raise ValueError(mode)
        self.mode, self.r_max, self.y_scale = mode, float(r_max), float(y_scale)
        self.emb = nn.Embedding(n_stations, emb_dim)
        d_ctx = emb_dim + n_static
        drop_rnn = dropout if layers > 1 else 0.0
        self.enc = nn.LSTM(n_in + d_ctx, hidden, num_layers=layers, batch_first=True, dropout=drop_rnn)
        self.dec = nn.LSTM(n_future + d_ctx, hidden, num_layers=layers, batch_first=True, dropout=drop_rnn)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden + d_ctx, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.day = nn.Linear(hidden, 4 if mode == "water_balance" else 1)
        self.gaps = nn.Linear(hidden, len(QUANTILES) - 1)
        if mode == "water_balance":           # start with inflow ~= release ~= 0.35 pp/day and a small loss
            with torch.no_grad():
                self.day.bias.copy_(torch.tensor([0.7, 0.0, 0.0, 0.0]))

    def forward(self, x, static, station, future):
        ctx = torch.cat([self.emb(station), static], dim=1)
        _, state = self.enc(torch.cat([x, ctx[:, None, :].expand(-1, x.shape[1], -1)], dim=2))
        out, _ = self.dec(torch.cat([future, ctx[:, None, :].expand(-1, future.shape[1], -1)], dim=2), state)
        h = self.mlp(torch.cat([self.drop(out), ctx[:, None, :].expand(-1, out.shape[1], -1)], dim=2))
        z = self.day(h)
        res = {}
        if self.mode == "water_balance":
            q_in = FLOW_SCALE * z[..., 0]
            q_out = FLOW_SCALE * F.softplus(z[..., 1])
            loss = LOSS_SCALE * F.softplus(z[..., 2])
            r = self.r_max * torch.tanh(z[..., 3])
            traj_pp = torch.cumsum(q_in - q_out - loss, dim=1) + r
            res = {"inflow": q_in / self.y_scale, "release": q_out / self.y_scale, "loss": loss / self.y_scale,
                   "residual": r / self.y_scale, "cum_in": q_in.sum(1) / self.y_scale,
                   "cum_out": q_out.sum(1) / self.y_scale}
        else:
            traj_pp = self.y_scale * z[..., 0]
        traj = traj_pp / self.y_scale
        m = traj[:, -1:]
        g = F.softplus(self.gaps(h[:, -1]))
        up = m + torch.cumsum(g[:, MEDIAN:], dim=1)
        lo = m - torch.cumsum(g[:, :MEDIAN].flip(1), dim=1).flip(1)
        res.update(q=torch.cat([lo, m, up], dim=1), traj=traj)
        return res


def build(kind, n_in, n_static, n_stations, length, **kw):
    if kind == "n4":
        return WaterBalanceED(n_in, n_static, n_stations, **kw)
    if kind in ("lstm", "gru"):
        return SharedRNN(n_in, n_static, n_stations, cell=kind, **kw)
    if kind == "tcn":
        kw = {k: v for k, v in kw.items() if k != "layers"}        # depth follows the receptive field instead
        return TCN(n_in, n_static, n_stations, length=length, **kw)
    raise ValueError(kind)


def pinball(q_pred, y, quantiles=QUANTILES):
    tau = torch.as_tensor(quantiles, dtype=q_pred.dtype, device=q_pred.device)[None, :]
    e = y[:, None] - q_pred
    return torch.maximum(tau * e, (tau - 1) * e).mean()


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

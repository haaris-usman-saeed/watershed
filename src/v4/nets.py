"""v4 PyTorch sequence models (requirements-v4.txt). Small by design; complexity is not assumed to help.

- N1 SharedRNN(cell="lstm"): one LSTM for all six reservoirs, with a reservoir embedding and static inputs
  concatenated to every time step and to the head (Kratzert et al. 2019 style "regional" model).
- N2 SharedRNN(cell="gru").
- N3 TCN: causal dilated 1-D convolutions over the same inputs.

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


def build(kind, n_in, n_static, n_stations, length, **kw):
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

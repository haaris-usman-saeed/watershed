"""v5 sequence data for the networks: src.v4.sequences.SequenceData extended with an explicit end date and optional
GEFS static inputs. With gefs=False and a panel ending before 2024-01-01 it produces exactly the v4 arrays (tested).

Static inputs per origin t:
- gefs=False (original v4 configurations): [log10(capacity) - 6]
- gefs=True (Experiment A): + [arcsinh(gefs d1-7 mm)/3, arcsinh(gefs d8-16 mm)/3, available (0/1), age_days/7], using the
  latest usable Monday cycle issued at or before t (age 0-6 days; src.v5.data.align_gefs); 0 with available = 0 otherwise.
All other inputs, the information rule, masks, targets and known-future decoder inputs are those of v4.
"""
import numpy as np
import pandas as pd

from src.v4.sequences import CFS_DAY_AF, CHANNELS, FUTURE, HORIZON, INNER_VAL_DAYS, KNOWN, _scaled
from src.v5 import STATIONS


class SeqData:
    def __init__(self, panel, end, lmax=365, gefs=False, stations=STATIONS):
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        assert panel.date.max() < end, f"panel must end before {end.date()}"
        self.lmax, self.stations, self.gefs, self.end = lmax, list(stations), gefs, end
        vals, seg, known, offs, frames = [], [], [], {}, []
        pos = 0
        for si, st in enumerate(self.stations):
            g = panel[panel.station_id == st].sort_values("date").reset_index(drop=True)
            assert g.date.diff().dropna().eq(pd.Timedelta(days=1)).all(), f"{st}: panel must be a full calendar"
            n = len(g)
            pad_dates = pd.date_range(end=g.date.iloc[0] - pd.Timedelta(days=1), periods=lmax, freq="D")
            all_dates = pad_dates.append(pd.DatetimeIndex(g.date))
            doy = all_dates.dayofyear.to_numpy()
            known.append(np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], axis=1))
            vals.append(np.vstack([np.full((lmax, len(CHANNELS)), np.nan), _scaled(g)]))
            seg.append(np.concatenate([np.full(lmax, -1), g.seg.to_numpy(int)]))
            offs[st] = pos + lmax
            s_af, cap = g.storage_af.to_numpy(float), g.capacity_af.to_numpy(float)
            q_in = g.inflow_cfs.to_numpy(float) * CFS_DAY_AF
            q_out = g.outflow_cfs.to_numpy(float) * CFS_DAY_AF
            frames.append(self._samples(st, si, g, s_af, cap, q_in, q_out, offs[st]))
            pos += lmax + n
        self.values = np.vstack(vals).astype(np.float32)
        self.observed = ~np.isnan(self.values)
        self.values = np.nan_to_num(self.values, nan=0.0)
        self.seg = np.concatenate(seg)
        self.known = np.vstack(known).astype(np.float32)
        self.offsets = offs
        self.samples = pd.concat(frames, ignore_index=True)
        self._targets = None
        self._static = self._static_array()

    @staticmethod
    def _samples(st, si, g, s_af, cap, q_in, q_out, off):
        dates, seg = g.date, g.seg.to_numpy(int)
        n, h = len(dates), HORIZON
        t = np.arange(n - h)
        y30 = 100 * (s_af[t + h] - s_af[t]) / cap[t]
        ok = np.isfinite(y30) & (seg[t] == seg[t + h])
        k = np.arange(1, h + 1)
        fut = s_af[t[:, None] + k[None, :]]
        traj = 100 * (fut - s_af[t][:, None]) / cap[t][:, None]
        traj_mask = np.isfinite(traj) & (seg[t[:, None] + k[None, :]] == seg[t][:, None])
        cin = q_in[t[:, None] + k[None, :]]
        cout = q_out[t[:, None] + k[None, :]]
        df = pd.DataFrame({"station_id": st, "station_idx": si, "row": off + t, "date": dates.to_numpy()[t],
                           "target_date": dates.to_numpy()[t + h], "y30": y30,
                           "cum_in": 100 * cin.sum(1) / cap[t], "cum_in_ok": np.isfinite(cin).all(1),
                           "cum_out": 100 * cout.sum(1) / cap[t], "cum_out_ok": np.isfinite(cout).all(1),
                           "s0_pct": 100 * s_af[t] / cap[t], "log_cap": np.log10(cap[t]) - 6.0})
        for c in ("gefs_p_d1_7_mm", "gefs_p_d8_16_mm", "gefs_age_days"):
            df[c] = g[c].to_numpy(float)[t] if c in g else np.nan
        df["traj"] = list(np.where(traj_mask, traj, 0.0).astype(np.float32))
        df["traj_mask"] = list(traj_mask)
        return df[ok].reset_index(drop=True)

    def _static_array(self):
        s = self.samples
        cols = [s.log_cap.to_numpy(np.float32)]
        if self.gefs:
            ok = s.gefs_p_d1_7_mm.notna().to_numpy() & s.gefs_p_d8_16_mm.notna().to_numpy()
            cols += [np.where(ok, np.arcsinh(s.gefs_p_d1_7_mm.fillna(0).to_numpy()) / 3.0, 0.0),
                     np.where(ok, np.arcsinh(s.gefs_p_d8_16_mm.fillna(0).to_numpy()) / 3.0, 0.0),
                     ok.astype(float), np.where(ok, s.gefs_age_days.fillna(0).to_numpy() / 7.0, 0.0)]
        return np.stack(cols, axis=1).astype(np.float32)

    @property
    def n_static(self):
        return self._static.shape[1]

    @property
    def n_inputs(self):
        return 2 * len(CHANNELS) + len(KNOWN)

    @property
    def n_future(self):
        return len(FUTURE)

    def future(self, idx):
        rows = self.samples.row.to_numpy()[np.asarray(idx)]
        g = rows[:, None] + np.arange(1, HORIZON + 1)[None, :]
        lead = np.broadcast_to((np.arange(1, HORIZON + 1, dtype=np.float32) / HORIZON)[None, :, None],
                               (len(rows), HORIZON, 1))
        return np.concatenate([self.known[g], lead], axis=2).astype(np.float32)

    def targets(self, idx):
        if self._targets is None:
            s = self.samples
            self._targets = {
                "y30": s.y30.to_numpy(np.float32),
                "traj": np.stack(s.traj.to_numpy()).astype(np.float32).reshape(len(s), HORIZON),
                "traj_mask": np.stack(s.traj_mask.to_numpy()).astype(bool).reshape(len(s), HORIZON),
                "cum_in": np.nan_to_num(s.cum_in.to_numpy(np.float32)), "cum_in_ok": s.cum_in_ok.to_numpy(bool),
                "cum_out": np.nan_to_num(s.cum_out.to_numpy(np.float32)), "cum_out_ok": s.cum_out_ok.to_numpy(bool),
                "s0_pct": s.s0_pct.to_numpy(np.float32)}
        i = np.asarray(idx)
        return {k: v[i] for k, v in self._targets.items()}

    def batch(self, idx, length, drop_sources=(), rng=None, source_dropout=0.0):
        """As src.v4.sequences.SequenceData.batch; static has n_static columns. Dropping the source group "gefs"
        zeroes the GEFS static inputs and their availability flag."""
        assert 1 <= length <= self.lmax
        idx = np.asarray(idx)
        rows = self.samples.row.to_numpy()[idx]
        gidx = rows[:, None] - length + 1 + np.arange(length)[None, :]
        vals, obs = self.values[gidx], self.observed[gidx].copy()
        lags = np.array([lag for _, lag, _ in CHANNELS])
        pos = np.arange(length)
        obs &= (pos[:, None] <= length - 1 - lags[None, :])[None, :, :]
        same_seg = self.seg[gidx] == self.seg[rows][:, None]
        obs[:, :, 0] &= same_seg
        groups = np.array([g for _, _, g in CHANNELS])
        for gname in drop_sources:
            if gname != "gefs":
                obs[:, :, groups == gname] = False
        if source_dropout > 0:
            rng = rng or np.random.default_rng(0)
            for gname in ("flow", "snow", "weather_dam", "weather_box"):
                drop = rng.random(len(rows)) < source_dropout
                obs[np.ix_(drop, np.ones(length, bool), groups == gname)] = False
        x = np.concatenate([np.where(obs, vals, 0.0), obs.astype(np.float32), self.known[gidx]], axis=2)
        static = self._static[idx].copy()
        if self.gefs and "gefs" in drop_sources:
            static[:, 1:] = 0.0
        return x.astype(np.float32), static, self.samples.station_idx.to_numpy(np.int64)[idx]


def split_at(samples, start, inner_val_days=INNER_VAL_DAYS, stride=2):
    """Fitting and early-stopping indices for a fold starting at `start` (the v4 rule, src.v4.sequences.fold_split):
    early-stopping origins are the `inner_val_days` before `start` with target before `start`; the fitting set keeps
    samples whose target precedes the first early-stopping origin; every `stride`-th day of the fitting set."""
    start = pd.Timestamp(start)
    cut = start - pd.Timedelta(days=inner_val_days)
    fit = samples.index[samples.target_date < cut].to_numpy()
    val = samples.index[(samples.date >= cut) & (samples.target_date < start)].to_numpy()
    if stride > 1:
        fit = fit[(samples.date.loc[fit] - samples.date.min()).dt.days.to_numpy() % stride == 0]
    if len(fit) and len(val):
        assert samples.target_date.loc[fit].max() < samples.date.loc[val].min()
    assert samples.target_date.loc[np.concatenate([fit, val])].max() < start
    return fit, val


def eval_index(samples, rows):
    """Sample indices of evaluation rows (station_id, date), in the rows' order; the sequence target must equal the
    tabular target on every row."""
    key = samples.reset_index().set_index(["station_id", "date"])["index"]
    idx = key.reindex(pd.MultiIndex.from_frame(rows[["station_id", "date"]])).to_numpy()
    if np.isnan(idx.astype(float)).any():
        raise ValueError(f"{int(np.isnan(idx.astype(float)).sum())} evaluation rows have no sequence sample")
    idx = idx.astype(int)
    diff = np.abs(samples.y30.to_numpy()[idx] - rows.target_delta_pp.to_numpy())
    assert diff.max() < 1e-6, f"sequence target differs from the tabular target by up to {diff.max()}"
    return idx

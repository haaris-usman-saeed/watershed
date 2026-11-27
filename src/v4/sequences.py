"""v4 leakage-safe sequence construction, targets and chronological folds (numpy only; no PyTorch needed).

Information rule for an origin t (a reservoir-day whose end-of-day storage is known):
- the history window is the L days t-L+1 .. t;
- CDEC storage, inflow and outflow are used up to and including day t (as storage is in v2/v3);
- basin snow is used up to t-1 (SNOW_LAG, as in v2);
- NASA POWER weather is used up to t-7 (WEATHER_LAG, as in v2/v3; near-real-time POWER lags a few days);
- storage on days in a different series-break segment from t is masked;
- calendar seasonality (day-of-year sine and cosine) is known for every day.
Nothing on or after t+1 is ever an input. Each channel has an explicit observed/allowed mask; a missing or
withheld value is 0 with mask 0, never filled.

Targets for origin t, all in percentage points of the capacity in force at t (the v2/v3 target):
- y30:   100 * (S[t+30] - S[t]) / cap[t]   (exact calendar day; same segment; else no sample)
- traj:  100 * (S[t+k] - S[t]) / cap[t] for k = 1..30, with a mask
- cum_in, cum_out: 100 * sum over days t+1..t+30 of flow (AF) / cap[t], with masks (auxiliary labels only)
- s0_pct: 100 * S[t] / cap[t], the storage level at the origin (known at t; used for N4's feasibility penalty)

Known-future decoder inputs (N4) for days t+1..t+30 are calendar only: day-of-year sine and cosine and the lead
time k/30. No observation after t is ever a decoder input.

Folds: for development water year W (Oct 1 of W-1 .. Sep 30 of W), training samples have target_date before
Oct 1 of W-1 (the v2 rule, so every training outcome is observed before the fold starts). Of those, origins in
the last 365 days before the fold form the early-stopping set, and the fitting set keeps only samples whose
target precedes that set's first origin (purged, so no target window overlaps it). Evaluation rows are the
v2 Monday rows of W. No date on or after DEV_END is ever used.
"""
import numpy as np
import pandas as pd

from src.v4 import DEV_END, STATIONS

HORIZON = 30
SNOW_LAG = 1
WEATHER_LAG = 7
CFS_DAY_AF = 1.983471
INNER_VAL_DAYS = 365
# name, days withheld before the origin, source group (for source dropout and robustness tests)
CHANNELS = [("storage_pct", 0, "storage"), ("inflow_pp", 0, "flow"), ("outflow_pp", 0, "flow"),
            ("swe_in", SNOW_LAG, "snow"),
            ("rain_mm", WEATHER_LAG, "weather_dam"), ("temp_c", WEATHER_LAG, "weather_dam"),
            ("rh_pct", WEATHER_LAG, "weather_dam"),
            ("rain_box_mm", WEATHER_LAG, "weather_box"), ("temp_box_c", WEATHER_LAG, "weather_box"),
            ("rh_box_pct", WEATHER_LAG, "weather_box")]
KNOWN = ["doy_sin", "doy_cos"]
FUTURE = KNOWN + ["lead"]
N_STATIC = 1


def _scaled(g):
    """Fixed physical scalings (no statistics from data, so nothing can leak across folds)."""
    cap = g.capacity_af.to_numpy(float)
    out = {"storage_pct": g.storage_af.to_numpy(float) / cap,
           "inflow_pp": np.arcsinh(100 * g.inflow_cfs.to_numpy(float) * CFS_DAY_AF / cap),
           "outflow_pp": np.arcsinh(100 * g.outflow_cfs.to_numpy(float) * CFS_DAY_AF / cap),
           "swe_in": g.swe_in.to_numpy(float) / 20.0}
    for suf in ("", "_box"):
        out[f"rain{suf}_mm"] = np.arcsinh(g[f"rain{suf}_mm"].to_numpy(float)) / 3.0
        out[f"temp{suf}_c"] = g[f"temp{suf}_c"].to_numpy(float) / 20.0
        out[f"rh{suf}_pct"] = g[f"rh{suf}_pct"].to_numpy(float) / 100.0
    return np.stack([out[c] for c, _, _ in CHANNELS], axis=1)


class SequenceData:
    """Per-reservoir daily arrays, concatenated with LMAX days of padding in front of each reservoir so any
    window can be gathered with one fancy index."""

    def __init__(self, panel, lmax=365, stations=STATIONS):
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        assert panel.date.max() < DEV_END, "v4 development data must end before 2024-01-01"
        self.lmax, self.stations = lmax, list(stations)
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
            offs[st] = pos + lmax                 # global row of this reservoir's first real day
            s_af, cap = g.storage_af.to_numpy(float), g.capacity_af.to_numpy(float)
            q_in = g.inflow_cfs.to_numpy(float) * CFS_DAY_AF
            q_out = g.outflow_cfs.to_numpy(float) * CFS_DAY_AF
            frames.append(self._samples(st, si, g.date, s_af, cap, g.seg.to_numpy(int), q_in, q_out, offs[st]))
            pos += lmax + n
        self.values = np.vstack(vals).astype(np.float32)
        self.observed = ~np.isnan(self.values)
        self.values = np.nan_to_num(self.values, nan=0.0)
        self.seg = np.concatenate(seg)
        self.known = np.vstack(known).astype(np.float32)
        self.offsets = offs
        self.samples = pd.concat(frames, ignore_index=True)
        self._targets = None

    @staticmethod
    def _samples(st, si, dates, s_af, cap, seg, q_in, q_out, off):
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
        df["traj"] = list(np.where(traj_mask, traj, 0.0).astype(np.float32))
        df["traj_mask"] = list(traj_mask)
        return df[ok].reset_index(drop=True)

    @property
    def n_inputs(self):
        return 2 * len(CHANNELS) + len(KNOWN)

    @property
    def n_future(self):
        return len(FUTURE)

    def future(self, idx):
        """Known-future decoder inputs for days t+1..t+HORIZON, [B, HORIZON, len(FUTURE)]: day-of-year sine and
        cosine of each future day, and the lead time k/HORIZON. Calendar only; nothing observed after t."""
        rows = self.samples.row.to_numpy()[np.asarray(idx)]
        g = rows[:, None] + np.arange(1, HORIZON + 1)[None, :]
        lead = np.broadcast_to((np.arange(1, HORIZON + 1, dtype=np.float32) / HORIZON)[None, :, None],
                               (len(rows), HORIZON, 1))
        return np.concatenate([self.known[g], lead], axis=2).astype(np.float32)

    def targets(self, idx):
        """Training labels for sample rows `idx` (pp of the capacity in force at t): y30; the daily trajectory and
        its mask; cumulative 30-day inflow and outflow (0 where the window is incomplete) and their masks; s0_pct.
        Labels are never inputs."""
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
        """Inputs for sample rows `idx` with history `length`: X [B, L, 2C+K] (values*mask, mask, known),
        static [B, 1], station [B]. `drop_sources` removes whole source groups (robustness tests);
        `source_dropout` removes each non-storage source group per sample with that probability (training)."""
        assert 1 <= length <= self.lmax
        s = self.samples.iloc[np.asarray(idx)]
        rows = s.row.to_numpy()
        gidx = rows[:, None] - length + 1 + np.arange(length)[None, :]
        vals, obs = self.values[gidx], self.observed[gidx].copy()
        lags = np.array([lag for _, lag, _ in CHANNELS])
        pos = np.arange(length)
        obs &= (pos[:, None] <= length - 1 - lags[None, :])[None, :, :]            # withheld days before t
        same_seg = self.seg[gidx] == self.seg[rows][:, None]
        obs[:, :, 0] &= same_seg                                                    # storage across a break
        groups = np.array([g for _, _, g in CHANNELS])
        for gname in drop_sources:
            obs[:, :, groups == gname] = False
        if source_dropout > 0:
            rng = rng or np.random.default_rng(0)
            for gname in ("flow", "snow", "weather_dam", "weather_box"):
                drop = rng.random(len(rows)) < source_dropout
                obs[np.ix_(drop, np.ones(length, bool), groups == gname)] = False
        x = np.concatenate([np.where(obs, vals, 0.0), obs.astype(np.float32), self.known[gidx]], axis=2)
        static = s[["log_cap"]].to_numpy(np.float32)
        return x.astype(np.float32), static, s.station_idx.to_numpy(np.int64)


def wy_bounds(wy):
    return pd.Timestamp(f"{wy - 1}-10-01"), pd.Timestamp(f"{wy}-10-01")


def fold_split(samples, wy, inner_val_days=INNER_VAL_DAYS, stride=1):
    """Daily-origin fitting and early-stopping sample indices for development water year `wy`."""
    start, _ = wy_bounds(wy)
    assert start <= DEV_END
    cut = start - pd.Timedelta(days=inner_val_days)
    fit = samples.index[samples.target_date < cut].to_numpy()
    val = samples.index[(samples.date >= cut) & (samples.target_date < start)].to_numpy()
    if stride > 1:
        fit = fit[(samples.date.loc[fit] - samples.date.min()).dt.days.to_numpy() % stride == 0]
    assert samples.target_date.loc[fit].max() < samples.date.loc[val].min() if len(fit) and len(val) else True
    assert samples.target_date.loc[np.concatenate([fit, val])].max() < start
    return fit, val


def monday_rows(samples, weekly, wy):
    """Sample indices for the v2 Monday evaluation rows of water year `wy`, in the weekly table's order.
    The v4 target must equal the v2 target on every row (checked)."""
    start, end = wy_bounds(wy)
    w = weekly[(weekly.date >= start) & (weekly.date < end)]
    assert w.target_date.max() < DEV_END
    key = samples.reset_index().set_index(["station_id", "date"])["index"]
    idx = key.reindex(pd.MultiIndex.from_frame(w[["station_id", "date"]])).to_numpy()
    if np.isnan(idx.astype(float)).any():
        raise ValueError(f"WY{wy}: {np.isnan(idx.astype(float)).sum()} Monday rows have no v4 sample")
    idx = idx.astype(int)
    diff = np.abs(samples.y30.to_numpy()[idx] - w.target_delta_pp.to_numpy())
    assert diff.max() < 1e-6, f"v4 target differs from the v2 target by up to {diff.max()}"
    return idx, w.reset_index(drop=True)

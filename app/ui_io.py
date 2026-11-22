"""File loading, validation and derived summaries. No Streamlit imports here, so the
contract checker (check_contract.py) can reuse exactly what the app runs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import ui_contract as C

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- data dir

def resolve_data_dir(env=None):
    """WATERSHED_DATA_DIR when set (relative paths resolve from CWD, then repo root),
    otherwise <repo>/outputs. Returns (path, source) where source is "env" or "default"."""
    env = os.environ if env is None else env
    raw = (env.get(C.ENV_VAR) or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            from_cwd = Path.cwd() / p
            p = from_cwd if from_cwd.exists() else REPO_ROOT / p
        return p.resolve(), "env"
    default = (REPO_ROOT / C.DEFAULT_DIR_NAME).resolve()
    if not default.exists():   # fresh public deployment: use the newest published release's UI files, if any
        releases = sorted(REPO_ROOT.glob("public_data/watershedshift-public-data-*/ui_outputs"))
        if releases:
            return releases[-1].resolve(), "public release"
    return default, "default"


def fingerprint(data_dir: Path):
    """Cache key: file names with mtime and size, so regenerated outputs reload."""
    out = []
    for name in C.ALL_FILES:
        try:
            s = (data_dir / name).stat()
            out.append((name, s.st_mtime_ns, s.st_size))
        except OSError:
            out.append((name, None, None))
    return tuple(out)


# ---------------------------------------------------------------- reports

@dataclass
class FileReport:
    name: str
    path: str
    found: bool = False
    usable: bool = False
    rows: int = 0
    error: str = ""
    resolved: dict = field(default_factory=dict)      # canonical -> actual column
    missing_required: list = field(default_factory=list)
    missing_optional: list = field(default_factory=list)
    extra_columns: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def status(self):
        if not self.found:
            return "missing"
        if not self.usable:
            return "invalid"
        return "warn" if self.warnings else "ok"


def _coerce(df, spec_map, report):
    for canon, spec in spec_map.items():
        if canon not in df.columns:
            continue
        s = df[canon]
        if spec.kind == "str":
            df[canon] = s.astype("string").str.strip()
        elif spec.kind == "float":
            before = s.notna().sum()
            df[canon] = pd.to_numeric(s, errors="coerce").astype("float64")
            lost = int(before - df[canon].notna().sum())
            if lost:
                report.warnings.append(f"{canon}: {lost} non-numeric values read as blank")
        elif spec.kind == "date":
            before = s.notna().sum()
            d = pd.to_datetime(s, errors="coerce")
            if getattr(d.dt, "tz", None) is not None:
                d = d.dt.tz_localize(None)
            df[canon] = d.dt.normalize()
            lost = int(before - df[canon].notna().sum())
            if lost:
                report.warnings.append(f"{canon}: {lost} unparseable dates read as blank")
    return df


def read_table(data_dir: Path, name: str, spec_map: dict):
    path = data_dir / name
    rep = FileReport(name=name, path=str(path))
    if not path.exists():
        return None, rep
    rep.found = True
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001  surfaced to the UI verbatim
        rep.error = f"could not parse CSV: {exc}"
        return None, rep
    rep.rows = len(raw)
    rename = {}
    for canon, spec in spec_map.items():
        actual = C.find_column(raw.columns, canon, spec)
        if actual is None:
            (rep.missing_required if spec.required else rep.missing_optional).append(canon)
        else:
            rep.resolved[canon] = str(actual)
            rename[actual] = canon
    used = set(rename)
    rep.extra_columns = [str(c) for c in raw.columns if c not in used]
    if rep.missing_required:
        rep.error = "missing required columns: " + ", ".join(rep.missing_required)
        return None, rep
    for canon, actual in rep.resolved.items():
        if actual != canon:
            rep.warnings.append(f"column '{actual}' read as '{canon}' (alias match; confirm against contract)")
    df = raw.rename(columns=rename)[[c for c in spec_map if c in rename.values()]].copy()
    df = _coerce(df, spec_map, rep)
    rep.usable = True
    return df, rep


def read_json(data_dir: Path, name: str):
    path = data_dir / name
    rep = FileReport(name=name, path=str(path))
    if not path.exists():
        return None, rep
    rep.found = True
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        rep.error = f"could not parse JSON: {exc}"
        return None, rep
    rep.usable = True
    return obj, rep


# ---------------------------------------------------------------- semantic checks

def _check_stations(df, rep):
    dup = df["station_id"].duplicated().sum()
    if dup:
        rep.warnings.append(f"{dup} duplicate station_id rows")
    bad_cap = int((df["capacity_af"].isna() | (df["capacity_af"] <= 0)).sum())
    if bad_cap:
        rep.warnings.append(f"{bad_cap} stations with missing or non-positive capacity_af")
    bad_geo = int((~df["lat"].between(32, 42.5) | ~df["lon"].between(-125, -113.5)).sum())
    if bad_geo:
        rep.warnings.append(f"{bad_geo} stations with coordinates outside California's bounding box")


def _check_history(df, stations, rep):
    dup = int(df.duplicated(["station_id", "date"]).sum())
    if dup:
        rep.warnings.append(f"{dup} duplicate (station_id, date) rows; the explorer shows the last of each")
    if stations is not None:
        unknown = sorted(set(df["station_id"].dropna()) - set(stations["station_id"]))
        if unknown:
            rep.warnings.append("station_id not in stations.csv: " + ", ".join(map(str, unknown)))
        if "storage_pct" in df.columns and df["storage_pct"].notna().any():
            over = int((df["storage_pct"] > 102).sum())
        else:
            cap = df.merge(stations[["station_id", "capacity_af"]], on="station_id", how="left")
            over = int((cap["storage_af"] > cap["capacity_af"] * 1.02).sum())
        if over:
            rep.warnings.append(f"{over} days above 102% of capacity in force (verify flood encroachment)")
    neg = int((df["storage_af"] < 0).sum())
    if neg:
        rep.warnings.append(f"{neg} negative storage values")


def _check_predictions(df, stations, rep):
    gap = (df["target_date"] - df["date"]).dt.days
    off = int((gap != C.HORIZON_DAYS).sum())
    if off:
        rep.warnings.append(f"{off} rows where target_date - date is not {C.HORIZON_DAYS} days")
    inverted = int((df["lower_90_pp"] > df["upper_90_pp"]).sum())
    if inverted:
        rep.warnings.append(f"{inverted} rows with lower_90_pp > upper_90_pp")
    outside = int(((df["pred_delta_pp"] < df["lower_90_pp"]) | (df["pred_delta_pp"] > df["upper_90_pp"])).sum())
    if outside:
        rep.warnings.append(f"{outside} rows where the point prediction lies outside its own interval")
    dup = int(df.duplicated(["station_id", "date"]).sum())
    if dup:
        rep.warnings.append(f"{dup} duplicate (station_id, date) forecast rows")
    if stations is not None:
        unknown = sorted(set(df["station_id"].dropna()) - set(stations["station_id"]))
        if unknown:
            rep.warnings.append("station_id not in stations.csv: " + ", ".join(map(str, unknown)))


def _check_metrics(obj, rep):
    if not isinstance(obj, dict):
        rep.usable = False
        rep.error = "metrics.json must be a JSON object"
        return
    if C.METRIC_KEYS["data_mode"] not in obj:
        rep.warnings.append("no data_mode key; the app treats the data as unverified")
    for m in C.MODELS:
        if m.metric_key not in obj:
            rep.missing_optional.append(m.metric_key)
    for k in ("coverage", "halfwidth", "test_count"):
        if C.METRIC_KEYS[k] not in obj:
            rep.missing_optional.append(C.METRIC_KEYS[k])


def normalize_features(obj):
    """Return a list of dicts with canonical keys, or None if the shape is unrecognised."""
    if obj is None:
        return None
    items = obj
    if isinstance(obj, dict):
        if isinstance(obj.get("features"), list):
            items = obj["features"]
        elif all(isinstance(v, str) for v in obj.values()):
            items = [{"name": k, "description": v} for k, v in obj.items()]
        elif all(isinstance(v, dict) for v in obj.values()):
            items = [{"name": k, **v} for k, v in obj.items()]
        else:
            return None
    if not isinstance(items, list):
        return None
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        low = {str(k).lower(): v for k, v in it.items()}
        row = {}
        for canon, aliases in C.FEATURE_FIELD_ALIASES.items():
            for a in aliases:
                if a in low:
                    row[canon] = low[a]
                    break
        known = {a for al in C.FEATURE_FIELD_ALIASES.values() for a in al}
        extras = {k: v for k, v in it.items() if str(k).lower() not in known}
        if extras:
            row["other"] = "; ".join(f"{k}: {v}" for k, v in extras.items())
        if row.get("name"):
            out.append(row)
    return out


# ---------------------------------------------------------------- load

def load_dir(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    reports = {}
    stations, reports[C.STATIONS_FILE] = read_table(data_dir, C.STATIONS_FILE, C.STATION_COLS)
    if stations is not None:
        _check_stations(stations, reports[C.STATIONS_FILE])
        stations = stations.drop_duplicates("station_id", keep="first").reset_index(drop=True)

    history, reports[C.HISTORY_FILE] = read_table(data_dir, C.HISTORY_FILE, C.HISTORY_COLS)
    if history is not None:
        history = history.dropna(subset=["station_id", "date"])
        _check_history(history, stations, reports[C.HISTORY_FILE])
        history = history.sort_values(["station_id", "date"]).reset_index(drop=True)

    preds, reports[C.PREDICTIONS_FILE] = read_table(data_dir, C.PREDICTIONS_FILE, C.PREDICTION_COLS)
    if preds is not None:
        preds = preds.dropna(subset=["station_id", "date"])
        _check_predictions(preds, stations, reports[C.PREDICTIONS_FILE])
        preds = preds.sort_values(["station_id", "date"]).reset_index(drop=True)
        if "reservoir_name" not in preds.columns and stations is not None:
            preds = preds.merge(stations[["station_id", "reservoir_name"]], on="station_id", how="left")

    metrics, reports[C.METRICS_FILE] = read_json(data_dir, C.METRICS_FILE)
    if metrics is not None:
        _check_metrics(metrics, reports[C.METRICS_FILE])
        if not reports[C.METRICS_FILE].usable:
            metrics = None

    feat_raw, reports[C.FEATURES_FILE] = read_json(data_dir, C.FEATURES_FILE)
    features = normalize_features(feat_raw)
    if feat_raw is not None and features is None:
        reports[C.FEATURES_FILE].usable = False
        reports[C.FEATURES_FILE].error = "unrecognised feature_catalog.json shape"

    data_mode = metrics.get(C.METRIC_KEYS["data_mode"]) if isinstance(metrics, dict) else None
    summary = summarize_stations(stations, history, preds) if stations is not None else None
    return dict(stations=stations, history=history, preds=preds, metrics=metrics,
                features=features, reports=reports, data_mode=data_mode, summary=summary)


# ---------------------------------------------------------------- derived

def _season_band(dates, pct, ref_date, window=7):
    """Values from other years within +/- window days of ref_date's day of year."""
    if ref_date is None or len(dates) == 0:
        return None
    doy = dates.dt.dayofyear.to_numpy()
    ref = ref_date.dayofyear
    dist = np.abs(doy - ref)
    dist = np.minimum(dist, 365 - dist)
    mask = (dist <= window) & (dates.dt.year.to_numpy() != ref_date.year)
    vals = pd.Series(pct.to_numpy()[mask], index=dates.dt.year.to_numpy()[mask]).dropna()
    if vals.empty:
        return None
    years = int(vals.index.nunique())
    if years < 2:
        return None
    return dict(lo=float(vals.min()), median=float(vals.median()), hi=float(vals.max()), years=years)


def summarize_stations(stations, history, preds):
    rows = []
    for st_row in stations.itertuples(index=False):
        sid = st_row.station_id
        cap = float(st_row.capacity_af) if pd.notna(st_row.capacity_af) else np.nan
        row = dict(station_id=sid, reservoir_name=st_row.reservoir_name, lat=st_row.lat, lon=st_row.lon,
                   capacity_af=cap, first_date=pd.NaT, last_date=pd.NaT, obs_days=0, span_days=0,
                   coverage_pct=np.nan, longest_gap_days=0, latest_af=np.nan, latest_pct=np.nan,
                   record_min_pct=np.nan, record_max_pct=np.nan, band=None, n_forecasts=0)
        if history is not None:
            h = history[(history["station_id"] == sid) & history["storage_af"].notna()]
            h = h.drop_duplicates("date", keep="last")
            if not h.empty:
                first, last = h["date"].iloc[0], h["date"].iloc[-1]
                span = int((last - first).days) + 1
                gaps = h["date"].diff().dt.days.dropna() - 1
                pct = h["storage_pct"].copy() if "storage_pct" in h.columns and h["storage_pct"].notna().any() else (h["storage_af"] / cap * 100 if cap and cap > 0 else pd.Series(np.nan, index=h.index))
                row.update(first_date=first, last_date=last, obs_days=int(len(h)), span_days=span,
                           coverage_pct=len(h) / span * 100, longest_gap_days=int(gaps.max()) if len(gaps) else 0,
                           latest_af=float(h["storage_af"].iloc[-1]), latest_pct=float(pct.iloc[-1]),
                           record_min_pct=float(pct.min()), record_max_pct=float(pct.max()),
                           band=_season_band(h["date"], pct, last))
        if preds is not None:
            row["n_forecasts"] = int((preds["station_id"] == sid).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def daily_series(history, station_id, capacity_af):
    """Full daily index for one station with storage left blank on unobserved days."""
    h = history[history["station_id"] == station_id].drop_duplicates("date", keep="last")
    if h.empty:
        return h.assign(storage_pct=pd.Series(dtype="float64"))
    idx = pd.date_range(h["date"].min(), h["date"].max(), freq="D")
    out = h.set_index("date").reindex(idx)
    out.index.name = "date"
    out["station_id"] = station_id
    if "storage_pct" not in out.columns or out["storage_pct"].isna().all():
        out["storage_pct"] = out["storage_af"] / capacity_af * 100 if capacity_af and capacity_af > 0 else np.nan
    return out.reset_index()


def break_gaps(df, date_col="date", factor=1.5):
    """Insert blank rows where consecutive dates jump more than factor x the typical step,
    so plotted lines break instead of bridging missing forecasts."""
    if len(df) < 3:
        return df
    d = df[date_col]
    step = d.diff().dt.days.median()
    if not step or np.isnan(step):
        return df
    jumps = d.diff().dt.days > step * factor
    if not jumps.any():
        return df
    pieces, prev = [], 0
    for i in np.flatnonzero(jumps.to_numpy()):
        pieces.append(df.iloc[prev:i])
        blank = df.iloc[[i - 1]].copy()
        for c in blank.columns:
            if c != date_col:
                blank[c] = pd.NaT if pd.api.types.is_datetime64_any_dtype(blank[c]) else np.nan
        blank[date_col] = d.iloc[i - 1] + pd.Timedelta(days=1)
        pieces.append(blank)
        prev = i
    pieces.append(df.iloc[prev:])
    return pd.concat(pieces, ignore_index=True)


def contiguous_segments(df, date_col="date", factor=1.5):
    """Split rows into runs without large date jumps (for drawing interval bands)."""
    if df.empty:
        return []
    d = df[date_col]
    step = d.diff().dt.days.median()
    if not step or np.isnan(step):
        return [df]
    cut = (d.diff().dt.days > step * factor).cumsum()
    return [g for _, g in df.groupby(cut, sort=True)]
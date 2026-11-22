"""Trusted rebuild of the frozen v2 primary model for research forecasting (no new model).

Run: python -m src.forecast_model --build    # refit, verify against the archive, write the artifact
     python -m src.forecast_model --verify   # re-check an existing artifact (hash, card, predictions)

The frozen run (src/train_v2.py) never saved its fitted model. This module refits it with the
identical code path: train_v2.load() with the frozen cutoff (target_date < 2024-01-01), the frozen
hyperparameters from config/v2_frozen.json, FEATURE_SETS["storage_snow"] and models.design().
It then predicts the 593 archived 2024-2025 test rows and REFUSES to write an artifact unless every
prediction matches results/v2/final/test_predictions_public.csv within TOL.

The fitted model is written to models/v2_forecast/ (git-ignored) together with the fixed
companions inference needs: core snow pillows (WY<=2023 rule), the pre-2024 snow climatology,
the frozen sym_season interval offsets and per-station historical interval coverage.
Nothing here reads or uses any 2024-2025 outcome for fitting; the archive is only compared with.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd

from src.features import core_snow_stations
from src.models import FEATURE_SETS, design, hgb, interval_offsets, snow_climatology
from src.train_v2 import CONFIG, frozen_hashes, load

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "models" / "v2_forecast"
MODEL_FILE, CARD_FILE = "model.joblib", "model_card.json"
TRAIN_END = pd.Timestamp("2024-01-01")
TEST_START, TEST_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
CAL_WYS = range(2008, 2024)
ARCHIVE = ROOT / "results/v2/final/test_predictions_public.csv"
RUN_RECORD = ROOT / "results/v2/final/RUN_RECORD.json"
OOF = ROOT / "results/v2/development/oof_predictions.csv"
TOL = 1e-9
SEASONS = ["DJF", "MAM", "JJA", "SON"]
PROVENANCE_FILES = ["data/processed/weekly_features.csv", "data/processed/snow_basin_index.csv",
                    "data/processed/snow_daily.csv", "data/raw_manifest.jsonl", "data/snow_stations.csv",
                    "data/capacity_history.csv", "data/series_breaks.csv", "data/spike_review.csv",
                    "results/v2/development/oof_predictions.csv", "config/v2_frozen.json"]


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_commit():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"commit": None, "clean": None}
    return {"commit": head.stdout.strip(), "clean": git("status", "--porcelain", "--untracked-files=no").stdout.strip() == ""}


def frozen_hyperparameters(cfg):
    hp = dict(cfg["primary_model"]["hyperparameters"])
    assert hp.pop("l2_regularization") == 1.0 and hp.pop("random_state") == 42, "frozen hgb() constants changed"
    return hp


def season_offsets():
    """Frozen sym_season 90% offsets: |OOF residual| 90th percentile (higher) per origin season, WY2008-2023."""
    cfg = json.loads(CONFIG.read_text())
    oof = pd.read_csv(OOF, parse_dates=["date"])
    oof = oof[oof.wy.isin(list(CAL_WYS))]
    cal = pd.DataFrame({"r": oof.target_delta_pp - oof["pred_HGB SWE no weather"], "season": oof.season,
                        "station_id": oof.station_id})
    apply = interval_offsets(cal, cfg["interval"]["method"])
    lo, hi = apply(pd.DataFrame({"season": SEASONS}))
    counts = cal.groupby("season").size().to_dict()
    return ({s: [float(a), float(b)] for s, a, b in zip(SEASONS, lo, hi)},
            {"method": cfg["interval"]["method"], "nominal": cfg["interval"]["nominal"],
             "calibration_water_years": [CAL_WYS.start, CAL_WYS.stop - 1], "n_residuals_by_season": counts})


def fit_frozen():
    """Refit exactly as train_v2.run('final') does. Returns (model, train, test, stations, columns)."""
    cfg = json.loads(CONFIG.read_text())
    train, test, _ = load(TRAIN_END, TEST_START, TEST_END)
    stations = sorted(train.station_id.unique())
    cols = FEATURE_SETS["storage_snow"]
    model = hgb(frozen_hyperparameters(cfg)).fit(design(train, cols, stations), train.target_delta_pp)
    return model, train, test, stations, cols


def verify_against_archive(model, test, stations, cols):
    arc = pd.read_csv(ARCHIVE, parse_dates=["date"])[["station_id", "date", "pred_delta_pp"]]
    got = test[["station_id", "date"]].assign(refit=model.predict(design(test, cols, stations)))
    m = got.merge(arc, on=["station_id", "date"], how="outer", indicator=True)
    if (m._merge != "both").any() or len(m) != len(arc):
        raise SystemExit(f"REFUSED: test rows differ from the archive ({(m._merge != 'both').sum()} unmatched).")
    diff = float(np.abs(m.refit - m.pred_delta_pp).max())
    if not diff <= TOL:
        raise SystemExit(f"REFUSED: refit does not reproduce the frozen predictions (max diff {diff:.3e} pp).")
    by_station = m.assign(d=(m.refit - m.pred_delta_pp).abs()).groupby("station_id").d.max().to_dict()
    return {"n_rows": int(len(m)), "max_abs_diff_pp": diff, "tolerance_pp": TOL,
            "origins": [str(m.date.min().date()), str(m.date.max().date())],
            "max_abs_diff_by_station_pp": {k: float(v) for k, v in by_station.items()}}


def companions(stations):
    snow = pd.read_csv(ROOT / "data/processed/snow_daily.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/snow_stations.csv")
    core = core_snow_stations(snow, meta).sort_values(["reservoir_id", "snow_station_id"])
    index = pd.read_csv(ROOT / "data/processed/snow_basin_index.csv", parse_dates=["date"])
    clim = snow_climatology(index, TRAIN_END)
    if sorted(clim) != stations:
        raise SystemExit(f"REFUSED: snow climatology stations {sorted(clim)} differ from model stations {stations}.")
    offsets, interval_meta = season_offsets()
    metrics = json.loads((ROOT / "results/v2/final/metrics.json").read_text())
    cov = {s: float(metrics["per_station"][s]["interval_90_coverage"]) for s in stations}
    return {"core_snow_pillows": {s: core.loc[core.reservoir_id == s, "snow_station_id"].tolist() for s in stations},
            "snow_climatology_in": {s: [float(v) for v in clim[s].reindex(range(1, 367)).to_numpy()] for s in stations},
            "interval_offsets_pp": offsets, "interval": interval_meta,
            "historical_interval_coverage": cov,
            "interval_reliable": {s: cov[s] >= 0.85 for s in stations}}


def build(out_dir=ART_DIR):
    for f in PROVENANCE_FILES:
        if not (ROOT / f).exists():
            raise SystemExit(f"REFUSED: required training input {f} is missing. The frozen model cannot be "
                             "reconstructed without it, and no substitute model is built.")
    rec = json.loads(RUN_RECORD.read_text())
    if rec["frozen_hashes"] != frozen_hashes():
        raise SystemExit("REFUSED: frozen config or code differs from results/v2/final/RUN_RECORD.json.")
    model, train, test, stations, cols = fit_frozen()
    check = verify_against_archive(model, test, stations, cols)
    feature_order = list(design(train.head(1), cols, stations).columns)
    if list(model.feature_names_in_) != feature_order:
        raise SystemExit("REFUSED: fitted feature order differs from design().")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / MODEL_FILE)
    cfg = json.loads(CONFIG.read_text())
    card = {"artifact": "WatershedShift v2 frozen primary model, reconstructed",
            "model_version": "v2-frozen", "model_name": cfg["primary_model"]["name"],
            "estimator": cfg["primary_model"]["estimator"], "hyperparameters": cfg["primary_model"]["hyperparameters"],
            "training_rule": cfg["train_rule"], "training_cutoff_exclusive": str(TRAIN_END.date()),
            "training_rows": int(len(train)), "training_origin_range": [str(train.date.min().date()), str(train.date.max().date())],
            "stations": stations, "feature_order": feature_order,
            "config_sha256": sha_file(CONFIG), "frozen_hashes": frozen_hashes(),
            "preregistered_run_commit": rec["git_commit_at_run"], "source_commit": source_commit(),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifact_sha256": sha_file(out_dir / MODEL_FILE),
            "training_data_provenance": {f: sha_file(ROOT / f) for f in PROVENANCE_FILES},
            "reproduction_check": check, **companions(stations),
            "note": ("Refit of the frozen v2 primary model with the frozen code, data and cutoff. Not retrained on "
                     "2024-2025 outcomes. Contains CDEC-derived values: keep private.")}
    (out_dir / CARD_FILE).write_text(json.dumps(card, indent=2))
    print(f"artifact written to {out_dir.relative_to(ROOT)}: reproduced {check['n_rows']} frozen predictions, "
          f"max diff {check['max_abs_diff_pp']:.2e} pp; sha256 {card['artifact_sha256'][:16]}...")
    return card


def load_artifact(art_dir=ART_DIR):
    """Load only an artifact written by build(): card present, model hash equal to the card, frozen
    hashes equal to the run record. Raises FileNotFoundError or ValueError otherwise."""
    card_p, model_p = Path(art_dir) / CARD_FILE, Path(art_dir) / MODEL_FILE
    if not card_p.exists() or not model_p.exists():
        raise FileNotFoundError(f"model artifact missing in {art_dir}: run python -m src.forecast_model --build")
    card = json.loads(card_p.read_text())
    if sha_file(model_p) != card.get("artifact_sha256"):
        raise ValueError("model.joblib SHA-256 does not match model_card.json; refusing to load it")
    if card.get("frozen_hashes") != json.loads(RUN_RECORD.read_text())["frozen_hashes"]:
        raise ValueError("model card was built from a configuration other than the frozen v2 run")
    model = joblib.load(model_p)   # trusted: our own file, hash-checked above
    if list(model.feature_names_in_) != card["feature_order"]:
        raise ValueError("model feature order does not match the card")
    return model, card


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
        return
    model, card = load_artifact()
    _, test, _ = load(TRAIN_END, TEST_START, TEST_END)
    check = verify_against_archive(model, test, card["stations"], FEATURE_SETS["storage_snow"])
    print(f"artifact OK: sha256 matches card, {check['n_rows']} frozen predictions reproduced "
          f"(max diff {check['max_abs_diff_pp']:.2e} pp), training cutoff {card['training_cutoff_exclusive']}")


if __name__ == "__main__":
    main()

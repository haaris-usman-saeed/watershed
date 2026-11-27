"""WatershedShift v5: HGB v5 (full information) and a standardised benchmark of every previously implemented model.

Nothing here changes, refits or replaces a frozen v1/v2/v3 model, a v4 result, an issued record or a public release.
It reads existing data, reuses v2/v3/v4 functions read-only, and writes only to:
- data/v5/      private panel, features, CDEC flow downloads 2024-2026, per-row predictions (git-ignored);
- models/v5/    every trained v5 model (git-ignored);
- results/v5/   aggregate tables, statistics, figures and run manifests (tracked).

Plan (written before any v5 result): docs/v5/BENCHMARK_PLAN.md. Its SHA-256 is recorded in every run manifest.

Populations (never pooled into one leaderboard)
- tuning folds WY2012-2015 and confirmation folds WY2016-2023: development data, dates before DEV_END only;
- retrospective windows HELD (origins 2024-01-01..2025-12-01) and T26 (origins 2026-01-05..2026-10-19): periods that
  were already viewed in v2/v3, so every result there is labelled post-hoc retrospective, never an untouched test.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "v5"
MODELS = ROOT / "models" / "v5"
RESULTS = ROOT / "results" / "v5"
PLAN = ROOT / "docs" / "v5" / "BENCHMARK_PLAN.md"

STATIONS = ("DNP", "EXC", "FOL", "NML", "ORO", "SHA")
DEV_END = pd.Timestamp("2024-01-01")            # exclusive end of development data (as in v4)
PANEL_END = pd.Timestamp("2026-11-21")          # exclusive end of the consolidated panel (last frozen storage 2026-11-20)
TUNE_WYS = (2012, 2013, 2014, 2015)
CONFIRM_WYS = tuple(range(2016, 2024))
SEEDS = (0, 1, 2)
# Retrospective windows: training targets before `train_end`, evaluation origins in [start, end], targets <= target_end.
RETRO = {
    "HELD": {"train_end": "2024-01-01", "start": "2024-01-01", "end": "2025-12-01", "target_end": "2025-12-31"},
    "T26": {"train_end": "2026-01-01", "start": "2026-01-05", "end": "2026-10-19", "target_end": "2026-11-20"},
}

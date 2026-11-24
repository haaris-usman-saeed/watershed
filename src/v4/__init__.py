"""WatershedShift v4 (NeuralHydro): a separate neural research programme.

Nothing in this package changes, imports for writing, or replaces the frozen v1/v2/v3 work. It only
reads the existing processed data. Development uses data before DEV_END: no origin, no target window
and no training label on or after 2024-01-01 (the v2 held-out period and the 2026 v3 test are never used
for development or tuning).

Modules
- flows:      CDEC daily inflow (sensor 76) and outflow (sensor 23), plus a water-balance audit
- dataset:    the daily panel (one row per reservoir-day)
- sequences:  leakage-safe windows, targets, masks and chronological folds (numpy only)
- nets:       PyTorch sequence models (needs requirements-v4.txt)
- train:      training, early stopping, seeds, loss curves (needs requirements-v4.txt)
- experiment: development experiments and baselines on identical Monday rows
"""
import pandas as pd

DEV_END = pd.Timestamp("2024-01-01")       # exclusive: every v4 development date is earlier than this
STATIONS = ("DNP", "EXC", "FOL", "NML", "ORO", "SHA")

"""Compare a reproduced metrics.json with the archived frozen one, number by number.

Run: python -m src.compare_metrics REPRODUCED.json [ARCHIVED.json]
     (ARCHIVED defaults to results/v2/final/metrics.json; use a copy saved before rerunning)

Timestamps (`generated_utc`) are ignored. Exit code 0 if every other value matches
(numbers within 1e-9 absolute), 1 otherwise. Read-only: nothing is written.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {"generated_utc"}
TOL = 1e-9


def walk(a, b, path=""):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: present in only one file")
            else:
                diffs += walk(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > TOL:
            diffs.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        diffs.append(f"{path}: differs")
    return diffs


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    rep = json.loads(Path(sys.argv[1]).read_text())
    arc = json.loads(Path(sys.argv[2] if len(sys.argv) == 3 else ROOT / "results/v2/final/metrics.json").read_text())
    diffs = walk(rep, arc)
    print("\n".join(diffs) if diffs else "IDENTICAL (ignoring generated_utc)")
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()

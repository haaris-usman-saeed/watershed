"""Deterministic, verifiable submission bundle for private assessment.

Run:
  python -m src.make_bundle                 # dist/watershedshift_v2_bundle.tar.gz (no raw data)
  python -m src.make_bundle --include-raw   # also includes data/raw/ (only if redistribution is confirmed)
  python -m src.make_bundle --verify dist/watershedshift_v2_bundle.tar.gz

Determinism: files are added in sorted order with fixed mtime (0), uid/gid 0, empty owner
names, mode 0644, and the gzip header timestamp fixed to 0, so identical inputs give a
byte-identical archive. SHA256SUMS.txt inside the bundle lists every file; the archive's
own SHA-256 is written next to it. The bundle is written to dist/, which is git-ignored.
"""
import argparse
import gzip
import hashlib
import io
from pathlib import Path
import sys
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
NAME = "watershedshift_v2_bundle"
INCLUDE = [
    "README.md", "STATUS.md", "stations.csv", "requirements.txt", "requirements-lock.txt",
    "config/v2_frozen.json", "docs/preregistration.md", "docs/UI_CONTRACT.md", "docs/originality_check.md",
    "data/provenance.md", "data/coverage_table.md", "data/raw_manifest.jsonl", "data/capacity_history.csv",
    "data/series_breaks.csv", "data/spike_review.csv", "data/snow_stations.csv", "data/water_year_types.csv",
    "data/processed/daily_joined.csv", "data/processed/snow_daily.csv", "data/processed/snow_basin_index.csv",
    "data/processed/weekly_features.csv", "data/processed/coverage.csv", "data/processed/station_year_audit.csv",
    "data/processed/discontinuity_scan.csv",
    "outputs/stations.csv", "outputs/history_daily.csv", "outputs/test_predictions.csv", "outputs/metrics.json",
    "outputs/feature_catalog.json", "ASSESSORS.md", "docs/reproducibility.md", "PROMPTS.md",
]
# results, the full engineering pipeline (src/), the tests and the synthetic UI fixtures they validate
INCLUDE_DIRS = ["results/v1", "results/v2", "src", "tests", "app/sample_data"]
EXCLUDE_PARTS = {"local_full", "ui_outputs_rehearsal", "__pycache__"}
EXCLUDE_SUFFIXES = {".pyc"}
LIMITATIONS = """WatershedShift v2 submission bundle: limitations

- Retrospective data. CDEC storage (provisional, revisable; 'r' flags), CDEC snow sensor 82 (an adjusted
  product revised after the fact) and NASA POWER (MERRA-2 archive vs GEOS-IT near real time) were retrieved
  on 2026-11-21. Rebuilding later from the same URLs can give slightly different values if the agencies
  revise their records; data/raw_manifest.jsonl holds the SHA-256 of every response used here.
- Raw responses are not included unless the bundle was built with --include-raw, because CDEC
  redistribution terms were not confirmed.
- The 2024-2025 test period was seen for v1 before v2 was designed (see docs/preregistration.md).
- Historical backtest only: not an operational forecast, not a water-safety tool.
"""


def collect(include_raw=False):
    files = [ROOT / f for f in INCLUDE if (ROOT / f).exists()]
    for d in INCLUDE_DIRS + (["data/raw"] if include_raw else []):
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and not EXCLUDE_PARTS & set(p.parts)
                  and p.suffix not in EXCLUDE_SUFFIXES]
    missing = [f for f in INCLUDE if not (ROOT / f).exists()]
    return sorted(set(files)), missing


def source_commit():
    """Commit the bundle was built from, and whether bundled tracked files differ from it."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return (f"source_commit: {head or 'not found'}\n"
            f"working_tree_clean_at_build: {'yes' if not dirty else 'no (bundle may differ from the commit)'}\n"
            "repository: https://github.com/haaris-usman-saeed/watershed\n")


def _add(tar, arcname, data):
    info = tarfile.TarInfo(arcname)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def build_archive(entries, out_path):
    """entries: list of (arcname, bytes). Writes a byte-deterministic .tar.gz."""
    entries = sorted(entries)
    sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in entries)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for n, b in entries + [("SHA256SUMS.txt", sums.encode())]:
            _add(tar, f"{NAME}/{n}", b)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(buf.getvalue())
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    out_path.with_suffix(out_path.suffix + ".sha256").write_text(f"{digest}  {out_path.name}\n")
    return digest


def verify(path):
    path = Path(path)
    side = path.with_suffix(path.suffix + ".sha256")
    ok = True
    if side.exists():
        ok &= side.read_text().split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(path, "r:gz") as tar:
        members = {m.name.split("/", 1)[1]: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    sums = members.pop("SHA256SUMS.txt").decode().splitlines()
    listed = {}
    for line in sums:
        h, n = line.split("  ", 1)
        listed[n] = h
    ok &= set(listed) == set(members)
    ok &= all(hashlib.sha256(members[n]).hexdigest() == h for n, h in listed.items())
    return ok, len(listed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-raw", action="store_true")
    ap.add_argument("--verify")
    a = ap.parse_args()
    if a.verify:
        ok, n = verify(a.verify)
        print(f"{'OK' if ok else 'FAILED'}: {n} files checked")
        sys.exit(0 if ok else 1)
    files, missing = collect(a.include_raw)
    if missing:
        sys.exit(f"missing required files (run the pipeline first): {missing}")
    entries = [(str(p.relative_to(ROOT)), p.read_bytes()) for p in files] + [
        ("LIMITATIONS.txt", LIMITATIONS.encode()), ("SOURCE_COMMIT.txt", source_commit().encode())]
    out = DIST / (NAME + ("_with_raw" if a.include_raw else "") + ".tar.gz")
    digest = build_archive(entries, out)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {len(entries)} files) sha256={digest}")


if __name__ == "__main__":
    main()

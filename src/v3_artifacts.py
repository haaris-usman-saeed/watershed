"""Package and install model artifacts for a deployment, verified by SHA-256 (no private datasets).

Run (trusted build machine):  python -m src.v3_artifacts pack --out dist/
Run (deployment):             python -m src.v3_artifacts install --url URL --sha256 HEX

The package contains only what inference needs:
  models/v2_forecast/model.joblib, models/v2_forecast/model_card.json   (Model A, frozen v2)
  models/v3/model_{A,B,B2,C}.joblib, models/v3/model_cards.json          (v3)
plus MANIFEST.json with each file's SHA-256. It contains no observation table, history, forecast or
residual file. Model cards do carry aggregate CDEC-derived values (snow climatology, interval
offsets), and the fitted models are fitted to CDEC values, so the package is governed by the CDEC
entry of config/data_release_policy.json.

install refuses unless: the archive's SHA-256 equals --sha256 (pinned by the owner); every member is
a regular file at an expected path (no links, no traversal); every file matches MANIFEST.json; and,
on a public deployment, the policy allows public display of CDEC-derived values.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = ["models/v2_forecast/model.joblib", "models/v2_forecast/model_card.json", "models/v3/model_cards.json"] + \
        [f"models/v3/model_{k}.joblib" for k in ("A", "B", "B2", "C")]
ALLOWED = set(FILES) | {"MANIFEST.json"}
MAX_BYTES = 200 * 1024 * 1024
INSTALLED_MANIFEST = "models/ARTIFACT_MANIFEST.json"
SELFCHECK_ROWS = 16


def runtime_versions():
    import joblib
    import numpy
    import sklearn
    return {"scikit-learn": sklearn.__version__, "numpy": numpy.__version__, "joblib": joblib.__version__}


def _selfcheck_inputs(features):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(20261122)
    x = pd.DataFrame(rng.normal(0, 3, size=(SELFCHECK_ROWS, len(features))), columns=features)
    st = [c for c in features if c.startswith("station_")]
    x[st] = 0.0
    for i in range(SELFCHECK_ROWS):
        x.loc[i, st[i % len(st)]] = 1.0
    return x


def _model_paths(root):
    import joblib
    out = {"models/v2_forecast/model.joblib": joblib.load(root / "models/v2_forecast/model.joblib")}
    for k in ("A", "B", "B2", "C"):
        out[f"models/v3/model_{k}.joblib"] = joblib.load(root / f"models/v3/model_{k}.joblib")
    return out


def selfcheck_reference(root=ROOT):
    """Predictions of each model on fixed synthetic inputs, stored in the manifest at packing time."""
    ref = {}
    for f, m in _model_paths(Path(root)).items():
        x = _selfcheck_inputs(list(m.feature_names_in_))
        ref[f] = [float(v) for v in m.predict(x)]
    return ref


def verify_selfcheck(root=ROOT, tol=1e-9):
    """After installation: every model must reproduce its packing-time predictions with the libraries installed
    here. Returns {file: error message} for models that fail (empty when all pass, or when no manifest is installed)."""
    mp = Path(root) / INSTALLED_MANIFEST
    if not mp.exists():
        return {}
    ref = json.loads(mp.read_text()).get("selfcheck", {})
    bad = {}
    for f, m in _model_paths(Path(root)).items():
        if f not in ref:
            bad[f] = "no self-check reference in the installed manifest"
            continue
        got = m.predict(_selfcheck_inputs(list(m.feature_names_in_)))
        if max(abs(a - b) for a, b in zip(got, ref[f])) > tol:
            bad[f] = (f"predictions differ from those recorded when the bundle was built "
                      f"(built with {json.loads(mp.read_text()).get('built_with')}, running {runtime_versions()})")
    return bad


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pack(out_dir, root=ROOT, selfcheck=True):
    missing = [f for f in FILES if not (root / f).exists()]
    if missing:
        raise SystemExit(f"REFUSED: missing artifacts {missing}; build them first.")
    man = {"files": {f: sha((root / f).read_bytes()) for f in FILES},
           "selfcheck": selfcheck_reference(root) if selfcheck else {}, "built_with": runtime_versions()}
    import gzip
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for name, data in [("MANIFEST.json", json.dumps(man, indent=1, sort_keys=True).encode())] + \
                          [(f, (root / f).read_bytes()) for f in sorted(FILES)]:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o444
            tar.addfile(ti, io.BytesIO(data))
    data = buf.getvalue()
    out = Path(out_dir) / f"watershedshift_models_{sha(data)[:12]}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    (out.parent / (out.name + ".sha256")).write_text(f"{sha(data)}  {out.name}\n")
    return out, sha(data)


def install(data, expected_sha256, dest=ROOT, mode=None):
    from src.v3_service import cdec_display_block_reason
    if (mode or os.environ.get("WATERSHED_DEPLOYMENT", "private")) == "public":
        why = cdec_display_block_reason(mode="public")
        if why:
            raise PermissionError("REFUSED: " + why)
    if len(data) > MAX_BYTES:
        raise ValueError("REFUSED: archive larger than expected")
    if sha(data) != expected_sha256.lower():
        raise ValueError("REFUSED: archive SHA-256 does not match the pinned value")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                if not m.isfile() or m.name not in ALLOWED:
                    raise ValueError(f"REFUSED: unexpected archive member {m.name!r}")
            blobs = {m.name: tar.extractfile(m).read() for m in members}
    except (tarfile.TarError, EOFError, OSError) as exc:      # truncated or corrupted archive
        raise ValueError(f"REFUSED: archive cannot be read ({type(exc).__name__}: {exc})")
    if "MANIFEST.json" not in blobs:
        raise ValueError("REFUSED: archive has no MANIFEST.json")
    try:
        man = json.loads(blobs["MANIFEST.json"])["files"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"REFUSED: MANIFEST.json cannot be read ({type(exc).__name__})")
    if set(man) != set(FILES) or any(sha(blobs.get(f, b"")) != h for f, h in man.items()):
        raise ValueError("REFUSED: archive contents do not match MANIFEST.json")
    for f in FILES + [INSTALLED_MANIFEST]:
        p = Path(dest) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(blobs["MANIFEST.json"] if f == INSTALLED_MANIFEST else blobs[f])
        tmp.replace(p)
    return man


DOWNLOAD_ATTEMPTS = 3
_ENSURE_LOCK = threading.Lock()


def download(url, headers=None, attempts=DOWNLOAD_ATTEMPTS, sleep=time.sleep):
    """GET the bundle. Retries only transient failures (connection errors, timeouts, HTTP 429 and 5xx), waiting
    2 s, then 4 s. Any other HTTP error is final. Returns the body bytes."""
    import requests
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=headers or {}, timeout=120)
            code = getattr(r, "status_code", None)
            if code is not None and (code == 429 or code >= 500):
                last = f"HTTP {code}"
            else:
                r.raise_for_status()
                return r.content
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if i < attempts - 1:
            sleep(2 * 2 ** i)
    raise RuntimeError(f"download failed after {attempts} attempts ({last})")


def ensure_artifacts(root=ROOT, sleep=time.sleep):
    """At app start: if artifacts are missing and WATERSHED_ARTIFACT_URL / _SHA256 are set (for example from
    Streamlit secrets), download and install them with every check above. Returns a status string.
    Concurrent calls are serialised, so two visitors never install at the same time. Transient download
    failures are retried here; a caller that keeps the process alive (ForecastService) may call this again later."""
    root = Path(root)
    with _ENSURE_LOCK:
        if all((root / f).exists() for f in FILES):
            return "present"
        url, pin = os.environ.get("WATERSHED_ARTIFACT_URL"), os.environ.get("WATERSHED_ARTIFACT_SHA256")
        if not url or not pin:
            return "missing (no WATERSHED_ARTIFACT_URL / WATERSHED_ARTIFACT_SHA256 configured)"
        headers = {"Authorization": f"Bearer {os.environ['WATERSHED_ARTIFACT_TOKEN']}"} if os.environ.get("WATERSHED_ARTIFACT_TOKEN") else {}
        try:
            install(download(url, headers, sleep=sleep), pin, dest=root)
            return "installed"
        except Exception as exc:  # noqa: BLE001  any failure leaves models absent; the app then refuses to forecast
            return f"install refused or failed: {exc}"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("pack")
    pk.add_argument("--out", default=str(ROOT / "dist"))
    ins = sub.add_parser("install")
    ins.add_argument("--url", default=os.environ.get("WATERSHED_ARTIFACT_URL"))
    ins.add_argument("--sha256", default=os.environ.get("WATERSHED_ARTIFACT_SHA256"))
    a = ap.parse_args()
    if a.cmd == "pack":
        out, h = pack(Path(a.out))
        print(f"{out}\nsha256 {h}")
        return
    if not a.url or not a.sha256:
        raise SystemExit("install needs --url and --sha256 (or WATERSHED_ARTIFACT_URL / WATERSHED_ARTIFACT_SHA256)")
    headers = {}
    if os.environ.get("WATERSHED_ARTIFACT_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['WATERSHED_ARTIFACT_TOKEN']}"
    man = install(download(a.url, headers), a.sha256)
    print(f"installed {len(man)} verified files")


if __name__ == "__main__":
    main()

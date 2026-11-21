"""Download ResOpsUS (Steyaert et al. 2022, Scientific Data; CC-BY 4.0) for an independent cross-check.

Run: python -m src.fetch_resopsus
Saves Zenodo record 5893641 ResOpsUS.zip unchanged to data/raw/resopsus/ (git-ignored) and logs
URL, retrieval time and SHA-256 in data/raw/fetch_log.jsonl. Used ONLY to verify our CDEC storage
series (src/compare_resopsus.py); never a model input.
"""
from datetime import datetime, timezone
import hashlib
import json

import requests

from src.fetch import RAW, ROOT, SESSION

RECORD = "5893641"
URL = f"https://zenodo.org/api/records/{RECORD}/files/ResOpsUS.zip/content"


def main():
    path = RAW / "resopsus" / "ResOpsUS.zip"
    if path.exists():
        print("cached:", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = SESSION.get(f"https://zenodo.org/api/records/{RECORD}", timeout=60).json()
    expected = {f["key"]: f for f in meta["files"]}["ResOpsUS.zip"]
    h = hashlib.sha256()
    tmp = path.with_suffix(".part")
    with SESSION.get(URL, stream=True, timeout=600) as r, open(tmp, "wb") as fh:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
            h.update(chunk)
    md5 = hashlib.md5(tmp.read_bytes()).hexdigest()
    if expected.get("checksum", "").startswith("md5:") and expected["checksum"][4:] != md5:
        raise SystemExit(f"checksum mismatch: zenodo {expected['checksum']} vs md5:{md5}")
    tmp.rename(path)
    with open(RAW / "fetch_log.jsonl", "a") as fh:
        fh.write(json.dumps({"file": str(path.relative_to(ROOT)), "url": URL, "status": 200,
                             "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "bytes": path.stat().st_size, "sha256": h.hexdigest(),
                             "zenodo_checksum": expected.get("checksum"), "license": "cc-by-4.0"}) + "\n")
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB) sha256={h.hexdigest()} zenodo {expected.get('checksum')} verified")


if __name__ == "__main__":
    main()

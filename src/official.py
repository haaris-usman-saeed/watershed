"""Official weekly forecast issuance, as defined in docs/prospective/PROTOCOL.md section 3.

Started by a person (the owner or an operator the owner designates). There is no scheduler.

  python -m src.official issue --dry-run                 # build and show this week's record; writes nothing
  python -m src.official issue                           # Tuesday or Wednesday (UTC) only; writes the record once
  python -m src.official record-missed --origin YYYY-MM-DD --reason "..."   # after the window closed unissued

Rules implemented here (protocol section 3)
- One record per week covering all six reservoirs. The origin is the Monday immediately before the issue day. The
  issue window is Tuesday 00:00 to Wednesday 23:59 UTC; the command refuses outside it, and refuses a second record
  (or a record after a recorded miss) for the same week.
- Forecasts come from the frozen models through src.v3_service.ForecastService (Model A frozen v2; B, B2, C v3), with
  their frozen information rules. If the service cannot use the week's Monday for a reservoir (it would fall back to an
  older origin, or no origin), the official record holds an explicit no-forecast entry with the reasons instead.
- The record (record_type "official_weekly_forecast") is written once to the persistent GitHub records repository
  (WATERSHED_RECORD_GITHUB_REPO / _TOKEN); the command refuses to issue without it. A read-only local copy goes to
  forecasts/official/ (git-ignored) for scoring, and a line with its SHA-256 is appended to
  docs/prospective/official_ledger.jsonl, which the operator then commits and pushes within the window.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "watershedshift-official-forecast/1.0"
RECORD_TYPE = "official_weekly_forecast"
LEDGER = ROOT / "docs" / "prospective" / "official_ledger.jsonl"
LOCAL = ROOT / "forecasts" / "official"
PREFIX = "official-records/"


class IssueError(RuntimeError):
    pass


def utcnow():
    return datetime.now(timezone.utc)


def week_origin(now):
    """The origin Monday for an issue at `now` (UTC), or IssueError outside Tuesday 00:00 - Wednesday 23:59 UTC."""
    d = pd.Timestamp(now.astimezone(timezone.utc).date())
    if d.dayofweek not in (1, 2):
        raise IssueError(f"{d.date()} ({d.day_name()}) is outside the issue window, Tuesday 00:00 to Wednesday 23:59 UTC")
    return d - pd.Timedelta(days=d.dayofweek)


def ledger_entries(ledger=LEDGER):
    p = Path(ledger)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def week_entry(origin, ledger=LEDGER):
    """The ledger line (issued record or missed week) for this origin, if any."""
    return next((e for e in ledger_entries(ledger) if e.get("origin_date") == str(pd.Timestamp(origin).date())), None)


def _no_forecast(station, reasons):
    return {"station_id": station, "status": "no_forecast", "reasons": list(reasons)}


def build_record(service, origin, issued, cfg, issuer="owner"):
    """Run the frozen models for every reservoir and assemble the week's official record (nothing is written)."""
    from src.v3_service import APPROVED_STATIONS
    want = str(pd.Timestamp(origin).date())
    entries = []
    for st in APPROVED_STATIONS:
        try:
            r = service.forecast(st, force_refresh=True)     # fresh data; the service's cooldown limits re-downloads
        except Exception as exc:  # noqa: BLE001  any failure is an explicit no-forecast entry, never a gap
            entries.append(_no_forecast(st, [f"{type(exc).__name__}: {exc}"]))
            continue
        if r.get("status") != "ok":
            entries.append(_no_forecast(st, r.get("reasons") or ["forecast unavailable"]))
        elif r.get("origin_date") != want:
            entries.append(_no_forecast(st, [
                f"The week's Monday origin {want} lacked complete inputs (the website service would have used "
                f"{r.get('origin_date')}). Official records never use an older origin."]))
        else:
            entries.append(r)
    return {"schema": SCHEMA, "record_type": RECORD_TYPE, "forecast_id": f"wsf-official-{want}", "origin_date": want,
            "issued_utc": issued.strftime("%Y-%m-%dT%H:%M:%SZ"), "issued_by": issuer, "is_heldout_test_prediction": False,
            "protocol_version": cfg["version"], "protocol_config_sha256": cfg["_config_sha256"],
            "stations_with_forecast": sum(e["status"] == "ok" for e in entries),
            "stations_without_forecast": sum(e["status"] != "ok" for e in entries), "stations": entries}


def issue(clock=utcnow, service=None, store=None, ledger=LEDGER, local_dir=LOCAL, dry_run=False, issuer="owner"):
    """Issue this week's official record. Returns (record, ledger_line); ledger_line is None for a dry run."""
    from src.prospective import load_protocol
    from src.v3_service import RecordStore
    cfg = load_protocol()                                   # refuses if the protocol or baseline was edited
    start = clock()
    origin = week_origin(start)
    prior = week_entry(origin, ledger)
    if prior:
        raise IssueError(f"the week of {origin.date()} already has a ledger entry ({prior.get('type')}); "
                         "one official record per week")
    store = store or RecordStore(root=local_dir, prefix=PREFIX)
    if not dry_run and store.backend != "github":
        raise IssueError("official records must be written to the persistent GitHub records repository: set "
                         "WATERSHED_RECORD_GITHUB_REPO and WATERSHED_RECORD_GITHUB_TOKEN (docs/v3/DEPLOYMENT.md), "
                         "or use --dry-run to preview")
    if service is None:
        from src.v3_artifacts import ensure_artifacts
        from src.v3_service import ForecastService
        service = ForecastService(artifact_installer=ensure_artifacts)
    rec = build_record(service, origin, start, cfg, issuer)
    if dry_run:
        return rec, None
    try:                                                     # building must not run past the window
        still_open = week_origin(clock()) == origin
    except IssueError:
        still_open = False
    if not still_open:
        raise IssueError("the issue window closed while the record was being built; nothing was written")
    body = json.dumps(rec, indent=1, allow_nan=False).encode()
    sha = hashlib.sha256(body).hexdigest()
    saved = store.save(rec, record_id=rec["forecast_id"])
    if saved["sha256"] != sha:
        raise IssueError("the stored record's SHA-256 differs from the local serialisation")
    local = Path(local_dir) / f"{rec['forecast_id']}.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "xb") as fh:
        fh.write(body)
    local.chmod(0o444)
    line = {"forecast_id": rec["forecast_id"], "type": RECORD_TYPE, "origin_date": rec["origin_date"],
            "issued_utc": rec["issued_utc"], "sha256": sha, "location": saved["location"],
            "records_repo_commit": saved.get("commit"), "stations_with_forecast": rec["stations_with_forecast"],
            "stations_without_forecast": rec["stations_without_forecast"], "protocol_version": cfg["version"]}
    Path(ledger).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as fh:
        fh.write(json.dumps(line) + "\n")
    return rec, line


def record_missed(origin, reason, clock=utcnow, ledger=LEDGER):
    """Append a missed-week line once that week's issue window has closed without a record."""
    o = pd.Timestamp(origin)
    if o.dayofweek != 0:
        raise IssueError(f"{o.date()} is not a Monday")
    if pd.Timestamp(clock().astimezone(timezone.utc).date()) < o + pd.Timedelta(days=3):
        raise IssueError(f"the issue window for {o.date()} has not closed yet (it ends Wednesday 23:59 UTC)")
    if week_entry(o, ledger):
        raise IssueError(f"the week of {o.date()} already has a ledger entry")
    if not reason.strip():
        raise IssueError("a reason is required")
    line = {"type": "missed_week", "origin_date": str(o.date()), "reason": reason.strip(),
            "recorded_utc": clock().strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(ledger, "a") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


def _brief(rec):
    out = {k: rec[k] for k in ("forecast_id", "origin_date", "issued_utc", "stations_with_forecast",
                               "stations_without_forecast")}
    out["stations"] = {}
    for e in rec["stations"]:
        if e["status"] != "ok":
            out["stations"][e["station_id"]] = {"no_forecast": e["reasons"]}
        else:
            out["stations"][e["station_id"]] = {k: (round(m["change_pp"], 2) if m.get("available") else "unavailable")
                                                for k, m in e["models"].items()}
    return out


def main():
    import warnings
    warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta", category=DeprecationWarning)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("issue")
    i.add_argument("--dry-run", action="store_true")
    i.add_argument("--issuer", default="owner", help="role of the person issuing (no names or contact details)")
    m = sub.add_parser("record-missed")
    m.add_argument("--origin", required=True)
    m.add_argument("--reason", required=True)
    a = ap.parse_args()
    if a.cmd == "record-missed":
        print(json.dumps(record_missed(a.origin, a.reason), indent=1))
        print("Now commit docs/prospective/official_ledger.jsonl on a branch and push it.")
        return
    rec, line = issue(dry_run=a.dry_run, issuer=a.issuer)
    print(json.dumps(_brief(rec), indent=1, default=str))
    if line is None:
        print("Dry run: nothing was written.")
    else:
        print(f"Issued {line['forecast_id']} -> {line['location']}\nLedger line appended. Commit "
              "docs/prospective/official_ledger.jsonl on a branch and push it before Wednesday 23:59 UTC.")


if __name__ == "__main__":
    main()

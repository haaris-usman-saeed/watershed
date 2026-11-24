# Issuing official weekly forecasts (operator guide)

This guide shows how to carry out section 3 of [PROTOCOL.md](PROTOCOL.md) with `src/official.py`. It does not change the protocol.

**There is no scheduler.** Every issue is started by the owner, or by an operator the owner designates.

## One-time setup

1. **The persistent GitHub records repository** described in `docs/v3/DEPLOYMENT.md` ("Persistent records in a GitHub repository"):
   - a separate repository;
   - a ruleset that blocks force pushes and deletions;
   - a fine-grained token limited to that repository, with Contents read and write.
2. **On the machine that issues** (the trusted build machine, which holds the model artifacts), set the variables in the shell only. Never commit them:

   ```bash
   export WATERSHED_RECORD_GITHUB_REPO="<owner>/watershed-records"
   ```

   ```bash
   export WATERSHED_RECORD_GITHUB_TOKEN="<fine-grained token>"
   ```

   The command refuses to issue without them. A dry run works without them.

## Each week

**When.** Any time from Tuesday 00:00 to Wednesday 23:59 UTC. The origin is that week's Monday.

**Issue late in the window.** CDEC posts daily storage with a lag of 1–2 days. In a dry run at 09:58 UTC on Tuesday 2026-11-24, Monday's storage was not yet posted for DNP, EXC and ORO, so they would have been no-forecast entries. Wednesday afternoon UTC gives the best chance that all six reservoirs have Monday data.

1. **Preview** (writes nothing):

   ```bash
   .venv/bin/python -m src.official issue --dry-run
   ```

2. **Issue** (writes once to the records repository, makes a read-only local copy in `forecasts/official/`, and appends a line to the ledger):

   ```bash
   .venv/bin/python -m src.official issue
   ```

3. **Commit and push the ledger line before Wednesday 23:59 UTC,** on a branch with a PR:

   ```bash
   git switch -c ledger/official-YYYY-MM-DD && git add docs/prospective/official_ledger.jsonl && git commit -m "Official forecast ledger: week of YYYY-MM-DD" && git push -u origin HEAD
   ```

   The records repository's own commit, timestamped by GitHub's server, is the primary evidence of when the record was issued. The push of the ledger line is a second record.

**What a record holds.**
- One record per week, `wsf-official-<Monday>`, with all six reservoirs.
- Each reservoir has either a forecast from models A, B, B2 and C, with each model's availability, or an explicit **no-forecast** entry with reasons.
- An older origin is never used. When the website would fall back to an older Monday, the official record says so and holds no forecast for that reservoir.

## Refusals

| Situation | Behaviour |
|---|---|
| Outside the window | Refuses |
| The week already has a record or a missed-week line | Refuses (one per week) |
| The GitHub records store is not configured | Refuses (use `--dry-run` to preview) |
| The window closes while the record is being built | Refuses; nothing is written |
| The protocol or baseline file differs from its registered hash | Refuses |

## A missed week

If no record was issued in the window, record that once the window has closed (from Thursday UTC):

```bash
.venv/bin/python -m src.official record-missed --origin YYYY-MM-DD --reason "short reason"
```

Then commit the ledger line as in step 3. Missed weeks count against availability (protocol section 5).

## Scoring

See [PROTOCOL.md](PROTOCOL.md) section 4. Use the local copy:
- provisional scores from target + 2 days;
- settled scores from target + 30 days.

```bash
.venv/bin/python -m src.prospective score forecasts/official/wsf-official-YYYY-MM-DD.json --stage settled
```

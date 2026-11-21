# Merge plan (owner actions)

The engineering agent does not merge either PR. Current state, checked 2026-11-21:
- `main` has no commits that `engineering` lacks.
- `engineering` is an ancestor of `review-v2`.

With merge commits, no conflicts are expected.

1. **Review and merge PR #1** (`engineering` into `main`): https://github.com/haaris-usman-saeed/watershed/pull/1. Use **"Create a merge commit"**, not squash or rebase. Squashing would rewrite the shared commits, and PR #2 would then show them as conflicts.
2. **Retarget PR #2 to `main`**: https://github.com/haaris-usman-saeed/watershed/pull/2. On the PR page, use Edit next to the title and change the base to `main`. Or run `gh pr edit 2 --base main`.
3. **Resolve any conflicts without altering frozen modelling files.** These files must stay byte-identical to commit `e47deab`:
   - `config/v2_frozen.json` and `docs/preregistration.md`;
   - `src/train_v2.py`, `src/models.py`, `src/features.py`, `src/evaluation.py` and `src/build_dataset.py`;
   - everything under `results/v2/final/` and `results/v1/`.

   Check with `git diff e47deab -- config/v2_frozen.json docs/preregistration.md src/train_v2.py src/models.py src/features.py src/evaluation.py src/build_dataset.py results/v2/final`, which must print nothing.
4. **Run the checks.**
   - CI (`.github/workflows/ci.yml`) runs automatically on the PR.
   - Locally, with the data present, run:
     - `python -m pytest -q tests`
     - `python -m src.contract outputs`
     - `python -m src.contract app/sample_data`
     - `python -m src.report_facts --check`
     - `python -m src.ui_export --verify`
5. **Merge PR #2**, again with a merge commit.
6. **Integrate and verify the frontend.**
   - Give `dist/outputs_v2_ui.zip` to Claude Web privately; it is not to be uploaded publicly while the CDEC terms are unresolved.
   - Claude Web commits its `app.py`, `app/` and `.streamlit/` changes. Only Claude Web or the owner edits those files.
   - Run the app against the real outputs: `streamlit run app.py`. Confirm that the Folsom interval warning, the strongest baseline (`bootstrap.best_baseline`, Smoothed DOY seasonal) and the CI are displayed.
7. **Tag the final integrated commit**: `git tag -a submission-v2 -m "WatershedShift v2 submission" && git push origin submission-v2`. Then rebuild the private bundle from that commit (`python -m src.make_bundle`, then `--verify`), so that its `SOURCE_COMMIT.txt` names the tagged commit.

Before tagging, also approve or edit the CDEC wording in README section "Attribution and terms", which is marked pending owner approval.

# Repository maintenance

## Cleanup audit (9 August 2026)

The pre-cleanup `HEAD` contained 1,120 tracked paths and about 262 MB. More than
95% was generated or local material in `.artifact-work/`, `.codex-tmp/`,
`.playwright-cli/`, `output/` and `outputs/`; it also contained two absolute
`node_modules` symlinks. The cleanup keeps only source, tests, configuration,
documentation and selected evidence; exact path and byte counts are intentionally
computed by repository tooling instead of copied into this document.

The repository now uses:

- `.gitignore` for local state (including `.venv-ocr`), build output, logs, browser traces and agent work;
- `.gitattributes` for stable line endings and binary classification;
- CI jobs for backend, frontend and repository hygiene;
- `scripts/check-repository-hygiene.sh` to reject generated paths and absolute
  symlinks if they become tracked again;
- `docs/evidence/` as the only canonical screenshot collection;
- `docs/benchmarks/` for small, explicitly dated or historical data snapshots.

High-confidence private-key/token patterns were not found in the cleaned working
tree. This is a hygiene check, not a substitute for secret rotation or a dedicated
scanner in release CI.

## Completed history rewrite

On 9 August 2026 the owner approved a full history scrub. The rewrite removed the
historical generated directories, report and presentation artifacts, duplicate
screenshots, `*.tsbuildinfo`, absolute dependency symlinks and personal academic
documents from every reachable commit. The rewritten `master` was verified and
published with an exact `--force-with-lease`.

The resulting repository contains the four surviving source-history commits and
six focused cleanup/hardening commits. Clones made before the rewrite must not
push their old history; they should be replaced with a fresh clone. The current
history is the canonical base for all future branches.

Do not run `git clean -fdX` in this repository: ignored `data/`, `.venv/` and model
caches are large, valuable local state.

## Project policy

The owner selected the MIT License; the canonical terms are in the repository
root `LICENSE` file. After CI is merged, protect the default branch with required
`backend`, `frontend`, `e2e` and `hygiene` checks and prefer small Conventional
Commit or squash-merge changes.

The approved post-cleanup architecture and implementation gates are recorded in
[`docs/system-rebuild.md`](system-rebuild.md). Runtime `data/` is user state and is
not part of repository cleanup; never remove it with broad Git or cache-cleaning
commands.

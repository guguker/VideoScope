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

## History is a separate operation

A normal cleanup commit makes future checkouts small, but the removed blobs remain
reachable from the five existing commits. Rewriting history is intentionally not
part of the cleanup commit: it changes commit IDs and requires owner approval,
backup, a clean clone, collaborator coordination and a force push.

If the owner chooses to scrub history, use `git filter-repo` in a fresh clone to
remove the historical generated directories, report artifacts, screenshots,
`*.tsbuildinfo` and any personal document selected for removal. Before pushing:

1. create an offline `git bundle` backup;
2. verify the exact path list and run the secret/provenance scan again;
3. run backend, frontend, build, E2E and repository-hygiene checks on the rewritten clone;
4. inspect `git fsck` and the resulting pack size;
5. push with an exact `--force-with-lease`, then require fresh clones.

Do not run `git clean -fdX` in this repository: ignored `data/`, `.venv/` and model
caches are large, valuable local state.

## Project policy

The owner selected the MIT License; the canonical terms are in the repository
root `LICENSE` file. After CI is merged, protect the default branch with required
`backend`, `frontend`, `e2e` and `hygiene` checks and prefer small Conventional
Commit or squash-merge changes.

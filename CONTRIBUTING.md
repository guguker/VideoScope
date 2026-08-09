# Contributing

VideoScope is currently maintained as a local-first research application. Before
opening a change, read the setup in `README.md` and dependency profiles in
`docs/dependencies.md`.

## Development checks

```bash
make install
make test
make build
pnpm --dir frontend e2e
bash scripts/check-repository-hygiene.sh
```

The browser suite starts a local Vite server. ML providers are optional for the
base unit suite; changes to a provider should include a focused fake/offline test
and a documented hardware smoke-test when applicable.

Keep changes small and use clear Conventional Commit messages. Do not commit
`data/`, source videos, model caches, generated reports, browser traces, build
state, absolute symlinks or personal academic material. Add durable screenshots
only to `docs/evidence/` and include the capture date and limitations.

HTTP changes must update the Pydantic request/response model, OpenAPI/API tests,
frontend TypeScript contract and a typed fixture together. Evaluation-methodology
changes must version or invalidate existing reports rather than silently reusing
old metrics.

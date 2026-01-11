## Backend Agent

Scope
- Work inside `backend/`
- Modify `frontend/` only when a backend change requires a matching API contract update

Primary mission
- Implement and maintain the FastAPI backend, storage, and 6-stage qEEG workflow.
- Integrate with CLIProxyAPI as the single upstream for model calls.

Project reality
- CLIProxyAPI is the only upstream. No direct provider SDK calls.
- No browser automation.

Key contracts
- CLIProxyAPI base URL comes from env `CLIPROXY_BASE_URL` with default `http://127.0.0.1:8317`
- Optional CLIProxyAPI auth key in `CLIPROXY_API_KEY`
- Model ids are discovered at runtime from `GET /v1/models`
- Prefer `POST /v1/chat/completions`
- Fallback once to `POST /v1/responses` when a model rejects chat completions

Commands
- Install deps: `uv sync`
- Run backend: `uv run python -m backend.main`
- Run tests: `uv run pytest -q`
- Format: `uv run ruff format`
- Lint: `uv run ruff check --fix`

Implementation rules
- Store metadata in SQLite and store large artifacts on disk.
- Avoid storing large text blobs inside SQLite.
- Keep artifact filenames deterministic:
  - `data/artifacts/<run_id>/stage-<n>/<model_id>.<ext>`
  - `.md` for markdown stages
  - `.json` for structured stages
- Keep the stage pipeline deterministic and restartable:
  - Each stage writes artifacts and updates run status before moving forward.
- SSE streaming uses a GET endpoint:
  - `GET /api/runs/{run_id}/stream` with `text/event-stream`
  - Emit stage transition events at minimum.

API surface
- `GET /api/health` checks:
  - CLIProxyAPI reachable
  - `/v1/models` returns a list
- `GET /api/models` returns:
  - configured council models
  - discovered upstream models
  - availability flags
- Patient and report endpoints:
  - `GET/POST/PUT /api/patients`
  - `POST /api/patients/{id}/reports`
  - `GET /api/patients/{id}/reports`
- Run endpoints:
  - `POST /api/runs`
  - `POST /api/runs/{run_id}/start`
  - `GET /api/runs/{run_id}`
  - `GET /api/runs/{run_id}/stream`
  - `GET /api/patients/{id}/runs`
  - `POST /api/runs/{run_id}/export`

Workflow requirements
- Stage 1: markdown analysis per model
- Stage 2: JSON peer reviews per reviewer, anonymized labels A/B/C per run
- Stage 3: markdown revisions per model
- Stage 4: markdown consolidation by consolidator
- Stage 5: JSON final review votes per model
- Stage 6: markdown final drafts per model
- Selection happens in UI, backend persists selected artifact reference and exports

Reliability
- Handle upstream errors with bounded retries:
  - 429, 502, 503 retry with exponential backoff and jitter
  - 401 marks run as needs-auth and surfaces error cleanly
- Partial completion is allowed:
  - Stage 1 can finish with missing models
  - Consolidation requires at least one revision

Do not do
- No scraping chat UIs
- No adding heavy infra dependencies like Redis
- No hidden background daemons beyond CLIProxyAPI
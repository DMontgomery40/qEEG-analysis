# qEEG Council

Local 6-stage deliberation workflow for analyzing qEEG reports with multiple LLMs, using **CLIProxyAPI** as the only upstream.

## Architecture

- Frontend: React + Vite (`http://localhost:5173`)
- Backend: FastAPI (`http://localhost:8000`)
- LLM upstream: CLIProxyAPI (`http://127.0.0.1:8317`)

## Prereqs

- Python 3.11+ and `uv`
- Node.js 18+
- CLIProxyAPI installed and logged in

## CLIProxyAPI

Start CLIProxyAPI (example):

```bash
cliproxyapi -config /opt/homebrew/etc/cliproxyapi.conf
```

Optional env vars:

```bash
export CLIPROXY_BASE_URL="http://127.0.0.1:8317"
export CLIPROXY_API_KEY=""
```

Check discovered models:

```bash
uv run python backend/scripts/cliproxy_models.py
```

Project-local CLIProxyAPI config (recommended):

- `./start.sh` will create `./.cli-proxy-api/cliproxyapi.conf` automatically.
- It disables client API-key auth (single-user localhost) and stores OAuth tokens under `./.cli-proxy-api/auth`.

## Install

Backend:

```bash
uv sync
```

Frontend:

```bash
cd frontend
npm install
cd ..
```

## Run

```bash
./start.sh
```

- Frontend: `http://localhost:5173`
- Backend health: `http://localhost:8000/api/health`

## Workflow (6 stages)

1. Initial analysis (per model, Markdown)
2. Peer review (per model, anonymized, JSON)
3. Revision (per model, Markdown)
4. Consolidation (single consolidator, Markdown)
5. Final review (per model vote, JSON)
6. Final drafts (per model, Markdown)

The UI lets you select either the consolidated report (Stage 4) or a final draft (Stage 6), then export to `final.md` and `final.pdf`.

## Tests

```bash
uv run pytest -q
```

## Notes

- Metadata is stored in `data/app.db` (SQLite).
- Large text, uploads, artifacts, and exports are stored under `data/`.

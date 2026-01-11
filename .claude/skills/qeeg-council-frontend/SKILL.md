---
name: qeeg-council-frontend
description: Build and maintain the qEEG Council React frontend: patients, reports, run wizard, 6-stage progress UI, SSE streaming, artifact viewers, and exports.
license: MIT
compatibility: Designed for Claude Code and Codex CLI. Requires node + npm. Backend default http://127.0.0.1:8000 and frontend dev http://localhost:5173.
metadata:
  author: dmontgomery
  version: "1.0"
---

# qEEG Council Frontend Skill

## Use this Skill for
- building React pages and components for patients, reports, runs
- wiring SSE progress updates into the UI
- rendering markdown and JSON artifacts per stage
- presenting model availability and source badges

## UI must-haves
- Patient list with search
- Patient detail showing reports and run history
- New run wizard: choose report, choose council models from discovered list, choose consolidator
- Run detail page:
  - 6-stage progress indicator
  - per-stage artifact views
  - per-model tabs for Stage 1, 3, 6
  - Stage 2 peer review view with anonymized labels plus UI de-anonymization
  - Stage 4 consolidated report panel
  - Stage 5 votes panel
  - Stage 6 final draft compare + selection
  - export buttons for MD and PDF

## Networking contract
- Model list: GET /api/models
- Runs:
  - POST /api/runs
  - POST /api/runs/{run_id}/start
  - GET /api/runs/{run_id}
  - SSE: GET /api/runs/{run_id}/stream

## SSE integration
- Use EventSource on the GET stream endpoint.
- Update stage UI on events.
- Keep a clear “needs-auth / upstream down / rate limited” error state.

## References
- UI checklist: [references/ui-checklist.md](references/ui-checklist.md)
- API quick map: [references/api-quick-map.md](references/api-quick-map.md)

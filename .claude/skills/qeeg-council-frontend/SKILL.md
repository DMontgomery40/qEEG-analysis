---
name: qeeg-council-frontend
description: Build and maintain the qEEG Council React frontend: patients, reports, run wizard, 6-stage progress UI, SSE streaming, artifact viewers, and exports.
license: MIT
compatibility: Designed for Claude Code and Codex CLI. Requires node + npm. Backend default http://127.0.0.1:8000 and frontend dev http://localhost:5173.
metadata:
  author: dmontgomery
  version: "2.0"
---

# qEEG Council Frontend Skill

## Use this Skill for
- building React pages and components for patients, reports, runs
- wiring SSE progress updates into the UI
- rendering markdown and JSON artifacts per stage
- presenting model availability and source badges

## Where the code lives
- API client: `frontend/src/api.js` (base URL via `VITE_BACKEND_URL`, default `http://127.0.0.1:8000`)
- UI: `frontend/src/components/*`

## UI must-haves
- Patient list with search
- Bulk upload page:
  - upload multiple report files
  - create a new patient per file (label = filename stem) and upload that file as the initial report
  - skip/report existing labels and batch duplicates
- Patient detail showing reports and run history
- Patient files in the patient detail:
  - upload/list/delete supporting files (PDF/Markdown/MP4)
- New run wizard: choose report, choose council models from discovered list, choose consolidator
- Report source viewer:
  - original PDF
  - extracted page images
  - extraction metadata (when available)
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
- Patients:
  - POST /api/patients/bulk_upload
  - GET /api/patients/{patient_id}/files
  - POST /api/patients/{patient_id}/files
  - GET /api/patient_files/{file_id}
  - DELETE /api/patient_files/{file_id}
- Runs:
  - POST /api/runs
  - POST /api/runs/{run_id}/start
  - GET /api/runs/{run_id}
  - SSE: GET /api/runs/{run_id}/stream
  - GET /api/runs/{run_id}/artifacts
  - POST /api/runs/{run_id}/select
  - POST /api/runs/{run_id}/export
  - GET /api/runs/{run_id}/export/final.md
  - GET /api/runs/{run_id}/export/final.pdf

- Reports:
  - POST /api/patients/{id}/reports
  - GET /api/reports/{report_id}/extracted
  - POST /api/reports/{report_id}/reextract
  - GET /api/reports/{report_id}/original
  - GET /api/reports/{report_id}/pages
  - GET /api/reports/{report_id}/pages/{page_num}
  - GET /api/reports/{report_id}/metadata

- CLIProxy helpers (used by the UI):
  - POST /api/cliproxy/start
  - POST /api/cliproxy/login
  - POST /api/cliproxy/install

## SSE integration
- Use EventSource on the GET stream endpoint.
- Update stage UI on events.
- Keep a clear “needs-auth / upstream down / rate limited” error state.

## References
- UI checklist: [references/ui-checklist.md](references/ui-checklist.md)
- API quick map: [references/api-quick-map.md](references/api-quick-map.md)

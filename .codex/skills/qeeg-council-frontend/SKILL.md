---
name: qeeg-council-frontend
description: "Build and maintain the qEEG Council React frontend: patients, reports, runs, SSE progress, artifact viewers, selection, and exports."
version: "1.0.0"
license: MIT
--- 

# qEEG Council Frontend Skill

## Purpose
Build and maintain the UI for:
- Patients (list, create, detail)
- Reports (upload, extracted text preview)
- Runs (new run wizard, 6-stage progress, artifact viewing)
- Final selection and export (Markdown and PDF)

## Required UI flows
### Patients
- Patient list with search and create
- Patient detail page with:
  - notes
  - reports list
  - run history list

### Reports
- Upload PDF or text
- Show extracted text preview returned by backend
- Store report_id and use it in run creation

### Runs
- New run wizard:
  - choose report_id
  - choose council models from `/api/models` discovered list
  - choose consolidator
  - create run and start it
- Run page:
  - 6-stage progress indicator
  - Stage 1, 3, 6: per-model markdown tabs
  - Stage 2 and 5: JSON viewer
  - Stage 4: consolidated markdown panel
  - Selection control that calls `POST /api/runs/{run_id}/select`
  - Export buttons that call `POST /api/runs/{run_id}/export` then download final.md and final.pdf

## Networking contract
Backend base URL default: `http://127.0.0.1:8000`

Endpoints used:
- `GET /api/health`
- `GET /api/models`
- `GET/POST/PUT /api/patients`
- `POST /api/patients/{id}/reports`
- `GET /api/patients/{id}/reports`
- `POST /api/runs`
- `POST /api/runs/{run_id}/start`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/stream` (SSE)
- `POST /api/runs/{run_id}/select`
- `POST /api/runs/{run_id}/export`
- `GET /api/runs/{run_id}/export/final.md`
- `GET /api/runs/{run_id}/export/final.pdf`

## SSE integration
- Use `EventSource` to `GET /api/runs/{run_id}/stream`
- Update stage progress on stage completion events
- Show error banners for upstream down, model unavailable, needs_auth, rate limited

## Rendering rules
- Render markdown artifacts with your existing markdown renderer
- Render JSON artifacts with a collapsible JSON viewer
- Show model `source` badge next to model name
- If a configured model is missing from discovered `/v1/models`, show it disabled

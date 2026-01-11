# Backend API Contract (minimal)

Base URL: http://localhost:8000

## Health
GET /api/health
- returns CLIProxyAPI reachability and basic status

## Models
GET /api/models
- returns:
  - discovered_models: list of ids from CLIProxyAPI /v1/models
  - configured_models: list of council models with availability flags

## Patients
GET /api/patients
POST /api/patients
GET /api/patients/{id}
PUT /api/patients/{id}
GET /api/patients/{id}/reports
GET /api/patients/{id}/runs

## Reports
POST /api/patients/{id}/reports
- multipart upload (pdf or text)
- returns report_id and extracted text preview

## Runs
POST /api/runs
- body: patient_id, report_id, council_model_ids, consolidator_model_id
POST /api/runs/{run_id}/start
GET /api/runs/{run_id}
GET /api/runs/{run_id}/stream (SSE)

## Exports
POST /api/runs/{run_id}/export
- creates final.md and final.pdf for selected artifact

# API quick map

Backend base: http://127.0.0.1:8000

GET /api/health
GET /api/models

POST /api/cliproxy/start
POST /api/cliproxy/login
POST /api/cliproxy/install

GET /api/patients
POST /api/patients
GET /api/patients/{id}
PUT /api/patients/{id}

POST /api/patients/{id}/reports
GET /api/patients/{id}/reports

GET /api/reports/{report_id}/extracted
POST /api/reports/{report_id}/reextract

POST /api/runs
POST /api/runs/{run_id}/start
GET /api/runs/{run_id}
GET /api/runs/{run_id}/stream

GET /api/runs/{run_id}/artifacts
POST /api/runs/{run_id}/select

POST /api/runs/{run_id}/export
GET /api/runs/{run_id}/export/final.md
GET /api/runs/{run_id}/export/final.pdf

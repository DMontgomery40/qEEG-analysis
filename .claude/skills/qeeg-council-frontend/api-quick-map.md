# API quick map

Backend base: http://127.0.0.1:8000

GET /api/health
GET /api/models

GET /api/patients
POST /api/patients
GET /api/patients/{id}
PUT /api/patients/{id}

POST /api/patients/{id}/reports
GET /api/patients/{id}/reports

POST /api/runs
POST /api/runs/{run_id}/start
GET /api/runs/{run_id}
GET /api/runs/{run_id}/stream

POST /api/runs/{run_id}/export

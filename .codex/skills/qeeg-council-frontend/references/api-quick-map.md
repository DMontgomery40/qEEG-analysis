# API quick map

Backend base: http://127.0.0.1:8000

GET /api/health
GET /api/models

POST /api/cliproxy/start
POST /api/cliproxy/login
POST /api/cliproxy/install

GET /api/patients
POST /api/patients
POST /api/patients/bulk_upload
GET /api/patients/{id}
PUT /api/patients/{id}

POST /api/patients/{id}/reports
GET /api/patients/{id}/reports
GET /api/patients/{id}/runs

GET /api/patients/{id}/files
POST /api/patients/{id}/files
GET /api/patient_files/{file_id}
DELETE /api/patient_files/{file_id}

GET /api/reports/{report_id}/extracted
POST /api/reports/{report_id}/reextract
GET /api/reports/{report_id}/original
GET /api/reports/{report_id}/pages
GET /api/reports/{report_id}/pages/{page_num}
GET /api/reports/{report_id}/metadata

POST /api/runs
POST /api/runs/{run_id}/start
GET /api/runs/{run_id}
GET /api/runs/{run_id}/stream

GET /api/runs/{run_id}/artifacts
POST /api/runs/{run_id}/select

POST /api/runs/{run_id}/export
GET /api/runs/{run_id}/export/final.md
GET /api/runs/{run_id}/export/final.pdf

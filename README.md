# qEEG Council

6-stage LLM deliberation workflow for analyzing qEEG/ERP reports. Uses CLIProxyAPI as the upstream router to OpenAI, Anthropic, and Google models.

## Architecture

```
Frontend (React/Vite)  →  Backend (FastAPI)  →  CLIProxyAPI  →  LLM providers
     :5173                    :8000                :8317
```

### Data Flow

1. Upload PDF → extract text + OCR + page images
2. Run 6-stage council workflow with selected models
3. Export consolidated report as Markdown/PDF

### Storage

- `data/app.db` - SQLite metadata (patients, reports, runs, artifacts)
- `data/reports/{patient_id}/{report_id}/` - uploaded files, extracted text, page images
- `data/artifacts/{run_id}/stage-{n}/` - stage outputs per model
- `data/exports/{run_id}/` - final exported reports
- `data/patient_files/{patient_id}/{file_id}/` - DB-tracked uploaded patient files (MP4/PDF/etc)
- `data/portal_patients/{patient_id}/` - clinician portal **sync folder** (drop final MP4s here)

## 6-Stage Workflow

| Stage | Output | Description |
|-------|--------|-------------|
| 1 | Markdown | Initial analysis (per model, multimodal for vision models) |
| 2 | JSON | Peer review (anonymized cross-evaluation) |
| 3 | Markdown | Revision (incorporate peer feedback) |
| 4 | Markdown | Consolidation (single model synthesizes all revisions) |
| 5 | JSON | Final review (vote APPROVE/REVISE with required changes) |
| 6 | Markdown | Final draft (apply changes, publication-ready) |

Vision-capable models (GPT-4o+, Claude 3+, Gemini 1.5+) receive PDF page images in Stage 1.

## Requirements

- Python 3.11+
- Node.js 18+
- CLIProxyAPI running on port 8317
- Tesseract (optional, for OCR)

## Install

```bash
uv sync                    # backend
cd frontend && npm install # frontend
```

## Run

```bash
./start.sh
```

Opens:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000/api/health

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | CLIProxyAPI status |
| GET | /api/models | Available models |
| GET/POST | /api/patients | List/create patients |
| POST | /api/patients/{id}/reports | Upload PDF |
| POST | /api/runs | Create analysis run |
| POST | /api/runs/{id}/start | Start 6-stage workflow |
| GET | /api/runs/{id}/stream | SSE progress events |
| POST | /api/runs/{id}/export | Generate final.md + final.pdf |

## Explainer Video Publishing (local-explainer-video)

This repo is the “ground truth + distribution” side of the patient explainer video pipeline:

- Ground truth artifacts live under `data/artifacts/<run_id>/` (Stage 4 consolidation + Stage 1 `_data_pack.json`)
- Final MP4s should be published into `data/portal_patients/<PATIENT_ID>/` so `thrylen` can sync them to the clinician portal

Recommended workflow:
- Generate the video in `../local-explainer-video`
- Run **QC + Publish** (Step 3) which verifies the narration + slide text against qEEG Council artifacts. Visual QC is **check-only by default** (writes `qc_visual_issues.json` when problems are found), and can optionally auto-fix slide text via image-edit (no regeneration). It then re-renders the MP4 and publishes:
  - `data/portal_patients/<PATIENT_ID>/<PATIENT_ID>.mp4`
  - Backend upload `POST /api/patients/{patient_uuid}/files` (DB-tracked)
- Image models (in the explainer repo): generate via `qwen/qwen-image-2512`, edit via `qwen/qwen-image-edit-2511` (or DashScope `qwen-image-edit-max` when configured).

## Tests

```bash
uv run pytest -q
```

## Environment

Optional `.env` variables:

```
CLIPROXY_BASE_URL=http://127.0.0.1:8317
CLIPROXY_API_KEY=
DEFAULT_CONSOLIDATOR=claude-opus-4-5-20251101
```

## License

Source-available. Noncommercial use only. No qEEG/EEG/ERP use (including research). Commercial licensing available by permission—contact the repository owner.

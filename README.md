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

Source-available. Noncommercial use only. No qEEG/EEG/ERP use (including research). Commercial licensing available by permission—contact dmontg@gmail.com.

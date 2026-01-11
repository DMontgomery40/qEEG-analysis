# CODEX CLI BUILD PROMPT: qEEG Council (forked from llm-council)

Repo state
- This repo is already a fork of karpathy/llm-council.
- Root-level docs and cascading agent instructions exist:
  - AGENTS.md (repo rules)
  - backend/AGENTS.md (backend rules)
  - frontend/AGENTS.md (frontend rules)
  - CLAUDE.md (technical notes)
- Repo-scoped Skills exist under .codex/skills and .claude/skills.
- Prompt skills were intentionally skipped, so you must create backend prompt files.

Your job
- Transform this fork into a working local app called qEEG Council.
- Replace OpenRouter usage with CLIProxyAPI as the ONLY LLM upstream.
- Add patient tracking, report upload and extraction, a 6-stage workflow, run history, artifact viewing, final selection, and export to Markdown and PDF.
- Produce a working end-to-end system that runs locally.

Hard constraints
- No browser automation, no scraping web UIs.
- No direct provider SDK calls.
- Only call CLIProxyAPI at CLIPROXY_BASE_URL (default http://127.0.0.1:8317).
- No encryption and no PHI guardrails.
- Keep ports:
  - frontend: 5173
  - backend: 8000
  - CLIProxyAPI: 8317

Upstream contract: CLIProxyAPI
- Must support:
  - GET  /v1/models
  - POST /v1/chat/completions
  - Optional POST /v1/responses fallback
- Optional auth:
  - If CLIPROXY_API_KEY is non-empty, send Authorization: Bearer <key>

Implementation decisions
- Metadata in SQLite (data/app.db).
- Large text and uploads stored as files in data/ (not blobs in SQLite).
- Workflow artifacts stored deterministically on disk.
- The UI reads artifacts via backend endpoints, not directly from filesystem.
- SSE streaming uses GET endpoint:
  - GET /api/runs/{run_id}/stream returns text/event-stream

Acceptance criteria
- Can create a patient.
- Can upload qEEG report as PDF or text.
- Can create a run and start it.
- Can see stage progress live.
- Can view each model’s outputs per stage.
- Can compare final drafts and select the final.
- Can export final report to final.md and final.pdf.
- /api/health verifies CLIProxyAPI reachability and model discovery.
- Minimal tests run and pass.

--------------------------------------------
PHASE 1: Backend refactor and CLIProxyAPI client
--------------------------------------------

1) Remove OpenRouter coupling
- Find the existing OpenRouter client module (likely backend/openrouter.py or similar).
- Replace it with backend/llm_client.py implementing AsyncOpenAICompatClient.

backend/llm_client.py requirements
- Use httpx.AsyncClient with a configurable timeout.
- Constructor:
  - base_url: str
  - api_key: str
  - timeout_s: float
- Methods:
  - async list_models() -> list[str]
  - async chat_completions(model_id: str, messages: list[dict], temperature: float, max_tokens: int, stream: bool) -> str
  - async responses(model_id: str, input_text: str, stream: bool) -> str
- Fallback behavior:
  - Try /v1/chat/completions first.
  - If the response indicates chat completions not supported for the model, retry once with /v1/responses.
  - Preserve error messages in raised exceptions for UI display.

2) Update backend/config.py
- Must include:
  - CLIPROXY_BASE_URL default http://127.0.0.1:8317
  - CLIPROXY_API_KEY default empty string
  - COUNCIL_MODELS: list of objects:
    - id: model id to send upstream
    - name: UI name
    - source: badge text (example “Subscription via CLIProxyAPI”)
    - endpoint_preference: “chat” or “responses”
  - DEFAULT_CONSOLIDATOR: model id string
  - DATA_DIR and subdirs:
    - REPORTS_DIR
    - ARTIFACTS_DIR
    - EXPORTS_DIR
- Do not hardcode model ids as truth.
- Implement discovery:
  - On startup, call CLIProxyAPI /v1/models and cache discovered ids.
  - Expose discovered ids in /api/models response.
  - Mark configured models as available/unavailable based on discovery.

3) Add SQLite storage layer (backend/storage.py)
- Use SQLAlchemy 2.x.
- Create tables:
  - Patient: id (uuid string), label, notes, created_at, updated_at
  - Report: id (uuid), patient_id, filename, mime_type, stored_path, extracted_text_path, created_at
  - Run: id (uuid), patient_id, report_id, status, council_model_ids_json, consolidator_model_id, label_map_json, started_at, completed_at, selected_artifact_id
  - Artifact: id (uuid), run_id, stage_num, stage_name, model_id, kind, content_path, content_type, created_at
- Store council model ids as JSON text, and stage label map as JSON text.
- Create DB at data/app.db automatically if missing.

4) Add report ingestion (backend/reports.py)
- Upload endpoint accepts PDF or text file.
- For PDF extraction, use pypdf.
- Save:
  - original file to data/reports/<patient_id>/<report_id>/original.<ext>
  - extracted text to data/reports/<patient_id>/<report_id>/extracted.txt
- Return report_id and extracted text preview (first 4000 chars).

5) Implement 6-stage workflow (backend/council.py)
Create QEEGCouncilWorkflow with:
- run_pipeline(run_id: str, on_event: callable | None) -> None
- Deterministic stages and artifact writing.

Stage behavior and artifacts
- Stage 1: initial analyses (parallel)
  - Output per model: Markdown .md
  - kind=analysis, stage_name=initial_analysis, stage_num=1

- Stage 2: peer review (parallel, anonymized)
  - Create a per-run label mapping A/B/C in a stable shuffled order.
  - Each reviewer sees the other analyses as “Analysis A”, “Analysis B”, etc.
  - Output per reviewer: JSON only
  - kind=peer_review, stage_name=peer_review, stage_num=2

- Stage 3: revision (parallel)
  - Each model sees its original analysis plus the peer review sections relevant to its label.
  - Output per model: Markdown .md
  - kind=revision, stage_name=revision, stage_num=3

- Stage 4: consolidation (single model)
  - Consolidator sees all Stage 3 revisions.
  - Output: Markdown .md
  - kind=consolidation, stage_name=consolidation, stage_num=4

- Stage 5: final review (parallel vote)
  - Each model reviews the consolidated report.
  - Output per model: JSON only with keys:
    - vote: APPROVE or REVISE
    - required_changes: string[]
    - optional_changes: string[]
    - quality_score_1to10: int
  - kind=final_review, stage_name=final_review, stage_num=5

- Stage 6: final drafts (parallel)
  - Each model rewrites consolidated report applying required changes without adding new facts.
  - Output per model: Markdown .md
  - kind=final_draft, stage_name=final_draft, stage_num=6

Event streaming
- Emit events per stage completion at minimum.
- on_event payload example:
  - { "run_id": "...", "stage_num": 3, "stage_name": "revision", "status": "complete" }

Resilience
- If a model fails in Stage 1, continue with remaining models.
- Consolidation requires at least one revision artifact.
- For 429, 502, 503 implement bounded retries with backoff.
- For 401 mark run status as needs_auth and stop.

6) FastAPI app and endpoints (backend/main.py)
- Fix the backend port to 8000.
- CORS allow origin http://localhost:5173.
- Endpoints:
  - GET /api/health
    - includes:
      - cliproxy_reachable: bool
      - discovered_model_count: int
  - GET /api/models
    - discovered_models: string[]
    - configured_models: [{id,name,source,endpoint_preference,available}]
  - Patients:
    - GET /api/patients
    - POST /api/patients
    - GET /api/patients/{id}
    - PUT /api/patients/{id}
  - Reports:
    - POST /api/patients/{id}/reports (multipart)
    - GET /api/patients/{id}/reports
  - Runs:
    - POST /api/runs
      - body: patient_id, report_id, council_model_ids, consolidator_model_id
    - POST /api/runs/{run_id}/start
    - GET  /api/runs/{run_id}
    - GET  /api/runs/{run_id}/artifacts
    - GET  /api/runs/{run_id}/stream (SSE)
  - Selection:
    - POST /api/runs/{run_id}/select
      - body supports either:
        - artifact_id
        - or {stage_num, model_id, kind}
  - Export:
    - POST /api/runs/{run_id}/export
      - generates final.md and final.pdf from selected artifact
    - GET /api/runs/{run_id}/export/final.md
    - GET /api/runs/{run_id}/export/final.pdf

PDF export
- Implement a simple PDF generator using reportlab.
- Render headings and paragraphs with readable spacing.
- No need to fully render markdown styling, but preserve line breaks and headings.

7) Create prompt templates (backend/prompts)
Create these files exactly:
- stage1_analysis.md
- stage2_peer_review.md
- stage3_revision.md
- stage4_consolidation.md
- stage5_final_review.md
- stage6_final_draft.md

Prompt rules
- Stage 2 and Stage 5 must demand JSON only, with no leading text.
- Markdown stages should use stable headings:
  - Findings
  - Interpretation
  - Clinical correlations
  - Recommendations
  - Uncertainties and limits

--------------------------------------------
PHASE 2: Frontend update for patients + runs + 6 stages
--------------------------------------------

1) Keep the llm-council “tabs per model” UX where it fits.
2) Add navigation:
- Sidebar patient list with search and “New patient”
- Patient detail page shows:
  - notes
  - uploaded reports
  - run history
3) Add New Run wizard:
- Select a report
- Select council models from /api/models discovered list
- Select consolidator
- Create run and start run
4) Run detail page:
- 6-stage progress indicator
- Stage views:
  - Stage 1: per-model markdown tabs
  - Stage 2: peer review JSON viewer with A/B/C and de-anonymized labels in UI
  - Stage 3: per-model markdown tabs
  - Stage 4: consolidated report markdown
  - Stage 5: vote JSON viewer and summary
  - Stage 6: final draft compare view (side-by-side or tabbed)
- Selection control:
  - Select final draft or consolidated report
  - Calls POST /api/runs/{run_id}/select
- Export controls:
  - Calls POST /api/runs/{run_id}/export
  - Downloads md and pdf
5) SSE integration
- Use EventSource on GET /api/runs/{run_id}/stream
- Update stage progress as events arrive
- Show error banners for:
  - CLIProxyAPI unreachable
  - model unavailable
  - run needs_auth
  - rate limited

--------------------------------------------
PHASE 3: Developer ergonomics and docs
--------------------------------------------

1) Update README.md to match qEEG Council and CLIProxyAPI usage.
2) Ensure start.sh starts backend and frontend in two processes, and prints URLs.
3) Add minimal tests
- backend unit tests:
  - model discovery parse
  - storage create and basic CRUD
  - artifact path creation
  - stage 2 and stage 5 JSON parsing sanity
4) Add a small script in backend/scripts:
- scripts/cliproxy_models.py prints discovered model ids
- scripts/smoke_api.py hits /api/health

--------------------------------------------
Deliverables
--------------------------------------------

- All backend code implemented and running on :8000
- Frontend implemented and running on :5173
- Works end-to-end with CLIProxyAPI running on :8317
- README updated
- Tests included and passing

Finish by running a local smoke test
- Start backend and frontend
- Confirm /api/health returns reachable true when CLIProxyAPI is running
- Create patient, upload report, create run, start run, observe stages, select final, export md and pdf

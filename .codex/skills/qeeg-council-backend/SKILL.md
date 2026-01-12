---
name: qeeg-council-backend
description: Build and maintain the qEEG Council FastAPI backend, CLIProxyAPI upstream client, SQLite persistence, and the 6-stage workflow artifacts and exports.
license: MIT
compatibility: Designed for Claude Code and Codex CLI. Requires python 3.11+, uv, and CLIProxyAPI reachable at CLIPROXY_BASE_URL (default http://127.0.0.1:8317).
metadata:
  author: dmontgomery
  version: "1.0"
---

# qEEG Council Backend Skill

## Use this Skill for
- implementing or refactoring FastAPI endpoints
- implementing the 6-stage workflow orchestration
- adding or changing artifact formats and storage layout
- integrating CLIProxyAPI model discovery and request routing
- implementing SSE progress streaming

## Ground rules
- Treat CLIProxyAPI as the only model upstream.
- Discover model IDs from `GET /v1/models` at startup and expose them via `/api/models`.
- Prefer `POST /v1/chat/completions`. Retry once with `POST /v1/responses` when a model rejects chat completions.
- Store metadata in SQLite and large text artifacts on disk.
- “WAVi” is vendor/report content (common in PDFs), not a code concept.

## Modes
- Real mode (default): backend calls CLIProxyAPI and uses real discovered model IDs.
- Mock mode (tests only): set `QEEG_MOCK_LLM=1` before starting the backend. Mock mode is not valid for evaluating report quality.

## Quick start checks
- CLIProxyAPI reachability + discovered models: `uv run python -m backend.cliproxy_status`
- Backend smoke test (requires backend running): `uv run python backend/scripts/smoke_api.py`
- CLIProxyAPI model list (direct): `uv run python backend/scripts/cliproxy_models.py`

## Canonical references
- Workflow spec: [references/workflow.md](references/workflow.md)
- API contract: [references/api-contract.md](references/api-contract.md)
- JSON shapes:
  - Stage 2 peer review schema: [assets/schemas/stage2_peer_review.schema.json](assets/schemas/stage2_peer_review.schema.json)
  - Stage 5 final review schema: [assets/schemas/stage5_final_review.schema.json](assets/schemas/stage5_final_review.schema.json)

## Implementation checklist
1. `backend/llm_client.py`
   - async httpx client to CLIProxyAPI
   - optional auth header via CLIPROXY_API_KEY
   - list_models + chat_completions + responses + fallback logic

2. `backend/storage.py`
   - SQLite tables for patients, reports, runs, artifacts
   - file layout under `data/` for report uploads, extracted text, artifacts, and exports

3. `backend/council.py`
   - deterministic 6-stage pipeline
   - per-run anonymization map (A/B/C)
   - artifact writing per stage
   - stage completion events for SSE
   - Stage 1 multimodal:
     - uses extracted text plus page images for vision-capable models
     - current implementation limits a single multimodal call to 10 pages; for PDFs >10 pages, implement multi-pass if “all data must be available”

4. `backend/main.py`
   - health endpoint validates CLIProxyAPI reachability
   - model endpoint returns configured + discovered models
   - run endpoints support start + SSE stream + export
   - report endpoints support extracted text viewing and re-extraction/OCR:
     - `GET /api/reports/{report_id}/extracted`
     - `POST /api/reports/{report_id}/reextract`

## Report storage gotcha (don’t miss)
- Report files live under `data/reports/<patient_id>/<upload_id>/...`
- The DB `report_id` is not guaranteed to equal `<upload_id>` (folder name).
- Always locate report assets via `stored_path` / `extracted_text_path`, not by constructing paths from ids.

## Validation utilities
- Stage 2 JSON validation: `python scripts/validate_stage2_peer_review.py path/to/file.json`
- Stage 5 JSON validation: `python scripts/validate_stage5_final_review.py path/to/file.json`

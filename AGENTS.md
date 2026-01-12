# Repository Agents (qEEG Council)

This file is for **AI agents working in this repo**. For architecture details, read `CLAUDE.md`.

## Start here (don’t skip)

1. Read `CLAUDE.md` for the real topology, endpoints, and data layout.
2. Understand that **“WAVi” is vendor/report content** inside PDFs, not a code module.
3. Confirm whether you are evaluating **real runs** vs **mock runs**:
   - Real runs: default behavior, calls CLIProxyAPI
   - Mock runs: `QEEG_MOCK_LLM=1` (tests only; not valid for “report quality”)

## Where to work

- Backend guidance: `backend/AGENTS.md`
- Frontend guidance: `frontend/AGENTS.md`
- Codex skill references: `.codex/skills/`

## Quick commands (common)

- Start everything: `./start.sh`
- Backend only: `uv run python -m backend.main`
- Backend tests: `uv run pytest -q`
- Frontend dev: `cd frontend && npm run dev`
- Frontend tests: `cd frontend && npm test`

## “All data must be available” rule of thumb

Before blaming models, verify the backend has actually extracted and stored:
- `extracted.txt`
- `extracted_enhanced.txt`
- `pages/page-*.png`

If missing/garbled, use `POST /api/reports/{report_id}/reextract` (or trigger it via the UI button “Re-extract (OCR)”).

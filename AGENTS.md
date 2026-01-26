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
- Skill references:
  - Codex: `.codex/skills/`
  - Claude Code: `.claude/skills/`

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

## Explainer Videos (cross-repo)

The patient-facing “explainer video” pipeline lives in `../local-explainer-video`, but it depends on this repo as the
ground truth + publishing target.

- Patient mapping is by **patient label**: `MM-DD-YYYY-N` (must match across repos)
- Narrative ground truth: **Stage 4 consolidation** artifact
- Numeric ground truth: **Stage 1 `_data_pack.json`** artifact
- Publish target folder (watched by `thrylen`): `data/portal_patients/<PATIENT_ID>/`
- Visual QC default is **check-only** (no automated image edits). When issues are found, the explainer repo writes:
  - `../local-explainer-video/projects/<PROJECT>/qc_visual_issues.json`

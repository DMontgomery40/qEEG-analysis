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
- In the explainer repo, **Generate/Regenerate** uses `qwen/qwen-image-2512` and **Edit Image** uses `qwen/qwen-image-edit-2511` (or DashScope `qwen-image-edit-max` when configured).

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **qEEG-analysis** (5903 symbols, 14667 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/qEEG-analysis/context` | Codebase overview, check index freshness |
| `gitnexus://repo/qEEG-analysis/clusters` | All functional areas |
| `gitnexus://repo/qEEG-analysis/processes` | All execution flows |
| `gitnexus://repo/qEEG-analysis/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

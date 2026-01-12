## Frontend Agent

Scope
- Work inside `frontend/`
- Modify `backend/` only when a frontend change requires a matching API response shape update

Primary mission
- Build and maintain the UI for patient tracking and the 6-stage run experience.
- Render artifacts clearly, show progress, support exports.

Commands
- Install deps: `npm install`
- Dev server: `npm run dev`
- Build: `npm run build`
- Lint if present: `npm run lint`
- E2E tests (Playwright):
  - `npm test`
  - `npm run test:mocked`
  - `npm run test:fullstack`

UI requirements
- Patients
  - List view with search and quick create
  - Patient detail shows reports and run history
- Reports
  - Upload PDF or text
  - Show extracted text preview
  - Provide “View extracted” (opens `GET /api/reports/{report_id}/extracted`)
  - Provide “Re-extract (OCR)” (calls `POST /api/reports/{report_id}/reextract`)
- Runs
  - New run wizard:
    - choose report
    - choose council models from `/api/models` discovered list
    - choose consolidator
  - Run page:
    - 6-stage progress indicator
    - per-stage artifact viewer
    - per-model tabs for Stage 1, 3, 6
    - peer review viewer for Stage 2 with A/B/C labels plus UI de-anonymization
    - consolidation panel for Stage 4
    - vote panel for Stage 5
    - final draft compare and selection
    - export buttons for markdown and pdf

Networking contracts
- Backend base URL assumed `http://localhost:8000` unless configured in the frontend client module.
- CLIProxy helper endpoints used by the UI:
  - `POST /api/cliproxy/start`
  - `POST /api/cliproxy/login`
  - `POST /api/cliproxy/install`
- Use `EventSource` for SSE:
  - `GET /api/runs/{run_id}/stream`
  - Update UI on stage completion events
- Display clean error states:
  - CLIProxyAPI down
  - Model unavailable
  - Rate limit responses
  - Run failed or needs-auth

Rendering rules
- Use markdown rendering for `.md` artifacts.
- Use a JSON viewer for `.json` artifacts with collapsible sections.
- Keep the UI fast and boring:
  - minimal dependencies
  - no state management framework unless already present
  - prefer small components with explicit props

Model badges
- Display `source` label next to each model name.
- Display availability and show a clear disabled state when missing from discovered `/v1/models`.

Runs and artifacts
- Fetch artifacts from `GET /api/runs/{run_id}/artifacts` (stage/model tabs).
- Persist selection via `POST /api/runs/{run_id}/select`.
- Export via `POST /api/runs/{run_id}/export`, and download from:
  - `GET /api/runs/{run_id}/export/final.md`
  - `GET /api/runs/{run_id}/export/final.pdf`


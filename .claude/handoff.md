# Agent Handoff: qEEG Council Project

## Project Overview

qEEG Council is a 6-stage LLM deliberation system for analyzing quantitative EEG (qEEG) and Event-Related Potential (ERP) clinical reports. Multiple LLMs analyze a patient's qEEG PDF report, peer-review each other's work anonymously, revise based on feedback, and produce a consolidated clinical document.

### Core Flow
```
PDF Upload → Text/OCR Extraction + Page Images → 6-Stage LLM Pipeline → Markdown/PDF Export
```

### Architecture
```
Frontend (React/Vite :5173) → Backend (FastAPI :8000) → CLIProxyAPI (:8317) → LLM Providers
                                      ↓
                              SQLite + Filesystem
```

**CLIProxyAPI** is a local proxy that routes requests to OpenAI, Anthropic, and Google. It handles authentication via browser-based OAuth. The backend talks ONLY to CLIProxyAPI, never directly to LLM providers.

---

## What Was Done This Session

### 1. Fixed PDF Extraction (Multimodal Support)

**Problem:** Agents only got OCR text, missing tables/graphs rendered as images.

**Solution:**
- Enhanced `backend/reports.py` to run OCR on ALL pages (not just empty ones)
- Added page image extraction as PNG files
- Store images at `data/reports/{patient_id}/{report_id}/pages/page-{n}.png`
- Store enhanced OCR at `data/reports/{patient_id}/{report_id}/extracted_enhanced.txt`

**Key functions added:**
- `extract_text_enhanced()` - OCR all pages
- `extract_pdf_with_images()` - returns (text, page_images)
- `get_page_images_base64()` - retrieve images for multimodal LLM input
- `get_enhanced_text()` - retrieve enhanced OCR text

### 2. Added Multimodal LLM Support

**Problem:** Vision-capable models couldn't "see" the PDF pages.

**Solution:**
- Added `VISION_CAPABLE_MODELS` set in `backend/config.py` with GPT-4o+, GPT-5+, Claude 3+, Gemini 1.5+
- Added `is_vision_capable(model_id)` function with substring matching
- Added `_call_model_multimodal()` in `backend/council.py` that sends images as base64 data URLs
- Stage 1 now sends page images to vision-capable models

**Format:** Uses OpenAI's multimodal format (works across all providers via CLIProxyAPI):
```python
{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}}
```

### 3. Context Propagation to All Stages

**Problem:** Stages 2-6 didn't have the original patient report, causing hallucination during peer review.

**Solution:** Modified all stage methods in `backend/council.py` to include original report text:
- `_stage2()` - adds report text for verification
- `_stage3()` - adds report text for fact-checking revision
- `_stage4()` - adds report text as source of truth for consolidation
- `_stage5()` - adds report text for final review verification
- `_stage6()` - adds report text for final polish

Pattern used:
```python
prompt_text = f"{prompt}\n\n---\n\nORIGINAL qEEG REPORT (for verification):\n\n{report_text}\n\n---\n\n..."
```

### 4. Increased max_tokens

| Stage | Before | After |
|-------|--------|-------|
| 1 | 2200 | 8000 |
| 2 | 1400 | 2000 |
| 3 | 2200 | 6000 |
| 4 | 2600 | 8000 |
| 5 | 1200 | 1500 |
| 6 | 2600 | 8000 |

### 5. Updated All Prompt Files

All 6 prompt files in `backend/prompts/` were rewritten with:
- Explicit word count requirements (2500-4000 words)
- Required 8-section structure
- Per-site P300 table requirement (C3, CZ, C4, P3, PZ, P4)
- Speculative Commentary section with competing hypotheses
- Quality checklists for review stages

### 6. Fixed Report ID Mismatch Bug

**Problem:** `save_report_upload()` generated one UUID for files, but `create_report()` generated a different UUID for the DB record. This broke `get_page_images_base64()`.

**Solution:**
- Modified `create_report()` in `backend/storage.py` to accept optional `report_id` parameter
- Modified `backend/main.py` to pass the same `report_id` to both functions

### 7. Updated README

Rewrote `README.md` to be accurate and concise:
- Architecture diagram
- Data flow description
- Storage paths
- 6-stage workflow table
- API endpoints table
- Environment variables

### 8. Added Playwright GUI Tests

Created `.playwright/` folder (gitignored) with comprehensive tests:
- 12 GUI tests (homepage, patients, models, upload, checkboxes, navigation)
- 4 API integration tests (health, models, patients endpoints)

---

## Project Structure

```
qEEG-analysis/
├── backend/
│   ├── main.py              # FastAPI app, all API routes
│   ├── council.py           # 6-stage workflow orchestration (QEEGCouncilWorkflow class)
│   ├── llm_client.py        # AsyncOpenAICompatClient for CLIProxyAPI
│   ├── storage.py           # SQLAlchemy models + CRUD (Patient, Report, Run, Artifact)
│   ├── reports.py           # PDF extraction, OCR, page images
│   ├── config.py            # VISION_CAPABLE_MODELS, paths, model config
│   └── prompts/
│       ├── stage1_analysis.md
│       ├── stage2_peer_review.md
│       ├── stage3_revision.md
│       ├── stage4_consolidation.md
│       ├── stage5_final_review.md
│       └── stage6_final_draft.md
├── frontend/
│   └── src/                  # React app
├── data/                     # (gitignored) runtime data
│   ├── app.db               # SQLite database
│   ├── reports/             # Uploaded PDFs, extracted text, page images
│   ├── artifacts/           # Stage outputs per run
│   └── exports/             # Final exported reports
├── examples/                 # Reference qEEG reports for quality comparison
├── .playwright/              # (gitignored) Playwright tests
│   ├── tests/gui.spec.js
│   ├── playwright.config.js
│   ├── package.json
│   └── node_modules/
├── .env                      # (gitignored) environment variables
├── pyproject.toml           # Python dependencies
├── start.sh                 # Starts backend + frontend + CLIProxyAPI
└── README.md
```

---

## Key Files Modified This Session

1. **backend/config.py** - Added `VISION_CAPABLE_MODELS`, `is_vision_capable()`
2. **backend/storage.py** - `create_report()` now accepts `report_id` parameter
3. **backend/main.py** - Passes `report_id` to `create_report()`
4. **backend/reports.py** - Added `extract_text_enhanced()`, `extract_pdf_with_images()`, `get_page_images_base64()`, `get_enhanced_text()`, `_render_page_image()`, `_ocr_image_bytes()`, `_ocr_available()`
5. **backend/council.py** - Added `_call_model_multimodal()`, updated all `_stage*()` methods with context propagation and increased max_tokens
6. **backend/prompts/*.md** - All 6 files completely rewritten
7. **pyproject.toml** - Added `pillow>=10.0.0` dependency
8. **README.md** - Complete rewrite
9. **.gitignore** - Added `.playwright/` entries

---

## 6-Stage Workflow Details

| Stage | Method | Output | Description |
|-------|--------|--------|-------------|
| 1 | `_stage1()` | Markdown | Initial analysis per model. Vision models receive page images. |
| 2 | `_stage2()` | JSON | Peer review. Models evaluate each other anonymously (Response A, B, C). |
| 3 | `_stage3()` | Markdown | Revision. Each model incorporates peer feedback. |
| 4 | `_stage4()` | Markdown | Consolidation. Single consolidator synthesizes all revisions. |
| 5 | `_stage5()` | JSON | Final review. Models vote APPROVE/REVISE with required_changes. |
| 6 | `_stage6()` | Markdown | Final draft. Apply changes, produce publication-ready document. |

---

## How to Run Tests

### Backend Python Tests
```bash
uv run pytest -q
```

### Playwright GUI Tests

**Prerequisites:** Frontend and backend must be running.

```bash
# Start servers (in separate terminal or use start.sh)
./start.sh

# Run Playwright tests
cd .playwright
npm install                    # First time only
npx playwright install chromium  # First time only
npx playwright test --reporter=list
```

**Expected output:** 16 passed tests

### Manual Pipeline Test

```python
import asyncio
from backend.storage import init_db, session_scope, create_run, get_report
from backend.council import QEEGCouncilWorkflow
from backend.llm_client import AsyncOpenAICompatClient
from backend.config import CLIPROXY_BASE_URL, CLIPROXY_API_KEY

init_db()

# Use existing patient/report IDs from DB
PATIENT_ID = '...'
REPORT_ID = '...'

with session_scope() as session:
    run = create_run(
        session,
        patient_id=PATIENT_ID,
        report_id=REPORT_ID,
        council_model_ids=["claude-opus-4-5-20251101", "gemini-3-pro-preview", "gpt-5.1"],
        consolidator_model_id="claude-opus-4-5-20251101"
    )
    run_id = run.id

async def run_pipeline():
    llm = AsyncOpenAICompatClient(base_url=CLIPROXY_BASE_URL, api_key=CLIPROXY_API_KEY, timeout_s=600.0)
    workflow = QEEGCouncilWorkflow(llm=llm)
    await workflow.run_pipeline(run_id=run_id)
    await llm.aclose()

asyncio.run(run_pipeline())
```

---

## Environment Setup

### Required Services
1. **CLIProxyAPI** on port 8317 (handles LLM auth)
2. **Backend** on port 8000
3. **Frontend** on port 5173

### Environment Variables (.env)
```
CLIPROXY_BASE_URL=http://127.0.0.1:8317
CLIPROXY_API_KEY=
DEFAULT_CONSOLIDATOR=claude-opus-4-5-20251101
```

### Dependencies
```bash
uv sync                    # Python backend
cd frontend && npm install # React frontend
```

### Optional: Tesseract OCR
For enhanced PDF text extraction:
```bash
brew install tesseract  # macOS
```

---

## Quality Verification

Final reports should have:
- **Word count:** 3000-4000 words
- **All 8 sections:** Dataset and Sessions, Key Empirical Findings, Performance Assessments, Auditory ERP, Background EEG, Speculative Commentary, Measurement Recommendations, Uncertainties
- **Per-site P300 table** with C3, CZ, C4, P3, PZ, P4
- **Competing hypotheses** in Speculative Commentary (not collapsed to single narrative)

Check artifact quality:
```python
content = open(f"data/artifacts/{run_id}/stage-6/{model_id}.md").read()
print(f"Words: {len(content.split())}")
print(f"Has P300 table: {'| Site |' in content}")
```

---

## Known Issues / Future Work

1. **No streaming:** Pipeline runs to completion before returning results
2. **No progress percentage:** SSE events show stage transitions but not per-model progress within stages
3. **No retry UI:** If a stage fails, must restart entire run
4. **Page image limit:** Multimodal capped at 10 pages to avoid token limits

---

## Useful Commands

```bash
# Check CLIProxyAPI models
curl -s http://127.0.0.1:8317/v1/models | python3 -m json.tool

# Check backend health
curl -s http://localhost:8000/api/health | python3 -m json.tool

# List patients
curl -s http://localhost:8000/api/patients | python3 -m json.tool

# Compile check
python -m py_compile backend/council.py backend/reports.py backend/config.py

# Run single Playwright test
cd .playwright && npx playwright test -g "loads homepage"
```

# HANDOFF: Multi-PDF Council Run for Patient 01-01-2013-0

## THE TASK

Patient `01-01-2013-0` has 3 WAVi PDF reports that together cover 5 sessions spanning Oct 2025 → Feb 2026. The qEEG Council pipeline is built for ONE PDF per run. You need to make it process all 5 sessions in ONE council run ($70/run, so no separate runs per PDF).

The council handles 3-5 sessions all the time — that's normal. The ONLY problem is the data is split across 3 PDFs instead of 1.

## THE SOLUTION

Extract the data from all 3 PDFs, combine/concatenate the extracted content into ONE unified input, and feed that single combined input through ONE council run. The council doesn't care that the data originally came from 3 files — it just sees extracted text + page images.

## THE 3 PDFs

Located in: `/Users/davidmontgomery/qEEG-analysis/data/portal_patients/01-01-2013-0/`

| File | Sessions | Dates | Pages |
|------|----------|-------|-------|
| `Dec_V_final qeeg_Redacted (1).pdf` | S1-S3 | Oct 24, Nov 14, Dec 8 2025 | 15 |
| `CV_ADHD_after 30 tx_Redacted.pdf` | S1-S4 | Oct 24, Nov 14, Dec 8, **Jan 14 2026** | 17 |
| `CV_ADHD_after 40 tx_Redacted.pdf` | S1-S2 | **Jan 14, Feb 18 2026** | 13 |

**IMPORTANT OVERLAP**: "After 30 tx" is a SUPERSET of the December report (adds Session 4). "After 40 tx" starts a new comparison series from Jan 14. Between "after 30 tx" and "after 40 tx" you get all 5 unique sessions. The December PDF is redundant (subset of "after 30 tx").

The PDFs are WAVi device exports — heavily image-based. Tables, topographic brain maps, spectrum plots, and coherence graphs are rendered as IMAGES, not text. That's why the extraction pipeline exists.

## EXISTING PIPELINE (DO NOT CHANGE)

### Extraction: `backend/reports.py`
- `extract_pdf_full(pdf_path: Path) -> PdfFullExtraction`
  - Runs 4 extraction engines per page: pypdf, PyMuPDF, Apple Vision OCR, Tesseract
  - Renders each page as a high-res PNG (4x zoom)
  - Returns: `.enhanced_text` (combined OCR text), `.page_images` (list of `{"page": int, "base64_png": str}`), `.per_page_sources`, `.metadata`
  - This is a STANDALONE function — no DB needed, just give it a Path

### Upload + DB registration: `backend/reports.py::save_report_upload()`
- Calls `extract_pdf_full()` internally
- Saves extracted text to `data/reports/<patient_id>/<report_id>/extracted_enhanced.txt`
- Saves page PNGs to `data/reports/<patient_id>/<report_id>/pages/page-N.png`
- Creates DB record via `storage.create_report()`

### Council run: `backend/council/workflow/core.py::QEEGCouncilWorkflow.run_pipeline(run_id)`
- Takes a `run_id` → looks up the run's `report_id` → reads that report's extracted text + page images
- Runs 6 stages of multi-model deliberation
- Stage 1 is KEY: vision-capable models look at the actual page IMAGES to extract data (not just OCR text)
- Produces: `_data_pack.json` (structured numeric data), `_vision_transcript.md`, stage 1-6 markdown artifacts

### How council gets report data:
- `backend/council/report_assets.py::_load_best_report_text(report)` — reads extracted text
- `backend/council/report_assets.py::_load_page_images(report)` — reads page PNGs
- Both look at the report's stored paths in DB

### Patient-facing PDF: `scripts/generate_patient_facing_writeups.py`
- Reads council Stage 6 artifacts
- Feeds them to Opus as source material
- Generates markdown → renders to PDF via `backend/patient_facing_pdf.py::render_patient_facing_markdown_to_pdf()`

## WHAT YOU NEED TO DO

### Step 1: Extract all 3 PDFs
Use `extract_pdf_full()` on each PDF. Get the `.enhanced_text` and `.page_images` from each.

### Step 2: Combine into one unified report
Concatenate the extracted text from all 3 PDFs (with clear section markers like "=== REPORT 1: December 2025 (Sessions 1-3) ===" etc.).

Combine all page images into one sequential list (renumber pages so they don't collide).

**Think about overlap**: "After 30 tx" already contains Sessions 1-3 data from December. You may want to use "after 30 tx" + "after 40 tx" only (skipping the redundant December PDF), or include all 3 and let the council handle deduplication. The council models are smart enough to notice overlap.

### Step 3: Register the combined report in the DB
Use `save_report_upload()` OR manually:
1. Create a report directory under `data/reports/<patient_id>/<new_report_id>/`
2. Write the combined extracted text to `extracted_enhanced.txt`
3. Write combined page PNGs to `pages/page-N.png`
4. Call `storage.create_report()` to register in DB

The cleanest approach might be to:
1. Concatenate the actual PDFs into one combined PDF using PyMuPDF (`fitz`)
2. Then call `save_report_upload()` with the combined PDF bytes — the extraction pipeline handles the rest automatically

```python
import fitz
combined = fitz.open()
for pdf_path in pdf_paths:
    src = fitz.open(str(pdf_path))
    combined.insert_pdf(src)
    src.close()
combined_bytes = combined.tobytes()
combined.close()
```

Then `save_report_upload(patient_id=..., report_id=new_uuid, filename="combined_5sessions.pdf", file_bytes=combined_bytes, ...)` does the rest.

**BUT WAIT** — the user explicitly said these aren't PDFs you can just "merge." The PAGE IMAGES from the extraction are what the council's vision models actually analyze. Merging the raw PDFs and re-extracting should work because `extract_pdf_full()` renders each page independently. The merged PDF would have all pages from all 3 PDFs, and the extraction would process each page. The question is whether the merged PDF's page rendering is faithful to the originals.

Test this: merge the PDFs, run `extract_pdf_full()` on the merged result, and verify the page images look correct before running council.

### Step 4: Run ONE council
```python
from backend.storage import create_run, session_scope
from backend.council import QEEGCouncilWorkflow
from backend.llm_client import AsyncOpenAICompatClient
from backend.config import CLIPROXY_BASE_URL, CLIPROXY_API_KEY, COUNCIL_MODELS

llm = AsyncOpenAICompatClient(base_url=CLIPROXY_BASE_URL, api_key=CLIPROXY_API_KEY, timeout_s=600.0)
workflow = QEEGCouncilWorkflow(llm=llm)

with session_scope() as session:
    run = create_run(
        session,
        patient_id=PATIENT_ID,
        report_id=COMBINED_REPORT_ID,
        council_model_ids=COUNCIL_MODELS,
        consolidator_model_id="claude-opus-4-6",
    )

await workflow.run_pipeline(run.id)
```

### Step 5: Generate patient-facing PDF
Use existing `scripts/generate_patient_facing_writeups.py`:
```bash
cd /Users/davidmontgomery/qEEG-analysis
uv run python scripts/generate_patient_facing_writeups.py \
  --patient-label 01-01-2013-0 \
  --model claude-opus-4-6 \
  --version v3-5sessions \
  --overwrite
```

This will pick up the new council run's artifacts automatically (it grabs the latest complete run).

## ENVIRONMENT

- Python: use `uv run python` from the qEEG-analysis repo root
- CLIProxyAPI: must be running at `http://127.0.0.1:8317` (check with `curl http://127.0.0.1:8317/v1/models`)
  - Binary: `/Users/davidmontgomery/.local/bin/cli-proxy-api-plus`
  - Config: `/Users/davidmontgomery/qEEG-analysis/.cli-proxy-api/cliproxyapi.conf`
  - Start: `cd /Users/davidmontgomery/qEEG-analysis && /Users/davidmontgomery/.local/bin/cli-proxy-api-plus -config .cli-proxy-api/cliproxyapi.conf &`
- DB: SQLite at `data/app.db`
- Patient label: `01-01-2013-0`

## WHAT NOT TO DO

1. **Do NOT run 3 separate council runs** — $70 each, totaling $210 for what should be one $70 run
2. **Do NOT modify any existing backend code** — write a script that imports existing functions
3. **Do NOT skip the extraction pipeline** — these PDFs are image-heavy, OCR + vision is required
4. **Do NOT assume you can "read" these PDFs with simple text extraction** — the multi-engine pipeline exists for a reason
5. **Do NOT permanently change the one-PDF-per-run assumption** in the pipeline — this is a one-off combination step before feeding into the standard pipeline

## KEY FILES

```
/Users/davidmontgomery/qEEG-analysis/
├── backend/
│   ├── reports.py              # extract_pdf_full(), save_report_upload()
│   ├── storage.py              # DB models, create_report(), create_run(), find_patients_by_label()
│   ├── council/
│   │   └── workflow/core.py    # QEEGCouncilWorkflow.run_pipeline()
│   ├── llm_client.py           # AsyncOpenAICompatClient
│   ├── patient_facing_pdf.py   # render_patient_facing_markdown_to_pdf()
│   └── config.py               # CLIPROXY_BASE_URL, COUNCIL_MODELS, etc.
├── scripts/
│   └── generate_patient_facing_writeups.py  # Patient-facing PDF generation
├── data/
│   ├── app.db                  # SQLite database
│   ├── reports/                # Extracted report data
│   └── portal_patients/
│       └── 01-01-2013-0/       # This patient's files
│           ├── Dec_V_final qeeg_Redacted (1).pdf
│           ├── CV_ADHD_after 30 tx_Redacted.pdf
│           ├── CV_ADHD_after 40 tx_Redacted.pdf
│           ├── council/3c5f0156-.../  # Existing council run (Dec PDF only)
│           └── 01-01-2013-0.md        # Existing council markdown output
└── CLAUDE.md                   # Project docs
```

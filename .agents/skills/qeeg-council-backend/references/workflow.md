# qEEG Council Workflow (6 stages)

## Shared source context (passed to all stages)
- Report text (prefer enhanced OCR): `extracted_enhanced.txt`
- Stage 1 structured facts: `data/artifacts/<run_id>/stage-1/_data_pack.json`
- Stage 1 vision transcript: `data/artifacts/<run_id>/stage-1/_vision_transcript.md`

## Stage 1: Initial analysis (parallel)
Input:
- extracted qEEG text (prefer enhanced OCR when available)
- PDF page images (`pages/page-*.png`) for vision-capable models

Outputs:
- Run-level (shared): structured data pack + vision transcript
  - `data/artifacts/<run_id>/stage-1/_data_pack.json`
  - `data/artifacts/<run_id>/stage-1/_vision_transcript.md`
- Per-model: Markdown analysis

Artifact kind: analysis
File ext: .md

Notes:
- Stage 1 is the only stage that currently supports multimodal “page images” input.
- Stage 1 must cover **ALL pages** when building the data pack / transcript:
  - chunking is controlled by `QEEG_VISION_PAGES_PER_CALL` (default 8) and is clamped to 10 pages/call
  - PDFs >10 pages will always require 2+ multimodal calls
- If extraction is missing key tables, regenerate via `POST /api/reports/{report_id}/reextract` (writes `extracted.txt`, `extracted_enhanced.txt`, and `pages/page-*.png`).

## Stage 2: Peer review (parallel, anonymized)
Input:
- other models' Stage 1 analyses labeled A/B/C
- shared source context (report text + data pack + vision transcript)
Output (per reviewer): JSON critique
Artifact kind: peer_review
File ext: .json
Required shape: assets/schemas/stage2_peer_review.schema.json

## Stage 3: Revision (parallel)
Input:
- model's Stage 1 analysis + peer feedback about that analysis
- shared source context (report text + data pack + vision transcript)
Output (per model): revised Markdown
Artifact kind: revision
File ext: .md

## Stage 4: Consolidation (single consolidator)
Input:
- all Stage 3 revisions
- shared source context (report text + data pack + vision transcript)
Output: consolidated Markdown report
Artifact kind: consolidation
File ext: .md

## Stage 5: Final review (parallel vote)
Input:
- consolidated report
- shared source context (report text + data pack + vision transcript)
Output (per model): JSON with vote + required edits + score
Artifact kind: final_review
File ext: .json
Required shape: assets/schemas/stage5_final_review.schema.json

## Stage 6: Final drafts (parallel)
Input:
- consolidated report + Stage 5 required edits
- shared source context (report text + data pack + vision transcript)
Output (per model): polished Markdown draft
Artifact kind: final_draft
File ext: .md

## Selection (UI)
User selects one output to become final.
Backend persists selected artifact reference and generates exports:
- final.md
- final.pdf

## Anonymization
- Each run assigns labels A/B/C to chosen model IDs.
- Store mapping in run metadata.
- Use labels in Stage 2 prompts only.
- De-anonymize only in UI.

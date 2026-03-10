# HANDOFF: Single-Agent 5-Session Patient-Facing Report

## Goal

Produce a **patient-facing qEEG report** for patient label `01-01-2013-0` using a **single strong agent/model workflow**, not the 6-stage council.

This is a one-off operational fallback for a real patient. Prioritize:

1. Getting a **good, readable, clinically careful narrative**
2. Capturing the **5 unique sessions**
3. Using **programmatic artifacts and rendering**
4. Allowing **human verification afterward**

Do **not** spend time trying to preserve the council architecture for this case.

## Non-goals

- Do not run the multi-model council
- Do not insist on strict OCR/data-pack completeness
- Do not hand-author a PDF from scratch
- Do not create a throwaway prose-only answer in chat and stop there

The output must be generated through code/scripts/files in this repo.

## Patient / files

Patient label: `01-01-2013-0`

Relevant source PDFs:

- `/Users/davidmontgomery/qEEG-analysis/data/portal_patients/01-01-2013-0/CV_ADHD_after 30 tx_Redacted.pdf`
- `/Users/davidmontgomery/qEEG-analysis/data/portal_patients/01-01-2013-0/CV_ADHD_after 40 tx_Redacted.pdf`

Ignore the December subset PDF unless you need it for cross-checking:

- `/Users/davidmontgomery/qEEG-analysis/data/portal_patients/01-01-2013-0/Dec_V_final qeeg_Redacted (1).pdf`

## Session truth

There are **5 unique sessions** total.

Use this chronology:

- Session 1 = `10/24/2025`
- Session 2 = `11/14/2025`
- Session 3 = `12/08/2025`
- Session 4 = `01/14/2026`
- Session 5 = `02/18/2026`

Important overlap:

- `CV_ADHD_after 30 tx_Redacted.pdf` contains Sessions 1-4
- `CV_ADHD_after 40 tx_Redacted.pdf` contains Sessions 4-5, but labels them locally as Session 1 / Session 2

So the second PDF must be remapped:

- local Session 1 -> global Session 4
- local Session 2 -> global Session 5

## Model / execution preference

Use a **single-agent workflow**.

Preferred plan:

1. Use the existing combined-report assets if helpful:
   - Combined manifest:
     [combined_5sessions.manifest.json](/Users/davidmontgomery/qEEG-analysis/data/portal_patients/01-01-2013-0/combined_5sessions.manifest.json)
   - Combined extracted report:
     [extracted_enhanced.txt](/Users/davidmontgomery/qEEG-analysis/data/reports/0c5c9b57-474e-460a-9d35-705c3898dcf1/0c21306e-aa18-4843-93a1-57dff348d8b2/extracted_enhanced.txt)
   - Combined page images folder:
     [/Users/davidmontgomery/qEEG-analysis/data/reports/0c5c9b57-474e-460a-9d35-705c3898dcf1/0c21306e-aa18-4843-93a1-57dff348d8b2/pages](/Users/davidmontgomery/qEEG-analysis/data/reports/0c5c9b57-474e-460a-9d35-705c3898dcf1/0c21306e-aa18-4843-93a1-57dff348d8b2/pages)

2. Use **OpenAI GPT-5.4** with reasoning effort `xhigh` if the local proxy exposes it.
3. If GPT-5.4 is unavailable locally, use the strongest actually available OpenAI model without blocking the task.
4. If one visual pass with Gemini is more reliable for extracting image-heavy values, that is acceptable.

This fallback does **not** require strict council-style multimodal verification.
Human review afterward is acceptable.

## Required deliverables

Create these programmatic outputs:

1. A machine-generated **analysis markdown**
2. A machine-generated **patient-facing markdown**
3. A programmatically rendered **patient-facing PDF**

Use the repo’s existing PDF/rendering flow where possible. Do not manually lay out a PDF.

## Recommended implementation path

### Option A: fastest acceptable route

Build a one-off script that:

1. Loads the combined extracted text and/or source PDFs
2. Optionally performs one best-effort multimodal pass over page images
3. Produces a single high-quality clinician-grade synthesis
4. Produces a patient-facing markdown report
5. Uses the existing renderer in `backend/patient_facing_pdf.py` to create the PDF

### Option B: leverage existing patient-facing pipeline

If easier, generate a single “source analysis” markdown artifact in the shape expected by:

- [generate_patient_facing_writeups.py](/Users/davidmontgomery/qEEG-analysis/scripts/generate_patient_facing_writeups.py)

Then drive the existing patient-facing generation flow from that artifact.

## Content requirements

The patient-facing report must:

- Explain what changed across **all 5 sessions**
- Be readable by a normal intelligent patient, not just a clinician
- Be careful about uncertainty
- Avoid fake certainty or made-up causal claims
- Avoid jargon overload
- Still include enough concrete detail that it feels grounded in the data

The report should clearly separate:

- what appears improved
- what appears mixed / uncertain
- what remains limited by the source data

## Verification expectations

Do a reasonable verification pass, not a perfectionist blockade.

Acceptable verification:

- confirm the report covers 5 sessions
- confirm Session 4 / Session 5 chronology is not collapsed
- confirm key dates match the truth above
- confirm the generated patient-facing markdown and PDF both exist
- spot-check a few important numeric claims against extracted text or page images

Do **not** fail the whole task just because a strict council-grade data pack is incomplete.

## Constraints

- Operate locally in `/Users/davidmontgomery/qEEG-analysis`
- Prefer modifying/adding scripts over ad hoc terminal-only output
- Preserve existing files unless replacement is intentional
- Do not update CLIProxyAPI-Plus during this task unless absolutely required

## What success looks like

By the end, there should be a usable patient-facing report for `01-01-2013-0` generated from code, covering all 5 sessions, with output files on disk and a short summary of what was generated and where.


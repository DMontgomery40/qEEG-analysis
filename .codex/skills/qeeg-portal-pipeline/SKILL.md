---
name: qeeg-portal-pipeline
description: Run and repair the qEEG portal upload pipeline when clinic uploads, portal patients, Netlify blobs, pipeline jobs, or missing analyses need auditing; use for real qEEG Council runs, worker failures, and patient label backfills. Do not use for mock-only tests or unrelated frontend styling.
---

# qEEG Portal Pipeline

Use this skill when a clinic-uploaded portal patient needs to be found, downloaded, analyzed, repaired, or verified end to end.

## Required Posture

- Real runs only. Do not use `QEEG_MOCK_LLM=1` for incident recovery or quality validation.
- Read `/Users/davidmontgomery/qEEG-analysis/plan.md` before changing pipeline code.
- Read project-local memory under `~/.codex/projects/-Users-davidmontgomery-qEEG-analysis/`.
- If explainer-video output is involved, also read `~/.codex/projects/-Users-davidmontgomery-local-explainer-video/MEMORY.md` and the linked mandatory validation notes.
- Treat Netlify job markers, local status files, DB rows, and generated artifacts as evidence. Do not report success from intent alone.

## Fast Commands

Audit all portal patients once:

```bash
uv run python scripts/portal_pipeline_worker.py --once --dry-run
```

Process one patient label:

```bash
uv run python scripts/portal_pipeline_worker.py --once --include-label MM-DD-YYYY-N
```

Run the existing council batch runner directly:

```bash
uv run python scripts/run_portal_council_batch.py --include-label MM-DD-YYYY-N
```

Check local worker status:

```bash
python3 .codex/skills/qeeg-portal-pipeline/scripts/check_status.py MM-DD-YYYY-N
```

## Workflow

1. Confirm CLIProxyAPI is reachable at `http://127.0.0.1:8317/v1/models`.
2. Confirm the patient exists in Netlify Blobs under `patients/<PATIENT_ID>/`.
3. Run `portal_pipeline_worker.py --once --include-label <PATIENT_ID>`.
4. If it fails, inspect `data/pipeline_jobs/<PATIENT_ID>.json` and remote `pipeline/status/<PATIENT_ID>.json`.
5. Confirm `data/portal_patients/<PATIENT_ID>/` contains source PDFs plus generated council/patient-facing artifacts.
6. Confirm complete runs in SQLite for each source report filename.
7. Update project-local memory and link the new memory note from `MEMORY.md`.

## References

- For the full runtime contract and failure handling, read `references/pipeline-contract.md`.

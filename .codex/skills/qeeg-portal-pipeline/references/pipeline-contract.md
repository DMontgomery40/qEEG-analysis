# Portal Pipeline Contract

## Runtime Pieces

- Clinic upload entrypoint: `/Users/davidmontgomery/thrylen/netlify/functions/qeeg-upload.js`
- Durable job marker prefix: `pipeline/jobs/<PATIENT_ID>/<UPLOADED_AT>.json`
- Local worker: `/Users/davidmontgomery/qEEG-analysis/scripts/portal_pipeline_worker.py`
- Existing batch runner: `/Users/davidmontgomery/qEEG-analysis/scripts/run_portal_council_batch.py`
- Local status: `/Users/davidmontgomery/qEEG-analysis/data/pipeline_jobs/<PATIENT_ID>.json`
- Remote status prefix: `pipeline/status/<PATIENT_ID>.json`
- Local artifacts: `/Users/davidmontgomery/qEEG-analysis/data/portal_patients/<PATIENT_ID>/`

## Guarantees

- Uploading a report PDF with `documentKind: "report"` creates a pending job marker.
- The local worker also audits all portal patient indexes, so old uploads without a marker are still recoverable.
- The worker downloads missing report PDFs from Netlify Blobs before running the council batch.
- The worker considers a source report incomplete until a matching local DB report filename has a complete run.
- Failures must be visible in local and remote status JSON.

## Failure Handling

- CLIProxyAPI unreachable: start or refresh CLIProxyAPI before retry.
- Netlify auth missing: authenticate the Netlify CLI used by the thrylen repo.
- Missing `patients/<PATIENT_ID>/$index.json`: repair portal metadata before running the worker.
- Batch runner nonzero: inspect stdout/stderr tails in `data/pipeline_jobs/<PATIENT_ID>.json`.
- Patient-facing output failure: inspect the auto generation logs and rerun only after source council artifacts exist.

## Verification

- Run narrow tests for the touched qEEG worker and thrylen upload marker.
- Run the repo-standard qEEG backend tests when changing worker behavior.
- Run thrylen Node tests when changing portal upload behavior.
- Validate this skill with:

```bash
python3 /Users/davidmontgomery/.codex/skills/.system/skill-creator/scripts/quick_validate.py /Users/davidmontgomery/qEEG-analysis/.codex/skills/qeeg-portal-pipeline
```

# Portal Upload Pipeline Reliability Plan

Date: 2026-04-12

## Immediate Incident

- Patient label `03-05-2010-0` existed in the thrylen portal blob store but had no local folder under `data/portal_patients/`.
- The missing local folder prevented the qEEG Council batch runner from discovering and processing the clinic-uploaded PDFs.
- Immediate remediation started by downloading the two source PDFs from Netlify Blobs into `data/portal_patients/03-05-2010-0/` and launching:
  - `uv run python scripts/run_portal_council_batch.py --include-label 03-05-2010-0`

## Patient-Facing PDF Backfill

- A later portal audit found missing patient-facing PDFs for `02-28-1978-0`, `09-23-1982-0`, `10-31-2008-0`, and `12-02-1985-0`.
- Operator explicitly excluded `09-23-1982-0` because it is another wrong-birthdate patient.
- Completed and verified patient-facing PDFs:
  - `data/portal_patients/02-28-1978-0/02-28-1978-0__patient-facing__auto-8d7e6f2c__2026-04-12.pdf`
  - `data/portal_patients/10-31-2008-0/10-31-2008-0__patient-facing__auto-7513a2d6__2026-04-12.pdf`
  - `data/portal_patients/12-02-1985-0/12-02-1985-0__patient-facing__auto-b2620384__2026-04-12.pdf`
- Current audit result: all portal patients have patient-facing PDFs except the intentionally excluded `09-23-1982-0`.
- New reliability fixes from this backfill:
  - `--force` must create a fresh run and bypass interrupted resume candidates.
  - Empty LLM text responses are retryable failures, not parseable content.
  - OpenAI Responses output blocks must be reconstructed when `output_text` is empty.
  - Claude chat calls omit `temperature`; live Claude chat calls use sync HTTP because the async path returned empty multimodal responses through CLIProxyAPI.
  - Patient-facing auto generation defaults to Sonnet and retries transient/empty upstream failures.
  - Portal batch fallback selection excludes Opus unless `QEEG_ALLOW_OPUS_MODELS` is explicitly truthy.

## Non-Negotiable Outcome

Every clinic-uploaded qEEG report PDF must become a durable pipeline job. A new portal patient or new report upload may not depend on a human noticing a missing local folder.

## Source-Backed Codex Harness Notes

- Codex hooks are experimental and require `[features] codex_hooks = true`.
- Hooks load from `hooks.json` next to active config layers, commonly `~/.codex/hooks.json` and `<repo>/.codex/hooks.json`.
- Current `PreToolUse` and `PostToolUse` only intercept Bash-shaped tool events and are guardrails, not a production scheduler.
- `Stop` hooks can force continuation with a JSON block decision, useful for preventing incomplete agent turns.
- Skills must have exactly one `SKILL.md`/`skill.md` manifest and clear YAML frontmatter with at least `name` and `description`.
- Skill scripts should be deterministic CLIs that fail loudly and write predictable outputs.

## Runtime Design

1. Portal upload creates durable work.
   - Update `thrylen/netlify/functions/qeeg-upload.js` to append a pipeline job marker whenever one or more uploaded files include qEEG report PDFs.
   - Store markers in the existing `qeeg-portal` Netlify Blob store under `pipeline/jobs/<patientId>/<uploadedAt>.json`.
   - Include `patientId`, uploaded report file keys, original/logical names, upload actor, and timestamps.
   - Job marker write failure must not silently disappear: return the upload response with an explicit pipeline warning field if marker creation fails.
   - qEEG report uploads must derive the canonical patient folder from the report DOB (`reportBirthdate`) when that metadata is available, even if the clinician selected the wrong existing patient folder.
   - Mixed report DOBs in one upload or report uploads without a parsed DOB must fail loudly instead of being routed into the wrong patient bucket.

2. Local qEEG worker pulls from portal and runs analysis.
   - Add a qEEG-analysis CLI worker that:
     - lists portal patient/job blobs,
     - downloads report PDFs missing from `data/portal_patients/<patientId>/`,
     - creates/updates local patient folders,
     - invokes the existing `scripts/run_portal_council_batch.py --include-label <patientId>`,
     - writes local state under `data/pipeline_jobs/`,
     - writes remote status under `pipeline/status/<patientId>.json`.
   - The worker must also audit all portal patients, not just job markers, so pre-existing misses like `03-05-2010-0` are recovered.
   - Job marker reports and patient-index reports must be merged, not substituted for each other.
   - Every report PDF must have report-level active/complete-run checks; patient-level artifacts cannot hide a second unprocessed report.
   - Same-session index reports must remain version-distinct; `logicalName` collisions cannot collapse v1/v2 into one job.
   - Even singleton index reports must use their collision-proof blob key as the local/report identity; completion checks must not fall back to a shared logical name.
   - An active run for one report may not starve other incomplete reports for the same patient.
   - If `$index.json` or job markers are missing/broken, the worker must still audit raw `patients/<patientId>/files/*.pdf` blobs.
   - Source report filters must not reject clinic PDFs just because the filename contains words like `analysis`; only known generated output naming patterns may be skipped.
   - Job marker completion checks must use the versioned blob key as the source filename so a new version cannot be skipped because an older logical/original name completed.
   - The worker must treat missing CLIProxyAPI, missing Netlify auth, failed extraction, failed run, or failed patient-facing generation as failed jobs with actionable status, not as success.
   - Final remote status publish failures must be recorded in the local status file.
   - Dry-run output must separate actual `downloaded` files from `would_download` paths.
   - Batch dry-run must remain an offline audit path and must not require CLIProxy model discovery.
   - Non-dry-run batch execution must take a local exclusive lock so concurrent agent/manual batches cannot contend for CLIProxy or resume each other's runs.
   - Default consolidation must use a chat-stable longform model. GPT-5.4 can remain in the council, but it is no longer the default Stage 4 consolidator after repeated CLIProxy 500/hang behavior.

3. Start path keeps worker alive.
   - Update qEEG-analysis startup to optionally launch the portal pipeline worker after CLIProxyAPI is reachable.
   - Default should be on for local operator runs unless disabled by env.
   - The worker must use a lock so multiple starts do not run duplicate council jobs.
   - Existing local `data/portal_patients/<PATIENT_ID>/` follow-up PDFs must also become durable work.
   - The backend raw portal watcher should detect stable local folder changes, check report-level DB completion by source PDF filename, and spawn `run_portal_council_batch.py --include-label <PATIENT_ID>` when a new or still-incomplete source report is present.
   - The local watcher should persist last-handled patient folder fingerprints so restarts catch post-change follow-ups without turning the first deploy into a mass backfill of every historical incomplete folder.

4. Agent harness guardrails reduce drift.
   - Add a repo-local Codex hook plan/config under `.codex/` that can:
     - remind on session start that portal upload automation and memory updates are mandatory,
     - stop a turn if `plan.md` exists but was not referenced or updated after mutating pipeline work,
     - stop a turn if qEEG-analysis/thrylen memory updates are missing after portal automation changes.
   - Hooks are not relied on for production execution.

5. Skill for repeatability.
   - Add a qEEG portal pipeline skill with correct YAML frontmatter.
   - The skill should route agents to:
     - run the real pipeline, never mock mode,
     - audit portal-vs-local patient state,
     - inspect/repair queue worker status,
     - validate `pipeline/jobs` and `pipeline/status` blobs,
     - update project-local memories.
   - Include deterministic scripts only where they reduce repeated error-prone shell steps.

6. Cathode follow-on stays patient-parameterized.
   - Any requested explainer-video follow-on uses `/Users/davidmontgomery/cathode/projects/<PATIENT_ID>`.
   - The current incident patient is `03-05-2010-0`; no reusable workflow, hook, skill, or memory may hardcode that ID.
   - After a final qEEG Council analysis completes, run the normal Cathode brief/director/plan pipeline for that patient and publish only after the legacy Streamlit QC path passes.

## Verification Gates

- For the immediate incident:
  - Confirm `03-05-2010-0` has local source PDFs.
  - Confirm a real run completes for each source report or record exact failures.
  - Confirm exported/patient-facing outputs and staged council artifacts land in `data/portal_patients/03-05-2010-0/`.
  - Confirm thrylen sync sees the generated outputs.

- For prevention code:
  - qEEG-analysis unit tests for portal worker discovery, missing-local download, report-level completion, same-session versioning, job/index/file fallback merging, active-run starvation prevention, dry-run honesty, status writes, locking, and batch invocation.
  - qEEG-analysis tests for per-run progress logs / tail-friendly progress events so operators can tell long multimodal work from silent failure.
  - thrylen function tests for collision-proof report keys, report-DOB canonical routing, job marker creation on report PDFs, and no marker for non-report uploads.
  - Skill validation with `quick_validate.py`.
  - Hook JSON/schema smoke test.
  - Narrow changed-surface tests plus each repo standard verify command.

## Open Risks

- Full automation depends on local machine services: Netlify CLI auth, CLIProxyAPI auth, and real model availability.
- Netlify functions cannot run the local qEEG Council workflow themselves, so the durable job marker plus local worker is the real reliability boundary.
- Existing worktrees contain many uncommitted changes; all integration must preserve and incorporate them rather than reverting.

## Audit and Orchestration Plan (2026-04-12)

### Summary

- Perform a repo-wide audit and remediation pass across backend workflow, prompt contract, model policy, patient-facing generation, frontend observability, and downstream orchestration with `../thrylen/qeeg` and `../cathode`.
- Preserve the dual-track product: auditable council deliberation remains, with separate patient/family and neuro-team value in separate sections/artifacts rather than forcing everything into a clinician-monograph template.
- Make evidence-first quality the default: hard-fail on factual errors, unsupported claims, contradictory logic, broken orchestration, and missing sourced essentials; treat word counts, rigid heading inventories, and ornamental table rules as soft guidance unless truly required.
- Append this full approved plan verbatim into `plan.md` as a new dated audit/remediation section, preserving the existing portal reliability content.

### Core Remediation Changes

- **Model policy by role**
  - Re-evaluate each role separately: strict extraction/data-pack, Stage 1 narrative analysis, consolidation, final review, and patient-facing rewrite.
  - Use live CLIProxy inventory plus official vendor docs to choose defaults; quality-first by role, Opus excluded by default for cost.
  - Replace stale/default vision assumptions such as `gemini-pro-vision`; prefer the strongest currently available Gemini multimodal model for shared extraction work, with explicit fallback order.
  - Surface role defaults and actual live availability in config and `/api/models`.

- **Prompt and workflow contract cleanup**
  - Remove fixed diagnosis/session assumptions except retained LUMIT framing; diagnosis/session/treatment specifics must come from report data or explicit metadata.
  - Rewrite Stage 1/2/4/5/6 prompts so they stop forcing literature-heavy filler, overblown uncertainty theater, and clinician-only template bloat.
  - Keep deliberation auditable, but focus outputs on verified trends, ratios, uncommon connections, practical caveats, and concise neuro-team gap-filling.
  - Relax Stage 2/5 and downstream repair logic so exact headings, exact word counts, and hardcoded table shapes no longer create false fails when the content is evidence-sound.
  - Add controlled fallback rules for downstream use: prefer Stage 4 consolidation for Cathode input; if Stage 4 is unavailable but Stage 3 is sound, allow Stage 3 fallback.

- **Stage 1/runtime reliability**
  - Audit the full Stage 1/data-pack path for hour-scale regressions: extraction, chunking, shared data-pack work, vision transcript, per-model multimodal notes, retries, and repair loops.
  - Remove hidden duplicate work where possible and make progress accounting truthful.
  - Persist structured progress for extraction/data-pack/transcript/model subtasks so UI progress can reflect real denominators and current subtask state.
  - Treat partial success honestly in run state and UI instead of burying it in logs or job JSON.

### Frontend and Orchestration Changes

- **Observability**
  - Add a patient orchestration/status surface backed by DB run state, `progress.jsonl`, `data/pipeline_jobs/*.json`, portal sync state, published portal artifacts, and Cathode handoff state.
  - Add a sidebar mini-indicator per patient showing latest pipeline/sync/downstream state at a glance.
  - Add a detailed patient-page orchestration panel showing report-level lifecycle: uploaded, extracted, council running, patient-facing generated, synced to `thrylen`, Cathode-ready / Cathode-synced, plus errors, elapsed time, and last update.
  - Upgrade run-page SSE rendering from coarse status to real stage/task/model/chunk progress; use percent only when the denominator is real, otherwise show phase + elapsed + current unit of work.

- **Sync/orchestration gap**
  - Treat the frontend as the control plane for the workflow that currently spans qEEG-analysis, `thrylen`, and Cathode.
  - Add backend endpoints for patient-scoped orchestration state and actions rather than leaving status trapped in scripts/files.
  - Add frontend actions for full orchestration through backend APIs: refresh state, retry portal sync, rerun patient-facing generation, rerun/export council artifacts, and trigger Cathode handoff.

### Public Interfaces

- Add a patient orchestration status shape that includes:
  - sidebar summary state
  - report-level pipeline states
  - current/last run progress with stage/task/model/chunk timing
  - portal sync status and last error
  - patient-facing generation status
  - Cathode handoff status and chosen source artifact
- Add patient-scoped orchestration actions:
  - refresh/recompute orchestration status
  - retry portal sync
  - retry patient-facing generation
  - trigger Cathode handoff using the approved council source artifact
- Extend run/event payloads to include enough structured progress for truthful UI progress bars/phase indicators.

### Validation and Completion Gate

- Add backend tests for:
  - role-based model resolution
  - evidence-first Stage 2/5 acceptance criteria
  - Stage 4 / Stage 3 Cathode fallback selection
  - patient orchestration aggregation from DB + file-backed state
  - real progress accounting across extraction/data-pack/transcript/model subtasks
- Add frontend tests for:
  - sidebar mini status
  - patient-page orchestration panel
  - truthful progress rendering from SSE/status payloads
  - retry/trigger actions and error states
  - stale/failed sync visibility
- Keep existing focused verification for `llm_client`, AI review agents, portal batch runner, and auto patient-facing generation, then extend with orchestration/status coverage.
- Completion is not done until:
  - the full approved plan has been appended into `plan.md`,
  - implementation has been verified on the changed surfaces,
  - and a low-context subagent review has been run at the end.
- **Required final subagent gate**
  - Spawn a reviewer subagent with minimal prior context.
  - Give it the appended `plan.md` audit section plus the recent code changes/diff only.
  - Ask it to report: plan drift, missed requirements, incorrect assumptions, overreach, and anything implemented that does not match the plan.
  - The main implementation pass must either fix the reported drift or explicitly document unresolved drift before closing.

### Assumptions and Defaults

- Dual-track remains the product direction.
- LUMIT framing stays, but hardcoded diagnosis/session assumptions are removed unless sourced.
- Evidence-first gates are the new default.
- Cathode continues consuming council markdown, with fallback preference `Stage 4 consolidation -> best Stage 3 revision`.
- `plan.md` remains the repo’s running plan file; this audit/remediation plan is appended as a new dated section rather than replacing the current portal reliability plan.

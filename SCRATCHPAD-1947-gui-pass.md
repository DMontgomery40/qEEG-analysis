# 1947 Patient GUI Pass Scratchpad

Date: 2026-05-07

Target patient/report:
- DOB seen in extracted report text: 05-13-1947
- Local report path found before GUI pass: `data/reports/45b5e1fb-de65-43bb-a422-622462a41ebd/a4efd45f-8e12-4d6c-babc-32eaa7687031/`

Running notes:
- Started after updating GPT-5.5 model defaults.
- Started GUI run `6963e2b1...` for `New Test Patient` / `LC_dementia_10tx_mid qeeg_Redacted.pdf`.
- Because CLIProxy discovery lacks GPT-5.5, manually selected council models `gpt-5.4`, `claude-sonnet-4-6`, `gemini-3.1-pro-preview`; consolidator `claude-sonnet-4-6`.
- Run entered Stage 1 initial analysis; live progress showed `data_pack_chunk` for `gemini-3.1-pro-preview`, chunk 1/2, 8%.
- Upgraded live CLIProxy on `:8317` from old `6.8.55-0-plus` to Homebrew `6.10.5`; direct `/v1/models` now includes `gpt-5.5`.
- Backed up old local Plus binary to `~/.local/bin/cli-proxy-api-plus.6.8.55-plus.bak` and pointed `~/.local/bin/cli-proxy-api-plus` / `cliproxyapi` at `/opt/homebrew/opt/cliproxyapi/bin/cliproxyapi` so future `start.sh` launches keep GPT-5.5.
- Follow-up defaults changed for future runs: GPT-5.5 now sends low reasoning by default; the model picker hides OpenAI models before GPT-5.5 and Gemini models before Gemini 3.
- Live backend still shows the previous picker/defaults until restart because the active `6963e2b1...` run is still executing inside that backend process. Restarting now would interrupt the run.

Bugs / Annoyances:
- Port collision / wrong-app trap: `http://127.0.0.1:4120/` is currently the NBA Signal Console, not qEEG. The qEEG frontend started on `http://127.0.0.1:5173/`. This is easy to miss because both are local operator consoles.
- Live qEEG backend was stale after code edits: `/api/models` still reported old defaults (`gpt-5.4`, `claude-sonnet-4-6`) until backend restart.
- Patient identity mismatch: the 05-13-1947 report is attached to a patient labeled `New Test Patient` (`45b5e1fb...`). The UI/sidebar gives no DOB clue, so "the 1947 patient" is not discoverable from the GUI without filesystem/API spelunking.
- Initial qEEG page rendered `No patients` while backend data existed; a clean reload after service checks populated the sidebar. It looked like a dead/empty database at first glance.
- Model availability/default mismatch during first pass: `/api/models` reported configured `gpt-5.5-xhigh` and default consolidator `gpt-5.5`, but old CLIProxy discovery did not include any `gpt-5.5` id. Fixed by upgrading CLIProxy and later changing GPT-5.5 default reasoning to low.
- New Run default is surprising/risky: consolidator dropdown selected `claude-opus-4-6` by default even though portal/batch rules exclude Opus unless explicitly allowed, and Opus is not in `COUNCIL_MODELS`.
- Disabled CTA lacks local explanation: `Create + Start` was disabled even with a report selected; the page did not say which required control was missing.
- Wrong vision default: Stage 1 vision transcript/checker was still using `gemini-3.1-pro-preview`; operator expected Gemini 3.1 Flash-family. The active run then hit `No capacity available for model gemini-3.1-pro-preview` on transcript chunk 5/7.
- Explainer target confusion: qEEG docs still mention the older `../local-explainer-video` path in places, but the current handoff default is `../cathode/projects`. Missing piece was automatic post-run launch into Cathode.
- Cathode queue was stale: `scripts/qeeg_patient_video_queue.py` still described/generated an all-Qwen static image queue and called the Anthropic API for storyboarding. Updated going-forward behavior to use Cathode, `claude -p` Sonnet 4.6 story writing, GPT-image-2 images through the Codex image lane, ffmpeg assembly, 6.5 minute default, and no generated videos/Remotion/overlays.
- qEEG post-run automation now prepares the Cathode handoff and spawns Cathode's queue as a detached job after patient-facing generation. Existing live backend must be restarted before future GUI runs get this new automatic Cathode hook.
- Manual 1947 handoff created at `../cathode/projects/05-13-1947-0/` from run `6963e2b1...` Stage 4 consolidation. Cathode queue is running in tmux session `cathode-qeeg-1947`; status/log files are `../cathode/projects/05-13-1947-0/qeeg_video_queue_status.json` and `../cathode/projects/05-13-1947-0/qeeg_video_queue.log`.
- Added Cathode queue post-render sync: successful MP4s copy to `data/portal_patients/<label>/<label>.mp4`, write `<label>__cathode_video.json`, then call `python -m backend.portal_sync --patient-label <label>` so Thrylen receives the video.
- Restarted qEEG backend/frontend after the Cathode automation patch. New backend `/api/models` reports configured models `gpt-5.5`, `claude-sonnet-4-6`, `gemini-3.1-flash-lite-preview`; default consolidator `gpt-5.5`; vision checker `gemini-3.1-flash-lite-preview`.
- Started 1980 patient run `c8148119-048e-4ebe-9bbf-44f4d3c94d65` for `10-27-1980-0` with council models `gpt-5.5`, `claude-sonnet-4-6`, `gemini-3.1-flash-lite-preview` and consolidator `gpt-5.5`. It reached Stage 1 vision transcript chunks.
- 1947 qEEG run `6963e2b1...` was interrupted by backend restart after Stage 5 completed and Stage 6 had started. It remains `running` in storage, but no in-memory task is attached after restart. Cathode is using the completed Stage 4 artifact, not waiting for Stage 6.
- 1947 Cathode completed successfully and synced to Thrylen; Thrylen portal shows `05-13-1947-0.mp4` with watch/download controls.
- Split-brain sync bug: Cathode/portal sync wrote the MP4 to `data/portal_patients/05-13-1947-0/`, but the local qEEG admin UI still showed no patient files because there was no `patient_files` DB row and no copy under `data/patient_files/...`.
- Added `scripts/register_patient_file.py` and wired Cathode queue completion to register rendered MP4s back into qEEG patient files after portal copy and before Thrylen sync. Backfilled 1947 as `patient_files.id=19f9ff20-71c2-4cec-a9bf-971886058ec2`, `filename=05-13-1947-0.mp4`, size `100623587`.
- 1980 run `c8148119...` completed but had no selected artifact, so export was not available. Selected the GPT-5.5 Stage 6 final draft and exported `10-27-1980-0.md` / `10-27-1980-0.pdf` into `data/portal_patients/10-27-1980-0/`.
- 1980 council internals manually staged under `data/portal_patients/10-27-1980-0/council/c8148119-048e-4ebe-9bbf-44f4d3c94d65/`. The broad staging script currently tries all patients and can stall behind unrelated portal sync work; it needs a narrow `--patient-label` option.
- Patient-facing generation for 1980 hung silently for several minutes in the Sonnet rewrite call and was stopped. That script needs visible progress/timeout handling so the GUI button does not feel dead.
- Started Cathode queue for 1980 in tmux session `cathode-qeeg-1980`; status file is `../cathode/projects/10-27-1980-0/qeeg_video_queue_status.json`, log is `../cathode/projects/10-27-1980-0/qeeg_video_queue.log`.
- Duplicate backend restart annoyance remains: PID `54620` is the actual listener on `:8000`; repeated restart attempts log `[Errno 48] address already in use` and exit, which is noisy but not a second live backend.
- User correctly pointed out the real portal failure: 1947 already had the video, but no patient-facing PDF; my earlier "fixed" status only solved local MP4 registration, not the missing Patient Summary deliverable.
- GUI verification on 5/7: local qEEG selected 1980 showed `Patient-facing regeneration requested` but still `Patient-facing: missing` / `Patient PDF: pending` until an actual `__patient-facing__...pdf` existed. Button-click success toast was not proof of completion.
- Thrylen portal contract: `public/qeeg/file-logic.js` only promotes a PDF to the Patient Summary hero when the PDF filename/logical name includes `patient-facing`. A generic `<patient>.pdf` export syncs but appears as archive/report material, not the hero summary.
- 1947 run status remains stale because it was interrupted after Stage 5 and before Stage 6 completion. GUI disables `Regenerate patient-facing` and `Export council artifacts`, leaving no GUI path to generate a Patient Summary from the completed Stage 4 consolidation. This is a product bug: stale-but-source-available runs need a rescue action.
- Stage 5 review for 1947 was split: Claude approved; GPT-5.4 voted REVISE with evidence-grounded corrections around CZ theta/beta power, PZ coherence availability, Session 1 P300 topography wording, and C-P latency spread. Those issues are at least partly real; the rescue patient summary avoided the disputed mechanistic language.
- The model-backed patient-facing generation path is unreliable operationally: GUI-triggered 1980 Sonnet job hung silently and later failed with `patient_facing_generation_failed`; direct 1947 GPT-5.5 generation hung from the operator perspective but did eventually write files before wrapper termination. Need visible progress, timeout, stderr surfaced in UI, and no silent background "requested" state.
- Created portal-recognized patient-facing PDFs for both active patients:
  - `data/portal_patients/05-13-1947-0/05-13-1947-0__patient-facing__manual-rescue__2026-05-07.pdf`
  - `data/portal_patients/10-27-1980-0/10-27-1980-0__patient-facing__manual-rescue__2026-05-07.pdf`
- Registered those PDFs into local qEEG patient files. GUI verified:
  - 1980 shows `Latest patient-facing PDF: 10-27-1980-0__patient-facing__manual-rescue__2026-05-07.pdf`, `Patient-facing: Ready`, report lifecycle `Patient PDF: ready`, and the PDF appears under Patient Files.
  - 1947 shows `Latest patient-facing PDF: 05-13-1947-0__patient-facing__manual-rescue__2026-05-07.pdf`, `Patient-facing: Ready`, and the PDF plus MP4 appear under Patient Files; however report lifecycle still says `Patient PDF: pending` because that row only marks ready when a complete run exists for the report. This is stale-run lifecycle logic, not missing files.

## 2026-05-07 1980 patient-facing correction
- Manual rescue patient-facing PDF for `10-27-1980-0` was wrong as a final answer. It has been removed from the portal folder and removed from `patient_files` DB tracking; archived outside `data/portal_patients` at `data/_manual_rescue_archive/10-27-1980-0/`.
- GUI-triggered `Regenerate patient-facing` did run the proper backend subprocess with `--version auto-c8148119`, but `claude-sonnet-4-6` returned empty text content through CLIProxyAPI after the real generator selected 3 Stage 6 source artifacts.
- Proper generator succeeded with `gpt-5.5`: `data/portal_patients/10-27-1980-0/10-27-1980-0__patient-facing__auto-c8148119__2026-05-07.pdf` plus `.md` and `__meta.json`.
- qEEG GUI verified patient-facing card now says Ready and names the auto-c8148119 PDF.
- Netlify blob list verified remote auto-c8148119 patient-facing PDF/MD/meta are present. Manual-rescue blobs and index entries were explicitly removed after an accidental archive-in-folder sync.
- Bug: auto patient-facing path swallowed useful stderr in the GUI; subprocess failure was only visible by rerunning foreground. Needs UI-visible failure log/operator hint.
- Bug/annoyance: portal sync mirrors nested archive folders under patient directories; archives must live outside `data/portal_patients` or they get uploaded.

## 2026-05-07 1980 Cathode scene correction
- User caught that the `What Is the P300?` scene still showed a female figure for a male patient. My first backend-side regenerate targeted the wrong internal scene index: GUI "Scene 7" is internal `scene_006`, not `scene_007`.
- Corrected `../cathode/projects/10-27-1980-0/plan.json` scene id `6` with a no-human chart-only visual prompt and regenerated only `../cathode/projects/10-27-1980-0/images/scene_006.png`.
- GUI verified at `http://127.0.0.1:9322/projects/10-27-1980-0/scenes`: selected `#6 What Is the P300?`; Visual Stage now shows a chart-only P300 waveform with no person.
- Reassembled through Cathode's ffmpeg static-image path only, not Remotion/browser rendering. New source video is `../cathode/projects/10-27-1980-0/10-27-1980-0.mp4`.
- Replaced qEEG portal copy at `data/portal_patients/10-27-1980-0/10-27-1980-0.mp4`; ffprobe shows duration `339.473307` seconds and size `80048074` bytes.
- Registered the refreshed MP4 into qEEG `patient_files` for patient label `10-27-1980-0` as id `3b6ed81a-22d8-42ee-9001-e266eb0e238d`, then synced the patient folder to Thrylen.
- Netlify blob index showed latest remote MP4 key `10-27-1980-0__10-27-1980-0__v3__2026-05-07.mp4` with size `80048074`.
- Pruned old remote MP4 blobs/index entries `v1` and `v2` for `10-27-1980-0` so portal downloads only expose the refreshed `v3` video.
- Bug/annoyance: Cathode scene controls use 1-based "Scene 7" while stored assets are zero-based `scene_006`, which makes surgical regenerates dangerously easy to aim at the wrong image unless the GUI selected title is cross-checked.

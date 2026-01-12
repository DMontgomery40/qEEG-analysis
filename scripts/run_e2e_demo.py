#!/usr/bin/env python
"""
Run a REAL end-to-end pipeline with mock LLM and persist artifacts.

This creates actual files in data/artifacts/ that you can inspect.
"""
import asyncio
import os
import sys
from pathlib import Path

# Set mock mode BEFORE importing backend modules
os.environ["QEEG_MOCK_LLM"] = "1"

async def main():
    from backend.config import ensure_data_dirs, ARTIFACTS_DIR
    from backend.storage import (
        init_db, session_scope,
        list_patients, list_reports,
        create_run, get_run, list_artifacts
    )
    from backend.council import QEEGCouncilWorkflow
    from backend.tests.fixtures.mock_llm import create_mock_transport
    from backend.llm_client import AsyncOpenAICompatClient

    # Initialize
    ensure_data_dirs()
    init_db()

    print(f"\n{'='*60}")
    print(f"REAL END-TO-END PIPELINE TEST")
    print(f"{'='*60}")

    # Find a patient with reports
    with session_scope() as session:
        patients = list_patients(session)
        if not patients:
            print("ERROR: No patients in database. Create one first.")
            sys.exit(1)

        # Find patient with reports
        patient = None
        report = None
        for p in patients:
            reports = list_reports(session, p.id)
            if reports:
                patient = p
                report = reports[0]
                break

        if not patient or not report:
            print("ERROR: No patients with reports found.")
            print("Available patients:")
            for p in patients:
                reps = list_reports(session, p.id)
                print(f"  - {p.label} ({p.id[:8]}): {len(reps)} reports")
            sys.exit(1)

        patient_id = patient.id
        report_id = report.id
        print(f"\nUsing patient: {patient.label} ({patient_id[:8]})")
        print(f"Using report: {report.filename} ({report_id[:8]})")

    # Create mock LLM client
    transport = create_mock_transport()
    llm = AsyncOpenAICompatClient(
        base_url="http://mock-cliproxy",
        api_key="",
        timeout_s=120.0,
        transport=transport,
    )

    # Create run
    with session_scope() as session:
        run = create_run(
            session,
            patient_id=patient_id,
            report_id=report_id,
            council_model_ids=["mock-council-a", "mock-council-b"],
            consolidator_model_id="mock-consolidator",
        )
        run_id = run.id
    print(f"\n✓ Created run: {run_id}")

    # Run the pipeline!
    print(f"\n{'─'*60}")
    print("Running 6-stage pipeline...")
    print(f"{'─'*60}")

    workflow = QEEGCouncilWorkflow(llm=llm)

    async def on_event(payload):
        stage = payload.get("stage_num", "")
        status = payload.get("status", "")
        if stage and status:
            print(f"  Stage {stage}: {status}")

    await workflow.run_pipeline(run_id, on_event=on_event)

    # Check results
    with session_scope() as session:
        run = get_run(session, run_id)
        artifacts = list_artifacts(session, run_id)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Run status: {run.status}")
    if run.error_message:
        print(f"Error: {run.error_message}")

    print(f"\nArtifacts created: {len(artifacts)}")

    artifact_dir = ARTIFACTS_DIR / run_id
    print(f"\n📁 Artifacts directory: {artifact_dir}")

    for art in sorted(artifacts, key=lambda a: (a.stage_num, a.model_id)):
        content_path = Path(art.content_path)
        size = content_path.stat().st_size if content_path.exists() else 0
        print(f"  Stage {art.stage_num} ({art.stage_name}) - {art.model_id}")
        print(f"    └── {content_path.name} ({size:,} bytes)")

    # Show where to find the final report
    stage6 = [a for a in artifacts if a.stage_num == 6]
    if stage6:
        print(f"\n{'='*60}")
        print("📄 FINAL REPORT LOCATIONS:")
        print(f"{'='*60}")
        for art in stage6:
            print(f"  {art.content_path}")

        # Show preview of final report
        final_path = Path(stage6[0].content_path)
        if final_path.exists():
            content = final_path.read_text()
            print(f"\n--- First 1000 chars of final report ---")
            print(content[:1000])
            print("\n...")

    print(f"\n✅ DONE! Check the artifacts at:")
    print(f"   {artifact_dir}")

    await llm.aclose()

if __name__ == "__main__":
    asyncio.run(main())

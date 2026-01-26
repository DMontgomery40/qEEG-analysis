#!/usr/bin/env python3
"""
Stage council artifacts for portal sync.

Copies key council artifacts from data/artifacts/<run_id>/ to
data/portal_patients/<patient_label>/council/<run_id>/ so they
can be synced to the thrylen portal.

Artifacts synced:
- stage-1/_data_pack.json (structured facts)
- stage-1/_vision_transcript.md (OCR output)
- stage-4/*.md (consolidation - narrative truth)
- stage-6/*.md (final drafts)

Usage:
    uv run python scripts/stage_council_artifacts.py
    uv run python scripts/stage_council_artifacts.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

from backend.storage import init_db, session_scope, Patient, Run


def get_completed_runs_by_patient_label() -> dict[str, list[tuple[str, str]]]:
    """
    Returns a dict mapping patient_label -> [(run_id, report_filename), ...]
    Only includes completed runs.
    """
    result: dict[str, list[tuple[str, str]]] = {}
    
    with session_scope() as session:
        # Get all patients with valid labels (MM-DD-YYYY-N format)
        patients = session.query(Patient).all()
        
        for patient in patients:
            label = patient.label
            # Skip invalid labels
            if not label or not _is_valid_patient_label(label):
                continue
            
            # Get completed runs for this patient
            completed_runs = [
                r for r in patient.runs 
                if r.status == "complete"
            ]
            
            if completed_runs:
                result[label] = [
                    (r.id, r.report.filename if r.report else "unknown")
                    for r in completed_runs
                ]
    
    return result


def _is_valid_patient_label(label: str) -> bool:
    """Check if label matches MM-DD-YYYY-N format."""
    import re
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})-(\d+)$", label)
    if not m:
        return False
    month, day, year, idx = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100 and idx >= 0


def stage_artifacts(dry_run: bool = False) -> None:
    """Copy council artifacts to portal_patients folders."""
    
    data_dir = Path(__file__).parent.parent / "data"
    artifacts_dir = data_dir / "artifacts"
    portal_dir = data_dir / "portal_patients"
    
    if not artifacts_dir.exists():
        print(f"Artifacts directory not found: {artifacts_dir}")
        return
    
    if not portal_dir.exists():
        print(f"Portal patients directory not found: {portal_dir}")
        return
    
    init_db()
    runs_by_label = get_completed_runs_by_patient_label()
    
    total_copied = 0
    total_skipped = 0
    
    for patient_label, runs in runs_by_label.items():
        patient_portal_dir = portal_dir / patient_label
        if not patient_portal_dir.exists():
            print(f"  Skipping {patient_label}: no portal folder")
            continue
        
        for run_id, report_filename in runs:
            run_artifacts_dir = artifacts_dir / run_id
            if not run_artifacts_dir.exists():
                continue
            
            # Destination: portal_patients/<label>/council/<run_id>/
            dest_dir = patient_portal_dir / "council" / run_id
            
            # Files to copy
            files_to_copy = []
            
            # Stage 1: _data_pack.json and _vision_transcript.md
            stage1_dir = run_artifacts_dir / "stage-1"
            if stage1_dir.exists():
                data_pack = stage1_dir / "_data_pack.json"
                if data_pack.exists():
                    files_to_copy.append((data_pack, dest_dir / "stage-1" / "_data_pack.json"))
                
                vision_transcript = stage1_dir / "_vision_transcript.md"
                if vision_transcript.exists():
                    files_to_copy.append((vision_transcript, dest_dir / "stage-1" / "_vision_transcript.md"))
            
            # Stage 4: consolidation (all .md files)
            stage4_dir = run_artifacts_dir / "stage-4"
            if stage4_dir.exists():
                for md_file in stage4_dir.glob("*.md"):
                    files_to_copy.append((md_file, dest_dir / "stage-4" / md_file.name))
            
            # Stage 6: final drafts (all .md files)
            stage6_dir = run_artifacts_dir / "stage-6"
            if stage6_dir.exists():
                for md_file in stage6_dir.glob("*.md"):
                    files_to_copy.append((md_file, dest_dir / "stage-6" / md_file.name))
            
            # Copy files
            for src, dst in files_to_copy:
                if dst.exists():
                    # Check if source is newer
                    if src.stat().st_mtime <= dst.stat().st_mtime:
                        total_skipped += 1
                        continue
                
                rel_dst = dst.relative_to(portal_dir)
                if dry_run:
                    print(f"  [dry-run] Copy {src.name} -> {rel_dst}")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    print(f"  Copy {src.name} -> {rel_dst}")
                total_copied += 1
    
    action = "Would copy" if dry_run else "Copied"
    print(f"\n{action} {total_copied} files, skipped {total_skipped} unchanged")


def main():
    parser = argparse.ArgumentParser(description="Stage council artifacts for portal sync")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying")
    args = parser.parse_args()
    
    print("Staging council artifacts for portal sync...")
    stage_artifacts(dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""
REAL validation: For each fact in data_pack, verify the value appears correctly
in the report in an appropriate context.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.storage import (
    find_patients_by_label,
    init_db,
    list_artifacts,
    list_runs,
    session_scope,
)

PORTAL_DIR = _REPO_ROOT / "data" / "portal_patients"


def get_data_pack_for_label(label: str) -> dict | None:
    """Get the data_pack.json for a patient label."""
    with session_scope() as session:
        patients = find_patients_by_label(session, label)
        if not patients:
            return None
        
        for p in patients:
            for r in list_runs(session, p.id):
                if (r.status or "") == "complete":
                    arts = list_artifacts(session, r.id)
                    for a in arts:
                        if a.kind == "data_pack" and a.stage_num == 1:
                            try:
                                return json.loads(Path(a.content_path).read_text())
                            except Exception:
                                pass
    return None


def find_value_in_tables(report: str, expected: float, metric: str, session: int) -> dict:
    """
    Find a value in markdown tables and verify it's in the right column (session).
    
    Returns dict with:
    - found: bool - whether we found the value
    - correct_position: bool - whether it's in the right session column
    - context: str - surrounding text
    """
    result = {"found": False, "correct_position": False, "context": ""}
    
    val_str = str(expected)
    if expected == int(expected):
        val_str = str(int(expected))
    
    # Find all table rows
    for line in report.split('\n'):
        if '|' not in line:
            continue
        if line.strip().startswith('|--'):
            continue
            
        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]
        
        if len(parts) < 4:
            continue
            
        row_label = parts[0].lower()
        
        # Check if this row could be for our metric
        metric_matches = False
        if 'reaction' in metric and 'reaction' in row_label:
            metric_matches = True
        elif 'trail_making_test_a' in metric and ('tmt-a' in row_label or 'simple' in row_label):
            metric_matches = True
        elif 'trail_making_test_b' in metric and ('tmt-b' in row_label or 'complex' in row_label or 'switching' in row_label):
            metric_matches = True
        elif 'p300_delay' in metric and ('timing' in row_label or 'latency' in row_label or 'delay' in row_label):
            metric_matches = True
        elif 'p300_voltage' in metric and ('strength' in row_label or 'amplitude' in row_label or 'voltage' in row_label):
            metric_matches = True
        elif 'theta_beta' in metric and ('theta' in row_label and 'beta' in row_label):
            metric_matches = True
        elif 'f3_f4_alpha' in metric and ('f3' in row_label and 'f4' in row_label):
            metric_matches = True
        elif 'frontal_peak' in metric and 'frontal' in row_label and ('freq' in row_label or 'peak' in row_label or 'rhythm' in row_label):
            metric_matches = True
        elif 'central_parietal_peak' in metric and 'central' in row_label and 'parietal' in row_label and '0.' not in parts[1]:
            # Exclude coherence rows (which have values like 0.95)
            metric_matches = True
        elif 'occipital_peak' in metric and 'occipital' in row_label and ('freq' in row_label or 'peak' in row_label or 'rhythm' in row_label or 'head' in row_label):
            metric_matches = True
            
        if not metric_matches:
            continue
        
        # Check if expected value is in the correct session column
        if session <= len(parts) - 1:
            cell = parts[session]
            
            # Extract number from cell
            match = re.search(r'(\d+\.?\d*)', cell)
            if match:
                found_val = float(match.group(1))
                
                if abs(found_val - expected) < 0.01:
                    result["found"] = True
                    result["correct_position"] = True
                    result["context"] = line.strip()[:100]
                    return result
                else:
                    # Found the row but value doesn't match
                    result["found"] = True
                    result["found_value"] = found_val
                    result["context"] = line.strip()[:100]
    
    # Also check if value appears anywhere (as a sanity check)
    if val_str in report:
        result["exists_somewhere"] = True
    
    return result


def validate_patient(label: str) -> dict:
    """Validate a patient's report against their data_pack."""
    result = {
        "label": label,
        "status": "unknown",
        "errors": [],
        "verified": 0,
        "total": 0,
        "not_found": [],
    }
    
    # Find the patient-facing markdown
    patient_dir = PORTAL_DIR / label
    md_files = list(patient_dir.glob("*__patient-facing__*.md"))
    if not md_files:
        result["status"] = "no_report"
        return result
    
    md_file = sorted(md_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    report = md_file.read_text()
    
    # Get data pack
    data_pack = get_data_pack_for_label(label)
    if not data_pack:
        result["status"] = "no_data_pack"
        return result
    
    # Validate each fact
    for fact in data_pack.get("facts", []):
        metric = fact.get("metric")
        value = fact.get("value")
        session = fact.get("session_index")
        
        if not metric or value is None or not session:
            continue
            
        result["total"] += 1
        
        check = find_value_in_tables(report, value, metric, session)
        
        if check.get("correct_position"):
            result["verified"] += 1
        elif check.get("found"):
            # Found the row but wrong value
            result["errors"].append({
                "metric": metric,
                "session": session,
                "expected": value,
                "found": check.get("found_value"),
                "context": check.get("context"),
            })
        else:
            result["not_found"].append(f"{metric} s{session}={value}")
    
    if result["errors"]:
        result["status"] = "errors"
    elif len(result["not_found"]) > result["total"] * 0.5:
        result["status"] = "many_not_found"
    else:
        result["status"] = "ok"
    
    return result


def main():
    init_db()
    
    labels = []
    for p in sorted(PORTAL_DIR.iterdir()):
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith("."):
            labels.append(p.name)
    
    print(f"Validating {len(labels)} patients - checking values are in correct session columns...\n")
    
    all_ok = True
    total_verified = 0
    total_facts = 0
    total_errors = 0
    
    for label in labels:
        result = validate_patient(label)
        total_verified += result.get("verified", 0)
        total_facts += result.get("total", 0)
        
        if result["status"] == "ok":
            pct = 100 * result["verified"] / result["total"] if result["total"] else 0
            print(f"✅ {label}: {result['verified']}/{result['total']} ({pct:.0f}%) values in correct positions")
        elif result["status"] == "many_not_found":
            print(f"⚠️  {label}: Only {result['verified']}/{result['total']} found in tables")
        elif result["status"] == "no_report":
            print(f"⏭️  {label}: No report")
        elif result["status"] == "no_data_pack":
            print(f"⚠️  {label}: No data_pack")
        elif result["status"] == "errors":
            all_ok = False
            total_errors += len(result["errors"])
            print(f"❌ {label}: {len(result['errors'])} WRONG VALUES:")
            for err in result["errors"]:
                print(f"      {err['metric']} session {err['session']}: expected {err['expected']}, found {err.get('found')}")
                if err.get("context"):
                    print(f"        in: \"{err['context'][:60]}...\"")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_verified}/{total_facts} values verified in correct positions")
    if total_errors:
        print(f"⚠️  {total_errors} values in WRONG positions (potential hallucinations or swapped values)")
    else:
        print("✅ No incorrect values detected")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

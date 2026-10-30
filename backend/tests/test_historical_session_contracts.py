"""REG-01/02/05 and reviewer SD pairing; synthetic data, actual engine authority.

These tests establish extraction/admission/filing lineage, not real-model report
quality or actual PDF/video inclusion. See the workbench coverage catalogue.
"""
from __future__ import annotations

import hashlib
import importlib

import pytest
from sqlalchemy import select

from backend import storage
from backend.tests.clinic_test_helpers import forbid_clinic_paid  # noqa: F401
from backend.tests.test_analysis_inputs import admission, source  # noqa: F401


@pytest.mark.parametrize("sd_form", ["({})", "(+{})", "(±{})"])
@pytest.mark.parametrize("units", ["all", "none", "alternating"])
@pytest.mark.parametrize("reverse", [False, True])
def test_REVIEW_01_different_sd_stays_with_mean_and_global_session(sd_form, units, reverse):
    from backend.council.report_text import _facts_from_report_text_summary

    sessions = [(1, 281, 19), (2, 317, 71), (3, 263, 23), (4, 299, 53)]
    # Two separate reports repeat local Session 1/2; later page order reverses.
    pages = [sessions[:2], sessions[2:]]
    if reverse:
        pages.reverse()
    text = []
    for page_number, page_sessions in enumerate(pages, 1):
        text.append(f"=== PAGE {page_number} / 2 ===")
        for local, (global_index, _, _) in enumerate(page_sessions, 1):
            text.append(f"[[QEEG_SESSION_ALIAS local={local} global={global_index}]]")
        cells = [str(mean) + " " + sd_form.format(sd) +
                 (" ms" if units == "all" or (units == "alternating" and i % 2) else "")
                 for i, (_, mean, sd) in enumerate(page_sessions)]
        text.append("Physical Reaction Time " + " ".join(cells) + " 252–362 ms")
    facts = _facts_from_report_text_summary("\n".join(text), expected_sessions=[1, 2, 3, 4])
    selected = [f for f in facts if f.get("metric") == "physical_reaction_time"]
    assert len(selected) == 4
    assert {f["session_index"]: (f["value"], f["sd_plus_minus"]) for f in selected} == {
        index: (mean, sd) for index, mean, sd in sessions
    }
    assert all(f["unit"] == "ms" and f["target_range"] == "252-362 ms" for f in selected)
    assert {f["local_session_index"] for f in selected} == {1, 2}


@pytest.mark.parametrize("metric,label,values,unit,target", [
    ("audio_p300_delay", "Audio P300 Delay", [351, None, 289], "ms", "265-344"),
    ("audio_p300_voltage", "Audio P300 Voltage", [4.2, None, 9.3], "uV", "7-18"),
])
def test_REG_05_missing_middle_session_is_not_shifted_or_filled_from_reference(metric, label, values, unit, target):
    from backend.council.report_text import _facts_from_report_text_summary

    text = "=== PAGE 1 / 1 ===\nSession 1 Session 2 Session 3\n" + label + " " + " ".join(
        "N/A" if value is None else f"{value} {unit}" for value in values
    ) + f" {target} {unit}\n"
    facts = _facts_from_report_text_summary(text, expected_sessions=[1, 2, 3])
    metric_facts = [f for f in facts if f.get("metric") == metric]
    assert len({f["session_index"] for f in metric_facts}) == len(metric_facts)
    selected = {f["session_index"]: f["value"] for f in metric_facts}
    # Missing evidence may be represented explicitly or omitted; known later
    # evidence must retain its session and ranges must never become measurements.
    assert selected.get(1) == values[0]
    assert selected.get(2) is None
    assert selected.get(3) == values[2]
    expected_target = f"{target} {'ms' if unit == 'ms' else 'µV'}"
    assert all(f["target_range"] == expected_target for f in metric_facts)


def test_REG_05_two_three_four_sessions_change_source_revision_and_keep_exact_inputs(admission):
    client, payload, root, _ = admission
    reports = [source(root, payload["patient_id"], date=f"2026-0{i}-02", pages=2, value=280+i)
               for i in range(1, 5)]
    runs = []
    for count in (2, 3, 4):
        ids = [r.id for r in reports[:count]]
        request = {**payload, "report_ids": ids, "operation_id": f"history-{count}",
                   "special_instructions": "Compare the full sequence; preserve all session evidence."}
        response = client.post("/api/runs", json=request)
        assert response.status_code == 200, response.text
        run = response.json()
        assert run["source_report_ids"] == ids
        assert [s["report_id"] for s in run["source_manifest"]["sources"]] == ids
        assert len(run["source_manifest"]["page_map"]) == 2 * count
        assert client.post("/api/runs", json=request).json()["id"] == run["id"]
        runs.append(run)
    assert len({run["analysis_input_fingerprint"] for run in runs}) == 3
    assert len({run["id"] for run in runs}) == 3
    with storage.session_scope() as session:
        assert len(list(session.scalars(select(storage.Run)))) == 3


def test_REG_01_REG_02_same_dob_same_initials_and_filenames_preserve_receipts_after_authority_reload(temp_data_dir):
    module = importlib.import_module("backend.clinic_intake")
    uploads = []
    for key, first_name, content in (("a", "SyntheticAda", b"A exact bytes"),
                                     ("b", "SyntheticAnne", b"B exact bytes")):
        kwargs = dict(key=key, identity={"firstName": first_name, "lastName": "Fixture", "birthdate": "01-01-1900"},
                      resolution={"forceNew": True}, files=[("same.txt", content, "text/plain")],
                      file_meta=[{"documentKind": "attachment"}], actor="synthetic-test")
        receipt = module.submit_upload(**kwargs)["upload"]
        # Reload the authority, re-read original receipt, and retry the same admission.
        module = importlib.reload(module)
        assert module.get_upload(receipt["uploadId"])["upload"] == receipt
        assert module.submit_upload(**kwargs)["upload"] == receipt
        assert receipt["items"][0]["sha256"] == hashlib.sha256(content).hexdigest()
        uploads.append(receipt)
    assert uploads[0]["patientId"] != uploads[1]["patientId"]
    assert uploads[0]["items"][0]["sourceId"] != uploads[1]["items"][0]["sourceId"]

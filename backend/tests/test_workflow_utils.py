import json

from backend import council


def test_artifact_path_is_deterministic_and_sanitized():
    p = council._artifact_path("run123", 2, "a/b:c", ".json")
    assert str(p).endswith("data/artifacts/run123/stage-2/a_b_c.json")


def test_stage5_validation_accepts_required_shape():
    payload = {
        "vote": "REVISE",
        "required_changes": ["x"],
        "optional_changes": [],
        "quality_score_1to10": 7,
    }
    council._validate_stage5(payload)


def test_json_loads_loose_extracts_object():
    payload = council._json_loads_loose("noise\n\n{ \"a\": 1 }\n\nmore")
    assert payload == {"a": 1}


from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
EXPECTED_AUDIT_HEADINGS = [
    "# Dataset and Sessions",
    "# Key Empirical Findings",
    "# Performance Assessments",
    "# Auditory ERP: P300 and N100",
    "# Background EEG Metrics",
    "# Speculative Commentary and Interpretive Hypotheses",
]


def _prompt_text(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def test_stage1_prompt_removes_hardcoded_diagnosis_and_session_assumptions():
    text = _prompt_text("stage1_analysis.md").lower()

    assert "mild cognitive impairment at session one" not in text
    assert "condition: mci" not in text
    assert "20 treatments spread evenly across all sessions" not in text
    assert "after 10 lumit sessions" not in text
    assert "do not assume diagnosis" in text
    assert "if the report or supplied metadata explicitly identifies lumit" in text
    assert "do not pad with generic literature review" in text


def test_review_prompts_are_evidence_first_not_word_count_first():
    stage2 = _prompt_text("stage2_peer_review.md").lower()
    stage5 = _prompt_text("stage5_final_review.md").lower()

    for text in (stage2, stage5):
        assert "arbitrary word count" in text

    assert "patient/family explanation" in stage2
    assert "patient or family explanation" in stage5
    assert "2500 word target" not in stage2
    assert "under 2000 words" not in stage2
    assert "word count is under 2500" not in stage5
    assert "missing structure/tables should always be" not in stage5


def test_stage3_to_stage6_keep_audit_headings_without_padding_language():
    for name in (
        "stage3_revision.md",
        "stage4_consolidation.md",
        "stage6_final_draft.md",
    ):
        text = _prompt_text(name)
        lowered = text.lower()
        for heading in EXPECTED_AUDIT_HEADINGS:
            assert heading in text
        assert "do not pad for word count" in lowered

    assert "at least 2500 words" not in _prompt_text("stage3_revision.md").lower()
    assert "3000-4000 words" not in _prompt_text("stage4_consolidation.md").lower()
    assert (
        "publication-ready clinical document"
        not in _prompt_text("stage6_final_draft.md").lower()
    )

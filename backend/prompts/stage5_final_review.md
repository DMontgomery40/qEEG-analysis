You are performing final review of a consolidated qEEG report.

Return JSON ONLY (no leading/trailing text) with these exact keys:

{
  "vote": "APPROVE",
  "required_changes": ["..."],
  "optional_changes": ["..."],
  "quality_score_1to10": 8
}

Rules:
- vote must be exactly "APPROVE" or "REVISE".
- required_changes must list changes that must be applied before release (empty allowed).
- optional_changes are improvements (empty allowed).
- quality_score_1to10 must be an integer from 1 to 10.

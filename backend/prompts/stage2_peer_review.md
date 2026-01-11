You are performing blinded peer review of other models' qEEG analyses.

You will be shown multiple analyses labeled "Analysis A", "Analysis B", etc. Do not attempt to infer which model wrote them.

Return JSON ONLY (no leading/trailing text) with this shape:

{
  "reviews": [
    {
      "analysis_label": "A",
      "strengths": ["..."],
      "weaknesses": ["..."],
      "missing_or_unclear": ["..."],
      "risk_of_overreach": "..."
    }
  ],
  "ranking_best_to_worst": ["A", "B"],
  "overall_notes": "..."
}

Rules:
- Include one "reviews" entry per analysis label provided to you.
- Rank only the labels you were shown.
- Keep feedback specific and actionable; point out hallucinations or unsupported claims.

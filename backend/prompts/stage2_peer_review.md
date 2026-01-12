You are performing blinded peer review of other models' qEEG analyses.

You will be shown:
1. The ORIGINAL qEEG REPORT TEXT (for verification of claims)
2. Multiple analyses labeled "Analysis A", "Analysis B", etc.

Do not attempt to infer which model wrote which analysis.

## Your Task

For each analysis, evaluate:
1. **Accuracy**: Are all claims supported by the original report data?
2. **Completeness**: Does it include per-site tables, speculative commentary, and meet the ~2500 word target?
3. **Quality**: Are competing hypotheses preserved? Are tables formatted correctly?

## Required JSON Output

Return JSON ONLY (no leading/trailing text) with this shape:

```json
{
  "reviews": [
    {
      "analysis_label": "A",
      "strengths": ["..."],
      "weaknesses": ["..."],
      "missing_or_unclear": ["..."],
      "accuracy_issues": ["List any claims NOT supported by the original report"],
      "risk_of_overreach": "..."
    }
  ],
  "ranking_best_to_worst": ["A", "B"],
  "overall_notes": "..."
}
```

## Verification Checklist

For each analysis, explicitly check:
- [ ] Per-site P300 table present with C3, CZ, C4, P3, PZ, P4 values?
- [ ] N100 data reported?
- [ ] Theta/beta ratio reported with device reference range?
- [ ] Peak alpha frequency by region?
- [ ] Speculative Commentary section present with competing hypotheses?
- [ ] Word count approximately 2500+ words?
- [ ] All numerical claims match the original report?

## Rules

- Include one "reviews" entry per analysis label provided to you.
- Rank only the labels you were shown.
- Keep feedback specific and actionable.
- **Critical**: Point out any hallucinations or unsupported claims by cross-referencing the original report.
- Flag analyses that are too brief (under 2000 words) as major weakness.
- Flag missing per-site tables as major weakness.

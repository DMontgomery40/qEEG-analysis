You are performing blinded peer review of other models' qEEG analyses.

You will be shown:
1. The ORIGINAL qEEG REPORT TEXT (for verification of claims)
2. Multiple analyses labeled "Analysis A", "Analysis B", etc.

## Data Precedence (Critical)

- The STRUCTURED DATA PACK is the authoritative numeric source.
- The MULTIMODAL VISION TRANSCRIPT is secondary support.
- Raw OCR report text is tertiary/context only and may contain OCR artifacts.
- If OCR text conflicts with Structured Data Pack values (for example decimal-drop artifacts), treat the Structured Data Pack as correct and do not flag this as an analysis hallucination by itself.

Do not attempt to infer which model wrote which analysis.

## Your Task

For each analysis, evaluate:
1. **Accuracy**: Are all claims supported by the original report data?
2. **Evidence-first usefulness**: Does it preserve the strongest verified findings, ratios or uncommon connections, and practical caveats?
3. **Quality**: Does it distinguish observations from interpretation, preserve competing hypotheses where warranted, and remain useful for downstream neuro-team review and patient/family explanation?
4. **Repair priority**: What, if anything, must change to improve correctness or usefulness?

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
- [ ] Strongest verified longitudinal findings are clearly stated?
- [ ] Ratios, asymmetries, gradients, or uncommon connections are discussed when supported by the data?
- [ ] Practical caveats are preserved (for example session validity, state dependence, device-definition quirks, N/A values)?
- [ ] N100, ERP, and background metrics are covered to the extent they materially matter in the source data?
- [ ] Competing hypotheses are preserved when the data reasonably supports more than one interpretation?
- [ ] All numerical claims match the original report?

Soft checks, not automatic failures:
- Per-site tables are only required when the source data makes them materially useful for evidence traceability.
- Exact headings are not required.
- Brevity is only a weakness when important supported findings or caveats are omitted.

## Rules

- Include one "reviews" entry per analysis label provided to you.
- Rank only the labels you were shown.
- Keep feedback specific and actionable.
- **Critical**: Point out any hallucinations or unsupported claims by cross-referencing the original report.
- Do NOT penalize primarily for missing exact headings, missing ornamental tables, or not hitting an arbitrary word count.
- Request more structure or tables only when they materially improve auditability, neuro-team usefulness, or later patient/family explanation.

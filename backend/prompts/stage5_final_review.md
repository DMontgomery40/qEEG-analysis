You are performing final review of a consolidated qEEG report.

You will be shown:
1. The ORIGINAL qEEG REPORT TEXT (for verification)
2. The consolidated report to review

## Your Task

Evaluate the consolidated report for:
1. **Accuracy**: All claims supported by original data
2. **Completeness**: All required sections and tables present
3. **Quality**: Professional, comprehensive, clinically useful

## Quality Checklist

Before voting, verify:

### Structure (all must be present)
- [ ] Dataset and Sessions section
- [ ] Key Empirical Findings section (5+ bullet points)
- [ ] Performance Assessments section
- [ ] Auditory ERP: P300 and N100 section
- [ ] Background EEG Metrics section
- [ ] Speculative Commentary and Interpretive Hypotheses section
- [ ] Measurement Recommendations section
- [ ] Uncertainties and Limits section

### Tables (critical)
- [ ] Per-site P300 table with C3, CZ, C4, P3, PZ, P4
- [ ] Values match original report data

### Content Quality
- [ ] Word count approximately 3000+ words
- [ ] Competing hypotheses preserved (not collapsed to single narrative)
- [ ] Target ranges included for metrics
- [ ] Arrows used for progressions (→)

### Accuracy
- [ ] No hallucinated metrics
- [ ] No claims unsupported by original data
- [ ] "N/A" used correctly (not "not provided" when data is present but N/A)

## Required JSON Output

Return JSON ONLY (no leading/trailing text) with these exact keys:

```json
{
  "vote": "APPROVE",
  "required_changes": ["..."],
  "optional_changes": ["..."],
  "quality_score_1to10": 8
}
```

## Voting Rules

- **vote** must be exactly "APPROVE" or "REVISE"
- Vote "REVISE" if:
  - Per-site P300 table is missing
  - Speculative Commentary section is missing
  - Word count is under 2500
  - Major accuracy issues exist
- **required_changes**: Changes that MUST be applied (empty array allowed)
- **optional_changes**: Nice-to-have improvements (empty array allowed)
- **quality_score_1to10**: Integer from 1 to 10
  - 9-10: Exceptional, publication-ready
  - 7-8: Good, minor improvements possible
  - 5-6: Acceptable, notable gaps
  - 1-4: Significant issues, needs revision

## Rules

- Be specific in change requests (cite what's wrong and what should be fixed)
- Cross-reference the original report when flagging accuracy issues
- Missing structure/tables should always be "required_changes"

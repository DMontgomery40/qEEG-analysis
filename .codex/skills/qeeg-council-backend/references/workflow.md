# qEEG Council Workflow (6 stages)

## Stage 1: Initial analysis (parallel)
Input: extracted qEEG text
Output (per model): Markdown analysis
Artifact kind: analysis
File ext: .md

## Stage 2: Peer review (parallel, anonymized)
Input: other models' Stage 1 analyses labeled A/B/C
Output (per reviewer): JSON critique
Artifact kind: peer_review
File ext: .json
Required shape: assets/schemas/stage2_peer_review.schema.json

## Stage 3: Revision (parallel)
Input: model's Stage 1 analysis + peer feedback about that analysis
Output (per model): revised Markdown
Artifact kind: revision
File ext: .md

## Stage 4: Consolidation (single consolidator)
Input: all Stage 3 revisions
Output: consolidated Markdown report
Artifact kind: consolidation
File ext: .md

## Stage 5: Final review (parallel vote)
Input: consolidated report
Output (per model): JSON with vote + required edits + score
Artifact kind: final_review
File ext: .json
Required shape: assets/schemas/stage5_final_review.schema.json

## Stage 6: Final drafts (parallel)
Input: consolidated report + Stage 5 required edits
Output (per model): polished Markdown draft
Artifact kind: final_draft
File ext: .md

## Selection (UI)
User selects one output to become final.
Backend persists selected artifact reference and generates exports:
- final.md
- final.pdf

## Anonymization
- Each run assigns labels A/B/C to chosen model IDs.
- Store mapping in run metadata.
- Use labels in Stage 2 prompts only.
- De-anonymize only in UI.

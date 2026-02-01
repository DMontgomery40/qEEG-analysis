# Grant Proposal: qEEG Educational Explainer Development Using Multi-Model AI Deliberation

---

## Project Summary

**Project Title:** Development of Patient-Facing and Clinician-Facing qEEG Educational Explainers Using Multi-Model AI Deliberation and Retrospective WAVi EEG Data

**Requesting Organization:** Neuro-Luminance Brain Health Centers, Denver, CO

**Clinical Lead:** Dr. Theodore Henderson, MD, PhD

**Project Duration:** 9 months

**Total Budget Requested:** $5,000

**Data Platform:** WAVi EEG System

**Dataset:** Approximately 400 retrospective, fully de-identified EEG records

---

## Organizational Background

Neuro-Luminance Brain Health Centers, based in Denver, Colorado, is a clinical practice specializing in brain health assessment and treatment under the leadership of Dr. Theodore Henderson. The organization has accumulated substantial clinical experience with quantitative electroencephalography (qEEG) and event-related potentials (ERP), utilizing the WAVi EEG platform for patient assessments.

Over years of clinical operation, Neuro-Luminance has amassed a retrospective dataset of approximately 400 EEG records. These records have been fully de-identified prior to any research use, containing no Protected Health Information (PHI), no patient identifiers, and no linkage keys that could enable re-identification. The dataset exists solely as aggregated pattern data suitable for educational content development.

---

## Problem Statement

Quantitative EEG (qEEG) reports are complex clinical documents containing multiple metrics, waveform data, statistical comparisons, and normative references. These reports present significant interpretation challenges:

**For Patients:**
- qEEG reports are dense with technical terminology (theta/beta ratios, P300 latencies, peak alpha frequencies)
- Patients struggle to understand what their results mean for their health
- Without clear explanation, patients cannot meaningfully participate in treatment decisions
- Existing educational materials are either too technical or overly simplified

**For Clinicians:**
- Interpreting qEEG reports requires synthesizing multiple data points across brain regions
- Different metrics may suggest competing hypotheses that require careful consideration
- Time constraints limit the depth of explanation possible during clinical encounters
- Consistent, high-quality interpretation support is needed

**The Core Challenge:**
Creating accurate, accessible educational explainers from qEEG data requires extreme precision. Any system that produces these explainers must:

1. **Never fabricate or hallucinate numeric values** - A wrong number in a clinical context could mislead patients or clinicians
2. **Preserve uncertainty and competing hypotheses** - Oversimplification can be as harmful as complexity
3. **Ground all statements in the actual source data** - Every claim must trace back to the original report
4. **Maintain consistency across multiple review passes** - Errors must be caught and corrected systematically

Traditional AI systems struggle with these requirements because single-model generation lacks the verification mechanisms needed for clinical accuracy.

---

## Project Objectives

### Primary Objective
Develop and validate a multi-model AI deliberation system capable of producing patient-facing and clinician-facing educational explainers from de-identified qEEG data that are:
- Factually accurate (all numeric values verified against source)
- Appropriately nuanced (competing hypotheses preserved)
- Accessible (patient-friendly language without oversimplification)
- Reproducible (deterministic validation prevents hallucination)

### Secondary Objectives
1. Document the multi-model deliberation methodology for potential broader application
2. Establish quality benchmarks for AI-assisted clinical document generation
3. Create a template framework for educational content development from complex medical data
4. Demonstrate the feasibility of using frontier AI models with appropriate verification safeguards

---

## Methods: Multi-Model Deliberation Architecture

### Overview: The qEEG Council System

This project employs a novel 6-stage deliberation workflow where multiple frontier large language models (LLMs) collaboratively analyze qEEG reports through a structured process of parallel analysis, peer review, revision, consolidation, voting, and final drafting. This architecture is specifically designed to prevent the hallucination of numeric values and ensure factual accuracy.

### The 6-Stage Workflow

#### Stage 1: Initial Analysis (Parallel, Multiple Models)
- **Process:** All council models (Claude, GPT, Gemini) analyze the same de-identified qEEG report in parallel
- **Input:** Original report text + page images for vision-capable models
- **Output per model:** Independent clinical analysis (~2,500 words)
- **Critical mechanism:** Multi-pass ingestion ensures ALL pages are processed; PDFs with more than 10 pages require 2+ multimodal passes

**Deterministic Data Extraction:**
Before any model generates analysis, the system performs deterministic extraction of numeric values directly from OCR text using strict regex patterns:
- Physical Reaction Time values
- Trail Making Test A/B scores
- Audio P300 delay and voltage measurements
- Theta/beta ratios
- Peak alpha frequencies by region
- Per-site P300 amplitudes (C3, CZ, C4, P3, PZ, P4)
- N100 central-frontal averages

This deterministic extraction creates a "Data Pack" (`_data_pack.json`) that serves as the authoritative ground truth. The Data Pack is tagged with `extraction_method: "deterministic_report_text"` to distinguish OCR-extracted values from any model-generated content.

**OCR Error Correction:**
The system includes automated correction for common OCR artifacts. For example, if OCR reads "1230" for what should be "120 ms" (a known N100 latency OCR error), the system detects this via latency window bounds (30-120 ms range) and corrects it automatically.

#### Stage 2: Peer Review (Parallel, Blinded)
- **Process:** Each model reviews all other models' Stage 1 analyses WITHOUT reviewing its own
- **Anonymization:** Analyses are labeled A, B, C... with stable label mapping; reviewers do not know which model wrote which analysis
- **Output:** Structured JSON critique for each analysis including:
  - Strengths and weaknesses
  - Accuracy issues (claims not supported by original report)
  - Missing or unclear elements
  - Risk of overreach assessment

**Verification Checklist (enforced at Stage 2):**
- Per-site P300 table present with all 6 electrode sites?
- N100 data reported?
- Theta/beta ratio with device reference range?
- Peak alpha frequency by region?
- All numerical claims match the original report?

**Temperature setting:** 0.1 (minimal variation for critical evaluation)

#### Stage 3: Revision (Parallel)
- **Process:** Each model revises its Stage 1 analysis incorporating peer feedback
- **Grounding requirement:** Must cite original report for ALL fact-checking
- **Rule:** Cannot add new facts not present in the original report text
- **Output:** Revised analysis (~2,500 words) that addresses peer critiques

#### Stage 4: Consolidation (Single Model)
- **Process:** A designated consolidator model synthesizes ALL revised analyses into one coherent report
- **Output requirements:**
  - Length: 3,000-4,000 words
  - 8 required sections (validated by system):
    1. Dataset and Sessions
    2. Key Empirical Findings
    3. Performance Assessments
    4. Auditory ERP: P300 and N100
    5. Background EEG Metrics
    6. Speculative Commentary and Interpretive Hypotheses
    7. Measurement Recommendations
    8. Uncertainties and Limits

**Structural validation:** The system checks for presence of all required headings and an end sentinel marker. If output is truncated, automated repair regenerates from the last complete heading (up to 3 calls total).

**Consolidation rules:**
- Preserves consensus points from all analyses
- Resolves disagreements explicitly (states the disagreement and resolution)
- Does NOT add any facts not present in the original report
- Every claim must be traceable to the original report text

#### Stage 5: Final Review & Voting (Parallel)
- **Process:** ALL council models independently vote on the consolidated report
- **Output:** Structured JSON with strict schema validation:
  - `vote`: Must be exactly "APPROVE" or "REVISE"
  - `required_changes`: Array of mandatory changes
  - `optional_changes`: Array of suggested improvements
  - `quality_score_1to10`: Integer 1-10

**Voting criteria (vote REVISE if):**
- Per-site P300 table is missing
- Speculative Commentary section is missing
- Word count under 2,500
- Major accuracy issues exist
- Numeric claims do not match the original report/Data Pack

**Temperature setting:** 0.1 (objective evaluation)

**Required Changes Aggregation:**
All `required_changes` from all voting models are collected and deduplicated. The union of all required changes is passed to Stage 6.

#### Stage 6: Final Draft (Parallel)
- **Process:** All models produce final output incorporating ALL aggregated required changes
- **Rule:** Keep all numeric claims aligned with the original report text / Data Pack
- **Output:** Publication-ready educational explainer

### Hallucination Prevention Mechanisms

The system employs multiple layers of protection against numeric hallucination:

**Layer 1: Deterministic Override**
- OCR-extracted facts are added FIRST and override any model outputs
- Facts tagged with `extraction_method: "deterministic_report_text"` take precedence
- Prevents models from substituting hallucinated numbers for real data

**Layer 2: Conflict Detection**
- System extracts numeric signatures from all facts
- Groups facts by key and detects when the same metric has different numeric values
- In strict mode, conflicts trigger runtime errors and require resolution

**Layer 3: Multi-Source Grounding**
- Every stage beyond Stage 1 receives:
  - Original report text (source of truth)
  - Data Pack JSON (authoritative transcription)
  - Vision transcript (page-grounded transcription)
- Creates multiple cross-checks that models must satisfy

**Layer 4: Strict Data Availability Mode**
- Requires ALL 4 extraction sources: pypdf, pymupdf, apple_vision (OCR), tesseract (OCR)
- Fails if any source is missing
- Multi-source metadata tracked for audit

**Layer 5: Prompt-Level Accuracy Requirements**
Every stage prompt includes explicit accuracy rules:
- "Point out any hallucinations or unsupported claims by cross-referencing the original report"
- "Do not add new facts that are not present in the report text"
- "No hallucinated metrics"
- "No claims unsupported by original data"

### Why Multiple Expensive Models?

This project deliberately employs multiple frontier models (Claude Opus/Sonnet, GPT-4o/GPT-5, Gemini Pro) rather than a single model for several reasons:

1. **Independent verification:** Each model's analysis is reviewed by peers who may catch errors the original model missed
2. **Diverse perspectives:** Different models may identify different valid interpretations, preserving clinical nuance
3. **Consensus filtering:** Only claims that survive multi-model peer review reach the final output
4. **Error amplification prevention:** A single model's systematic bias is corrected by other models

The deliberation process ensures that:
- A numeric error in one model's output will be flagged by peer reviewers
- Hallucinated claims will fail the "supported by original report" check
- Competing hypotheses are preserved rather than collapsed to a single narrative

---

## What This Project Is NOT

To ensure clarity for reviewers, this project explicitly excludes:

1. **NOT new data collection:** All data already exists and has been collected in the course of standard clinical care
2. **NOT an intervention study:** No patients will receive any treatment as part of this project
3. **NOT diagnostic tool development:** The system produces educational content, not clinical diagnoses
4. **NOT autonomous medical decision-making:** All outputs are educational materials for human review
5. **NOT re-identification research:** The data has no linkage keys and cannot be re-identified
6. **NOT human subjects research:** Uses only secondary analysis of de-identified data

---

## Ethical & Data Governance Statement

### Data Status
- All ~400 EEG records are fully de-identified
- No Protected Health Information (PHI)
- No patient identifiers
- No linkage keys enabling re-identification
- Data exists as aggregated patterns only

### IRB Determination
This project qualifies for IRB exemption or "not human subjects research" determination because:
- It involves only secondary use of existing de-identified data
- There is no interaction with human subjects
- There is no collection of identifiable private information
- Re-identification is not possible

**Note:** Final IRB determination will be obtained from the appropriate institutional review body prior to project initiation.

### AI Tooling Governance
- AI tools are used as assistive educational infrastructure only
- No autonomous clinical decision-making
- All outputs subject to human clinical review
- Multi-model deliberation with deterministic validation prevents hallucination
- Complete audit trail maintained for all processing

### Data Security
- Data remains on secured local infrastructure
- API calls to AI services transmit only de-identified content
- No PHI transmitted to any external service
- Processing logs retained for audit purposes

---

## AI Tooling Justification

### Why AI-Assisted Generation?

Creating high-quality educational content from complex qEEG data is time-intensive when done manually. A single comprehensive qEEG report may require hours of expert interpretation to translate into patient-friendly explanations. With ~400 records and the need for both patient-facing and clinician-facing content, manual creation is impractical within the project timeline and budget.

### Why This Specific Architecture?

**Single-model generation is insufficient for clinical accuracy requirements:**
- Individual LLMs can hallucinate plausible-sounding but incorrect numeric values
- Single models may oversimplify or miss important competing hypotheses
- No built-in verification mechanism exists

**The 6-stage multi-model deliberation addresses these limitations:**
- Deterministic extraction provides ground truth for all numeric values
- Peer review catches errors before they propagate
- Consensus mechanisms filter out unsupported claims
- Structural validation ensures completeness
- Voting and required changes ensure quality standards

### Model Selection Rationale

The project uses three frontier model families:

1. **Anthropic Claude (Opus, Sonnet):** Strong reasoning capabilities, careful about uncertainty
2. **OpenAI GPT (GPT-4o, GPT-5):** Excellent at structured output, strong vision capabilities
3. **Google Gemini (Pro, Flash):** Extended context windows, multimodal strength

Using models from different providers ensures:
- No single-vendor bias in outputs
- Different training data leads to diverse perspectives
- Redundancy if one service experiences issues

### Cost Efficiency

The $5,000 budget covers 9 months of subscription access to:
- Claude Pro Max (Anthropic)
- GPT Pro (OpenAI)
- Gemini Pro Ultra (Google)

These subscription tiers provide sufficient API throughput for processing ~400 records through the 6-stage pipeline while remaining dramatically more cost-effective than manual expert interpretation.

---

## Deliverables

### Primary Deliverables

1. **Patient-Facing Educational Explainers**
   - Clear, accessible explanations of qEEG findings
   - Appropriate for patients without medical training
   - Preserves important nuance without overwhelming detail
   - Format: PDF documents suitable for patient portals

2. **Clinician-Facing Interpretive Summaries**
   - Comprehensive analyses including all technical detail
   - Preserves competing hypotheses and uncertainty
   - Includes measurement recommendations
   - Format: Markdown/PDF documents

3. **Validated Multi-Model Deliberation System**
   - Documented 6-stage workflow
   - Deterministic validation mechanisms
   - Reproducible processing pipeline

### Secondary Deliverables

4. **Methodology Documentation**
   - Technical specification of deliberation architecture
   - Hallucination prevention mechanisms
   - Quality assurance procedures

5. **Quality Metrics Report**
   - Accuracy validation results
   - Peer review statistics
   - Voting and consensus data

---

## Timeline (9 Months)

### Months 1-2: Infrastructure and Validation
- Configure multi-model deliberation system
- Validate deterministic extraction accuracy
- Process initial batch of 50 records
- Refine prompts and validation thresholds

### Months 3-5: Primary Processing
- Process remaining ~350 records through 6-stage pipeline
- Ongoing quality monitoring and adjustment
- Begin patient-facing content adaptation

### Months 6-7: Content Refinement
- Review and refine educational content
- Clinical review of outputs
- Develop patient-friendly formatting

### Months 8-9: Documentation and Completion
- Compile methodology documentation
- Quality metrics analysis
- Final deliverable preparation
- Project report completion

---

## Budget

### Total Requested: $5,000

| Category | Item | Monthly Cost | Duration | Total |
|----------|------|--------------|----------|-------|
| AI Services | Claude Pro Max Subscription | $200/month | 9 months | $1,800 |
| AI Services | GPT Pro Subscription | $200/month | 9 months | $1,800 |
| AI Services | Gemini Pro Ultra Subscription | $150/month | 9 months | $1,350 |
| Contingency | Infrastructure/Misc | - | - | $50 |
| **TOTAL** | | | | **$5,000** |

### Budget Justification

**AI Service Subscriptions ($4,950):**
The multi-model deliberation architecture requires access to multiple frontier AI services. Subscription tiers (rather than pay-per-token API access) provide:
- Predictable monthly costs
- Sufficient throughput for batch processing
- Access to latest model versions
- Extended context windows needed for comprehensive qEEG analysis

The 6-stage pipeline processes each record through multiple models:
- Stage 1: 3 models in parallel
- Stage 2: 3 models peer reviewing
- Stage 3: 3 models revising
- Stage 4: 1 consolidator model
- Stage 5: 3 models voting
- Stage 6: 3 models finalizing

This represents approximately 16 model invocations per record, with ~400 records requiring substantial but manageable throughput within subscription limits.

**Contingency ($50):**
Minor infrastructure costs (cloud storage for artifacts, backup systems).

### In-Kind Contributions (Not Charged to Grant)
- Clinical expertise (Dr. Henderson) - Donated
- Existing de-identified dataset - Previously collected
- Computing infrastructure - Organization-provided
- Administrative support - Organization-provided

---

## Impact & Sustainability

### Immediate Impact
- Approximately 400 educational explainers produced
- Validated methodology for AI-assisted clinical content generation
- Framework adaptable to other clinical documentation needs

### Broader Impact
- Demonstrates feasibility of multi-model deliberation for clinical accuracy
- Provides template for similar projects in other clinical domains
- Contributes to understanding of AI verification mechanisms in healthcare contexts

### Sustainability
- System design documented for continued use beyond grant period
- Methodology transferable to new data as clinic operations continue
- No ongoing grant funding required after initial development

---

## Funder-Specific Alignment

### [Likely Interested Funder 1]

*[This section to be customized based on specific funder priorities and guidelines]*

**Alignment with Funder Priorities:**
- [Describe how project aligns with funder's stated mission]
- [Reference specific funder initiatives or focus areas]
- [Highlight aspects of project most relevant to funder]

**Addressing Funder Evaluation Criteria:**
- Innovation: Novel multi-model deliberation architecture with deterministic validation
- Feasibility: Clear timeline, modest budget, experienced clinical leadership
- Impact: Scalable methodology with potential for broad application
- Sustainability: System operational beyond grant period without additional funding

---

### [Likely Interested Funder 2]

*[This section to be customized based on specific funder priorities and guidelines]*

**Alignment with Funder Priorities:**
- [Describe how project aligns with funder's stated mission]
- [Reference specific funder initiatives or focus areas]
- [Highlight aspects of project most relevant to funder]

**Addressing Funder Evaluation Criteria:**
- [Customize based on funder's specific evaluation framework]

---

## Appendix A: Technical Specification Summary

### Data Pack Schema (Version 1)

The deterministic Data Pack contains:
- `schema_version`: 1
- `pages_seen`: Array of processed page numbers
- `page_inventory`: Structured inventory of each page's contents
- `facts`: Array of extracted metrics with:
  - `fact_type`: performance_metric | evoked_potential | state_metric | peak_frequency | p300_cp_site | n100_central_frontal_average
  - `metric`: Specific metric identifier
  - `session_index`: Session number
  - `value`: Numeric value
  - `unit`: ms | V | Hz | ratio
  - `target_range`: Reference range
  - `source_page`: Page number
  - `extraction_method`: deterministic_report_text | vision_llm
- `derived`: Pre-computed markdown tables

### Stage Temperature Settings
- Stage 1 (Analysis): 0.2
- Stage 2 (Peer Review): 0.1
- Stage 3 (Revision): 0.2
- Stage 4 (Consolidation): 0.2
- Stage 5 (Voting): 0.1
- Stage 6 (Final Draft): 0.2

Lower temperatures (0.1) are used for critical evaluation stages where variation could introduce errors.

---

## Appendix B: Stage 5 Voting Schema

```json
{
  "vote": "APPROVE" | "REVISE",
  "required_changes": ["string array of mandatory changes"],
  "optional_changes": ["string array of suggested improvements"],
  "quality_score_1to10": integer (1-10)
}
```

Validation rules enforced programmatically:
- `vote` must be exactly "APPROVE" or "REVISE"
- `required_changes` must be array of strings
- `optional_changes` must be array of strings
- `quality_score_1to10` must be integer in range [1, 10]

---

## Fields Requiring Manual Editing Before Submission

1. **[Likely Interested Funder 1]** - Replace with actual funder name and customize alignment section
2. **[Likely Interested Funder 2]** - Replace with actual funder name and customize alignment section
3. **IRB determination** - Confirm specific exemption category based on institutional guidance
4. **Subscription cost verification** - Confirm current pricing for AI service subscriptions
5. **Organization details** - Verify and update any organizational information as needed
6. **Clinical lead credentials** - Confirm Dr. Henderson's current title and credentials
7. **Dataset count** - Verify exact number of de-identified records available

---

*Document prepared for grant submission. All technical specifications reflect the implemented qEEG Council multi-model deliberation system.*

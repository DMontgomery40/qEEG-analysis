"""Mock LLM responses for each pipeline stage.

These responses are designed to:
1. Pass validation checks (especially Stage 5's _validate_stage5)
2. Have the correct structure for each stage
3. Be deterministic and realistic enough to exercise real code paths
"""

import json

# Stage 1: Initial Analysis (Markdown, ~2500 words)
STAGE1_MARKDOWN = """# Dataset and Sessions

This analysis is based on qEEG data collected using the WAVi device. The dataset includes multiple recording sessions.

**Data Source**: WAVi EEG System
**Patient Demographics**: Adult male
**Sessions Analyzed**: 3 sessions

Self-report instruments (HAM-A, PHQ-9) were not completed for these sessions. Subjective labels such as "improvement" are not supported by this dataset alone.

# Key Empirical Findings

- **P300 latency progression**: Central-parietal P300 delay showed improvement across sessions: 344 ms → 316 ms → 264 ms. This represents a 80 ms reduction from baseline.

- **Theta/beta ratio normalization**: The theta/beta ratio decreased from elevated levels (2.8) toward the target range (1.5-2.5), suggesting improved frontal regulation patterns.

- **Peak alpha frequency stability**: Occipital peak alpha frequency remained stable at approximately 10.2 Hz across all sessions, within the healthy range (9.0-11.0 Hz).

- **Reaction time variability**: Physical reaction time showed increased variability: 246 ms (±37) → 280 ms (±43) → 293 ms (±40), which may reflect state factors.

- **N100 amplitude changes**: N100 amplitudes showed a redistribution pattern with increased left hemisphere activity in later sessions.

# Performance Assessments

**Physical reaction time** (ms): 246 (±37) → 280 (±43) → 293 (±40). Target range: 253–364 ms.

Real-world framing: the mean increases by 47 ms from Session 1 to Session 3 (about five hundredths of a second). This is small in daily life but large enough to reflect a meaningful shift on computerized reaction-time tasks.

**Trail Making Test A** (seconds): 41 → 33 → 47. Target range: 38–64.

**Trail Making Test B** (seconds): 44 → 70 → 38. Target range: 43–84.

Interpretation constraints: the Session 3 pattern (TMT-B faster than TMT-A) is unusual and should be treated as a validity concern rather than a definitive executive improvement claim. Replication under standardized conditions is recommended.

# Auditory ERP: P300 and N100

## Device Definitions

The WAVi system defines P300 delay as the earliest qualifying central-parietal latency in the 240–499 ms window. This means the delay metric can change because a different site becomes the earliest qualifying site, not necessarily because that site's latency changed.

## Per-Site P300 Data

| Site | Session 1 (µV / ms) | Session 2 (µV / ms) | Session 3 (µV / ms) |
|------|---------------------|---------------------|---------------------|
| C3   | 5.2 / 356           | 6.1 / 328           | 7.0 / 276           |
| CZ   | 6.8 / 344           | 7.2 / 316           | 8.1 / 264           |
| C4   | 4.9 / 362           | 5.8 / 334           | 6.5 / 282           |
| P3   | 5.5 / 348           | 6.4 / 320           | 7.3 / 268           |
| PZ   | 7.1 / 340           | 7.8 / 312           | 8.5 / 260           |
| P4   | 5.0 / 358           | 5.9 / 330           | 6.7 / 278           |

## Topographic Observations

The P300 shows a clear central-parietal distribution with PZ and CZ showing the highest amplitudes across all sessions. There is a slight left-lateralized pattern with P3 amplitudes consistently higher than P4.

Latency improvements are relatively uniform across all sites, suggesting a global processing speed improvement rather than site-specific changes.

## N100 Data

N100 latency: 98 ms → 95 ms → 92 ms (target range: 85-115 ms)
N100 amplitude at CZ: -4.2 µV → -4.8 µV → -5.1 µV

The N100 shows amplitude increases across sessions, potentially reflecting enhanced early auditory processing or attention allocation.

# Background EEG Metrics

**Theta/Beta Ratio**: 2.8 → 2.2 → 1.9. Device reference range: 1.5–2.5.

The theta/beta ratio shows progressive normalization, moving from above-range values toward the center of the target range.

**F3/F4 Alpha Ratio**: 1.05 → 1.02 → 0.98.

Alpha lateralization shows slight normalization toward symmetry across sessions.

**Peak Alpha Frequency by Region**:
- Frontal: 9.8 Hz → 9.9 Hz → 10.0 Hz
- Central-Parietal: 10.1 Hz → 10.2 Hz → 10.2 Hz
- Occipital: 10.2 Hz → 10.2 Hz → 10.3 Hz

All regions show stable peak alpha frequencies within the healthy range (9.0-11.0 Hz).

**Band Magnitudes** (note: these are artifact-sensitive):
- Delta: Stable across sessions
- Theta: Slight decrease (consistent with theta/beta ratio changes)
- Alpha: Stable with good occipital prominence
- Beta: Slight increase in frontal regions

**Coherence Summary**:
- Alpha coherence: Strong interhemispheric coherence maintained
- Beta coherence: Moderate, with frontal predominance
- Theta coherence: Decreased from baseline (potentially positive)

# Speculative Commentary and Interpretive Hypotheses

*The following are interpretive possibilities, not diagnoses, not treatment advice, and not causal claims.*

## Competing Hypotheses

**Generator Migration Hypothesis**: The latency improvements may reflect optimization of neural generator locations, with processing shifting toward more efficient cortical pathways.

**Compensatory Division of Labor Hypothesis**: The slight lateralization changes could represent the brain redistributing processing load across hemispheres for improved efficiency.

**State Plus Metric Sensitivity Hypothesis**: Session-to-session variations may reflect both genuine neural changes and state-dependent factors (caffeine, sleep, arousal) that influence EEG metrics.

**Network Constraints Hypothesis**: The parallel improvements in P300 and theta/beta ratio may reflect coordinated changes in frontal-parietal network connectivity.

## Topographic Interpretation

The central-parietal dominance pattern is consistent with attentional processing networks. The slight left lateralization in P300 may reflect verbal processing strategies during the auditory task.

## Stage Opposition Patterns

The relationship between N100 and P300 changes deserves attention: N100 amplitudes increased while P300 latencies decreased. This could reflect enhanced early sensory gating (N100) freeing up resources for faster stimulus classification (P300).

# Measurement Recommendations

- Standardize time of day for future acquisitions
- Complete HAM-A and PHQ-9 self-report instruments before each session
- Document caffeine, sleep, and medication status
- Consider additional sessions to establish within-subject variability
- Ensure consistent electrode impedances across sessions
- Document any interventions between sessions with specific timing

# Uncertainties and Limits

This analysis is limited by:

- Absence of self-report data to correlate with EEG changes
- Unknown intervention timing and specifics between sessions
- Potential state effects (alertness, caffeine, etc.) not controlled
- Device-specific reference ranges that may not generalize
- The unusual TMT-B < TMT-A pattern in Session 3 raises validity concerns
- Coherence metrics can be influenced by volume conduction artifacts

The P300 and theta/beta improvements are encouraging but require replication and should not be interpreted as diagnostic or prognostic without additional clinical context.
"""

# Stage 2: Peer Review (JSON format)
STAGE2_JSON = json.dumps({
    "reviews": [
        {
            "analysis_label": "A",
            "strengths": [
                "Comprehensive per-site P300 table included",
                "Appropriate use of competing hypotheses in speculative commentary",
                "Good quantitative detail with arrows for progressions"
            ],
            "weaknesses": [
                "Could expand on coherence interpretation",
                "TMT validity concern could be more prominent"
            ],
            "missing_or_unclear": [
                "Specific electrode impedance values not mentioned"
            ],
            "accuracy_issues": [],
            "risk_of_overreach": "Low - maintains appropriately cautious framing"
        },
        {
            "analysis_label": "B",
            "strengths": [
                "Good structure following required format",
                "Includes key empirical findings with quantitative data"
            ],
            "weaknesses": [
                "Per-site table values could be more complete",
                "Speculative commentary section somewhat brief"
            ],
            "missing_or_unclear": [
                "Alpha lateralization interpretation"
            ],
            "accuracy_issues": [],
            "risk_of_overreach": "Low"
        }
    ],
    "ranking_best_to_worst": ["A", "B"],
    "overall_notes": "Both analyses demonstrate solid empirical grounding. Analysis A provides slightly more comprehensive coverage and better preserves competing hypotheses."
})

# Stage 3: Revision (Markdown, similar structure to Stage 1)
STAGE3_MARKDOWN = """# Dataset and Sessions

This analysis is based on qEEG data collected using the WAVi device. The dataset includes multiple recording sessions for longitudinal comparison.

**Data Source**: WAVi EEG System, software version as specified in original report
**Patient Demographics**: Adult male
**Sessions Analyzed**: 3 sessions with dates as specified in original report

Self-report instruments (HAM-A, PHQ-9) were not completed for these sessions. Subjective labels such as "improvement" are not supported by this dataset alone.

# Key Empirical Findings

- **P300 latency progression**: Central-parietal P300 delay showed substantial improvement across sessions: 344 ms → 316 ms → 264 ms. This represents an 80 ms (23%) reduction from baseline, with CZ and PZ showing the largest improvements.

- **Theta/beta ratio normalization**: The theta/beta ratio decreased progressively: 2.8 → 2.2 → 1.9, moving from above the device reference range (1.5-2.5) to within-range values.

- **Peak alpha frequency stability**: Occipital peak alpha frequency remained stable at approximately 10.2 Hz across all sessions, within the healthy range (9.0-11.0 Hz).

- **Reaction time variability**: Physical reaction time increased: 246 ms → 280 ms → 293 ms with increasing variability, potentially reflecting state factors that should be investigated.

- **N100 amplitude enhancement**: N100 amplitudes at CZ showed progressive increases: -4.2 µV → -4.8 µV → -5.1 µV, suggesting enhanced early auditory processing.

- **TMT validity concern**: Session 3 showed TMT-B faster than TMT-A (38s vs 47s), which is unusual and flags a validity concern for that session's executive function data.

# Performance Assessments

**Physical reaction time** (ms): 246 (±37) → 280 (±43) → 293 (±40). Target range: 253–364 ms.

Real-world framing: the mean increases by 47 ms across sessions. While still within target range, the increasing trend and variability warrant monitoring.

**Trail Making Test A** (seconds): 41 → 33 → 47. Target range: 38–64.
**Trail Making Test B** (seconds): 44 → 70 → 38. Target range: 43–84.

**CRITICAL VALIDITY NOTE**: The Session 3 pattern (TMT-B = 38s, TMT-A = 47s) is highly unusual. TMT-B typically takes longer than TMT-A due to increased cognitive load. This inverted pattern should be treated as a validity concern rather than interpreted as executive improvement. Recommend replication under standardized conditions.

# Auditory ERP: P300 and N100

## Device Definitions

The WAVi system defines P300 delay as the earliest qualifying central-parietal latency in the 240–499 ms window. This "winner-take-all" approach means the delay metric can change because a different site becomes the earliest qualifier, not necessarily because processing speed changed uniformly.

## Per-Site P300 Data

| Site | Session 1 (µV / ms) | Session 2 (µV / ms) | Session 3 (µV / ms) |
|------|---------------------|---------------------|---------------------|
| C3   | 5.2 / 356           | 6.1 / 328           | 7.0 / 276           |
| CZ   | 6.8 / 344           | 7.2 / 316           | 8.1 / 264           |
| C4   | 4.9 / 362           | 5.8 / 334           | 6.5 / 282           |
| P3   | 5.5 / 348           | 6.4 / 320           | 7.3 / 268           |
| PZ   | 7.1 / 340           | 7.8 / 312           | 8.5 / 260           |
| P4   | 5.0 / 358           | 5.9 / 330           | 6.7 / 278           |

## Topographic Analysis

PZ and CZ show the highest amplitudes and earliest latencies across all sessions, consistent with the expected central-parietal topography for auditory P300.

Slight left lateralization: P3 amplitudes are consistently ~0.5 µV higher than P4, which may reflect verbal processing strategies.

Latency improvements are relatively uniform (76-84 ms reduction across all sites), suggesting global processing speed enhancement rather than site-specific reorganization.

## N100 Analysis

| Metric | Session 1 | Session 2 | Session 3 |
|--------|-----------|-----------|-----------|
| Latency (ms) | 98 | 95 | 92 |
| CZ Amplitude (µV) | -4.2 | -4.8 | -5.1 |

Target range for N100 latency: 85-115 ms (all sessions within range).

# Background EEG Metrics

**Theta/Beta Ratio**: 2.8 → 2.2 → 1.9
Device reference range: 1.5–2.5

Progressive normalization from above-range to within-range values.

**F3/F4 Alpha Ratio** (alpha lateralization): 1.05 → 1.02 → 0.98
Values near 1.0 suggest relatively symmetric alpha distribution between hemispheres.

**Peak Alpha Frequency by Region**:
- Frontal: 9.8 Hz → 9.9 Hz → 10.0 Hz
- Central-Parietal: 10.1 Hz → 10.2 Hz → 10.2 Hz
- Occipital: 10.2 Hz → 10.2 Hz → 10.3 Hz
- Target range: 9.0-11.0 Hz (all within range)

**Coherence Patterns**:
- Alpha band: Strong interhemispheric coherence maintained
- Beta band: Moderate frontal coherence, consistent across sessions
- Theta band: Slight decrease from Session 1, potentially reflecting reduced slow-wave synchronization

Note: Coherence metrics can be influenced by volume conduction and should be interpreted cautiously.

# Speculative Commentary and Interpretive Hypotheses

*The following are interpretive possibilities, not diagnoses, not treatment advice, and not causal claims.*

## Competing Hypotheses

**1. Generator Migration Hypothesis**
The uniform latency improvements across all central-parietal sites could reflect optimization of neural generator efficiency rather than anatomical reorganization. The preserved topographic pattern suggests the same generators are active, but operating more quickly.

**2. Compensatory Division of Labor Hypothesis**
The slight left-right asymmetries (P3 > P4 for P300, symmetric alpha) may represent task-specific hemispheric specialization. The stability of this pattern suggests an established processing strategy rather than ongoing reorganization.

**3. State Plus Metric Sensitivity Hypothesis**
Session-to-session variations in reaction time and TMT scores may substantially reflect state factors (sleep quality, time of day, caffeine intake, arousal level) rather than stable trait changes. The ERP improvements may be more robust to state effects than behavioral measures.

**4. Frontal-Parietal Network Hypothesis**
The parallel improvements in P300 latency and theta/beta ratio suggest coordinated changes in the frontal-parietal attention network. Enhanced frontal regulation (lower theta/beta) may facilitate faster parietal stimulus classification (faster P300).

## Unresolved Questions

- Are the P300 improvements stable or state-dependent?
- What accounts for the unusual TMT pattern in Session 3?
- Would these patterns replicate with different tasks?

# Measurement Recommendations

1. **Standardize acquisition conditions**: Same time of day, consistent pre-session instructions
2. **Complete self-report instruments**: HAM-A, PHQ-9 before each session to correlate with EEG changes
3. **Document state factors**: Caffeine, sleep duration, medications, stress level
4. **Additional sessions**: At least 2 more sessions to establish within-subject variability bounds
5. **Investigate TMT validity**: Repeat TMT under standardized conditions with additional practice trials
6. **Document interventions**: Specific timing and nature of any treatments between sessions
7. **Electrode preparation**: Record impedance values to ensure consistent data quality

# Uncertainties and Limits

**Data Gaps**:
- No self-report instruments to correlate EEG changes with subjective experience
- Unknown specifics of interventions between sessions
- State factors not controlled or documented

**Validity Concerns**:
- TMT-B < TMT-A in Session 3 is highly unusual and unexplained
- Reaction time increases despite P300 improvements are somewhat paradoxical

**Technical Limitations**:
- Device-specific reference ranges may not generalize to other EEG systems
- Coherence metrics subject to volume conduction artifacts
- Band magnitude measures are artifact-sensitive

**Interpretive Constraints**:
- These data cannot establish causation or diagnosis
- Improvements may not generalize to real-world function
- Replication is required before drawing conclusions

The P300 latency and theta/beta improvements are consistent patterns across sessions, but their functional significance remains to be established through behavioral correlates and replication.
"""

# Stage 4: Consolidation (Markdown, comprehensive ~3000 words)
STAGE4_MARKDOWN = """# Dataset and Sessions

**Data Source**: WAVi EEG System
**Patient**: Adult male, demographic details as specified in original report
**Sessions**: 3 longitudinal recording sessions

All analyses agreed on these session parameters. Self-report instruments (HAM-A, PHQ-9) were not completed for any session. This absence limits interpretation of subjective experience correlates.

**Key Agreement**: All council members confirmed the data source and session structure match the original report.

# Key Empirical Findings

The council reached consensus on the following key findings:

- **P300 latency improvement (consensus)**: Central-parietal P300 delay showed consistent improvement across sessions: 344 ms → 316 ms → 264 ms. This 80 ms (23%) reduction was noted by all analyses and is well-supported by the per-site data.

- **Theta/beta ratio normalization (consensus)**: All analyses confirmed the progressive decrease: 2.8 → 2.2 → 1.9, moving from above the device reference range (1.5-2.5) to within-range values.

- **Peak alpha frequency stability (consensus)**: Occipital peak alpha remained stable at ~10.2 Hz (target: 9.0-11.0 Hz) across all sessions.

- **Reaction time variability (consensus)**: Physical reaction time showed increasing trend: 246 ms → 280 ms → 293 ms with increasing variability. All analyses flagged this as requiring investigation.

- **N100 amplitude enhancement (consensus)**: Progressive increases at CZ: -4.2 µV → -4.8 µV → -5.1 µV, suggesting enhanced early auditory processing.

- **TMT validity concern (consensus)**: All analyses flagged the Session 3 TMT-B < TMT-A pattern (38s vs 47s) as a significant validity concern.

# Performance Assessments

**Physical reaction time** (ms): 246 (±37) → 280 (±43) → 293 (±40)
Target range: 253–364 ms

All analyses noted the 47 ms increase across sessions remains within target range but warrants monitoring. The increasing variability (±37 to ±40) was highlighted as potentially reflecting state factors.

**Trail Making Test A** (seconds): 41 → 33 → 47
Target range: 38–64

**Trail Making Test B** (seconds): 44 → 70 → 38
Target range: 43–84

**Council Agreement on Validity Concern**: The Session 3 pattern (TMT-B faster than TMT-A) was unanimously flagged as unusual. This inverted pattern contradicts the expected cognitive load difference and should be treated as a validity concern requiring replication under standardized conditions.

# Auditory ERP: P300 and N100

## Device Definition (Council Consensus)

The WAVi system defines P300 delay as the earliest qualifying central-parietal latency in the 240–499 ms window. This "winner-take-all" approach means the delay metric can change when a different site becomes the earliest qualifier. All analyses correctly noted this measurement consideration.

## Per-Site P300 Data (Verified Against Original)

| Site | Session 1 (µV / ms) | Session 2 (µV / ms) | Session 3 (µV / ms) |
|------|---------------------|---------------------|---------------------|
| C3   | 5.2 / 356           | 6.1 / 328           | 7.0 / 276           |
| CZ   | 6.8 / 344           | 7.2 / 316           | 8.1 / 264           |
| C4   | 4.9 / 362           | 5.8 / 334           | 6.5 / 282           |
| P3   | 5.5 / 348           | 6.4 / 320           | 7.3 / 268           |
| PZ   | 7.1 / 340           | 7.8 / 312           | 8.5 / 260           |
| P4   | 5.0 / 358           | 5.9 / 330           | 6.7 / 278           |

## Topographic Consensus

All analyses agreed on the following topographic observations:

1. **Central-parietal maximum**: PZ and CZ show highest amplitudes and earliest latencies
2. **Slight left lateralization**: P3 amplitudes consistently ~0.5 µV higher than P4
3. **Uniform latency improvement**: 76-84 ms reduction across all sites, suggesting global rather than site-specific changes
4. **Preserved topographic pattern**: The distribution shape remains consistent across sessions

## N100 Data (Council Verified)

| Metric | Session 1 | Session 2 | Session 3 |
|--------|-----------|-----------|-----------|
| Latency (ms) | 98 | 95 | 92 |
| CZ Amplitude (µV) | -4.2 | -4.8 | -5.1 |

Target range for N100 latency: 85-115 ms (all sessions within range)

All analyses noted the progressive N100 amplitude increase, interpreted as potentially reflecting enhanced early auditory processing.

# Background EEG Metrics

**Theta/Beta Ratio** (Council Consensus): 2.8 → 2.2 → 1.9
Device reference range: 1.5–2.5

All analyses agreed this represents progressive normalization from above-range to within-range values, potentially reflecting improved frontal regulation.

**F3/F4 Alpha Ratio**: 1.05 → 1.02 → 0.98
Values near 1.0 indicate relatively symmetric alpha distribution. Minor disagreement existed on interpretation significance, but all agreed the values suggest no concerning asymmetry.

**Peak Alpha Frequency by Region** (Consensus):
- Frontal: 9.8 Hz → 9.9 Hz → 10.0 Hz
- Central-Parietal: 10.1 Hz → 10.2 Hz → 10.2 Hz
- Occipital: 10.2 Hz → 10.2 Hz → 10.3 Hz
- All within target range: 9.0-11.0 Hz

**Coherence Patterns**:
- Alpha band: Strong interhemispheric coherence maintained
- Beta band: Moderate frontal coherence, consistent across sessions
- Theta band: Slight decrease from Session 1

All analyses appropriately cautioned that coherence metrics can be influenced by volume conduction.

# Speculative Commentary and Interpretive Hypotheses

*The following are interpretive possibilities, not diagnoses, not treatment advice, and not causal claims. This section preserves the competing hypotheses from all council analyses.*

## Hypothesis 1: Generator Migration (All Analyses)

The uniform latency improvements across central-parietal sites could reflect optimization of neural generator efficiency. The preserved topographic pattern suggests the same generators are active but operating more quickly, rather than processing shifting to new cortical regions.

## Hypothesis 2: Compensatory Division of Labor (Multiple Analyses)

Hemispheric asymmetries (P3 > P4, symmetric alpha) may represent task-specific specialization. Two analyses noted this could reflect verbal processing strategies during the auditory task.

## Hypothesis 3: State vs. Trait Sensitivity (Council Consensus)

There was strong agreement that session-to-session behavioral variations (reaction time, TMT) may substantially reflect state factors rather than stable trait changes. ERP measures may be more robust to state effects than behavioral measures. This was cited as a key uncertainty.

## Hypothesis 4: Frontal-Parietal Network Integration (Multiple Analyses)

The parallel improvements in P300 latency and theta/beta ratio suggest coordinated changes in frontal-parietal attention networks. Enhanced frontal regulation (lower theta/beta) may facilitate faster parietal stimulus classification (faster P300).

## Resolved Disagreement

One analysis emphasized the paradox of P300 improvements alongside reaction time increases. The council resolved this by noting that P300 reflects stimulus classification speed while reaction time includes motor execution and response selection—these can dissociate.

## Unresolved Questions

- Are improvements stable or state-dependent?
- What explains the unusual TMT Session 3 pattern?
- Would patterns replicate with different tasks?

# Measurement Recommendations

The council consolidated the following recommendations:

1. **Standardize acquisition conditions**: Same time of day, consistent pre-session instructions, controlled environment
2. **Complete self-report instruments**: HAM-A, PHQ-9 before each session
3. **Document state factors**: Caffeine intake, sleep duration/quality, medications, stress level, time since last meal
4. **Additional sessions**: At least 2 more sessions to establish within-subject variability bounds
5. **Investigate TMT validity**: Repeat TMT under standardized conditions with practice trials and explicit timing
6. **Document interventions**: Specific timing, dosing, and nature of any treatments between sessions
7. **Electrode quality**: Record impedance values for each electrode at each session

# Uncertainties and Limits

**Data Limitations** (Council Consensus):
- Absence of self-report instruments prevents correlation with subjective experience
- Intervention details between sessions are unknown
- State factors were not controlled or documented

**Validity Concerns**:
- TMT-B < TMT-A in Session 3 is unexplained and concerning
- Reaction time increases despite P300 improvements are paradoxical

**Technical Constraints**:
- Device-specific reference ranges may not generalize
- Coherence measures subject to volume conduction artifacts
- Band magnitudes are artifact-sensitive

**Interpretive Boundaries**:
- These data cannot establish causation or diagnosis
- Improvements may not generalize to real-world function
- Replication required before drawing firm conclusions

The P300 and theta/beta improvements are consistent patterns supported by all council analyses. Their functional significance remains to be established through behavioral correlates, self-report data, and replication.
"""

# Stage 5: Final Review (JSON format)
STAGE5_JSON = json.dumps({
    "vote": "APPROVE",
    "required_changes": [],
    "optional_changes": [
        "Consider adding more specific electrode impedance documentation guidance",
        "Could expand on the state vs trait hypothesis with specific examples"
    ],
    "quality_score_1to10": 8
})

# Stage 6: Final Draft (Markdown, polished version)
STAGE6_MARKDOWN = """# Dataset and Sessions

**Data Source**: WAVi EEG System
**Patient**: Adult male
**Sessions**: 3 longitudinal recording sessions

Self-report instruments (HAM-A, PHQ-9) were not completed for any session. This absence limits interpretation of subjective experience correlates. Subjective labels such as "improvement" are not supported by this dataset alone.

# Key Empirical Findings

- **P300 latency improvement**: Central-parietal P300 delay showed consistent improvement across sessions: 344 ms → 316 ms → 264 ms. This 80 ms (23%) reduction is well-supported by per-site data across all central-parietal electrodes.

- **Theta/beta ratio normalization**: Progressive decrease from 2.8 → 2.2 → 1.9, moving from above the device reference range (1.5-2.5) to within-range values.

- **Peak alpha frequency stability**: Occipital peak alpha remained stable at approximately 10.2 Hz (target: 9.0-11.0 Hz) across all sessions.

- **Reaction time variability**: Physical reaction time showed increasing trend: 246 ms → 280 ms → 293 ms with increasing variability (±37 → ±40). While within target range, this warrants monitoring.

- **N100 amplitude enhancement**: Progressive increases at CZ: -4.2 µV → -4.8 µV → -5.1 µV, suggesting enhanced early auditory processing.

- **TMT validity concern**: Session 3 showed TMT-B faster than TMT-A (38s vs 47s), an unusual pattern flagged as a validity concern requiring replication.

# Performance Assessments

**Physical reaction time** (ms): 246 (±37) → 280 (±43) → 293 (±40)
Target range: 253–364 ms

Real-world framing: the 47 ms increase represents approximately five hundredths of a second—small in daily life but meaningful on computerized tasks.

**Trail Making Test A** (seconds): 41 → 33 → 47. Target range: 38–64.
**Trail Making Test B** (seconds): 44 → 70 → 38. Target range: 43–84.

**Validity Note**: The Session 3 pattern (TMT-B = 38s faster than TMT-A = 47s) contradicts the expected cognitive load relationship and should be treated as a validity concern rather than interpreted as executive improvement.

# Auditory ERP: P300 and N100

## Device Definition

The WAVi system defines P300 delay as the earliest qualifying central-parietal latency in the 240–499 ms window. This "winner-take-all" approach means the delay metric reflects whichever site is earliest, not an average.

## Per-Site P300 Data

| Site | Session 1 (µV / ms) | Session 2 (µV / ms) | Session 3 (µV / ms) |
|------|---------------------|---------------------|---------------------|
| C3   | 5.2 / 356           | 6.1 / 328           | 7.0 / 276           |
| CZ   | 6.8 / 344           | 7.2 / 316           | 8.1 / 264           |
| C4   | 4.9 / 362           | 5.8 / 334           | 6.5 / 282           |
| P3   | 5.5 / 348           | 6.4 / 320           | 7.3 / 268           |
| PZ   | 7.1 / 340           | 7.8 / 312           | 8.5 / 260           |
| P4   | 5.0 / 358           | 5.9 / 330           | 6.7 / 278           |

## Topographic Analysis

- **Central-parietal maximum**: PZ and CZ show highest amplitudes and earliest latencies
- **Left lateralization**: P3 amplitudes consistently ~0.5 µV higher than P4
- **Uniform improvement**: 76-84 ms latency reduction across all sites suggests global processing speed enhancement
- **Preserved pattern**: Topographic distribution remains consistent across sessions

## N100 Data

| Metric | Session 1 | Session 2 | Session 3 |
|--------|-----------|-----------|-----------|
| Latency (ms) | 98 | 95 | 92 |
| CZ Amplitude (µV) | -4.2 | -4.8 | -5.1 |

All N100 latencies within target range (85-115 ms). Amplitude increases suggest enhanced early auditory processing.

# Background EEG Metrics

**Theta/Beta Ratio**: 2.8 → 2.2 → 1.9
Device reference range: 1.5–2.5

Progressive normalization from above-range to within-range values.

**F3/F4 Alpha Ratio**: 1.05 → 1.02 → 0.98
Values near 1.0 indicate symmetric alpha distribution between hemispheres.

**Peak Alpha Frequency by Region**:
- Frontal: 9.8 Hz → 9.9 Hz → 10.0 Hz
- Central-Parietal: 10.1 Hz → 10.2 Hz → 10.2 Hz
- Occipital: 10.2 Hz → 10.2 Hz → 10.3 Hz

All within target range (9.0-11.0 Hz).

**Coherence Summary**:
- Alpha: Strong interhemispheric coherence maintained
- Beta: Moderate frontal coherence, consistent across sessions
- Theta: Slight decrease from baseline

Note: Coherence metrics can be influenced by volume conduction artifacts.

# Speculative Commentary and Interpretive Hypotheses

*The following are interpretive possibilities, not diagnoses, not treatment advice, and not causal claims.*

## Generator Efficiency Hypothesis

Uniform latency improvements across central-parietal sites suggest optimization of existing neural generators rather than anatomical reorganization. The preserved topographic pattern supports this interpretation.

## Hemispheric Specialization Hypothesis

The consistent left lateralization (P3 > P4) may reflect verbal processing strategies during the auditory task. The symmetric alpha distribution suggests this is task-specific rather than a general hemispheric imbalance.

## State vs. Trait Sensitivity Hypothesis

Behavioral measures (reaction time, TMT) may be more sensitive to state factors (sleep, caffeine, arousal) than ERP measures. This could explain the paradox of P300 improvements alongside reaction time increases.

## Frontal-Parietal Network Hypothesis

Parallel improvements in P300 latency and theta/beta ratio suggest coordinated changes in frontal-parietal attention networks. Enhanced frontal regulation may facilitate faster parietal stimulus classification.

# Measurement Recommendations

1. Standardize acquisition conditions (time of day, environment, instructions)
2. Complete HAM-A and PHQ-9 self-report instruments before each session
3. Document state factors (caffeine, sleep, medications, stress level)
4. Add sessions to establish within-subject variability bounds
5. Repeat TMT under standardized conditions to investigate validity concern
6. Document intervention timing and specifics between sessions
7. Record electrode impedance values for data quality assurance

# Uncertainties and Limits

**Data Limitations**:
- No self-report instruments to correlate with EEG changes
- Unknown intervention details between sessions
- State factors not controlled or documented

**Validity Concerns**:
- TMT-B < TMT-A in Session 3 remains unexplained
- Reaction time increases alongside P300 improvements require interpretation

**Technical Constraints**:
- Device-specific reference ranges may not generalize
- Coherence measures subject to volume conduction artifacts

**Interpretive Boundaries**:
- Cannot establish causation or diagnosis
- Improvements may not generalize to real-world function
- Replication required before drawing conclusions

The P300 and theta/beta improvements are consistent patterns. Their functional significance requires behavioral correlates, self-report data, and replication to establish.
"""

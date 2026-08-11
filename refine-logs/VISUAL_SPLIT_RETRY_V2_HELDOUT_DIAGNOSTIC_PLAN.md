# Visual Grounding Split-retry v2 Heldout Diagnostic

**Problem**: The locked child `peripheral_detail_verification_accuracy` achieved a high discovery-90 Specialized ACC (76.40%), while the accepted complete v2 child set reached only 70.79%. We must test whether the strong child generalizes and whether the three repaired siblings add useful heldout signal or dilute it.

**Method thesis**: Evaluate the already frozen v2 Visual Grounding children on heldout-500 without generating a new Manager proposal, changing any description, or using heldout for selection.

**Date**: 2026-08-10

## Claim Map

| Claim | Why it matters | Minimum evidence | Block |
|---|---|---|---|
| C1 | The locked child is a transferable local visual expert rather than a discovery-only outlier. | Its heldout Specialized ACC exceeds the parent-only Visual Grounding subtree, with corrected > harmed. | B1 |
| C2 | The full repaired Split is helpful only if its added children do not erase the locked child's gain. | Full v2 versus locked-only paired corrected/harmed and subtree ACC. | B1 |

**Anti-claim ruled out**: choosing, revising, activating, or rejecting children after inspecting heldout. This is an exploratory paired diagnostic because heldout-500 has already been accessed by earlier development experiments.

## Block B1 — Frozen Visual Grounding Paired Heldout Evaluation

- **Dataset/task**: the fixed RLHF-V heldout-500 pairwise dataset.
- **Worker**: Qwen3-VL-8B-Instruct, P05, one replicate, routed only to `vllm-8000`.
- **Manager**: no Manager requests. No ErrorSignature, clustering, child generation, Refine, or attribution.
- **Frozen source**:
  - v2 experiment: `phase8_visual_split_retry_locked_v2/`;
  - base rubric: `frozen_source/before_rubric.json` (SHA256 `b5fc4e…`);
  - source heldout pairwise artifact: `phase7_split_refine_evolution_v1/heldout500/combined_pairwise.json`.
- **New predictions**: exactly four v2 child descriptions × 500 samples = **2,000** requests:
  1. `peripheral_detail_verification_accuracy` (locked, exact description);
  2. `main_subject_factuality_and_grounding`;
  3. `relational_compositional_priority`;
  4. `visual_premise_validation_and_consistency`.
- **Reused predictions**: the Visual parent is byte-identical between v2's discovery base and the Phase7 heldout source and must be projected exactly. Full M1 is deliberately replayed in the frozen Phase7-final heldout context: all non-Visual outputs remain identical across P/L/F. (One non-Visual Phase7-final description differs from v2's epoch-start discovery rubric, so it is a fixed contextual baseline rather than a claim of exact whole-rubric replay.)

### Frozen systems

| ID | System | Visual Grounding aggregation |
|---|---|---|
| P | Parent only | parent vote only |
| L | Parent + locked child | locked child when decisive; otherwise parent fallback |
| F | Parent + locked + three repaired children | current v2 children majority vote; tie/all-None parent fallback |

For each system, replay both:

1. the Visual Grounding local subtree on the fixed parent decisive scope; and
2. full five-root M1, replacing only the Visual Grounding subtree while retaining every non-Visual source output.

### Metrics

Primary paired comparisons:

| Comparison | Main metrics |
|---|---|
| L vs P | subtree Specialized ACC, corrected/harmed, net corrected, exact McNemar |
| F vs P | same |
| F vs L | same; direct dilution/complementarity test |

Each system reports full M1 ACC, Coverage, covered ACC, tie rate, correct count and Wilson 95% CI. Each individual child reports heldout support, coverage, ACC, same-support parent ACC, net corrected, and leave-one-child-out effect within F. Sibling conflict is reported pairwise.

Heldout has no discovery cluster labels; therefore target/non-target cluster accuracy is deliberately not reported or inferred.

### Interpretation

| Outcome | Interpretation | Next action |
|---|---|---|
| L > P and F < L | Strong child generalizes; weak siblings dilute it. Prioritize Refine of weak siblings with locked-child conflict feedback. |
| L > P and F >= L | Children are complementary on heldout despite mixed discovery metrics. Preserve full Split as a viable candidate. |
| L <= P | The discovery 76.40% result is not stable enough to support strong-child claims; inspect its coverage and sample-level harms before changing Split. |
| F <= P | Current repair set does not generalize; do not use its heldout result to select replacements. Return to discovery-only Split/Refine improvements. |

McNemar non-significance does not overturn directional evidence at n=500; results must be described as exploratory paired evidence.

## Freeze and Artifact Protocol

New output subtree, without overwriting v2 discovery artifacts:

```text
phase8_visual_split_retry_locked_v2/
  heldout500_diagnostic/
    frozen_manifest.json
    child_predictions.json
    systems/{parent_only,locked_only,full_v2}/
      combined_pairwise.json
      subtree_execution.json
      m1_execution.json
    report.json
    report.md
```

The freeze manifest records source/v2 hashes, the four child descriptions and hashes, heldout dataset hash and sample fingerprints, source prediction hash, P05 request identity, endpoint route, and `selection_after_heldout_forbidden=true`. Any source artifact, criterion description, dataset, or request identity drift hard-fails.

## Run Order and Milestones

| Milestone | Stage | Gate | Estimated cost |
|---|---|---|---:|
| M0 | `split-retry-v2-heldout-freeze` | Four exact child requests; no Worker calls | seconds |
| M1 | `split-retry-v2-heldout-run` | Four complete 500-sample child outputs, resumable | ~45–75 min |
| M2 | `split-retry-v2-heldout-report` | Three local + full M1 replays and paired report | seconds |

## Acceptance Checklist

- [ ] Exactly 2,000 new Worker requests at most through `vllm-8000`; source nodes are never re-requested.
- [ ] Locked child description is byte-identical to its discovery v2 version.
- [ ] Parent-only, locked-only and full-v2 use the same parent scope and aggregation definitions as discovery.
- [ ] Every system passes local subtree and full-M1 replay validation.
- [ ] No Manager call or post-heldout mutation occurs.
- [ ] Report clearly marks exploratory status and does not recommend a selected checkpoint or child set.

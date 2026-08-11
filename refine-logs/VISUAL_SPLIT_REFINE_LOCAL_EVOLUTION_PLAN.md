# Visual Grounding Split-to-Refine Local Evolution Plan

**Problem**: The locked strong child in the accepted Visual Grounding Split v2 improves
the subtree on its own, but the complete four-child majority-voting subtree remains
worse than the locked-only system on heldout-500.  We need to test whether Refine can
improve the existing children without confounding the result with another Split.

**Method thesis**: Starting from a fixed accepted Split subtree, role-aware Refine can
improve eligible child descriptions and improve the collective Visual Grounding
decision without regenerating its child structure.

**Date**: 2026-08-10

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked block |
|---|---|---|---|
| C1 | Refine can improve Split-produced local experts without re-running Split. | At least one child has strict discovery self-ACC improvement with support >= 15. | B1 |
| C2 | Improvements to individual children can improve the fixed child set as a whole. | Final subtree improves over the source full-v2 subtree on heldout-500, with corrected > harmed. | B2 |

**Anti-claim to rule out**: Any observed gain is caused by re-clustering, new child
generation, a changed aggregation rule, or selection on heldout data.

## Frozen Protocol

### Source and scope

- Source experiment: `phase8_visual_split_retry_locked_v2`.
- Source rubric: `final/rubric_committed.json`, SHA-256
  `b11c39f5dd90c53abee90a766200211e51288c44549e4f08bc380e5a1f69e6dc`.
- Root: `init_02_visual_grounding_and_details` only.
- Candidate scope: the four committed children only.  The parent is frozen and is not
  a Refine candidate.
- Split is disabled: no ErrorSignature, clustering, child generation, replacement,
  or topology edit is allowed.
- Node ID, criterion name, score, parent, edges, examples, and all other rubric nodes
  remain unchanged.  Refine may replace only a child description.

### Locked-child semantics

`peripheral_detail_verification_accuracy` remains Split-locked: a future Split cannot
regenerate or replace it.  This does **not** exclude it from Refine.  As agreed, every
child satisfying the role-aware trigger is scheduled for Refine, including this locked
child.

### Refine trigger and competition

For each child `c`, schedule Refine exactly when:

\[
0.5 < \operatorname{Acc}(c) < 0.80,\qquad
|\mathcal S(c)|\ge15,\qquad
\operatorname{Wrong}(c)\ge5.
\]

On the frozen source discovery-90 outputs, all four children qualify:

| Child | ACC | Support | Wrong | Split state |
|---|---:|---:|---:|---|
| `peripheral_detail_verification_accuracy` | 0.7808 | 73 | 16 | locked |
| `main_subject_factuality_and_grounding` | 0.6575 | 73 | 25 | unlocked |
| `relational_compositional_priority` | 0.6207 | 58 | 22 | unlocked |
| `visual_premise_validation_and_consistency` | 0.6203 | 79 | 30 | unlocked |

Each candidate is accepted only if its own discovery support-based accuracy strictly
improves and its new support is at least 15:

\[
\operatorname{Acc}(c')>\operatorname{Acc}(c),\qquad
|\mathcal S(c')|\ge15.
\]

Coverage, description length, full subtree accuracy, and M1 are diagnostics only;
they must not change a Refine accept/reject decision.

## Experiment Blocks

### B0: Offline freeze and eligibility audit

- **Purpose**: freeze the v2 source, discovery-90 identities, source child prediction
  hashes, worker identity, role-aware thresholds, and endpoint protocol before any new
  request is made.
- **Required checks**:
  - source result is accepted and contains exactly the stated Visual subtree;
  - source rubric, discovery dataset, parent/root prediction, and all reused child
    predictions match their frozen hashes;
  - all four source children pass the trigger values listed above;
  - no heldout dataset, metric, or prediction can be read in this stage.
- **Output directory**:
  `phase9_visual_split_refine_local_v1/`.
- **Artifacts**: `frozen_manifest.json`, `source_snapshot/`,
  `eligibility_audit.json`, and `stage_status.json`.

### B1: Three-epoch child-only Refine evolution

- **Claim tested**: C1.
- **Manager**: `Qwen/Qwen3.5-397B-A17B`, Refine v1 prompt and
  `global_rubric_v1` epoch-start snapshot.
- **Worker**: Qwen3-VL-8B-Instruct, P05, single replicate, 8000 endpoint.
- **Epoch protocol**:
  1. Freeze the epoch-start rubric and one Global Rubric Memory snapshot.
  2. Recompute every child trigger from the committed epoch-start predictions.
  3. For every triggered or valid-rejected retryable child, construct Refine evidence
     from that child alone and generate one candidate description.
  4. Generate 90 new pairwise predictions only for each formed candidate; reuse all
     unchanged nodes' frozen/current predictions.
  5. Apply strict self-competition independently for all candidates.
  6. A valid rejected candidate receives mandatory 397B natural-language attribution;
     only then is it eligible for retry next epoch.  Schema-invalid proposals are
     recorded structurally but are not fabricated as natural-language history.
  7. Commit all accepted description patches synchronously at epoch end.  No candidate
     sees another candidate from the same epoch.
  8. Recompute fixed-tree subtree diagnostics after the synchronized commit.
- **Duration**: maximum 3 epochs; early-stop after an epoch if no child is triggered
  and no valid rejected child remains retryable.
- **Required history**: each retry receives its node-specific old/new metrics,
  support, wrong count, corrected/harmed IDs, abstain transitions, sibling overlap and
  conflict diagnostics, previous proposal, and mandatory failure attribution.

### B2: Final heldout-500 diagnostic

- **Claim tested**: C2.
- **Access rule**: only after B1 final rubric, final descriptions, and all hashes are
  frozen.  The shared heldout-500 was previously used, so this is explicitly an
  exploratory paired diagnostic, not a new confirmatory test.
- **New predictions**: generate only the final descriptions that differ from the v2
  source; reuse the source heldout predictions for unchanged children and every other
  rubric node.  All requests use 8000.
- **Systems compared**:

| System | Purpose |
|---|---|
| Parent-only | original Visual Grounding root baseline |
| Locked-only | strong-child reference / current local upper bound |
| Split-v2 full children | fixed source before Refine |
| Split-v2 + final Refine | primary system |

- **Metrics**: Visual parent-scope Parent/Specialized ACC and coverage, full M1 ACC
  and coverage, correct count, corrected/harmed/net corrected, exact McNemar,
  child-level ACC/support/coverage, sibling conflict and overlap.
- **No post-heldout selection**: heldout results cannot select descriptions, epochs,
  children, or another retry.

## Success and Interpretation

| Outcome | Interpretation |
|---|---|
| At least one accepted local Refine and final full subtree improves vs source full-v2 on heldout | Evidence that Refine repairs part of the collective-child problem. |
| Children improve locally but full subtree is flat or worse | Self-competition is insufficient to control interaction; investigate conflict-aware diagnostics/gates later. |
| No candidate passes strict self-competition | Current evidence/prompt is inadequate for Split children; analyze required attributions before changing the operator. |
| Final full subtree remains below locked-only | Strong-child preservation remains useful, but other-child Refine has not yet recovered the ensemble penalty. |

## Execution Order

| Milestone | Stage | API use | Decision gate |
|---|---|---|---|
| M0 | Freeze + audit | none | exact source/identity/trigger match |
| M1 | Epoch 1 Refine | 4 x 90 Worker max | inspect only discovery diagnostics |
| M2 | Epoch 2–3 retry | at most 4 x 90 per epoch | stop early when no eligible/retryable child |
| M3 | Final heldout | at most 4 x 500 Worker | final descriptions frozen first |
| M4 | Final report | none | paired comparison, no selection |

## Cost and Risks

- Discovery maximum: 12 candidate descriptions x 90 = 1,080 Worker requests, plus up
  to 12 Manager generations and attributions for valid rejections.
- Heldout maximum: four final changed descriptions x 500 = 2,000 Worker requests.
- Expected wall time on 8000: roughly 1.5–3 hours, dominated by final heldout; actual
  duration depends on connection retries and concurrency.
- Main risk: local self-ACC increases through narrower applicability / `None` outputs
  while the combined vote remains harmful.  Preserve this as a diagnostic rather than
  silently changing the acceptance rule.

## Required Tests

- Source rubric/predictions, discovery hashes, and 8000 request identity freeze and
  resume without drift.
- Exactly four source children are eligible in epoch 1.
- The locked child is excluded from Split regeneration but included in Refine when it
  satisfies the trigger.
- Parent/topology/name/score/edges cannot change.
- Refine self-competition is strict, support-based, and independent per child.
- All candidates in an epoch use one identical Global Memory hash and synchronous
  commits; accepted descriptions become visible only next epoch.
- Valid rejection cannot retry without natural-language attribution.
- No heldout data can be read before M3; final heldout only refreshes changed children.
- Full subtree/M1/conflict diagnostics do not alter individual Refine decisions.

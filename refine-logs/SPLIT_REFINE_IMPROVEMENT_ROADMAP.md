# Split–Refine Improvement Roadmap

**Problem**: Split retry cannot reliably recover a useful child set, while the uniform Refine trigger leaves high-coverage, low-accuracy children without a repair operator.

**Method thesis**: Preserve useful local experts during Split retry and use role-aware Refine scheduling so that Split discovers subdomains and Refine sharpens committed children.

**Date**: 2026-08-10

## Claim map

| Claim | Minimum convincing evidence | Linked block |
|---|---|---|
| C1. Child-level failure feedback improves Split retry quality. | On the known Visual Grounding failure case, strong children are retained, sibling conflict falls, and best Specialized ACC improves without changing Split acceptance. | B1 |
| C2. Role-aware child Refine repairs useful but overly broad children missed by the uniform trigger. | On one frozen Split rubric, newly eligible children improve node ACC and do not reduce final heldout M1. | B2 |

Anti-claim to rule out: gains come only from changing several mechanisms together or selecting variants using heldout-500.

## What can be implemented together

Implement the following behind independent configuration flags in one code cycle:

1. `split_retry_feedback_mode = aggregate_v1 | child_diagnostic_v2`
2. `refine_trigger_mode = uniform_v1 | role_aware_v2`

This shares metric and artifact plumbing, but the first experiments must enable only one treatment at a time. Split voting, Specialized ACC, Refine self-competition, Worker, Global Memory, and heldout protocol remain frozen.

## Block B1 — Visual Grounding Split-retry diagnostic

- **Claim tested**: fine-grained child feedback helps Manager preserve a strong semantic expert and repair the rest of a rejected set.
- **Why first**: Visual Grounding failed five times, but an archived decomposition passes the current local gate. It is the clearest known search-recall failure and is cheaper than another five-root run.
- **Source**: discovery-90, frozen Visual Grounding parent prediction and frozen 397B ErrorSignatures.
- **Control**: replay the current aggregate failure-history trajectory.
- **Treatment**: child-level diagnostics plus strong-child preservation guidance.
- **Strong-child evidence**:
  - child ACC and parent ACC on the same child support;
  - support, coverage, corrected/harmed;
  - target/non-target activation;
  - pairwise sibling overlap/conflict;
  - leave-one-child-out change in Specialized ACC.
- **Retry instruction**:
  - identify `preserve`, `refine_description`, and `replace` children;
  - preserve exact name/description when evidence is strong;
  - regenerate weak or conflicting children;
  - still submit one complete child set for unchanged collective competition.
- **Run order**:
  1. Offline metric/artifact replay.
  2. One-seed, at most three-attempt smoke.
  3. Only if direction is positive, run three Manager seeds for stability.
- **Primary metrics**: best/final Specialized ACC on fixed parent scope, acceptance attempt, corrected/harmed, sibling conflicts.
- **Success criterion**: treatment improves best Specialized ACC over the current retry baseline and demonstrably retains at least one strong child; strongest evidence is acceptance within three attempts.
- **Heldout**: not used for proposal selection. Existing heldout artifacts may be reported only after the discovery decision and labelled exploratory.
- **Priority**: MUST-RUN.

## Block B2 — Role-aware child Refine trigger

- **Claim tested**: removing the child Coverage ceiling allows high-coverage children to become local experts through Refine.
- **Why separate**: this changes scheduler eligibility, not Split generation, and must not be confounded with B1.
- **Source**: frozen `phase6_split_only_evolution_global_memory_v1` final rubric.
- **Control trigger**:

  \[
  0.55 < Acc(c) < 0.80,\quad Cov(c) \le 0.80,\quad |S(c)| \ge 15.
  \]

- **Treatment trigger**:
  - roots keep the Control trigger;
  - children use

  \[
  0.5 < Acc(c) < 0.80,\quad |S(c)| \ge 15,\quad Wrong(c) \ge 5.
  \]

- **Protocol**: Refine-only, 2–3 synchronous epochs, same 397B Manager, Global Memory, 8001 Worker, self-competition, and failure history.
- **Primary metrics**: eligible/attempted/accepted children, node ACC, support, corrected/harmed, final discovery M1.
- **Secondary metrics**: subtree diagnostics, None transitions, support contraction, heldout M1.
- **Success criterion**: at least one newly eligible child is accepted with strict node-ACC improvement and the final heldout result does not regress materially from the frozen source.
- **Priority**: MUST-RUN after B1 smoke.

## Block B3 — Integrated Split + Refine v2

- **Claim tested**: the two individually validated changes improve the complete evolutionary trajectory.
- **Systems**:
  1. frozen current `phase7_split_refine_evolution_v1`;
  2. Split retry feedback v2 only;
  3. role-aware Refine trigger only;
  4. both changes enabled.
- **Execution economy**: reuse existing Control and targeted B1/B2 diagnostics; launch a new five-root run only for the combined system unless an isolated treatment is ambiguous.
- **Primary metric**: heldout-500 equal-root M1 ACC, reported as exploratory because heldout has been accessed previously.
- **Secondary metrics**: discovery M1, accepted operations, Visual Grounding acceptance, node/subtree alignment, cost.
- **Success criterion**: all local acceptance constraints hold; combined heldout corrected exceeds harmed relative to v1, with no material coverage loss.
- **Priority**: MUST-RUN only after B1 and B2 pass their gates.

## Block B4 — Refine-then-Split scheduling ablation

- **Claim tested**: changing an unproductive parent description before regenerating subdomains helps roots stuck after repeated Split failure.
- **Preferred treatment**: fallback after three consecutive failed Splits, not unconditional Refine-before-Split.
- **Flow**:

  ```text
  3 failed Split attempts
  -> Refine parent
  -> if Refine accepted: refresh parent predictions and decisive-wrong IDs
  -> regenerate multimodal ErrorSignatures
  -> Split with prior history tagged to the old parent version
  ```

- **Control**: continue Split retry with child-diagnostic history.
- **First target**: Visual Grounding only.
- **Success criterion**: higher Split acceptance or Specialized ACC than continued retry at comparable cost.
- **Stop rule**: if B1 already accepts a strong Visual split consistently, defer this block to the appendix.
- **Priority**: NICE-TO-HAVE / later ablation.

## Explicitly deferred

- Unconditional Refine-before-every-Split: too expensive and changes every parent before the Split mechanism is measured.
- Heldout-driven child selection or weight search: invalid for operator selection.
- Recursive child Split, Drop, or Merge: separate operators and unnecessary for the present diagnosis.
- Simultaneously changing acceptance Fitness, voting, or Worker: would destroy attribution.

## Run order and decision gates

| Milestone | Goal | Decision gate | Relative cost | Main risk |
|---|---|---|---:|---|
| M0 | Add flags, diagnostics, deterministic artifacts and tests | Control replay unchanged | Low | accidental protocol drift |
| M1 | Visual one-root B1 smoke | best Specialized ACC improves and strong-child handling is visible | Medium | one-seed Manager variance |
| M2 | Frozen-rubric B2 Refine-only test | newly eligible child accepted without final regression | Medium | support shrinkage / adaptive overfit |
| M3 | Combined five-root v2 | corrected > harmed vs v1 | High | interactions between accepted operations |
| M4 | Three-failure Refine→Split ablation | fallback beats continued retry | Medium–high | refreshed signatures increase cost |

## Verification requirements

- Control request identities and outputs remain unchanged when new modes are disabled.
- Failure attribution receives every child's metrics and per-pair sibling conflict.
- Preserved child artifacts are hash-identical and their Pairwise predictions are reused.
- Changed/refined children receive new request identities and predictions.
- Root and child trigger boundary tests cover 0.5, 0.55, 0.8, support 15, and wrong 5 exactly.
- Refine acceptance remains strict node-ACC improvement with support at least 15.
- Parent Refine acceptance invalidates Split signatures unless parent predictions and decisive-wrong IDs remain identical.
- No discovery stage reads heldout artifacts.

## Recommended immediate action

Implement M0, then run only B1 on Visual Grounding. Do not launch another full five-root evolution until the enhanced retry can show that it uses the existing strong child signal more effectively.

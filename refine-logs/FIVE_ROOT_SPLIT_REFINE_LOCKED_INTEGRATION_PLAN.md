# Five-root Locked-Split + Role-aware Refine Integration Plan

**Problem.** Split-retry v2 validated preservation of a strong child during a failed Split retry, while role-aware Refine expanded the set of repairable children. This experiment tests whether they compose into a better final five-root rubric.

**Method thesis.** Preserve a strong child in failed Split retries, then apply role-aware Refine to committed children, without changing Pairwise voting or Split competition.

**Date:** 2026-08-11

**Protocol:** `five-root-locked-split-role-aware-refine-v1`

## 1. Claims and frozen scope

| Claim | Minimum evidence |
|---|---|
| C1: locked retry preserves useful expertise | A natural retry reuses a locked child unchanged and evaluates a complete revised child set. |
| C2: integration improves final aggregation | Final heldout five-root M1 exceeds frozen Split-only + Global Memory control with positive net corrected. |

Start afresh from the five initial roots and zero children; do not continue a previous final rubric. Discovery uses the frozen 90 samples. Heldout-500 is inaccessible until final rubric, request identities, and reports are frozen.

Output: `output/evolving_structured_rubrics/rubric_evolution_phase5/phase10_five_root_locked_split_refine_v1/`.

Manager: `Qwen/Qwen3.5-397B-A17B`, seed 42, epoch-start `global_rubric_v1`. Pairwise Worker: Qwen3-VL-8B-Instruct, P05, one replicate, **8001 only**. Run at least three and at most five epochs; after epoch 3, stop only when no operation is runnable or retryable.

## 2. Shared epoch semantics

- Every candidate sees the same epoch-start committed rubric, prediction, and Global Memory snapshot. Same-epoch candidates cannot see one another.
- Accepted patches commit synchronously at epoch end. New Split children first enter Global Memory and Refine scheduling next epoch.
- A node receives at most one operator per epoch. A pending Split retry takes precedence over Refine on its parent root.
- Worker prompts contain only the criterion description and sample; examples, gold, Manager memory, and failure history never enter Worker prompts.
- Cross-root child-name collisions reject every conflicting Split candidate.

## 3. Split with locked-child retry

Split is restricted to the five initial roots and triggers when `ACC < 0.70` and `Coverage > 0.80`. The first attempt remains: `397B multimodal ErrorSignature → 397B clustering → 397B child generation → 8001 child Pairwise → parent-scope Specialized Accuracy competition`.

Children vote first on the fixed parent decisive scope; A/B ties and all-`None` fall back to the parent. Accept the complete child set exactly when `Specialized Accuracy >= Parent Accuracy`.

After a formed candidate is rejected, record each child’s parent-scope support, individual specialized ACC, corrected/harmed, net corrected, overlap, and conflict. A child is strong when `support >= 15` and `net_corrected >= 3`.

Lock at most one strong child: largest net corrected, then individual specialized ACC, then support, then lexical criterion name. On retry, preserve that child’s description and Pairwise artifact hash-identically, reuse the prior clusters, and ask the Manager to replace only unlocked children using the full diagnostic table and mandatory failure attribution. Re-evaluate the complete set with the unchanged Specialized-ACC rule.

The locked child remains a pending candidate artifact until the complete Split set succeeds. It is **not** committed alone after exhausted retries; partial Split acceptance is deliberately excluded from v1. A valid rejected Split requires a 397B natural-language attribution before retry. ErrorSignatures are reusable only when parent description, parent predictions, and decisive-wrong IDs match exactly.

## 4. Role-aware Refine

Refine changes only a committed node description. Roots retain Refine v1 trigger: `0.55 < ACC < 0.80`, `Coverage <= 0.80`, and `support >= 15`. Committed children—including a formerly Split-locked child after its full set commits—use role-aware trigger: `0.5 < ACC < 0.80`, `support >= 15`, and `wrong >= 5`.

Each selected node gets one 397B candidate and 90 new 8001 Pairwise predictions. Accept only if its own valid A/B support has strict improvement and new support is at least 15. Subtree ACC, full M1, coverage, and length are diagnostics only. A valid rejection requires 397B failure attribution before retry; a proposal-invalid event is structural history only.

## 5. Stages and artifacts

### B0 — offline freeze and audit

Freeze initial rubric, discovery data, thresholds, Manager/Worker request specs, 8001 identity, and Global Memory mode. Verify heldout isolation. Audit Split and Refine eligibility separately. Existing ErrorSignatures are read-only reusable only after item-level identity validation.

### B1 — integrated discovery evolution

For every epoch save rubric memory/hash, scheduling decisions, Split/Refine attempts, child diagnostics, locked-child decision and reuse identity, proposals, predictions, competition/self-competition, attributions, committed rubric, feedback, M1 execution, and summary. Evolution history must distinguish ordinary Split success, locked-and-retried, locked-and-accepted, no-strong-child, and exhausted-without-acceptance.

### B2 — one final heldout-500 diagnostic

After final freeze, generate 8001 predictions only for descriptions differing from reusable artifacts. Do not rerun epochs, select an epoch, or alter descriptions after heldout access.

| System | Role |
|---|---|
| Initial five roots | Initialization baseline |
| Split-only + Global Memory | Frozen component control |
| Locked-Split + Role-aware Refine final | Integrated treatment |

Report M1 ACC, correct count, coverage, Wilson CI, corrected/harmed, net corrected, and exact McNemar. Also report per-child ACC/support/coverage, `None` transitions, sibling overlap/conflict, leave-one-out subtree effect, API cost, cache reuse, and node growth. Heldout is exploratory.

## 6. Run order and gates

| Milestone | Action | Gate |
|---|---|---|
| M0 | Offline tests, freeze, audit | identities, resume behavior, and heldout isolation pass |
| M1 | Discovery epochs 1–3 | at least one complete synchronous epoch commits |
| M2 | Conditional epochs 4–5 | run only while retryable work remains |
| M3 | Discovery report/final freeze | final hash and accepted patch set fixed |
| M4 | Final heldout/report | exactly one exploratory heldout pass |

Discovery cost is dominated by first-wave Split children and early accepted children that enter Refine: budget roughly 15–30 new descriptions × 90 8001 requests, plus Split children. Final heldout cost is changed final descriptions × 500. 8001 can run independently of the 8000 Visual-local experiment, although the shared 397B Manager may serialize generation.

## 7. Acceptance checklist

- [ ] Split parent scope, fallback, and competition are unchanged.
- [ ] Locked retry preserves one child artifact exactly and does not regenerate it.
- [ ] No partial acceptance occurs after exhausted Split retries.
- [ ] Only committed children enter role-aware Refine.
- [ ] Root and child Refine triggers are distinct and auditable.
- [ ] Same-epoch commits are synchronous and order-independent.
- [ ] Retryable rejection has mandatory natural-language attribution.
- [ ] Discovery cannot read heldout; heldout is one final exploratory pass.

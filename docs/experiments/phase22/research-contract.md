# Phase22 Research Contract

**Date**: 2026-09-02\
**Experiment**: `phase22_all_sample_subtree_adaptive_recluster_evolution_v1`\
**Source plan**: [冻结协议](protocol.md)

> 冻结的实验前研究约束；实验已完成，观察结果与限制见[实验索引](../README.md)。

## Research question

When Phase21 keeps its complete-root-subtree candidate unit, triggers, budgets,
worker runtime, synchronous commit, and diagnostic-only Global Arbiter, does a
combined change to all-sample net-gain acceptance and failure-aware Split
reclustering produce a better evolution trajectory?

## Frozen intervention

For every Discovery sample $i$, let the epoch-start root prediction be $p_i^0$,
the candidate prediction be $p_i^1$, and the dataset label be $y_i$. The
sample-level gain is

$$
g_i = \mathbf{1}[p_i^1=y_i] - \mathbf{1}[p_i^0=y_i].
$$

The complete candidate root subtree is accepted if and only if

$$
G_r = \sum_i g_i > 0.
$$

`None` is scientifically wrong. An unresolved technical failure pauses the
attempt and cannot be converted into a scientific error or decision.

After a scientifically rejected Split, only
`cluster_or_decomposition_error` causes a new ClusterProposal. All other
validated scientific failure types reuse the existing clusters while
regenerating the complete child bundle. Technical retries do neither.

## Claims under test

- **C1**: The all-sample $+1/-1/0$ ledger is a valid, reproducible, common
  acceptance rule for complete-root Split and Refine candidates.
- **C2**: Failure-aware reclustering can change the Split search space when the
  validated failure is a cluster/decomposition error, without weakening atomic
  competition or increasing the Phase21 attempt budget.

## Minimum evidence

- Every candidate has a ground-truth-based all-sample ledger whose corrected,
  harmed, unchanged, `None` transitions, total gain, and digest are
  independently recomputable.
- Every accepted Split or Refine satisfies $G_r>0$; ties and losses reject.
- Every recluster action is traceable to a validated
  `cluster_or_decomposition_error`; other scientific failures preserve the
  prior cluster hash.
- Recluster attempts discard the old ClusterProposal and regenerate all
  children as one atomic bundle.
- Trigger IDs and error-signature IDs remain identical to Phase21 for an
  identical epoch-start state.
- Dev, heldout, VL-RewardBench, and Global-Arbiter diagnostics never influence
  candidate selection, retry action, or rollback.

## Anti-claims

This experiment does not claim that local optima guarantee a global optimum,
that data distribution effects disappear, that adaptive reclustering always
helps, or that Phase22-versus-Phase21 differences identify the separate causal
effect of the metric and reclustering mechanisms. Earlier fact experiments are
not selection evidence for this trajectory.

## Data isolation and decision roles

- Discovery100: candidate generation and acceptance.
- Dev150: epoch-level read-only diagnostics.
- heldout500: final-only exploratory evaluation.
- VL-RewardBench: final external benchmark; selection forbidden.
- Dataset ground truth is the only correctness target. No model output is used
  as a surrogate label.

## Completion rule

The implementation milestone is complete when focused and full offline tests
pass, a fresh-agent code review has no blocking issue, and the Phase22 sanity
stage emits auditable metric and retry artifacts. Full model-backed deployment
may begin only after that sanity gate.

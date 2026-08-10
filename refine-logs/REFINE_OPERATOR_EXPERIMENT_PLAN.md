# Refine v1 Operator Experiment Plan

## Claim and protocol

Refine v1 tests whether error-guided rewriting of one existing criterion description can improve that node's selective Pairwise accuracy without changing its identity or Rubric topology.

- Trigger: `0.55 < ACC < 0.80`, `Coverage <= 0.80`, `support >= 15`.
- Manager: `Qwen/Qwen3.5-397B-A17B` with epoch-start `global_rubric_v1` memory.
- Worker: Qwen3-VL-8B-Instruct, P05, one replicate, endpoint `vllm-8001` only.
- Candidate: one description; fixed node ID/name/score/parent/edges.
- Acceptance: `ACC(new) > ACC(old)` on each description's own A/B support, with `support(new) >= 15`.
- Diagnostics only: subtree accuracy, complete M1, coverage, corrected/harmed, overlap/conflict and scope expansion.
- A rejected valid candidate enters history only after durable 397B natural-language attribution.

## Forced smoke

Source: completed `phase6_split_only_evolution_global_memory_v1`.

Target:

```text
node_id = init_01_completeness_and_coverage__verified_existence_over_hallucinated_volume__dbf3c657ab
criterion = verified_existence_over_hallucinated_volume
```

The target is deliberately outside the automatic trigger and is recorded as `forced=true`. Run stages in order:

```text
refine-freeze
refine-smoke
refine-smoke-heldout
refine-smoke-report
```

The full experiment may start only when the report emits `go`. Heldout-500 is exploratory and cannot change the discovery decision.

## Split + Refine evolution

The full run starts again from the five initial roots. Split v1 remains frozen and only targets initial roots. Refine targets all committed nodes. Both operators see the same epoch-start Rubric and memory; accepted patches are committed synchronously. New Split children become Refine candidates only in the next epoch.

```text
split-refine-freeze
split-refine-run
split-refine-report
split-refine-heldout
split-refine-final-report
```

The experiment runs at least three and at most five epochs. The primary comparison is Initial five-root vs Split-only + Global Memory vs Split + Refine + Global Memory on heldout-500 equal-root M1.

## Artifact boundary

- Forced smoke: `output/evolving_structured_rubrics/rubric_evolution_phase5/phase7_refine_operator_v1/`
- Full evolution: `output/evolving_structured_rubrics/rubric_evolution_phase5/phase7_split_refine_evolution_v1/`
- Discovery is the sole acceptance set; heldout is inaccessible before its explicit stage.
- Existing Phase 6 outputs are read-only and never overwritten.

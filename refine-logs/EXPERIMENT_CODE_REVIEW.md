# Unified-Subtree + Global-Arbiter Aligned Evolution Code Review

**Date:** 2026-08-28
**Scope:** Phase19 aligned evolution, incremental system runtime, VL-RewardBench evaluator, and Split/Refine integration.
**Review policy:** Only issues capable of changing the scientific result or blocking the run were considered.

## Independent review findings and resolution

| Finding | Severity | Resolution |
| --- | --- | --- |
| Joint competition regenerated winner subtrees at `temperature=0.5`, mixing sampling noise into candidate interaction | Major | Fixed. Joint and leave-one-root-out evaluations now compose the already frozen candidate root reports and rerun only Global Arbiter. |
| Worker checks validated endpoint IDs but not the live model/checkpoint identity | Major | Fixed. Freeze records live endpoint identities; smoke/run/retry/heldout require exact identity replay. |
| A technical failure in the epoch-start baseline could create false `corrected` transitions | Major | Fixed. Epoch-0, epoch-start, and committed system artifacts must have zero unresolved calls before evolution state advances. |
| Refine could call Manager after system attribution removed nearly all wrong evidence | Medium | Fixed. Child Refine requires five attributed wrong samples; a root requires at least one. Insufficient evidence returns `not_eligible`. |
| Accepted/rejected transition summary had no durable `accepted` field | Medium | Fixed. Every stored system evaluation now records `accepted` and `failure_type`; focused tests cover this behavior. |

## Deliberate scope boundary

The implementation reports the failure types it can currently establish from frozen evidence (`no_system_effect`, `subtree_evidence_regression`, same-root supersession, and joint interaction regression). It does not claim a complete six-way causal taxonomy for every harmed sample.

## Verification

- Focused aligned-evolution tests: passed.
- Affected Split/Refine/Phase17 tests: passed.
- Full `unittest` discovery: 418 tests passed.
- `compileall`: passed.
- `git diff --check`: passed (line-ending warnings only).

**Review disposition:** code ready for model-backed smoke and full experiment; no known unresolved issue that should change the planned claim.

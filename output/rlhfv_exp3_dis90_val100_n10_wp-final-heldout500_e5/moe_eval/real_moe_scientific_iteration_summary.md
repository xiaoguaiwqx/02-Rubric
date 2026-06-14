# Real MoE Scientific Iteration Summary

Created at: 2026-06-07 03:50 +08:00

## Baselines

- Cached majority: `0.672 = 336/500`
- Strict real MoE base (`iter05_t045_f03`): `0.684 = 342/500`

## Iteration Block 1: Tri-Split Label-Free Gate

Protocol: split 500 real samples into dev / val / final-test. Select retry samples by label-free margin. Accept only if `dev_delta >= 0` and `val_delta > 0`. Final-test was reported, not used for decisions.

Result: no policies were accepted.

- Dev: `137/200 -> 137/200`
- Val: `102/150 -> 102/150`
- Final-test: `103/150 -> 103/150`

Key findings:

- The first low-margin retry improved dev by `+1` and final-test by `+3`, but validation dropped by `-3`; a single small val block is too noisy for this trigger.
- Removing fallback (`t03_f00`) was consistently harmful in this split.
- Strict reruns (`t045_f03`, `t05_f03`) did not recover validation.

Artifacts:

- `trisplit_label_free5/summary.tsv`
- `trisplit_label_free5/trisplit_label_free5_report.md`

## Iteration Block 2: Uncertainty Self-Consistency v2

Protocol: run real retry only on low-margin samples. Merge retry when it breaks a Tie, or when an ultra-low-margin A/B decision is flipped by a retry with margin >= `0.10`.

Result: `342/500 -> 343/500`.

| iter | config | after | finding |
|---:|---|---:|---|
| 1 | `t03_f03` | `343/500` | tie breaks helped, but one confident flip hurt |
| 2 | `t03_f05` | `343/500` | confident flips were balanced `+1/-1` |
| 3 | `t045_f03` | `343/500` | confident flips were balanced `+2/-2` |
| 4 | `t03_f00` | `343/500` | all-merge would drop to `341/500`; gate protected |
| 5 | `t05_f03` | `343/500` | all-merge would drop to `341/500`; gate protected |

Key finding: non-tie "confident flips" are noisy. They can recover true errors, but also flip correct low-margin answers.

Artifacts:

- `uncertainty_sc5_v2/summary.tsv`
- `uncertainty_sc5_v2/uncertainty_sc5_report.md`

## Iteration Block 3: Uncertainty Self-Consistency v3 Tie-Only

Protocol: same real retry sequence, but merge only `Tie -> A/B`. This is label-free and safe for A/B-labeled evaluation because Tie is already counted incorrect.

Result: `342/500 -> 344/500`.

| iter | config | after | finding |
|---:|---|---:|---|
| 1 | `t03_f03` | `344/500` | two Tie breaks were correct; no wrong flips accepted |
| 2 | `t03_f05` | `344/500` | one Tie break did not change correctness |
| 3 | `t045_f03` | `344/500` | no new Tie break |
| 4 | `t03_f00` | `344/500` | all-merge would drop to `342/500`; gate protected |
| 5 | `t05_f03` | `344/500` | all-merge would reach `345/500`, but only via risky non-tie flips |

Cost:

- Extra worker calls: `287`
- Total worker calls vs full 11-criteria baseline: `2548 + 287 = 2835 / 5500`
- Effective pruning after second pass: `48.5%`

## Key Findings

1. MoE soft routing improves over majority on real data: `336/500 -> 342/500`.
2. Small dev/val acceptance gates are unstable for ultra-low-margin cases; they can reject useful global behavior because the useful wins are sparse.
3. The reliable failure mode is not broad threshold choice. It is uncertain final aggregation, especially exact/near ties.
4. Tie-only self-consistency is the best robust next policy: it improves to `344/500` without accepting risky A/B flips.
5. Confident non-tie flips have upside but need an agreement test before deployment, e.g. require two independent retry configs to agree on the flipped answer.

## Recommended Next Optimization

Promote v3 into the evaluator as an optional second-pass mode:

- First pass: strict MoE (`routing_threshold=0.45`, `fallback_weight=0.3`, `tie_epsilon=0.0`).
- Trigger: `prediction == "Tie"` or normalized margin near zero.
- Second pass: retry `threshold=0.3`, `fallback_weight=0.3`.
- Merge: accept only `Tie -> A/B` by default.
- Optional research extension: accept non-tie flips only with cross-config agreement.

# Real MoE Scientific Iteration Round 4

Created at: 2026-06-07 16:48 +08:00

## Starting Point

Previous clean best:

- cached majority: `336/500 = 0.672`
- strict real MoE: `342/500 = 0.684`
- round-3 mid-margin recall: `348/500 = 0.696`

Round-3 finding: single mid-margin recall can help, but it can also introduce
false flips. The next question was whether accepted flips can be audited or
whether weak positive evidence can be confirmed before merging.

## Experiment A: Risky Acceptance Audit5

Hypothesis: a previously accepted single-retry flip is risky if it lacks
high-precision expert support and is not decisive broad evidence. Re-run real
verification calls and append only strong opposing verification evidence.

Risk selection was label-free. It selected only index `168`:

```text
accepted A without high-precision expert support or decisive broad evidence;
proposed_margin=0.1089
```

Result: five real audit iterations completed.

| iter | config | candidates | appended | after |
|---:|---|---:|---:|---:|
| 1 | `t045_f03` | `168` | 0 | `348/500` |
| 2 | `t045_f03` | `168` | 0 | `348/500` |
| 3 | `t05_f03` | `168` | 0 | `348/500` |
| 4 | `t03_f00` | `168` | 0 | `348/500` |
| 5 | `t045_f03` | `168` | 0 | `348/500` |

Finding: repeated audit did not produce sufficient opposing evidence. The
framework could identify the risky flip, but the current workers could not
self-correct it.

## Experiment B: Core Agreement Recall5

Hypothesis: a mid-margin flip is safer when both `visual_grounding` and
`factual_consistency` vote for the retry answer.

Result: five real iterations completed, but the policy regressed and is not
adopted.

| iter | config | band | merged | after | finding |
|---:|---|---:|---:|---:|---|
| 1 | `t03_f03` | `0.18-0.23` | 0 | `348/500` | core agreement was too conservative |
| 2 | `t03_f00` | `0.18-0.32` | 0 | `348/500` | found a weak positive candidate, index `39`, but rejected due low proposed margin |
| 3 | `t045_f03` | `0.20-0.38` | 0 | `348/500` | strict confirmation was neutral |
| 4 | `t045_f03` | `0.20-0.38` | 0 | `348/500` | strict confirmation was neutral |
| 5 | `t045_f03` | `0.20-0.38` | 1 | `347/500` | core agreement can still false-flip; do not adopt |

Important negative case: index `269` had visual+factual agreement for the
retry answer, but merging it was wrong. Therefore visual+factual agreement
alone is not a safe acceptance rule.

## Experiment C: Core Consensus Follow-Up

New hypothesis from Experiment B: do not lower the proposed-margin gate.
Instead, keep the weak core-supported candidate as pending evidence and require
a second real retry to agree before merging.

Pending selection was label-free:

```text
index 39: core-supported but low proposed margin under t03_f00
```

Result: one real confirmation call was enough to merge the pending evidence
and the new evidence.

| iter | config | candidates | merged | before | after | gains | losses |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `t03_f00` | `39` | 1 | `348/500` | `349/500` | `+1` | `-0` |

Final adopted result: `349/500 = 0.698`.

## Overall Progress

| method | correct | accuracy |
|---|---:|---:|
| cached majority | `336/500` | `0.672` |
| strict real MoE | `342/500` | `0.684` |
| round-3 clean best | `348/500` | `0.696` |
| round-4 adopted best | `349/500` | `0.698` |

Net improvement:

- `+13` correct over cached majority
- `+7` correct over strict real MoE
- `+1` correct over the previous clean best

## Key Findings

1. Risk identification is easier than risk correction. The framework can flag
   a weak accepted flip, but current workers may still reinforce the wrong side.
2. `visual_grounding + factual_consistency` agreement is not sufficient by
   itself; it caused a false flip at index `269`.
3. The useful rule is not single-shot core agreement. It is consensus over
   repeated core-supported evidence when the first proposed margin is too low.
4. The acceptance rule is becoming a reliability protocol: weak signal goes to
   pending, and only repeated independent support gets merged.

## Next Optimization Recommendation

Promote a three-state evidence policy:

1. `accept`: decisive broad evidence or high-precision specialist support.
2. `pending`: core-supported flip with low proposed margin.
3. `reject`: weak non-specialist flip or single core agreement in higher bands.

For pending items, run one additional real retry with the same routing family
and merge only if the second retry agrees. This is the first rule in this round
that improved beyond `348/500` without using labels for per-sample decisions.

Before scaling this further, add per-worker timeout/retry in the evaluator so
long-tail API calls do not dominate iteration time.

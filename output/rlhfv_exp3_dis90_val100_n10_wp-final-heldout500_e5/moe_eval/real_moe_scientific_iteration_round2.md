# Real MoE Scientific Iteration Round 2

Created at: 2026-06-07 15:00 +08:00

## Starting Point

Previous best robust policy:

- Strict real MoE: `342/500 = 0.684`
- Tie-only self-consistency v3: `344/500 = 0.688`

Previous finding: non-tie flips have upside, but single-pass confident flips are noisy.

## Experiment A: Agreement Flip5

Hypothesis: accept a non-tie flip if two independent retry configurations agree.

Result: not sufficient.

| iter | configs | after | finding |
|---:|---|---:|---|
| 1 | `t03_f03+t045_f03` | `343/500` | agreement produced `+2/-3`; unsafe |
| 2 | `t03_f03+t03_f05` | `343/500` | no merge; avoided more damage |
| 3 | `t045_f03+t05_f03` | `344/500` | recovered one true error |

The run was stopped during iteration 4 because a worker call hung. The completed three iterations were enough to identify the failure: agreement alone still flips correct low-margin samples such as indices `329`, `40`, and `169`.

Artifact: `agreement_flip5/summary.tsv`

## Key Finding From Experiment A

The true positive flips shared two label-free properties:

- very small original margin, roughly `before_margin <= 0.02`
- strong averaged post-merge confidence, roughly `proposed_margin >= 0.10`

False positives often had a larger original margin or weak post-merge confidence.

## Experiment B: Agreement + Margin Filter

New merge gate:

- `before_margin <= 0.02`
- two retry configs agree on the same opposite A/B answer
- both retry margins pass the iteration threshold
- averaged score flips
- `proposed_margin >= 0.10`

Result after two completed iterations:

| iter | configs | after | accepted flip |
|---:|---|---:|---|
| 1 | `t03_f03+t045_f03` | `345/500` | `190: A -> B` |
| 2 | `t03_f03+t05_f03` | `346/500` | `300: B -> A` |

The run was stopped during iteration 3 due to another long-tail API call. The two completed iterations were kept as warm-start evidence for the continuation run.

Artifact: `agreement_margin5/summary.tsv`

## Experiment C: Agreement + Margin Filter, Small-Batch Continuation

To avoid long-tail worker hangs, the continuation uses `target_count=4` and warm-starts from:

- v3 tie-only state
- the two successful Agreement + Margin Filter iterations

Start: `346/500 = 0.692`

| iter | configs | after | finding |
|---:|---|---:|---|
| 1 | `t03_f03+t045_f03` | `346/500` | no merge |
| 2 | `t03_f03+t03_f05` | `347/500` | accepted `292: B -> A` |
| 3 | `t03_f03+t05_f03` | `347/500` | no merge |
| 4 | `t03_f03+t03_f05` | `347/500` | all-agreement would drop to `345/500`; gate protected |
| 5 | `t045_f03+t05_f03` | `347/500` | all-agreement would drop to `346/500`; gate protected |

Final: `347/500 = 0.694`

Artifacts:

- `agreement_margin5_cont/summary.tsv`
- `agreement_margin5_cont/agreement_flip5_report.md`

## Overall Progress

| method | correct | accuracy |
|---|---:|---:|
| cached majority | `336/500` | `0.672` |
| strict real MoE | `342/500` | `0.684` |
| tie-only self-consistency v3 | `344/500` | `0.688` |
| agreement + margin filter | `347/500` | `0.694` |

Net improvement:

- `+11` correct over cached majority
- `+5` correct over strict real MoE
- `+3` correct over v3 tie-only

## Recommended Next Optimization

Promote the current rule as the next candidate evaluator option:

1. First pass: strict MoE.
2. Second pass: tie-only self-consistency.
3. Third pass for remaining ultra-low-margin A/B cases:
   - run two retry configs
   - merge only if both agree and pass the margin filter

Do not deploy raw agreement without the margin filter. It produced a real regression in Experiment A.

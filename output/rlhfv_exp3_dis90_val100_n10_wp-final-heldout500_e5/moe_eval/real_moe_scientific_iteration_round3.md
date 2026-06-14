# Real MoE Scientific Iteration Round 3

Created at: 2026-06-07 16:08 +08:00

## Starting Point

Clean label-free state from the previous round:

- cached majority: `336/500 = 0.672`
- strict real MoE: `342/500 = 0.684`
- tie-only + agreement/margin gate: `347/500 = 0.694`

The previous finding was that ultra-low-margin samples are mostly exhausted:
agreement alone is unsafe, and a margin gate is necessary.

## Experiment A: Mid-Margin Recall5

Hypothesis: remaining recoverable errors live outside the ultra-low-margin
region. Probe `0.08-0.45` margin bands with real router + active-worker calls,
and merge only high-confidence opposite retries.

Result: completed five real iterations.

| iter | config | band | merged | after | finding |
|---:|---|---:|---:|---:|---|
| 1 | `t03_f03` | `0.08-0.16` | 2 | `347/500` | one true rescue and one false rescue; single recall retry is still unsafe |
| 2 | `t045_f03` | `0.18-0.40` | 0 | `347/500` | strict-ish confirmation did not produce usable flips |
| 3 | `t05_f03` | `0.20-0.45` | 0 | `347/500` | strict rerun was safe but neutral |
| 4 | `t05_f03` | `0.20-0.45` | 1 | `348/500` | rescued index `453` with decisive broad evidence |
| 5 | `t03_f03` | `0.28-0.40` | 0 | `348/500` | expanding after the gain did not add more reliable flips |

Final: `348/500 = 0.696`.

Artifacts:

- `midmargin_recall5/summary.tsv`
- `midmargin_recall5/midmargin_recall5_report.md`
- `run_midmargin_recall5.py`

## Key Finding From Experiment A

The first iteration exposed the next bottleneck:

- index `127` was a true rescue; the retry had support from high-precision
  specialist criteria such as `calibrated_uncertainty` and
  `interpretive_accuracy_under_ambiguous_context`.
- index `168` was a false rescue; the retry was driven mostly by broad or
  lower-prior criteria.
- index `453` was a true rescue despite no specialist activation because the
  strict retry was decisive: retry margin `1.0`, proposed margin about `0.37`.

This suggests the next merge rule should be criterion-aware:

1. accept specialist-supported flips even at moderate retry margins;
2. accept broad flips only when the retry is decisive;
3. reject weak non-specialist recall flips.

## Experiment B: Specialist Recall5 Follow-Up

Hypothesis: require a flip to be supported by a high-precision specialist
criterion, or by extremely decisive broad evidence.

Completed real iterations before stopping a long-tail API call:

| iter | config | band | merged | after | finding |
|---:|---|---:|---:|---:|---|
| 1 | `t03_f03` | `0.08-0.16` | 0 | `347/500` | specialist gate prevented the previous broad false flip, but no rescue reproduced |
| 2 | `t05_f03` | `0.20-0.45` | 0 | `347/500` | strict decisive probe was neutral |
| 3 | `t05_f03` | `0.20-0.45` | 0 | `347/500` | strict decisive probe did not reproduce index `453` this time |

Iteration 4 encountered a long-tail real worker/API call and was stopped. Logs
were preserved. This reinforces the engineering need for per-worker timeout and
retry inside `MoEPairEvaluator`, not only at the outer subprocess level.

Artifacts:

- `specialist_recall5/summary.tsv`
- `specialist_recall5/RUNNING_STATUS.txt`
- `run_specialist_recall5.py`

## Overall Progress

| method | correct | accuracy |
|---|---:|---:|
| cached majority | `336/500` | `0.672` |
| strict real MoE | `342/500` | `0.684` |
| previous clean best | `347/500` | `0.694` |
| mid-margin recall5 | `348/500` | `0.696` |

Net improvement:

- `+12` correct over cached majority
- `+6` correct over strict real MoE
- `+1` correct over previous clean best

## Scientific Conclusion

The useful frontier has moved from "routing threshold tuning" to
"criterion-aware evidence acceptance." Plain repeated sampling is not enough:
some retries are confidently wrong. The next credible method should use
criterion-level support, especially high-precision specialist criteria, as part
of the acceptance rule.

## Next Optimization Recommendation

Promote a criterion-aware retry policy:

1. base pass: strict MoE;
2. uncertainty pass: tie-only and low-margin agreement gate;
3. evidence audit: reject weak non-specialist flips;
4. specialist retry: accept moderate flips only if high-precision criteria
   support the retry answer;
5. decisive broad retry: accept broad-only flips only when retry margin and
   proposed margin are both very high.

Before running larger experiments, add per-worker timeout/retry so one slow API
call cannot block an iteration.

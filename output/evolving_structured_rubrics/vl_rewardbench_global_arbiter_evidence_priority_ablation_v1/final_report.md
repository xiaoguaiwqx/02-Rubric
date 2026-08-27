# Global Arbiter Evidence/Priority Ordered Ablation

V0 and V1 use the same neutral Arbiter prompt. V2 is the exact frozen full-report factuality-first result and receives no new model calls.

| System | Strict ACC | OverallAcc | MacroAcc | Coverage | Correct |
|---|---:|---:|---:|---:|---:|
| v0_label_only_neutral | 65.76% | 65.76% | 60.86% | 100.00% | 820 |
| v1_full_report_neutral | 71.13% | 71.42% | 66.17% | 99.60% | 887 |
| v2_full_report_factuality_first | 71.13% | 71.59% | 66.24% | 99.36% | 887 |

## Registered paired comparisons

| Comparison | Delta Strict | 95% paired bootstrap CI | Corrected | Harmed | Flip rate | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_report_vs_label_only_under_neutral | +5.37% | [+3.61%, +7.14%] | 104 | 37 | 11.39% | 1.46871e-08 | 2.93742e-08 |
| factuality_first_vs_neutral_under_full_report | +0.00% | [-0.96%, +0.96%] | 19 | 19 | 3.45% | 1 | 1 |

These are ordered conditional comparisons, not independent causal contributions. Strict ACC counts final None as incorrect.

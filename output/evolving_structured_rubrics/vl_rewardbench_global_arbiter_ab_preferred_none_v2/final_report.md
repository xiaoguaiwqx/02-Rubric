# VL-RewardBench Single-Prompt Global Arbiter v2 (A/B-preferred, None-tolerant)

The Arbiter is prompted to choose A or B, while A, B, and None are all parsed as valid semantic outputs. Five same-replicate subtree reports are synthesized before final K=3.

| System | Strict ACC | OverallAcc | MacroAcc | Coverage | Correct |
|---|---:|---:|---:|---:|---:|
| s0_explicit_recursive | 70.01% | 70.29% | 64.42% | 99.60% | 873 |
| s3_unified_subtree | 66.72% | 68.42% | 62.97% | 97.51% | 832 |
| s4_global_arbiter | 67.44% | 71.64% | 66.11% | 94.15% | 841 |
| s5_legacy_ab_only_reference | 71.45% | 71.45% | 66.19% | 100.00% | 891 |
| s5_v2_global_arbiter_ab_preferred_none_tolerant | 71.13% | 71.59% | 66.24% | 99.36% | 887 |

## Paired comparisons

| Comparison | Corrected | Harmed | Net | Exact McNemar p |
|---|---:|---:|---:|---:|
| s5_vs_s4 | 54 | 8 | 46 | 1.70933e-09 |
| s5_vs_s3 | 74 | 19 | 55 | 7.72092e-09 |
| s5_vs_s0 | 62 | 48 | 14 | 0.214977 |
| s5_v2_vs_legacy_s5 | 15 | 19 | -4 | 0.607591 |

## Former S4 None subset

- Samples: 73
- S5 correct: 39
- S5 accuracy: 53.42%
- Correct needed to match S0 from the frozen S4 count: 32
- Correct needed to exceed S0 from the frozen S4 count: 33

## Efficiency

- New S5 Arbiter requests: 3741
- Full S5 system requests: 22446
- Request reduction vs S0: 77.78%
- Full-run wall time: 2085.0s
- Full-run throughput: 107.66 new requests/min

A/B/None are valid semantic outputs. OverallAcc is computed on covered A/B decisions; Strict ACC counts final None as incorrect.

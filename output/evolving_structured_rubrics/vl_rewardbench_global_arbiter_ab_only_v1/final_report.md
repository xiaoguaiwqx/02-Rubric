# VL-RewardBench Global Arbiter A/B-only v1

All 3,741 S5 arbiter requests are regenerated with an A/B-only answer space. Five same-replicate subtree reports are synthesized before final K=3.

| System | Strict ACC | OverallAcc | MacroAcc | Coverage | Correct |
|---|---:|---:|---:|---:|---:|
| s0_explicit_recursive | 70.01% | 70.29% | 64.42% | 99.60% | 873 |
| s3_unified_subtree | 66.72% | 68.42% | 62.97% | 97.51% | 832 |
| s4_global_arbiter | 67.44% | 71.64% | 66.11% | 94.15% | 841 |
| s5_global_arbiter_ab_only | 71.45% | 71.45% | 66.19% | 100.00% | 891 |

## Paired comparisons

| Comparison | Corrected | Harmed | Net | Exact McNemar p |
|---|---:|---:|---:|---:|
| s5_vs_s4 | 58 | 8 | 50 | 1.79515e-10 |
| s5_vs_s3 | 75 | 16 | 59 | 2.65172e-10 |
| s5_vs_s0 | 69 | 51 | 18 | 0.120328 |

## Former S4 None subset

- Samples: 73
- S5 correct: 37
- S5 accuracy: 50.68%
- Correct needed to match S0 from the frozen S4 count: 32
- Correct needed to exceed S0 from the frozen S4 count: 33

## Efficiency

- New S5 Arbiter requests: 3741
- Full S5 system requests: 22446
- Request reduction vs S0: 77.78%
- Full-run wall time: 2048.1s
- Full-run throughput: 109.60 new requests/min

S5 has no semantic abstention. OverallAcc equals Strict ACC and Coverage is 100%.

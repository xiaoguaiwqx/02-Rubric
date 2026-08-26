# Unified Full-Rubric Worker v1

One model call per sample/replicate. The complete Phase17 E4 rubric is used as one decision policy; no node/subtree votes, routing, Arbiter, or fallback.

## Internal K=1

| Split | System | Strict ACC | OverallAcc | MacroAcc | Coverage |
|---|---|---:|---:|---:|---:|
| discovery100 | phase17_e4_explicit_recursive | 65.00% | 66.33% | 64.94% | 98.00% |
| discovery100 | s6_unified_full_rubric | 63.00% | 63.00% | 62.92% | 100.00% |
| dev150 | phase17_e4_explicit_recursive | 73.33% | 73.83% | 73.33% | 99.33% |
| dev150 | s6_unified_full_rubric | 72.67% | 72.67% | 72.67% | 100.00% |
| heldout500 | phase17_e5_explicit_recursive_reference | 74.00% | 74.45% | - | 99.40% |
| heldout500 | s6_unified_full_rubric | 72.20% | 72.20% | - | 100.00% |

## VL-RewardBench K=3

| System | Strict ACC | OverallAcc | MacroAcc | Coverage | Correct |
|---|---:|---:|---:|---:|---:|
| s0_explicit_recursive | 70.01% | 70.29% | 64.42% | 99.60% | 873 |
| s3_unified_subtree | 66.72% | 68.42% | 62.97% | 97.51% | 832 |
| clean_s5_v2_global_arbiter | 71.13% | 71.59% | 66.24% | 99.36% | 887 |
| s6_unified_full_rubric | 62.07% | 62.07% | 59.51% | 100.00% | 774 |

## Paired VL-RewardBench comparisons

| Comparison | Corrected | Harmed | Net | Exact McNemar p |
|---|---:|---:|---:|---:|
| s6_vs_s0_explicit_recursive | 68 | 167 | -99 | 8.6162e-11 |
| s6_vs_s3_unified_subtree | 80 | 138 | -58 | 0.000103785 |
| s6_vs_clean_s5_v2_global_arbiter | 69 | 182 | -113 | 6.31339e-13 |

## VL-RewardBench category Strict ACC

| System | General | Hallucination | Reasoning |
|---|---:|---:|---:|
| s0_explicit_recursive | 49.17% | 76.50% | 66.56% |
| s3_unified_subtree | 46.96% | 72.36% | 64.67% |
| clean_s5_v2_global_arbiter | 54.14% | 77.84% | 64.98% |
| s6_unified_full_rubric | 48.07% | 63.28% | 67.19% |

## Efficiency

| System | VL-RB logical requests | Reduction vs S0 |
|---|---:|---:|
| S0 Explicit Recursive | 101,007 | - |
| S3 Unified Subtree | 18,705 | 81.48% |
| Clean S5-v2 | 22,446 | 77.78% |
| S6 Unified Full-Rubric | 3,741 | 96.30% |

VL-RewardBench main-run wall time: 1486.1s; throughput: 151.04 requests/min and 50.35 samples/min.

Unresolved technical failures: 0.
Semantic None is valid and lowers Coverage/Strict ACC; only technical failures are retried.

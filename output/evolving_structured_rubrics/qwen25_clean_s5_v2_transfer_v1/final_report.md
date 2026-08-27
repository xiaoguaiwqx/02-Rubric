# Qwen2.5 Clean S5-v2 transfer

The Phase17 epoch-4 rubric and Clean S5-v2 aggregation protocol are frozen. Only the worker model changes to Qwen2.5-VL-7B-Instruct.

## Internal K=1

| Split | Model / aggregation | Strict ACC | OverallAcc | MacroAcc | Coverage |
|---|---|---:|---:|---:|---:|
| discovery100 | qwen3_clean_s5_v2 | 65.00% | 67.71% | 65.00% | 96.00% |
| discovery100 | qwen25_clean_s5_v2 | 51.00% | 55.43% | 50.89% | 92.00% |
| dev150 | qwen3_clean_s5_v2 | 62.67% | 63.09% | 62.67% | 99.33% |
| dev150 | qwen25_clean_s5_v2 | 58.67% | 63.31% | 58.67% | 92.67% |
| heldout500 | qwen3_clean_s5_v2 | 73.40% | 73.99% | - | 99.20% |
| heldout500 | qwen25_clean_s5_v2 | 62.60% | 64.80% | - | 96.60% |

## VL-RewardBench K=3

| Model / aggregation | Strict ACC | OverallAcc | MacroAcc | Coverage | Correct |
|---|---:|---:|---:|---:|---:|
| qwen3_explicit_recursive_e4 | 70.01% | 70.29% | 64.42% | 99.60% | 873 |
| qwen3_clean_s5_v2 | 71.13% | 71.59% | 66.24% | 99.36% | 887 |
| qwen25_explicit_recursive_e4 | 56.94% | 57.07% | 52.95% | 99.76% | 710 |
| qwen25_clean_s5_v2 | 58.06% | 59.69% | 54.72% | 97.27% | 724 |

## Model-by-aggregation effect

| Worker | Explicit Recursive | Clean S5-v2 | Aggregation gain |
|---|---:|---:|---:|
| Qwen3-VL-8B | 70.01% | 71.13% | +1.12% |
| Qwen2.5-VL-7B | 56.94% | 58.06% | +1.12% |

Difference-in-differences: +0.00%.

## Paired VL-RewardBench comparisons

| Comparison | Corrected | Harmed | Net | Exact McNemar p |
|---|---:|---:|---:|---:|
| qwen25_clean_vs_qwen25_explicit | 112 | 98 | 14 | 0.369712 |
| qwen25_clean_vs_qwen3_clean | 92 | 255 | -163 | 7.56222e-19 |
| qwen3_clean_vs_qwen3_explicit | 62 | 48 | 14 | 0.214977 |

## Category Strict ACC

| Model / aggregation | General | Hallucination | Reasoning |
|---|---:|---:|---:|
| qwen3_explicit_recursive_e4 | 49.17% | 76.50% | 66.56% |
| qwen3_clean_s5_v2 | 54.14% | 77.84% | 64.98% |
| qwen25_explicit_recursive_e4 | 35.91% | 59.28% | 63.41% |
| qwen25_clean_s5_v2 | 37.57% | 62.62% | 58.99% |

## Efficiency

Internal logical requests: 4,500.
VL-RewardBench logical requests: 22,446.
Internal main-run wall time: 1330.3s; VL-RewardBench main-run wall time: 5793.7s.

Unresolved technical failures: 61.
Semantic None is a valid decision; only parse/transport failures are retried.

# Internal K=1 Single-Prompt Global Arbiter

One original-order inference per sample; no A/B swap and no replicate ensemble.

| Split | System | Strict ACC | OverallAcc | MacroAcc | Coverage | Semantic None | Technical failures |
|---|---|---:|---:|---:|---:|---:|---:|
| discovery100 | phase17_e4_explicit_recursive | 65.00% | 66.33% | 64.94% | 98.00% | 2 | 0 |
| discovery100 | implicit_subtree_equal_root | 58.00% | 63.04% | 57.93% | 92.00% | 8 | 0 |
| discovery100 | single_prompt_global_arbiter | 65.00% | 67.71% | 65.00% | 96.00% | 4 | 0 |
| dev150 | phase17_e4_explicit_recursive | 73.33% | 73.83% | 73.33% | 99.33% | 1 | 0 |
| dev150 | implicit_subtree_equal_root | 66.00% | 69.23% | 66.00% | 95.33% | 7 | 0 |
| dev150 | single_prompt_global_arbiter | 62.67% | 63.09% | 62.67% | 99.33% | 1 | 0 |
| heldout500 | phase17_e5_explicit_recursive_reference | 74.00% | 74.45% | - | 99.40% | 3 | 0 |
| heldout500 | implicit_subtree_equal_root | 73.00% | 74.80% | - | 97.60% | 12 | 0 |
| heldout500 | single_prompt_global_arbiter | 73.40% | 73.99% | - | 99.20% | 4 | 0 |

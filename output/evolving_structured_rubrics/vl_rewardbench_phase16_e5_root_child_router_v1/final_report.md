# Phase16 E5 Root Router + Child Gate

Exploratory K=3 VL-RewardBench routing evaluation. Pairwise Prompt-v2 predictions are reused; only Router/Gate calls are new.

| System | OverallAcc | MacroAcc | Coverage |
|---|---:|---:|---:|
| all_roots_all_children | 66.50% | 60.58% | 99.12% |
| root_router_only | 66.21% | 60.34% | 93.99% |
| child_gate_only | 60.59% | 56.94% | 98.48% |
| root_router_child_gate | 62.22% | 58.13% | 93.18% |

Root activation reduction: 35.86%
Child activation reduction: 74.26%
Parse valid rate: 99.98%

# VL-RewardBench Full Child-Gate v1

Exploratory K=3 transfer evaluation. Pairwise predictions are frozen; each Gate replicate is independently sampled.

| System | OverallAcc | MacroAcc | Coverage | Strict ACC |
|---|---:|---:|---:|---:|
| parent_only | 0.5812 | 0.5460 | 0.9824 | 0.5710 |
| all_children | 0.6953 | 0.6337 | 0.9896 | 0.6881 |
| full_child_gate | 0.6099 | 0.5671 | 0.9848 | 0.6006 |
| oracle_child_routing_upper_bound | 0.8121 | 0.7613 | 0.9944 | 0.8075 |
| single_root_gate::init_01_completeness_and_coverage | 0.6917 | 0.6278 | 0.9936 | 0.6872 |
| single_root_gate::init_02_visual_grounding_and_details | 0.6818 | 0.6205 | 0.9904 | 0.6752 |
| single_root_gate::init_03_factuality_no_hallucination | 0.6929 | 0.6333 | 0.9872 | 0.6840 |
| single_root_gate::init_04_creativity_and_expressiveness | 0.6815 | 0.6219 | 0.9920 | 0.6760 |
| single_root_gate::init_05_clarity_and_coherence | 0.6769 | 0.6160 | 0.9928 | 0.6720 |

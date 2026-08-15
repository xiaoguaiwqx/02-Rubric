# VL-RewardBench Prompt v2 Transfer Report

Exploratory K=3 evaluation using the frozen Phase10 rubric and order schedule.

| System | OverallAcc | MacroAcc | Coverage | Strict ACC |
|---|---:|---:|---:|---:|
| initial_five_root_equal | 0.5812 | 0.5460 | 0.9824 | 0.5710 |
| phase10_visual_only | 0.6885 | 0.6231 | 0.9679 | 0.6664 |
| phase10_final_weighted | 0.6866 | 0.6225 | 0.9928 | 0.6816 |
| phase10_final_equal | 0.6953 | 0.6337 | 0.9896 | 0.6881 |

## Historical references

| System | OverallAcc | MacroAcc | Coverage | Strict ACC |
|---|---:|---:|---:|---:|
| Prompt v1 native_vlrb_prompt | 0.5452 | 0.5356 | 0.9928 | 0.5413 |
| Prompt v1 initial_five_root_equal | 0.4504 | 0.4775 | 0.9775 | 0.4403 |
| Prompt v1 phase10_final_equal | 0.6431 | 0.5905 | 0.9864 | 0.6343 |

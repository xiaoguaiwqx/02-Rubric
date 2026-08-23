# Qwen2.5-VL Phase17 E4 Worker Transfer

Exploratory K=3 model-swap evaluation on VL-RewardBench.

| System | OverallAcc | MacroAcc | Coverage | Strict ACC |
|---|---:|---:|---:|---:|
| qwen25_native_vlrb_prompt | 0.4130 | 0.4473 | 1.0000 | 0.4130 |
| qwen25_initial_five_root_prompt_v2 | 0.4362 | 0.4498 | 0.9487 | 0.4138 |
| qwen25_phase17_epoch_04_equal | 0.5707 | 0.5295 | 0.9976 | 0.5694 |
| qwen3_native | 0.5452 | 0.5356 | 0.9928 | 0.5413 |
| qwen3_initial_five_root_prompt_v2 | 0.5812 | 0.5460 | 0.9824 | 0.5710 |
| qwen3_phase17_epoch_04_equal | 0.7029 | 0.6442 | 0.9960 | 0.7001 |

# Phase16 Checkpoint VL-RewardBench Transfer Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Freeze and audit | Phase16 epoch 2 + epoch 3 | VL-RewardBench 1,247 | hashes, K=3 schedule, request identity | MUST | PASSED | 9 unique historical description versions; K=3 balance 624 ABA / 623 BAB. |
| R002 | M1 | Smoke | Epoch 2 + epoch 3 | First 20 frozen records | parse validity, endpoint use, consistency | MUST | PASSED | 27 variant×replicate shards completed; both checkpoint composites have 100% coverage. |
| R003 | M2 | External diagnosis | Epoch 2 + epoch 3 | Full VL-RewardBench K=3 | OverallAcc, MacroAcc, category ACC | MUST | TODO | Reuse only identical historical descriptions. |
| R004 | M3 | Repair/report | Epoch 2 + epoch 3 | Retry shards | failed count, paired corrected/harmed, McNemar | MUST | TODO | Compare each primarily to epoch 5. |

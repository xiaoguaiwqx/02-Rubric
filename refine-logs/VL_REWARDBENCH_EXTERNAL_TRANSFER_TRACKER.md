# VL-RewardBench External Transfer Experiment Tracker

| Run ID | Stage | System | Scope | Metric / Gate | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| VLRB-001 | freeze | all | 1,247 pairs, K=3 schedule | hashes, response mapping, port identity | MUST | TODO | overlap audit explicitly skipped |
| VLRB-002 | smoke | native / initial / epoch-1 | 20 deterministic pairs x 3 orders | image, parser, remapping, M1 replay | MUST | TODO | no metric selection |
| VLRB-003 | run | native prompt | 1,247 x 3 | strict ACC, parser validity | MUST | TODO | official prompt, T=0.2, top-p=0.2 |
| VLRB-004 | run | initial five-root M1 | 1,247 x 3 x 5 nodes | strict ACC, coverage | MUST | TODO | frozen P05 Worker |
| VLRB-005 | run | role-aware epoch 1 M1 | 1,247 x 3 x 17 nodes | strict ACC, coverage | MUST | TODO | frozen 17-node checkpoint |
| VLRB-006 | report | all | completed artifacts | macro, paired corrected/harmed, McNemar | MUST | TODO | no post-benchmark selection |

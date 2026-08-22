# Phase17 Checkpoint VL-RewardBench Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R001 | M0 | Freeze checkpoints and controls | Phase17 E2/E3/E4; E5; Phase10 | Offline | hashes, node/version counts | MUST | PASSED | 1,247 records; E2/E3/E4 frozen; 18 unique historical descriptions |
| R002 | M1 | Audit exact reuse | 18 historical description variants | Offline | reuse count, generated count, E5 reconstruction | MUST | PASSED | K=3 schedule 624 ABA / 623 BAB; E5 logical votes reconstructed exactly |
| R003 | M2 | End-to-end smoke | E2/E3/E4 | First 20 frozen VLRB records, K=3 | parse validity, composite completeness, endpoint calls | MUST | PASSED | Composite inference and exact description-keyed reuse completed |
| R004 | M3 | Full external diagnosis | E2/E3/E4 | VL-RewardBench 1,247, K=3 | OverallAcc, MacroAcc, Coverage, Strict ACC | MUST | PASSED | E2/E3/E4 OverallAcc 67.45% / 67.74% / 70.29% |
| R005 | M4 | Repair technical failures | Failed variant requests only | Retry manifest | recovered, unresolved, retry calls | MUST | PASSED | Recovered 63/64 failed requests with 97 new model calls; one unresolved criterion-level prediction |
| R006 | M4 | Final paired report | E2/E3/E4/E5/Phase10 | Full benchmark | corrected, harmed, net, McNemar, category/root results | MUST | PASSED | E4 was the highest observed checkpoint; +15 net correct vs Phase10, p=0.142 |
| R007 | Optional | Complete epoch curve | Phase17 E1 | Full benchmark, K=3 | same metrics | NICE | DEFERRED | Run only if full E0–E5 figure is needed |

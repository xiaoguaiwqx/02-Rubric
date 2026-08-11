# Role-aware Refine Checkpoint Heldout Tracker

| Run ID | Milestone | Purpose | Systems | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| RARC-M0-01 | M0 | Freeze three checkpoint identities and changed-node map | S0 / S1 / S3 | heldout-500 metadata only | hashes, 9 unique descriptions | MUST | DONE | Frozen 2026-08-10; 9 nodes / 4,500 requests; no API requests |
| RARC-M1-01 | M1 | Generate only missing description predictions | S1 / S3 over S0 cache | heldout-500 | valid rate, request count, time | MUST | DONE | 4,500 requests completed; checkpoint artifacts replayed successfully |
| RARC-M2-01 | M2 | Paired checkpoint report | S1 vs S0; S3 vs S0; S3 vs S1 | heldout-500 | ACC, Coverage, Wilson CI, corrected/harmed, McNemar | MUST | TODO | Exploratory; no post-heldout selection |
| RARC-M3-01 | Follow-up | One-accept-per-node scheduler ablation | conditional | discovery then heldout | trajectory stability | CONDITIONAL | TODO | Only if S1 > S0 and S3 < S1 |

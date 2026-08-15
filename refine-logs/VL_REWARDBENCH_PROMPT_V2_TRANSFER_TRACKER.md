# VL-RewardBench Prompt v2 Transfer Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Requests | Metrics | Priority | Status | Notes |
|---|---|---|---|---:|---|---|---|---|
| VLRB-PV2-001 | M0 | Freeze protocol | dataset + K=3 + Prompt v2 + Phase10 | 0 | hashes, identities | MUST | DONE | 1,247 rows、22 nodes、82,302 requests frozen |
| VLRB-PV2-002 | M0 | Offline audit | 5-root subset and four aggregations | 0 | node/criterion equality | MUST | DONE | 5 roots exact；624 ABA / 623 BAB；no inference |
| VLRB-PV2-003 | M1 | End-to-end smoke | 20 pairs × 22 nodes × K=3 | 1,320 | validity, mapping, dual endpoint | MUST | DONE | valid=99.85%；8000/8001 calls=638/693 |
| VLRB-PV2-004 | M2 | Full inference | Phase10 22 nodes, Prompt v2 | 82,302 | completion, latency, tokens | MUST | TODO | available-slot dynamic pool |
| VLRB-PV2-005 | M3 | Technical retry | failed/invalid requests only | variable | recovered, unresolved | MUST | TODO | max10, no gold access |
| VLRB-PV2-006 | M4 | Evolution comparison | Final equal v2 vs Initial v2 | 0 | Overall/Macro/strict, paired | MUST | TODO | primary result |
| VLRB-PV2-007 | M4 | Prompt comparison | Final v2 vs Final v1 | 0 | delta, McNemar, order gap | MUST | TODO | supporting result |
| VLRB-PV2-008 | M4 | Offline variants | Visual-only and weighted v2 | 0 | Overall/Macro/Coverage | SHOULD | TODO | no extra inference |
| VLRB-PV2-009 | M4 | Efficiency report | Prompt v2 runtime | 0 | req/min, P50/P90/P99 | SHOULD | TODO | descriptive comparison |

# VL-RewardBench Phase10 External Transfer Tracker

| Run ID | Milestone | Purpose | System / Scope | Metric / Gate | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| VLRB2-001 | M0 | Freeze latest protocol | 1,247 rows, Phase10 hash, K=3, weights | all hashes and mappings exact | MUST | TODO | new output directory |
| VLRB2-002 | M0 | Freeze dual endpoints | 8000 + 8001 identities and deterministic shards | compatible models, near 1:1 schedule | MUST | TODO | no random round-robin |
| VLRB2-003 | M0 | Offline audit | labels, duplicate IDs, prompt leakage, aggregation | audit passed | MUST | TODO | no model calls |
| VLRB2-004 | M1 | End-to-end smoke | 20 rows × 3 orders × native/22 nodes | complete, resumable, valid mapping | MUST | TODO | about 1,400 requests |
| VLRB2-005 | M1 | Endpoint parity diagnostic | 20 structured requests on both endpoints | transport/schema valid on both | MUST | TODO | vote agreement diagnostic only |
| VLRB2-006 | M2 | Native baseline | 1,247 × K=3 | strict ACC, parser validity | MUST | TODO | deterministic dual-endpoint shards |
| VLRB2-007 | M3 | Phase10 node inference | 22 × 1,247 × K=3 | all criterion/endpoint shards complete | MUST | TODO | shared by equal and weighted |
| VLRB2-008 | M4 | Initial replay | exact five-root prediction reuse | rubric/request identity exact | MUST | TODO | regenerate on mismatch |
| VLRB2-009 | M4 | Final aggregation | Initial/equal/weighted/native | ACC, Coverage, paired tests | MUST | TODO | no inference in report |
| VLRB2-010 | M4 | Frozen diagnostics | source groups, roots, Visual-only, order bias | report-only | MUST | TODO | no model selection |

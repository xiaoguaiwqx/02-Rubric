# Pairwise Worker Cache-oriented Prompt Ablation Tracker

| Run ID | Milestone | Purpose | Variant | Split | Key Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| PCACHE-001 | M0 | Prompt v2 template/hash audit | S2 | offline | template hash, block order, no dataset name | MUST | DONE | real freeze/audit passed; heldout not accessed |
| PCACHE-002 | M0 | Scheduler determinism and endpoint affinity | S1/S2 | offline | route hash, seed-before-fanout invariant | MUST | DONE | 1,980 routes; each sample stays on one endpoint |
| PCACHE-003 | M0 | Control compatibility | S0 | offline | byte-identical prompt/request spec | MUST | DONE | v1 unchanged; v1/v2 identities distinct; full unittest 249/249 passed |
| PCACHE-004 | M1 | Control smoke | S0 | discovery-10 | valid rate, requests/min, TTFT | MUST | TODO | fresh cache |
| PCACHE-005 | M1 | Scheduler-only smoke | S1 | discovery-10 | cache hit, affinity, seed timing | MUST | TODO | same prompt as S0 |
| PCACHE-006 | M1 | Prompt-v2 smoke | S2 | discovery-10 | cache hit, valid rate, corrected/harmed | MUST | TODO | criterion-last |
| PCACHE-007 | M2 | Control baseline | S0 | discovery-90 | M1, cache, TTFT, wall time | MUST | TODO | 1,980 logical requests |
| PCACHE-008 | M2 | Scheduler isolation | S1 | discovery-90 | delta vs S0 | MUST | TODO | 1,980 logical requests |
| PCACHE-009 | M2 | Full cache-oriented protocol | S2 | discovery-90 | efficiency + quality non-inferiority | MUST | TODO | 1,980 logical requests |
| PCACHE-010 | M3 | Fresh heldout control | S0 | heldout-500 | M1, cache, TTFT, wall time | MUST | TODO | 11,000 logical requests |
| PCACHE-011 | M3 | Prompt-v2 heldout | S2 | heldout-500 | efficiency, M1, corrected/harmed | MUST | TODO | exploratory |
| PCACHE-012 | M4 | System-role/order diagnosis | conditional | discovery/heldout | isolate source of quality shift | NICE | BLOCKED | run only if S2 shifts quality |
| PCACHE-013 | M4 | Final report | all | all | decision and provenance | MUST | TODO | adopt v2/scheduler/none |
| PCACHE-014 | M2 | Prompt-v2 with original scheduler | S3 | discovery-90 | wall time, effective concurrency, M1, corrected/harmed | MUST | FROZEN | additive extension; reuses frozen S0/S2 only for comparison |
| PCACHE-015 | M3 | S3 semantic-drift and speed diagnostic | S0/S3 | heldout-500 | M1, coverage, valid rate, corrected/harmed, node deltas, wall time | MUST | READY | 22,000 logical requests; exploratory reused heldout |

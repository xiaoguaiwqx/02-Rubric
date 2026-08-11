# Five-root Locked-Split + Role-aware Refine Integration Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Freeze protocol and source identities | `phase10_five_root_locked_split_refine_v1` | discovery-90 | hashes, trigger audit, heldout isolation | MUST | TODO | 8001 worker; no online predictions |
| R002 | M0 | Offline regression | locked retry + role-aware Refine | offline | tests, cache/reuse, synchronous commit | MUST | TODO | Protect existing Split/Refine protocols |
| R003 | M1 | Integrated evolution epochs 1–3 | five initial roots | discovery-90 | Split/Refine decisions, M1, local diagnostics | MUST | TODO | Global Memory; maximum five epochs |
| R004 | M2 | Conditional continuation | retryable roots/nodes only | discovery-90 | same as R003 | MUST | TODO | Only if work remains after epoch 3 |
| R005 | M3 | Freeze discovery result | final committed rubric | discovery-90 | final hash, report completeness | MUST | TODO | No heldout access |
| R006 | M4 | Final integrated comparison | initial / split-only-memory / integrated final | heldout-500 | M1, coverage, corrected/harmed, McNemar | MUST | TODO | One-pass exploratory diagnostic |

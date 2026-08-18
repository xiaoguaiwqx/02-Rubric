# Root Boundary Pre-Refine → Split+Refine Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| RB-001 | M0 | Protocol freeze | `root-boundary-pre-refine-split-refine-v3` | discovery-90 | hashes, request identities, heldout isolation | MUST | DONE | Dual endpoint pool and max_tokens=2048 frozen |
| RB-001B | M0 | Capped five-root baseline | Initial five roots | discovery-90 | 450 Pairwise, valid rate, endpoint counts | MUST | DONE | 450/450 valid in 142.6s; endpoint calls 221/230 including one retry |
| RB-002 | M0 | Eligibility and baseline audit | Initial five roots | discovery-90 | per-root ACC/support/Coverage, overlap, conflict | MUST | DONE | Five roots verified; heldout not accessed |
| RB-003 | M0 | Focused regression | Root Refine + Split/Refine integration | offline | trigger bypass, schema, sync commit, resume | MUST | DONE | 296 full tests passed; Gate remains disabled |
| RB-003S | M0 | Online sanity | one forced root, diagnostic only | discovery-90 | proposal, 90 Pairwise, self-competition | MUST | DONE | 90/90 valid; endpoint calls 49/43; capped identity exact |
| RB-004 | M1 | Root boundary epoch 1 | forced Refine for all five roots | discovery-90 | self-ACC, support, `None` transitions | MUST | PAUSED | User handoff after roots 1–2 completed/rejected with attribution; root 3 evidence staged |
| RB-005 | M1 | Root boundary retry epochs 2–3 | rejected roots only | discovery-90 | attribution, proposal change, self-ACC | MUST | TODO | Accepted roots locked |
| RB-006 | M2 | Root-only checkpoint report | Initial vs Root-pre-refined | discovery-90 | M1, overlap/conflict, selective abstention | MUST | TODO | No heldout access |
| RB-007 | M3 | Integrated epochs 1–3 | Root-pre-refined → Locked-Split + Refine | discovery-90 | operator trajectory, M1, node growth | MUST | TODO | Frozen Phase 10 semantics |
| RB-008 | M3 | Conditional epochs 4–5 | retryable operations only | discovery-90 | accepted/rejected/retry history | MUST | TODO | Early stop after epoch 3 allowed |
| RB-009 | M3 | Final discovery freeze | Treatment final rubric | discovery-90 | final M1, root/subtree metrics, hashes | MUST | TODO | Freeze heldout manifest next |
| RB-010 | M4 | Heldout comparison | Initial / Root-only / Phase 10 / Treatment | heldout-500 | ACC, Coverage, corrected/harmed, McNemar | MUST | TODO | One exploratory access; no reselection |
| RB-011 | M5 | Prompt-v2 deployment diagnostic | Phase 10 vs Treatment | heldout-500 | same-prompt ACC and paired deltas | NICE | TODO | Run only after main freeze |
| RB-012 | M5 | Boundary prompt ablation | ordinary Refine vs boundary-aware Refine | discovery-90 | accepted roots, overlap/conflict, M1 | NICE | TODO | Only if main treatment is positive |
| RB-013 | M5 | Final report | All frozen systems | discovery + heldout | claim table, costs, failure analysis | MUST | TODO | Separate operator and prompt effects |

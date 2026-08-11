# Visual Grounding Split-to-Refine Local Evolution Tracker

| Run ID | Milestone | Purpose | System / variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| VSR-R001 | M0 | Freeze source and audit eligibility | Split-v2 final Visual subtree | discovery-90 | hashes, four trigger decisions | MUST | TODO | No API / no heldout |
| VSR-R002 | M1 | First child-only Refine epoch | four eligible children, synchronous commit | discovery-90 | child self-ACC, support, subtree diagnostics | MUST | TODO | Worker via 8000 |
| VSR-R003 | M2 | Retry / continued Refine | epoch 2–3, max three total epochs | discovery-90 | accepted/rejected, attribution, conflict | MUST | TODO | Early-stop allowed |
| VSR-R004 | M3 | Final frozen-rubric diagnostic | parent-only, locked-only, source full-v2, final Refine | heldout-500 | ACC, coverage, corrected/harmed, McNemar | MUST | TODO | Exploratory paired use only |
| VSR-R005 | M4 | Report and diagnosis | final experiment report | discovery + heldout | child/subtree/M1 trajectory | MUST | TODO | No post-heldout selection |

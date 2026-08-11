# Visual Grounding Split-retry v2 Heldout Diagnostic Tracker

| Run ID | Milestone | Purpose | Systems | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| VSR2H-M0-01 | M0 | Freeze v2 children, source pairwise artifact and request identity | P / L / F | heldout metadata only | hashes, 4 child descriptions, request count | MUST | DONE | Frozen 2026-08-10; 2,000 requests; no API calls |
| VSR2H-M1-01 | M1 | Evaluate exactly four frozen children | L / F over P cache | heldout-500 | valid rate, child ACC/support, runtime | MUST | TODO | 2,000 8000 requests; resume via cache |
| VSR2H-M2-01 | M2 | Produce local subtree and full-M1 paired report | L vs P; F vs P; F vs L | heldout-500 | ACC, coverage, Wilson CI, corrected/harmed, McNemar, conflicts | MUST | TODO | Exploratory; no post-heldout selection |

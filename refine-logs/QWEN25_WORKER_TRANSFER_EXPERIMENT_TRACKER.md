# Qwen2.5-VL Worker Transfer Experiment Tracker

| Run ID | Milestone | System | Requests | Metrics | Priority | Status |
| --- | --- | --- | ---: | --- | --- | --- |
| R001 | Freeze/Audit | Qwen2.5 endpoints + Phase17 E4 + historical Qwen3 controls | 0 | 1,247 samples；27 nodes；E4/root projection/dual endpoints passed | MUST | DONE |
| R002 | Smoke | Qwen2.5 Native + E4, first20, K=3；397B parser probe | ≤1,681 | parse validity, endpoint usage, wall time | MUST | TODO |
| R003 | Structured run | Qwen2.5 Phase17 E4; derive Initial offline | 101,007 | Overall/Macro/Strict, categories, roots | MUST | TODO |
| R004 | Native run | Qwen2.5 VL-RewardBench native prompt + regex parser | 3,741 | Overall/Macro/Strict、首次解析率 | MUST | TODO |
| R005 | Native recovery | 397B text-only fallback；仍失败时原请求最多重跑10次 | variable | fallback/retry recovered、unresolved | MUST | TODO |
| R006 | Final report | Qwen2.5 internal + Qwen2.5/Qwen3 comparison | 0 | paired McNemar, model/rubric deltas | MUST | TODO |
| R007 | Future | Qwen2.5-specific full Split+Refine evolution | TBD | Discovery/Dev/heldout/VLRB | DEFERRED | TODO |

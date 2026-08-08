# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M1 | 生成 treatment signatures | Qwen3.5-397B multimodal | frozen 27 wrong samples | valid rate, provenance | MUST | DONE | `split-signature-qwen35-generate` |
| R002 | M2 | paired quality audit | 8B vs 397B | same 27 samples | direction, facts, consistency, actionability | MUST | DONE | `split-signature-qwen35-report` 后填写模板 |
| R003 | M3 | 重新聚类 | 397B signatures | discovery-90 wrong domain | cluster coherence | MUST | DONE | `split-signature-qwen35-cluster` |
| R004 | M3 | 重新生成 children | new clusters/signatures | discovery-90 | applicability quality | MUST | DONE | `split-signature-qwen35-propose` |
| R005 | M3 | Pairwise/Fitness | full treatment | parent domain | collective Fitness, overlap, target/non-target | MUST | DONE | `split-signature-qwen35-evaluate`; 当前严格对照先保持1 replicate |
| R006 | M3 | final comparison | control vs treatment | discovery-90 | M1 ACC, corrected/harmed | MUST | DONE | `split-signature-qwen35-compare`; 不访问 heldout |
| R007 | M4 | 一次性 Visual subtree 泛化 | parent/all/spatial+direct/full-M1 | heldout-500 | ACC, coverage, CI, McNemar, activation | MUST | TODO | `split-signature-qwen35-heldout-visual`; 仅8001；报告后禁止调参 |

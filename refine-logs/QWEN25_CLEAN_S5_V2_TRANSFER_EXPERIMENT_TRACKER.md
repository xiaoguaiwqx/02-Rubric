# Qwen2.5-VL Clean S5-v2 跨模型聚合实验追踪表

| Run ID | Milestone | Purpose | System / Variant | Split | Logical requests | Priority | Status | Notes |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| Q25S5-001 | M0 | 冻结 Phase17 E4、Prompt、数据、schedule 和 Qwen2.5 endpoints | Offline | all | 0 | MUST | TODO | 变量只有 backbone |
| Q25S5-002 | M0 | 校验 Qwen3/Qwen2.5 Controls 与 request identity | Offline | all | 0 | MUST | TODO | Qwen3 inference reuse=0 |
| Q25S5-003 | M1 | 验证5子树→Arbiter完整 bundle | Qwen2.5 Clean S5-v2 | smoke | bounded | MUST | TODO | 两个 endpoint，技术 gate |
| Q25S5-004 | M2 | 内部演化数据诊断 | Qwen2.5 Clean S5-v2 | Discovery100 | 600 | MUST | TODO | K=1，无 swap |
| Q25S5-005 | M2 | 内部多源泛化诊断 | Qwen2.5 Clean S5-v2 | Dev150 | 900 | MUST | TODO | K=1，无 swap |
| Q25S5-006 | M2 | RLHF-V 历史 heldout 诊断 | Qwen2.5 Clean S5-v2 | heldout500 | 3,000 | MUST | TODO | K=1，无 swap |
| Q25S5-007 | M3 | 外部主要评测 | Qwen2.5 Clean S5-v2 | VL-RewardBench | 22,446 | MUST | TODO | K=3 counterbalanced |
| Q25S5-008 | M4 | 恢复技术失败 | Same Prompt retry | all | variable | MUST | TODO | 最多额外10次，无 rescue |
| Q25S5-009 | M4 | 生成2×2、类别、推翻、位置和效率报告 | Qwen2.5/Qwen3 × Explicit/S5 | all | 0 | MUST | TODO | Strict ACC为主要指标 |

## Implementation status

- 2026-08-27：实验协议与实施计划已冻结为草案。
- 2026-08-27：已实现独立 runner、配置块、CLI stages、缓存隔离、同 Prompt
  技术重试、内部/VL-RewardBench 报告与 focused tests。
- 离线 manifest 审计确认 Phase17 E4 Rubric hash 为
  `007bd32557007aebc6f952cc63a49135f262648aff8943fbe99addecb4674b4d`，
  请求预算为 internal=4,500、VL-RewardBench=22,446、total=26,946。
- 已通过 399 个完整 `unittest` 与 compile checks；正式模型推理尚未启动。
- 预计总逻辑请求数：26,946；预计总墙钟时间：3–5小时。

# Unified Full-Rubric Worker v1 实验追踪表

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| UFR-001 | M0 | 冻结 Rubric、Prompt、数据与 K=3 schedule | Unified Full-Rubric v1 | offline | hashes / node coverage | MUST | COMPLETED | Phase17 E4，5 roots / 27 nodes |
| UFR-002 | M0 | 校验对照复用身份和 heldout 隔离 | S0 / S3 / Clean S5-v2 | offline | provenance checks | MUST | COMPLETED | 对照只读复用，无模型请求 |
| UFR-003 | M1 | 验证完整 Rubric Prompt、图片和解析 | Unified Full-Rubric v1 | smoke20 | parse rate / latency / endpoints | MUST | COMPLETED | 两个 endpoint 均完成请求 |
| UFR-004 | M2 | 内部演化数据诊断 | Unified Full-Rubric v1 | Discovery100 | Strict / Coverage / domain | MUST | COMPLETED | 63.00% Strict，100% Coverage |
| UFR-005 | M2 | 内部独立开发集诊断 | Unified Full-Rubric v1 | Dev150 | Strict / Coverage / domain | MUST | COMPLETED | 72.67% Strict，100% Coverage |
| UFR-006 | M2 | RLHF-V 历史 heldout 诊断 | Unified Full-Rubric v1 | heldout500 | Strict / Coverage | MUST | COMPLETED | 72.20% Strict，100% Coverage |
| UFR-007 | M3 | 外部主消融 | Unified Full-Rubric v1 | VL-RewardBench | Strict / Overall / Macro / Coverage | MUST | COMPLETED | 62.07% Strict，59.51% Macro，100% Coverage |
| UFR-008 | M4 | 恢复技术失败 | Same Prompt retry | all | recovered / unresolved | MUST | COMPLETED | 初次38个失败全部恢复，无 rescue |
| UFR-009 | M4 | 生成配对、类别和效率报告 | S0 / S3 / Clean S5-v2 / S6 | all | paired tests / time / tokens | MUST | COMPLETED | 4,491个逻辑请求，最终解析率100% |

## Implementation preflight

- 2026-08-26：runner、配置、CLI、缓存/恢复、进度文件和报告链路已实现。
- 2026-08-26：focused tests 7/7、相关聚合回归 tests 27/27、完整 unittest 393/393 通过。
- 2026-08-27：内部 K=1 与 VL-RewardBench K=3 全部完成；同 Prompt retry 后未解决技术失败为0。
- 主要结论：S6 将 VL-RewardBench 逻辑请求降至3,741次，但 Strict ACC 为62.07%，显著低于 S0 的70.01%和 Clean S5-v2 的71.13%，因此作为“完整 Rubric 文本不能替代结构化执行”的负消融保留。

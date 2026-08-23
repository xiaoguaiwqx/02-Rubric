# Qwen2.5-VL Worker Transfer Experiment Plan

**Problem**: 已演化 Rubric 的收益是否依赖 Qwen3-VL-8B-Instruct，还是能够迁移到较早的 Qwen2.5-VL-7B-Instruct Worker？

**Method thesis**: 固定数据、Prompt、顺序、Rubric 和聚合，仅替换 Pairwise Worker，可以把 backbone 能力差异与 Rubric 的跨 Worker 泛化能力分开测量。

**Date**: 2026-08-23

## Claim Map

| Claim | Minimum convincing evidence |
| --- | --- |
| C1：结构化 Rubric 能跨 Worker 迁移 | Qwen2.5 + Phase17 E4 显著优于 Qwen2.5 Initial five-root |
| C2：收益不只是 Qwen3 backbone 带来的 | Qwen2.5 上仍有正向 corrected/harmed 与 Overall/Macro/Strict ACC 增益 |
| Anti-claim：比较中混入了 Prompt、顺序或聚合变化 | 所有非模型身份完全冻结，并复用同一 K=3 schedule |

## Block 1：严格 Worker model swap

### 固定项

- 数据：VL-RewardBench 完整1,247对；
- 顺序：沿用既有 seed=42、K=3 counterbalanced A/B schedule；
- Structured Prompt：Pairwise Worker Prompt v2 `1.1.0-cache-pilot`；
- decoding：`temperature=0.5`、`max_tokens=2048`；
- 聚合：root 内 subtree aggregation 不变，五个 roots 等权 M1；
- parser、技术重试和 OverallAcc/MacroAcc/Strict ACC 口径不变；
- 不使用 Gate、Root Router、权重搜索或 benchmark 反馈。

### Native 输出解析与重试协议

Qwen2.5 Native 必须严格复现历史 Qwen3 Native 的最终评测链路，避免把模型差异与输出恢复策略差异混在一起：

1. 使用完全相同的 VL-RewardBench 原生 judge prompt 和 K=3 顺序；
2. 首先使用 `vl_rewardbench_overall_judgment_regex_v1` 从原始回答解析偏好；
3. 正则无法解析时，使用 `Qwen/Qwen3.5-397B-A17B` 作为仅文本 fallback parser，根据原始 judge 回答识别最终偏好；保持历史协议的最多3次 parser 尝试，不得向 parser 提供图像、候选回答或 gold；
4. fallback parser 仍无法恢复时，使用同一 Native prompt、图像、候选回答、A/B 顺序和 decoding 重新推理，最多10次，并在首次得到可解析偏好时停止；
5. 最终报告同时记录首次解析成功率、397B parser 恢复数、重新推理恢复数、剩余技术失败数和最终 Coverage；
6. Native 的最终逻辑票应与历史协议一致，以完成上述恢复后的 `combined/logical_votes.json` 为准。

Native 仅作为裸 judge 能力的辅助基线；主要因果比较仍是共享 Prompt v2、顺序、Worker 与模型调用的 `Qwen2.5 Phase17 E4` 对 `Qwen2.5 Initial five-root`。

### 唯一 Treatment 变化

- Worker：`Qwen/Qwen2.5-VL-7B-Instruct`；
- endpoints：`vllm-8000 + vllm-8001` available-slot pool；
- 两端 `max_model_len=25800`，freeze 时记录实际 checkpoint root、vLLM version 和完整 endpoint identity。

### Rubric

- 主 Treatment：Phase17 Epoch 4 committed Rubric；
- canonical rubric SHA256：`007bd32557007aebc6f952cc63a49135f262648aff8943fbe99addecb4674b4d`；
- 结构：5 roots、22 children，共27 nodes；
- 选择原因：E4 是五个 roots 首次全部完成 Split 的 checkpoint，并在既有 Qwen3 Worker 上达到当前最高 VL-RewardBench OverallAcc 70.29%。

### 比较系统

| 系统 | Worker | Prompt / Rubric | 作用 |
| --- | --- | --- | --- |
| Qwen2.5 Native | Qwen2.5-VL-7B | VL-RewardBench 原生 judge prompt | 测量 Qwen2.5 裸 judge 能力 |
| Qwen2.5 Initial five-root | Qwen2.5-VL-7B | Prompt v2 + 初始5 roots | Qwen2.5 结构化基线 |
| Qwen2.5 Phase17 E4 | Qwen2.5-VL-7B | Prompt v2 + E4完整27 nodes | 主要 Treatment |
| Qwen3 Native | Qwen3-VL-8B | 历史结果只读复用 | backbone 对照 |
| Qwen3 Initial five-root | Qwen3-VL-8B | 历史结果只读复用 | 严格 model-swap 对照 |
| Qwen3 Phase17 E4 | Qwen3-VL-8B | 历史结果只读复用 | 最佳 Rubric 对照 |

Qwen2.5 E4 只生成一套27-node predictions；Initial five-root 从相同预测中只取五个 root 离线重算，保证两个 structured 系统共享完全相同的模型调用。Native 需要独立运行原生 prompt。

### 指标

Primary：

- Qwen2.5 E4 vs Qwen2.5 Initial 的 OverallAcc、MacroAcc、Strict ACC；
- paired corrected、harmed、net corrected、exact McNemar；
- General、Hallucination、Reasoning covered accuracy。

Secondary：

- Qwen2.5 vs Qwen3 在 Native、Initial、E4 三个对应系统上的差值；
- Coverage、K=3 order disagreement、每个 root subtree ACC；
- 请求数、重试数、端点负载和 wall time。

### 解释规则

- E4 在 Qwen2.5 上明显优于 Initial：Rubric 具有跨 Worker 泛化；
- E4 有收益但小于 Qwen3：Rubric 可迁移，但部分规则与 Qwen3 的视觉/推理能力耦合；
- E4 与 Initial 持平：Rubric 的可执行收益依赖 Worker 能力；
- E4 下降：不能直接迁移，应为 Qwen2.5 重新演化；
- Qwen2.5 绝对分数低于 Qwen3 不构成失败，核心问题是同一 Qwen2.5 内部的 Rubric 增益。

## Execution stages

```text
vlrb-qwen25-transfer-freeze
vlrb-qwen25-transfer-audit
vlrb-qwen25-transfer-smoke
vlrb-qwen25-transfer-run
vlrb-qwen25-transfer-retry
vlrb-qwen25-transfer-report
```

输出目录：

```text
output/evolving_structured_rubrics/vl_rewardbench_qwen25_phase17_e4_transfer_v1/
```

## Run order

1. Freeze：验证两个端点确为同一 Qwen2.5 checkpoint，冻结 E4 rubric、数据和 K=3 schedule；
2. Audit：验证除 model identity 外与 Qwen3 对照一致，禁止读取 benchmark gold 构造 prompt；
3. Smoke：20条，执行 Native 与 E4 structured，检查两个端点、Native 正则/397B fallback 解析链路、解析率和离线 Initial 派生；
4. Full run：先运行 E4 27-node structured，再运行 Native；
5. Retry：Structured 只重试技术失败请求；Native 依次执行正则解析、397B 文本 fallback，并对仍失败的原始 Native judge 请求最多重新推理10次；
6. Report：生成 Qwen2.5 内部增益、Qwen2.5/Qwen3 model swap 和类别/root 诊断。

最坏请求量：

- Structured：`1247 × 3 × 27 = 101,007`；
- Native：`1247 × 3 = 3,741`；
- 合计：104,748 logical requests，不含技术重试。

按既有 Qwen3 约232 requests/min 的记录估计约7.5小时；Qwen2.5 实际速度需由 smoke 更新，建议预留6–10小时。

## Decision gate for Qwen2.5 full evolution

完成 model-swap 后再决定是否运行完整演化：

- 若 E4 跨 Worker 增益稳定：完整演化用于判断 Qwen2.5-specific Rubric 能否进一步超过 transferred E4；
- 若 E4 不迁移：完整演化用于判断 Rubric 是否必须与 Worker 共同适配；
- Qwen2.5 完整演化必须从初始五 roots 重新开始，使用 Qwen2.5 的 fresh predictions、fresh ErrorSignatures 和 fresh Split/Refine competition；不得复用 Qwen3 ErrorSignatures 或候选接受历史；
- Manager、Discovery100、Dev150、Prompt v2、Global Memory、Locked-Child Split、Role-aware Refine、epoch/seed 保持与 Phase17 相同；
- 最终比较 Qwen2.5 Initial、transferred Phase17 E4、Qwen2.5-specific evolved Rubric，并在 heldout-500 exploratory 与 VL-RewardBench 上评测。

## Final checklist

- [ ] 两端 Qwen2.5 endpoint identity 完全冻结
- [ ] Qwen3 controls 只读复用且数据/schedule hash 一致
- [ ] Initial 与 E4 由同一套 Qwen2.5 structured predictions 派生
- [ ] Native prompt、regex parser、397B fallback parser 与最多10次重新推理协议均与历史 Native protocol 一致
- [ ] 技术失败单独重试并报告
- [ ] 结果不用于事后修改 Phase17 E4
- [ ] 完整演化在 model-swap 结果完成后单独立项

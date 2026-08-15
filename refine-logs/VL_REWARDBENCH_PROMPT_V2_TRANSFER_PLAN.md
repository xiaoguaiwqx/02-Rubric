# VL-RewardBench Prompt v2 Transfer Experiment Plan

**Problem**: Prompt v2 在 heldout-500 上将 Phase 10 M1 从71.4%提高到76.6%，但现有 VL-RewardBench 结果全部由 Prompt v1 产生，尚不能代表当前默认 Pairwise Worker 协议。

**Method thesis**: 将 criterion 放在候选之后并使用稳定 System prompt，可以减少候选位置偏置，使演化 Rubric 在外部 benchmark 上更可靠地调用 Worker 的视觉判断能力。

**Date**: 2026-08-14

**Experiment ID**: `vl_rewardbench_phase10_prompt_v2_transfer_v1`

## 1. Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked block |
|---|---|---|---|
| C1：演化 Rubric 在 Prompt v2 下仍具有独立外部迁移收益 | 排除 heldout 增益只来自 Prompt 改写、而非 Rubric 演化 | 在相同 Prompt v2、模型、K=3 schedule 下，Final equal 明显优于 Initial five-root，且 paired corrected > harmed | B1 |
| C2：Prompt v2 的收益可迁移到 VL-RewardBench | 验证5.2 pp并非只适用于开发用 heldout-500 | Final equal v2 不低于 Final equal v1，并改善位置对称性、Coverage或有效率 | B2 |

**Anti-claim to rule out**: 最终提升完全来自 Prompt v2 的通用 judge 改善，演化 children 不再提供额外收益。

## 2. Frozen Protocol

### 2.1 Data and order

- 数据固定为 VL-RewardBench 1,247 pairs；dataset SHA-256=`500a33ec...b59`，record SHA-256=`42ff026c...f20`。
- 只读复用 `vl_rewardbench_phase10_transfer_v2_max2048` 已冻结的 K=3 schedule：seed=42，protocol=`balanced_b_1minusb_b`。
- 每个判断的三次输出先映射回原始 answer index，再做多数投票；Tie/None单独进入 Coverage，strict ACC 中计为错误。
- 不重新选择 checkpoint、criterion、root权重、Prompt或重试策略；VL-RewardBench 不反馈给演化流程。

### 2.2 Model and Prompt

- Pairwise Worker：`Qwen/Qwen3-VL-8B-Instruct`。
- Prompt：`STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1_1`，prompt version=`1.1.0-cache-pilot`，冻结 hash=`7b94a9b...7df`。
- User content顺序固定为：`image → question → Candidate A → Candidate B → criterion → final question`。
- Decoding：`temperature=0.5`、`max_tokens=2048`。
- 两个 API：`vllm-8000`和`vllm-8001`，各20并发，总并发40；使用 available-slot 动态调度。
- Freeze时重新验证两个端点的模型、checkpoint和运行身份；中途身份漂移必须暂停，不允许静默继续。

### 2.3 Rubrics and systems

Prompt变化会改变 request identity，因此不能复用 Prompt v1 的节点预测。新实验只生成一次 Phase 10 完整22节点预测，再从同一 artifact 离线得到以下系统：

| System | Nodes / aggregation | New model calls |
|---|---|---:|
| Initial five-root v2 | Phase 10 Rubric中的5个原始 roots，等权 M1 | 由22节点预测子集离线聚合 |
| Visual Grounding subtree v2 | Visual root及其children | 离线聚合 |
| Final weighted v2 | 22节点；Visual=0.4，其余roots各0.15 | 离线聚合 |
| Final equal v2 | 22节点；五个root子树等权 | 离线聚合，主要结果 |

Phase 10 Rubric固定为5 roots、17 children，rubric SHA-256=`17ad7a0b6350338b100384b701303008746463911662f4a628b90c4632cdf7ef`。Initial five-root的5个root必须与该Rubric中的对应root逐字段相同，否则Freeze硬失败。

以下结果只作为既有对照读取，不重新推理：

- Prompt v1 Initial five-root：OverallAcc=45.04%，MacroAcc=47.75%。
- Prompt v1 Final equal：OverallAcc=64.31%，MacroAcc=59.05%。
- Native VL-RB prompt：OverallAcc=54.52%，MacroAcc=53.56%；它不使用结构化Worker Prompt，因此无需重跑。

## 3. Experiment Blocks

### B0. Freeze, audit and smoke

- **Claim tested**: 新运行与旧实验除Prompt外完全可比。
- **Runs**:
  - 验证dataset、1,247 IDs、K=3 schedule、Rubric和endpoint identities；
  - 验证Prompt v2 hash与heldout Prompt ablation完全一致；
  - 20 pairs × 22 nodes × K=3，共1,320个逻辑请求；
  - 检查A/B映射、JSON解析、K=3多数票、离线四系统聚合和双端调用。
- **Go gate**: 1,320/1,320 requests形成artifact；valid rate≥99%；两个端点均被实际调用；离线重放hash一致。
- **Priority**: MUST-RUN。

### B1. Evolution gain under Prompt v2

- **Claim tested**: 演化 Rubric 的收益不依赖 Prompt v1。
- **Primary comparison**: Final equal v2 vs Initial five-root v2。
- **Metrics**:
  - OverallAcc、MacroAcc、Coverage、strict ACC、正确数；
  - General/Hallucination/Reasoning covered ACC；
  - paired corrected、harmed、net corrected、exact McNemar；
  - 每个root子树及各node的support、covered/strict ACC和冲突率。
- **Success criterion**: Final equal v2 的OverallAcc和MacroAcc均高于Initial v2，corrected > harmed，Coverage下降不超过1 pp。
- **Failure interpretation**: 若两者接近，说明此前外部迁移收益主要可能来自执行Prompt，而非演化children；若Final更差，则children聚合在Prompt v2下产生负干扰。
- **Table target**: 主结果表。
- **Priority**: MUST-RUN。

### B2. Prompt transfer and position calibration

- **Claim tested**: Prompt v2 的heldout收益能迁移到外部benchmark。
- **Primary comparison**: Final equal v2 vs既有Final equal v1。
- **Secondary comparisons**: Initial v2 vs Initial v1；Visual-only v2与Final equal v2；equal与预设weighted聚合。
- **Metrics**:
  - OverallAcc、MacroAcc、Coverage和strict ACC变化；
  - paired corrected/harmed和exact McNemar；
  - 原始gold-A/gold-B准确率、预测A/B比例；
  - K=3中正序/反序准确率、swap consistency、order gap；
  - parse-invalid、达到2048-token上限、重试次数。
- **Success criterion**: Final v2不低于64.31%，最好获得净正paired改进；gold-A/gold-B gap与order gap不扩大。
- **Failure interpretation**: 若heldout改善但VL-RB下降，Prompt v2可能针对RLHF-V式样本分布校准，不能作为通用默认协议。
- **Table target**: Prompt迁移与顺序稳健性表。
- **Priority**: MUST-RUN。

### B3. Efficiency diagnostic

- **Claim tested**: Prompt v2在K=3外部评测中仍具有工程收益。
- **Metrics**: wall time、推理次数/分钟、P50/P90/P99 latency、额外模型调用、输入/输出tokens、两端请求数。
- **Comparison**: 与旧Prompt v1 VL-RewardBench运行记录作描述性比较；由于服务状态不同，不将其解释为严格系统benchmark。
- **Priority**: NICE-TO-HAVE，但由主运行自动产出。

## 4. Execution Stages

建议新增独立stages，不覆盖已有VL-RewardBench结果：

```text
vlrb-prompt-v2-freeze
vlrb-prompt-v2-audit
vlrb-prompt-v2-smoke
vlrb-prompt-v2-run
vlrb-prompt-v2-retry
vlrb-prompt-v2-report
```

输出目录：

```text
output/evolving_structured_rubrics/
  vl_rewardbench_phase10_prompt_v2_transfer_v1/
```

Stage语义：

1. `freeze`：冻结数据、schedule、Rubric、Prompt、解码、端点和既有baseline hashes。
2. `audit`：纯离线验证5 roots是22节点的精确子集、四种聚合可由同一预测生成，且没有benchmark信息进入Prompt。
3. `smoke`：运行20 pairs完整K=3链路；不得写入正式run cache。
4. `run`：生成22×1,247×3=`82,302`个逻辑判断，支持按sample/node/replicate恢复。
5. `retry`：只重试技术失败或parse-invalid请求，最多10次；不读取gold决定是否重试。仍失败则保留invalid并进入Coverage/strict ACC。
6. `report`：离线聚合四个结构化系统，并与既有Prompt v1/Native逐样本比较。

## 5. Run Order and Compute Budget

| Milestone | Goal | Requests | Decision gate | Estimated time |
|---|---|---:|---|---:|
| M0 | Freeze + audit | 0 | 所有hash与子集关系一致 | <5 min |
| M1 | Smoke | 1,320 | valid rate≥99%，双端与K=3映射正确 | 5–15 min |
| M2 | Full Prompt v2 inference | 82,302 | 全部shards完成或明确标记invalid | 理想约5.2 h；预算6–10 h |
| M3 | Technical retry | 仅失败项，最多10次 | unresolved尽可能接近0 | 0–2 h |
| M4 | Offline report | 0 | 四系统重放及paired统计一致 | <10 min |

时间估计以heldout Prompt v2的262.1次推理/分钟为理想吞吐：`82,302 / 262.1 ≈ 314 min`。VL-RewardBench图像和回答长度分布更复杂，因此实际预算按6–10小时估计。

## 6. Artifact and Validation Requirements

必须保存：

```text
frozen_manifest.json
offline_audit.json
smoke/report.json
structured/prompt_v2/replicate_01..03/
structured/combined_pairwise.json
systems/initial_equal.json
systems/visual_only.json
systems/final_weighted.json
systems/final_equal.json
retry/report.json
final_report.json
final_report.md
```

必须验证：

- Prompt v2 hash、Rubric hash、dataset hash和K=3 schedule hash冻结后不可漂移；
- Prompt v1 cache不能命中Prompt v2请求；
- Initial与Final共享的5个root只推理一次，离线复用必须逐项相同；
- equal、weighted、Visual-only只改变离线聚合，不产生额外模型请求；
- swap后A/B输出正确映射回原始answer index；
- 失败重试不读取gold，且不能覆盖已有成功结果；
- Native baseline不重跑、不因Prompt v2修改；
- 运行阶段不根据VL-RewardBench结果选择系统或参数；
- full unittest、focused VL-RB/Prompt tests、compileall和`git diff --check`通过。

## 7. Reporting Boundary

本实验应报告为 **exploratory external protocol validation**。Prompt v2是在heldout-500上确定的，不是在VL-RewardBench上选择的；但VL-RewardBench此前已用于Prompt v1评估，因此它不再是首次、完全无偏的外部测试。无论结果正负，都不得据此继续修改Prompt v2后再把同一benchmark称为confirmatory结果。

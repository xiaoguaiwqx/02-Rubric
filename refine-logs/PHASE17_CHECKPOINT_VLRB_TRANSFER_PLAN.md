# Phase17 中间 Checkpoint 的 VL-RewardBench 泛化诊断计划

**问题**：Phase17 在 Dev150 上于 Epoch 2 达到峰值，之后持续下降；但目前只有最终 Epoch 5 在 VL-RewardBench 上完成评测，无法判断后期 Refine 是否真的损害外部泛化。

**方法主张**：在不重新演化、不改变 Prompt、聚合或 benchmark 协议的条件下，评测预先冻结的 Epoch 2、3、4，并与已有 Epoch 5 和 Phase10 Control 做逐样本比较，从而判断 Phase17 的外部最优点是否出现在中间 epoch。

**日期**：2026-08-22

## 1. Claim Map

| Claim | 为什么重要 | 最低可信证据 | 实验块 |
| --- | --- | --- | --- |
| C1：Phase17 后期 Refine 是否造成外部泛化退化 | 决定后续是否需要 Dev early stopping、Refine patience 或重复 Refine 限制 | E2/E3/E4/E5 在相同 VL-RewardBench K=3 协议下的 OverallAcc、MacroAcc、Strict ACC 和 paired corrected/harmed | B1 |
| C2：观察到的差异只来自 checkpoint Rubric | 排除 Prompt、数据顺序、随机方向、Worker 或聚合变化 | 冻结 Rubric/request/schedule hash；仅对不同 description 补预测；完全相同请求逐项复用 | B2 |

**Anti-claim**：Dev150 的下降可能只是150条样本的方差或 Dev/VL-RewardBench 分布不一致，并不代表真正的外部过拟合。

## 2. Checkpoint 预注册

在读取任何新 checkpoint 的 VL-RewardBench 结果前固定以下系统：

| 系统 | 选择理由 | 状态 |
| --- | --- | --- |
| Phase17 Epoch 2 | Dev150 最佳：75.33% | 新评测 |
| Phase17 Epoch 3 | Discovery100 最佳：66.00% | 新评测 |
| Phase17 Epoch 4 | 五个初始 roots 首次全部完成 Split | 新评测 |
| Phase17 Epoch 5 | 正式 final checkpoint | 已完成，只读复用 |
| Phase10 + Prompt v2 | 当前强历史 Control | 已完成，只读复用 |

Epoch 1 不纳入主实验：它只完成部分 Split，Discovery100 和 Dev150 均未改善。若后续确实需要绘制完整 E0–E5 曲线，可作为 appendix 的独立补充，不能在看到 E2–E4 结果后用于挑选新的最佳点。

本实验为 exploratory checkpoint diagnosis。即使某个中间 epoch 最优，也不能将其追认为新的无偏正式结果；它用于检验 early stopping 假设并设计下一版协议。

## 3. 冻结对照

### 3.1 数据与任务

- 数据：VL-RewardBench 固定1,247个 preference pairs；
- 任务：结构化多模态 Pairwise Worker 对每个 Rubric node 输出 `A/B/None`；
- Replicates：K=3，沿用已有 Phase17 的 frozen A/B counterbalanced schedule；
- 最终聚合：每个 replicate 内执行相同的完整 five-root equal-weight M1，三个 replicate 映射回 source orientation 后多数投票。

### 3.2 Worker 与执行身份

- Worker：`Qwen/Qwen3-VL-8B-Instruct`；
- Prompt：Pairwise Worker Prompt v2，版本 `1.1.0-cache-pilot`；
- decoding：`temperature=0.5`、`max_tokens=2048`，其余 request spec 与 Phase17 E5 完全一致；
- endpoints：`vllm-8000 + vllm-8001` available-slot pool；
- 调度：sample-major dynamic scheduling；
- 不生成新的 Manager、ErrorSignature、Split、Refine、Gate 或 Root Router 请求。

### 3.3 结果口径

- `OverallAcc`：在具有明确最终 A/B 判断的覆盖样本上计算；
- `MacroAcc`：General、Hallucination、Reasoning 三类 covered accuracy 的宏平均；
- `Coverage`：产生明确最终 A/B 判断的比例；
- `Strict ACC`：正确数除以全部1,247条，未覆盖计错。

主表必须同时报告四项，禁止只报告可能因弃权增加而上升的 covered accuracy。

## 4. 精确复用与新增请求

以 Phase17 E5 的完整 K=3 Pairwise artifacts 为只读基线。复用键必须至少包含：

```text
dataset/sample identity
replicate and orientation
node_id
criterion name
description SHA-256
Prompt/parser/request spec
model and decoding identity
```

仅 criterion name 相同但 description 不同不得复用。中间 checkpoint 中 E5 不存在的旧 description 必须重新推理。

离线审计得到：

| Checkpoint | Nodes | 相对 E5 不同的 descriptions |
| --- | ---: | ---: |
| E2 | 23 | 11 |
| E3 | 23 | 7 |
| E4 | 27 | 6 |

跨 E2/E3/E4 去重后共有18个 unique historical description variants，因此最大新增逻辑请求为：

\[
18\times1,247\times3=67,338.
\]

相同 historical description 在多个 checkpoint 出现时只生成一次，再分别组装 checkpoint prediction。E5 和 Phase10 不产生新 Pairwise 请求。

## 5. Experiment Blocks

### B1：中间 Checkpoint 外部泛化曲线

- **Claim tested**：Dev150 的 E2 峰值及后续下降是否在 VL-RewardBench 上复现。
- **Compared systems**：Phase17 E2/E3/E4/E5 与 Phase10 Control。
- **Primary metrics**：OverallAcc、MacroAcc、Strict ACC；每个 checkpoint 相对 E5 和 Phase10 的 corrected、harmed、net corrected、exact McNemar。
- **Secondary metrics**：Coverage、General/Hallucination/Reasoning ACC、每个 source subset ACC、每个 root subtree 的 OverallAcc/Strict ACC。
- **成功解释**：
  - `E2 > E3 > E4 > E5` 且方向与 Dev150 一致：支持后期 Refine 外部过拟合；
  - E2/E3 最佳、E4/E5 接近：说明主要收益在前两轮完成，后续 Refine 边际无效；
  - E5 最佳：Dev150 下降更可能来自采样方差或分布错配；
  - 不同类别在不同 epoch 达峰：说明发生领域间能力迁移，而不是单一全局过拟合。
- **优先级**：MUST-RUN。

### B2：协议与复用审计

- **Claim tested**：所有 checkpoint 只改变 Rubric node set/description。
- **检查内容**：Rubric hash、node/version manifest、VL-RewardBench dataset hash、K=3 schedule hash、Prompt/request identity、reuse provenance、E5 结果精确重建、技术失败统计。
- **失败条件**：任何 request identity drift、错误 description 复用、E5 重建不一致或 benchmark gold 进入 prompt 时硬失败。
- **优先级**：MUST-RUN，先于 B1。

### B3：完整 Epoch 1 补充

- **用途**：仅在需要 E0–E5 完整曲线时补充。
- **新增成本**：E1 相对 E5 有4个不同 descriptions；若与现有 historical variants 去重后仍缺失，再生成对应 K=3 predictions。
- **优先级**：NICE-TO-HAVE；不得延迟 B1/B2。

## 6. 输出与 Stages

建议独立输出目录：

```text
output/evolving_structured_rubrics/
  vl_rewardbench_phase17_checkpoint_transfer_v1/
```

建议 stages：

```text
vlrb-phase17-checkpoint-freeze
vlrb-phase17-checkpoint-audit
vlrb-phase17-checkpoint-smoke
vlrb-phase17-checkpoint-run
vlrb-phase17-checkpoint-retry
vlrb-phase17-checkpoint-report
```

主要 artifacts：

```text
frozen_manifest.json
checkpoint_rubrics/epoch_02.json
checkpoint_rubrics/epoch_03.json
checkpoint_rubrics/epoch_04.json
variant_manifest.json
reuse_manifest.json
smoke/report.json
run/variant_predictions/
run/checkpoint_predictions/
retry/report.json
final_report.json
final_report.md
```

## 7. Run Order and Milestones

| Milestone | 目标 | 决策门槛 | 预计成本 | 主要风险 |
| --- | --- | --- | --- | --- |
| M0 Freeze | 冻结 E2/E3/E4 Rubric、E5/Phase10 control、schedule 与18个 variants | 所有 hash 可复现 | 离线，分钟级 | checkpoint 路径或 hash 漂移 |
| M1 Audit | 验证精确复用和 E5 重建 | E5 logical/final votes 逐项一致 | 离线，分钟级 | 按 name 错误复用 description |
| M2 Smoke | 20条样本覆盖三个 checkpoint、两个 endpoints | parse valid 100%，两个端口均被调用，composite 完整 | 最多 `20×18×3=1,080` 请求 | cache/path/resume 错误 |
| M3 Full run | 只补18个 historical variants | 无静默 fallback；可断点恢复 | 最多67,338请求 | 长尾输出与连接重试 |
| M4 Retry/report | 最多10次重试技术失败并生成 paired report | unresolved 明确报告；所有口径一致 | retry-only | covered/strict 指标混淆 |

按 Phase17 E5 实测约232次逻辑推理/分钟估算，67,338个请求约需4.8小时纯推理时间；考虑 smoke、长尾和重试，建议预留约5–7小时。实际时间取决于两个 vLLM 服务的并发与稳定性。

## 8. 判定与证据边界

### 8.1 Primary decision

优先判断 E2/E3/E4 是否在 OverallAcc、MacroAcc 和 Strict ACC 中一致优于 E5，并检查 paired net corrected 与 McNemar。单独一个 covered metric 上升而 Coverage/Strict ACC 下降，不视为更优 checkpoint。

### 8.2 后续协议含义

- 若 E2 或 E3 明确优于 E5：下一步单独设计“Dev early stopping / Refine patience”实验，而不是直接替换正式 final；
- 若差异不显著但 E2–E4 均不低于 E5：说明5轮计算可能没有必要，可研究更短演化；
- 若 E5 仍最好：保留固定5轮，重新审视 Dev150 的代表性；
- 若类别峰值分裂：优先研究按领域的 discovery/aggregation，而非单一 early stopping。

### 8.3 明确禁止

- 不根据新 benchmark 结果重新生成或 Refine criterion；
- 不新增 root 权重、删 root、Gate 或 Router；
- 不只报告最优 epoch，必须完整报告 E2/E3/E4/E5；
- 不将最佳中间 checkpoint 称为新的 confirmatory 主结果。

## 9. Tests and Acceptance

- E2/E3/E4 Rubric hash 与 Phase17 committed artifacts 完全一致；
- checkpoint 角色与选择理由在 inference 前冻结；
- unique historical description 数严格为18，否则 freeze 硬失败；
- E5 与 Phase10 为 generated=0 的只读 control；
- 完全相同请求才可复用，description hash 漂移硬失败；
- E5 reconstructed votes、metrics 与既有 final report 完全一致；
- K=3 orientation schedule 与既有 Phase17 完全相同；
- sample-major available-slot 调度实际使用8000与8001；
- retry 只重跑技术/解析失败，不重跑成功的随机输出；
- OverallAcc、MacroAcc、Coverage、Strict ACC 和 paired metrics 口径测试通过；
- focused unittest、完整 unittest、compileall 与 `git diff --check` 通过。

## 10. Final Checklist

- [ ] E2/E3/E4 checkpoint 及角色已冻结
- [x] E5/Phase10 control 可只读精确重建
- [x] 18个 historical description variants 已审计
- [x] Prompt v2、K=3、schedule、模型与 decoding identity 一致
- [x] Smoke 同时覆盖两个 endpoints
- [x] 技术失败完成重试并单独报告
- [x] E2/E3/E4/E5 全部进入最终主表
- [x] 结果明确标记为 exploratory checkpoint diagnosis

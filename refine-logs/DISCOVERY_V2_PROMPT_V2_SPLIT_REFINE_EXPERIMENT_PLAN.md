# Discovery-v2 Prompt-v2 Split+Refine 实验计划

## 1. 实验目的

本实验建立一条独立于旧 Discovery90 的 Phase17 演化轨迹，回答两个问题：

1. 将演化数据替换为覆盖视觉事实、多模态推理和通用偏好的多源 Discovery100 后，能否学到比旧轨迹更可迁移的结构化 Rubric？
2. 在完全不参与模型选择的 Dev150 上，Rubric 的逐轮变化能否揭示 discovery 过拟合、领域偏移和准则退化？

核心假设是：扩大少量偏好经验的任务分布后，Split 与 Refine 能把这些经验编码成更通用的决策准则，从而改善 VL-RewardBench 的 General、Hallucination 和 Reasoning 综合表现。RLHF-V heldout-500 只用于观察域内行为变化，不参与主要成功判定。

本实验属于 exploratory pilot。当前 Discovery100 与 Dev150 的 gold 来自清洗后的 source label，并非全部人工复核，因此结果不能表述为最终确认性结论。

## 2. Claim–Evidence 结构

### 2.1 主要主张

| ID | 主张 | 主要证据 | 反证条件 |
|---|---|---|---|
| C1 | 多源 Discovery100 能提高演化 Rubric 的跨分布迁移能力 | 最终 Rubric 在 VL-RewardBench 的 OverallAcc、MacroAcc 和三类 ACC；与 Phase16 同提示词、同聚合的逐样本比较 | VL-RewardBench 不升，或提升仅来自 Prompt/推理协议变化 |
| C2 | Dev150 能诊断各 epoch 的泛化轨迹而不污染演化 | epoch 0–5 的 Dev M1、分领域/分来源结果、corrected/harmed；manifest 证明 Dev 未进入 Manager、触发器、竞争与早停 | Dev 被用于接受、重试、早停或 checkpoint 选择 |

### 2.2 明确不主张

- 不声称当前 source label 版本等价于人工高质量偏好集。
- 不用 Dev150、heldout-500 或 VL-RewardBench 选择最佳 epoch。
- 不把结果归因于新的算子、Gate、Root Router 或聚合权重；这些均不在本实验中改变。
- 不声称一次 seed=42 轨迹足以证明统计稳定性。

## 3. 数据冻结与隔离

### 3.1 输入数据

| 数据 | 数量 | 作用 | Manager 可见 | 影响接受/早停 |
|---|---:|---|---|---|
| Discovery100 | 100 | ErrorSignature、Split、Refine、竞争 | 是 | 是 |
| Dev150 | 150 | 每轮独立诊断 | 否 | 否 |
| RLHF-V heldout | 500 | 最终域内外推测试 | 否 | 否 |
| VL-RewardBench | 1,247 | 最终外部迁移测试 | 否 | 否 |

输入文件预期为：

```text
data/discovery_v2_demo_v3/discovery_100.jsonl
data/discovery_v2_demo_v3/dev_150.jsonl
data/RLHF-V/heldout_validation_500_pair.jsonl
data/VL_RewardBench/
```

### 3.2 已知 heldout 重叠及解释边界

离线审计发现当前导出版本存在 8 个与 heldout-500 完全相同的图像—问题实例：

- Discovery100：2 条；
- Dev150：6 条。

本轮按用户授权不替换这 8 条，因为 heldout-500 只作为简单的探索性回归检查，VL-RewardBench 才是主要外部评测。freeze 必须完整记录这些重叠及对应 sample ID，但不将其作为阻塞条件。因此：

- heldout-500 结果不得描述为无偏测试或主要证据；
- 不允许根据 heldout 结果选择 epoch、修改 Rubric 或调整聚合；
- VL-RewardBench 的 sample ID、image SHA 和 unordered pair SHA 交集必须为零；
- Discovery 与 Dev 之间的 sample/image/question/pair/source ID 交集仍必须为零。

当前与 VL-RewardBench 存在少量相同通用问题文本，但图像和回答对不同；该数量需要报告，不作为强泄漏键。

### 3.3 冻结内容

freeze 保存：

- 四个评测集合的路径、数量、文件 hash 和逐样本指纹；
- Discovery/Dev 的 source、domain、A/B gold 分布；
- 五个初始 roots 与 initial rubric hash；
- Manager、Worker、Prompt、decoding、endpoint 和 seed identity；
- Split/Refine 触发、竞争、失败历史与同步提交协议；
- `dev_visible_to_manager=false`、`selection_by_dev=false`、`heldout_access=final_only`。

## 4. 冻结的演化协议

### 4.1 唯一实验变量

Treatment 复用 Phase16 的方法协议，只替换演化数据并增加 Dev 诊断：

| 项目 | Phase16 Control | Phase17 Treatment |
|---|---|---|
| 演化数据 | Discovery90，视觉事实偏重 | 多源 Discovery100 |
| Dev | 无独立逐轮 Dev | Dev150，只读诊断 |
| Pairwise Worker | Prompt v2 | 相同 |
| Split / locked retry | Phase16 协议 | 相同 |
| Role-aware Refine | Phase16 协议 | 相同 |
| Global Rubric Memory | `global_rubric_v1` | 相同 |
| 聚合 | 五-root 等权 M1 | 相同 |
| seed / epoch | 42 / 3–5 | 相同 |

不启用 Create、Root Boundary Pre-Refine、Gate Worker、Root Router、后验权重搜索或 semantic gate。

### 4.2 模型与请求

- Manager：`Qwen/Qwen3.5-397B-A17B`。
- Pairwise Worker：`Qwen/Qwen3-VL-8B-Instruct`。
- Worker Prompt：`PAIRWISE_WORKER_PROMPT_V2_CACHE_PILOT_VERSION`，版本 `1.1.0-cache-pilot`。
- Worker decoding：temperature `0.5`，max tokens `2048`，seed `42`。
- Worker endpoint：`vllm-8000 + vllm-8001` available-slot pool。
- 调度：sample-major；同一样本的不同准则连续提交，以利用 Prompt v2 前缀缓存。
- ErrorSignature 必须从 Discovery100 上的 Prompt-v2 parent 错误重新生成；禁止复用 Discovery90 signatures。

### 4.3 Split

- 仅五个初始 roots 可执行 Split。
- 触发阈值沿用 Phase16：`ACC < 0.75` 且满足冻结的 coverage/support 条件。
- Split 失败后沿用 locked-child retry：最多锁定 1 个强 child；强 child 至少 support 15、net corrected 3。
- 锁定 child 不再重新生成；其余 children 使用原 cluster、所有 child 指标和失败归因重试。
- Split 接受仍由 specialized subtree 与 parent 在固定 decisive scope 上竞争；不允许部分提交。

不强制每个 root 都 Split。若新的数据分布使某 root 不满足触发器，应如实记录，而不是为获得更多 children 放宽协议。

### 4.4 Role-aware Refine

- roots 沿用既有 root trigger。
- children 使用：

\[
0.5 < \operatorname{Acc}(c) < 0.80,\quad
|\mathcal S(c)|\ge 15,\quad
\operatorname{Wrong}(c)\ge 5.
\]

- 每次只生成一个新 description，node ID、name、score 和 topology 不变。
- 新旧 node 使用各自 A/B 支持集做 self-competition；严格提升且 support 不低于 15 才接受。
- Dev、heldout 和 VL-RewardBench 结果不得参与 Refine 接受。

### 4.5 Epoch 与提交

- 最少 3 epoch、最多 5 epoch。
- 每个 epoch 的 Split/Refine candidate 基于同一 epoch-start rubric 和 Global Memory。
- 同轮 accepted patches 同步提交；下一轮才可见。
- 第 3 轮以后，仅当 Discovery100 上没有 eligible 或 retryable 操作时早停。
- 最终模型固定为最后一个 committed epoch，不选择 Dev 最优 checkpoint。

## 5. Dev150 逐轮验证

### 5.1 执行时点

对 epoch 0 初始 Rubric 和每个 epoch 提交后的 Rubric 运行 Dev150：

```text
epoch commit
  -> freeze committed rubric/hash
  -> evaluate Dev150
  -> write diagnostic only
  -> next epoch
```

Dev artifact 存在于输出目录，但演化 runner 的 Manager/evidence/trigger/competition loader 不得引用该目录。中断恢复时只恢复同一 epoch 的 Dev 诊断，不能改变已经提交的 Rubric。

### 5.2 缓存策略

- criterion name + description + request spec 完全相同：复用该 node 的 Dev prediction；
- 新增或 Refine 后 description 改变：只生成缺失预测；
- 不允许按 criterion name 复用不同 description 的输出；
- 每个 epoch 组装完整 Pairwise artifact 后离线执行同一 M1。

### 5.3 Dev 指标

- Overall M1 ACC、Coverage、covered ACC、correct count；
- visual/reasoning/general 分领域 ACC；
- 六个 source family 的 ACC；
- 每个 root subtree 与每个 node 的 ACC/support/coverage/wrong；
- 相对 epoch 0 和上一 epoch 的 corrected、harmed、net corrected；
- description 版本、node 数量、接受/拒绝操作轨迹。

Dev 的作用仅是回答“演化在哪一轮、哪个领域开始过拟合”，而不是挑选 checkpoint。

## 6. 最终评测与对照

### 6.1 对照系统

所有系统必须在相同 Prompt v2、Worker、decoding 和等权 M1 下比较：

1. Initial five-root M1；
2. Phase10 final Rubric：当前 VL-RewardBench 的强历史基线；
3. Phase16 final Rubric：同算法、旧 Discovery90 的直接数据分布对照；
4. Phase17 Discovery-v2 final Rubric：本实验 Treatment。

若 criterion description 与已冻结 artifact 完全一致，可复用预测；否则必须重新推理。不得混用 Prompt v1 输出。

### 6.2 heldout-500

仅在 Discovery final report 与 final rubric hash 冻结后开放：

- 单次 Prompt-v2 Pairwise 推理，与历史 heldout 协议一致；
- 报告严格 ACC（分母 500）、Coverage、covered ACC；
- 报告 Phase17 相对 Initial、Phase10 和 Phase16 的 corrected/harmed/net corrected；
- 使用 exact McNemar，并给总体 ACC Wilson 95% CI；
- 不测试 epoch 1–4，不用 heldout 选择 checkpoint。

### 6.3 VL-RewardBench

- 1,247 个样本；
- K=3 独立 replicate，seed `42/43/44`；每个 replicate 按冻结 schedule 随机 A/B orientation；
- Prompt v2、sample-major、双端口 available-slot 调度；
- 三次原始顺序映射回 source orientation 后多数投票；tie/abstain 在 strict ACC 中计错；
- 对技术或解析失败最多重试 10 次，并单独报告 unresolved 数；
- 不使用 Gate、Root Router 或 benchmark 自带 judge prompt。

主要指标：

- OverallAcc（strict，分母 1,247）；
- MacroAcc（General/Hallucination/Reasoning 三类 strict ACC 的宏平均）；
- General、Hallucination、Reasoning ACC；
- Coverage、covered ACC、position disagreement；
- Phase17 对 Phase16/Phase10 的逐样本 corrected、harmed、net corrected 和 exact McNemar。

## 7. 成功判据与结果解释

### 7.1 Primary

最有说服力的正向结果是：

- Phase17 的 VL-RewardBench OverallAcc 与 MacroAcc 均高于 Phase16；
- General/Reasoning 至少一个明显改善，Hallucination 不发生大幅退化；
- Phase17 对 Phase16/Phase10 获得正的 paired net corrected；显著性不足时只表述为 exploratory evidence。

heldout-500 不设 pass/fail 门槛；它只报告域内回归方向，并明确包含已知重叠。

### 7.2 解释规则

- VL-RewardBench 提升且 heldout 稳定：支持“更广 discovery 分布提高迁移”。
- Dev 提升但两个最终测试不升：说明 Dev 分布仍不足以代表外部泛化。
- heldout 提升而 VL-RewardBench 下降：仍偏向视觉事实数据，不能支持分布扩展主张。
- VL-RewardBench 提升但 heldout 明显下降：发生能力重分配，需要作为 trade-off 报告。
- 指标变化小或 McNemar 不显著：只表述为 pilot evidence。

## 8. Stages 与输出

### 8.1 演化 stages

```text
discovery-v2-evolution-freeze
discovery-v2-evolution-audit
discovery-v2-evolution-smoke
discovery-v2-evolution-run
discovery-v2-evolution-report
discovery-v2-evolution-heldout
discovery-v2-evolution-final-report
```

输出目录：

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase17_discovery_v2_prompt_v2_split_refine_v1/
```

建议目录结构：

```text
frozen_manifest.json
offline_audit.json
epoch_00/
  rubric_committed.json
  discovery/
  dev150/
epochs/epoch_NN/
  rubric_memory.json
  operations/
  rubric_committed.json
  discovery_report.json
  dev150/report.json
final/
  rubric.json
  discovery_report.json
  dev_trajectory.json
heldout500/
  frozen_manifest.json
  predictions.json
  report.json
final_report.json
```

### 8.2 VL-RewardBench stages

```text
vlrb-discovery-v2-freeze
vlrb-discovery-v2-audit
vlrb-discovery-v2-smoke
vlrb-discovery-v2-run
vlrb-discovery-v2-retry
vlrb-discovery-v2-report
```

输出目录：

```text
output/evolving_structured_rubrics/
  vl_rewardbench_phase17_discovery_v2_prompt_v2_v1/
```

## 9. 测试与验收门槛

必须覆盖：

- Discovery100/Dev150 数量分别严格为 100/150，图片全部可读，gold 为 A/B。
- A/B 位置平衡、dataset hash 和逐样本 fingerprint 确定。
- Discovery/Dev 强去重键交集为零；VL-RewardBench 的 sample/image/pair 强重叠为零；heldout 的 8 条已知重叠被完整记录且不作为主要证据。
- initial prediction 确实由 Discovery100 + Prompt v2 新生成。
- ErrorSignature 只来自 Discovery100 的当前 parent 错误，旧 signatures 命中数为零。
- Dev 不出现在 Manager prompt、Rubric Memory、failure history、trigger 或 acceptance artifact 中。
- 修改 Dev label 或 prediction 不改变任何 accepted operation、最终 rubric hash 或停止 epoch。
- 每个 epoch 同一 committed rubric 对应唯一 Dev report；中断恢复不重复生成已缓存的 node version。
- Split/locked retry、Refine trigger、自竞争和同步提交与 Phase16 regression 一致。
- discovery 阶段无法读取 heldout 或 VL-RewardBench。
- heldout/VL-RB stage 验证 final rubric、dataset、Prompt 和 endpoint identity。
- VL-RB 三个 replicate 不跨 replicate 复用随机输出，但相同 frozen baseline artifact 可只读复用。
- 技术失败与模型 abstain 分开统计。
- focused unittest、完整 unittest、compileall 和 `git diff --check` 通过。

## 10. 运行顺序、成本与停止条件

### 10.1 必跑顺序

1. Offline freeze + audit，记录而不阻塞 8 条 heldout overlap。
2. Smoke：Discovery/Dev 各 20 条的 Prompt-v2、双数据 loader 与 M1 链路；禁止 heldout/VL-RB。
3. 完整 3–5 epoch 演化与逐轮 Dev。
4. Discovery/Dev final report，冻结最后 committed rubric。
5. heldout-500 探索性回归测试。
6. VL-RewardBench smoke、K=3 完整运行、技术失败重试和报告。

### 10.2 预计成本

- Discovery baseline：约 `100 × 5 = 500` 个 Worker 请求。
- Dev baseline：约 `150 × 5 = 750` 个 Worker 请求；后续只刷新新增/改写 node。
- 演化期 Worker 请求取决于生成的 child/node 版本，预计数千级；397B 主要用于 ErrorSignature、聚类、child/refine 生成和失败归因。
- heldout 只刷新 Phase17 新增或改写准则，通常为数千至约一万请求。
- VL-RewardBench 最昂贵。若最终为 23 个 nodes，K=3 为 `1,247 × 23 × 3 = 86,043` 个逻辑 Worker 请求；实际随最终 node 数线性变化。
- 两个 Worker endpoint 均健康时，演化与 Dev 可按小时计，VL-RewardBench 通常需要长时间连续运行，应支持断点恢复。

### 10.3 停止条件

- 数据重叠、hash 或请求身份不一致：停止，禁止在线修补。
- Smoke schema/缓存/Dev 隔离失败：停止实现修复。
- Manager transport failure：保存 pause state，恢复后继续。
- 结果不理想不是工程停止条件；仍完成冻结协议内的 discovery report，再决定是否授权最终测试。

## 11. 后续但不纳入本实验

- 人工复核后的 Discovery-v2 confirmatory replay；
- 多 seed 演化稳定性；
- Create 初始化 root；
- 用 Dev 进行 checkpoint selection 的独立消融；
- Gate/Router 与新 Rubric 的组合测试；
- 新聚合权重或学习式 aggregation。

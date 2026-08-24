# Qwen2.5-VL Full Split+Refine Evolution Experiment Plan

**Problem**: Phase17 E4 Rubric 从 Qwen3-VL-8B 迁移到 Qwen2.5-VL-7B 后仍带来显著收益，但尚不清楚：根据 Qwen2.5 自身错误重新执行完整 Split+Refine，能否得到比直接迁移 E4 更适配该 Worker 的 Rubric。

**Method thesis**: 固定 Discovery-v2 数据、Manager、Prompt v2、Split/Refine 协议、epoch 和 seed，只让 Qwen2.5 从 Initial five-root 开始产生自己的预测、错误经验和候选竞争，可以区分“通用 Rubric 迁移”与“Worker-specific 共同适配”的增量价值。

**Date**: 2026-08-23

## 1. Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
| --- | --- | --- | --- |
| C1：Qwen2.5 能从自身错误经验中完成有效演化 | 证明完整算法不依赖 Qwen3 Worker | Qwen2.5-specific Final 在 VL-RewardBench 显著优于 Qwen2.5 Initial；Discovery/Dev 轨迹与算子接受记录完整 | B1、B2、B4 |
| C2：Worker-specific 演化是否优于直接 Rubric 迁移 | 决定未来应为每个 Worker 重跑演化，还是复用通用 Rubric | Qwen2.5-specific Final 与 transferred Phase17 E4 的配对 Overall/Macro/Strict ACC 比较 | B4 |
| Anti-claim：收益来自 Qwen3 历史 artifact 或 benchmark 选择 | 保证闭环因果有效 | fresh Qwen2.5 predictions、fresh ErrorSignatures、fresh candidates/history；Dev/heldout/VL-RewardBench 均不参与接受或 checkpoint 选择 | B1–B4 |

## 2. 冻结对照与唯一变量

### 2.1 固定项

- 初始化：相同五个 initial roots；
- Discovery：`data/discovery_v2_demo_v3/discovery_100.jsonl`，100条；
- Dev：`data/discovery_v2_demo_v3/dev_150.jsonl`，150条，只读诊断；
- Manager：`Qwen/Qwen3.5-397B-A17B`；
- Global Memory：`global_rubric_v1`；
- Worker Prompt：Pairwise Worker Prompt v2 `1.1.0-cache-pilot`；
- Worker decoding：`temperature=0.5`、`max_tokens=2048`；
- 调度：`vllm-8000 + vllm-8001` sample-major available-slot pool，全局并发40；
- 演化：Locked-Child Split + Role-aware Refine，同步 epoch commit；
- epoch：最少3、最多5；seed=42；
- Split trigger：`ACC < 0.75` 且使用既有高 Coverage 条件；
- Split retry：复用当前错误聚类，最多锁定1个强 child；
- Refine：保持 Phase17 的 root/child role-aware trigger、self-competition 和失败归因不变；
- 聚合：root 内现有 subtree aggregation，五个 roots 等权 M1；
- 不使用 Gate、Root Router、Root Boundary Pre-Refine、权重搜索或 benchmark feedback。

### 2.2 唯一核心变化

Pairwise Worker 改为 `Qwen/Qwen2.5-VL-7B-Instruct`，并从 Initial five-root 重新开始完整演化。以下 artifact 必须全部由 Qwen2.5 当前轨迹产生：

- Discovery100 initial predictions 与 decisive-wrong IDs；
- ErrorSignatures；
- Split children 的 Worker predictions 与竞争结果；
- Refine old/new predictions 与 self-competition；
- Dev150 每 epoch diagnostics；
- 所有失败历史和 Manager 下一轮反馈中的 child metrics。

禁止复用 Qwen3 Phase17 的 ErrorSignatures、clusters、candidate descriptions、accepted patches、失败归因或 Discovery/Dev predictions。即使 sample ID 和 criterion name 相同，也只有完整 request identity 与 Qwen2.5 model identity 均一致时才允许复用本实验自己的缓存。

## 3. 比较系统

| 系统 | Worker | Rubric 来源 | 作用 |
| --- | --- | --- | --- |
| Qwen2.5 Initial | Qwen2.5-VL-7B | 初始五 roots | 同模型演化基线 |
| Qwen2.5 Transferred E4 | Qwen2.5-VL-7B | Qwen3 Phase17 E4，既有结果只读复用 | 通用 Rubric 直接迁移基线 |
| **Qwen2.5-specific Final** | Qwen2.5-VL-7B | 本实验从 Initial 完整演化 | 主要 Treatment |
| Qwen3 Phase17 E4 | Qwen3-VL-8B | 既有结果只读复用 | 能力上限背景，不作为主因果对照 |

Qwen2.5 Native 已在 transfer 实验中完成，只作背景结果，不重复运行。主因果比较是 `Qwen2.5-specific Final vs Qwen2.5 Initial`；`specific Final vs transferred E4` 用于判断重新演化是否必要。

## 4. Experiment Blocks

### B1：Fresh Qwen2.5 闭环审计与 smoke

- **Claim tested**：演化输入确实来自 Qwen2.5，而非 Qwen3 历史 artifact。
- **Dataset**：Discovery100；smoke 固定一个 root 和小规模候选链路。
- **Checks**：两个 endpoint identity、Prompt v2 request spec、fresh initial predictions、fresh signature policy、Global Memory、Manager specs、heldout isolation。
- **Success criterion**：五个初始 roots 的 Qwen2.5 ACC/Coverage 可重算；smoke 完成 signature→cluster→candidate→competition；Qwen3 artifact reuse count=0。
- **Failure interpretation**：任何身份混用都使主实验无效，必须修复后重新 freeze。
- **Priority**：MUST-RUN。

### B2：Discovery100 + Dev150 五轮完整演化

- **Claim tested**：Qwen2.5 能根据自身错误完成有效 Split+Refine。
- **Acceptance data**：仅 Discovery100。
- **Diagnostic data**：Dev150；Manager 不可见，不早停、不接受候选、不选择 checkpoint。
- **Metrics**：每 epoch Discovery/Dev Overall、Macro、三领域 ACC、Coverage、corrected/harmed；每 root/node ACC、support、wrong；Split/Refine 接受轨迹、locked children、失败历史、node count。
- **Formal output**：协议自然停止时的最后 committed Rubric；不使用 Dev 最佳点替换。
- **Success criterion**：至少形成一个有效结构变化，Final 相对 Initial 在 Discovery 上净改善；Dev 不作为硬 gate，但若持续明显下降必须在报告中标记适配风险。
- **Failure interpretation**：若无法超过 Initial，说明当前错误反馈/阈值对 Qwen2.5 不适配，而不是证明 transferred E4 无效。
- **Priority**：MUST-RUN。

### B3：RLHF-V heldout-500 探索性回归

- **Claim tested**：Qwen2.5-specific 演化是否保留视觉幻觉同域能力。
- **Systems**：Qwen2.5 Initial、Transferred E4、Qwen2.5-specific Final。
- **Metrics**：Strict ACC、Covered ACC、Coverage、corrected/harmed/net corrected、五个 root subtree。
- **Selection rule**：heldout 只在 formal Final 冻结后运行，禁止改变接受结果或 checkpoint。
- **Evidence boundary**：该 heldout 已被多次使用，且 Discovery-v2 有少量已知重叠，只能作为 exploratory regression。
- **Priority**：MUST-RUN，但不是主要结论。

### B4：VL-RewardBench 主要外部验证

- **Claim tested**：Qwen2.5-specific Rubric 是否在外部分布上优于 Initial，以及是否超过直接迁移 E4。
- **Dataset**：VL-RewardBench 完整1,247对。
- **Protocol**：Prompt v2、K=3、相同 seed=42 counterbalanced A/B schedule、Qwen2.5 双端点、等权五-root M1。
- **Primary metrics**：OverallAcc、MacroAcc、Coverage、Strict ACC；paired corrected/harmed/net corrected 与 exact McNemar。
- **Secondary metrics**：General/Hallucination/Reasoning；每 root subtree；最强/最弱节点；位置偏置；技术解析率与请求成本。
- **Success criterion**：
  - 对 Initial：Strict/Overall 明确提高且 paired net corrected > 0；`p<0.05` 为强证据；
  - 对 Transferred E4：提高且 paired net corrected > 0 表明 Worker-specific 共适配有额外价值；若持平，则通用迁移已经足够；若下降，则重新演化发生过适配或数据过拟合。
- **Selection rule**：benchmark 不得选择 epoch、root subset、权重或节点；任何组合分析只能标记 post-hoc。
- **Priority**：MUST-RUN。

### B5：机制与失败诊断

- **Claim tested**：收益来自哪些错误类型和结构变化。
- **Analyses**：Qwen2.5 与 Qwen3 的 initial wrong overlap；fresh children 与 transferred E4 的语义对应；accepted Split/Refine 在三领域的 corrected/harmed；root 冲突与位置偏置；Final 与 Transferred E4 的节点数量和推理成本。
- **Failure interpretation**：若 Final 只改善 Hallucination、General/Reasoning 不变，应限定为视觉事实偏好适配，而非通用 judge 提升。
- **Priority**：MUST-RUN 的离线诊断，不增加模型请求。

## 5. 结果解释矩阵

| Qwen2.5-specific Final 相对结果 | 结论 |
| --- | --- |
| 高于 Initial，且高于 Transferred E4 | Qwen2.5 可从自身错误中进一步共同适配；每个 Worker 独立演化有价值 |
| 高于 Initial，与 Transferred E4 持平 | Rubric 有效，但 Qwen3 演化的规则已充分迁移；重复完整演化不是必要成本 |
| 高于 Initial，但低于 Transferred E4 | 演化机制有效，但 Qwen2.5-specific trajectory/Discovery100 出现过适配；通用强 Worker 生成的 Rubric 更优 |
| 与 Initial 持平或下降 | 当前 Split/Refine 阈值或 Manager evidence 不适合 Qwen2.5，不能声称完整算法跨 Worker 稳健 |

## 6. Stages 与 Artifact

建议新增演化 stages：

```text
qwen25-evolution-freeze
qwen25-evolution-audit
qwen25-evolution-smoke
qwen25-evolution-run
qwen25-evolution-report
qwen25-evolution-heldout
qwen25-evolution-final-report
```

演化输出：

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase18_qwen25_discovery_v2_prompt_v2_split_refine_v1/
```

建议新增外部评测 stages：

```text
vlrb-qwen25-evolved-freeze
vlrb-qwen25-evolved-audit
vlrb-qwen25-evolved-smoke
vlrb-qwen25-evolved-run
vlrb-qwen25-evolved-retry
vlrb-qwen25-evolved-report
```

外部评测输出：

```text
output/evolving_structured_rubrics/
  vl_rewardbench_qwen25_phase18_evolved_v1/
```

每个 epoch 继续保存 rubric memory、operation schedule、attempt artifacts、committed rubric、Discovery prediction、Dev diagnostics 和 evolution history。最终报告必须记录 Final rubric hash、dataset hash、Worker request spec、两端 endpoint identity，以及所有复用 artifact 的来源和 hash。

## 7. Run Order and Decision Gates

| Milestone | Goal | Runs | Decision gate | Estimated cost | Main risk |
| --- | --- | --- | --- | --- | --- |
| M0 | 冻结身份与数据 | freeze → audit | Qwen3 evolution artifact reuse=0；heldout未访问 | <5 min，离线 | request identity 掩盖 |
| M1 | 验证完整链路 | smoke | fresh signatures、candidate 和 Qwen2.5 competition 均有效 | 10–30 min + Manager | 397B网络波动 |
| M2 | 完整演化 | run → report | formal Final Rubric 冻结；Dev仅诊断 | 约3–8 h | 多轮 Manager 串行、失败重试 |
| M3 | 同域回归 | heldout → final-report | 不改变 Final 选择 | 约0.5–2 h | 新节点较多导致请求增长 |
| M4 | 外部主结果 | VL-RB freeze→audit→smoke→run→retry→report | Initial/Transferred/Specific 三方结果完整 | 约5–9 h | 位置偏置、解析重试 |

总计建议预留约9–19小时墙钟时间，主要由实际 node 数、Refine attempts 和397B Manager拥塞决定。若两个端点并行稳定，可在一天内完成；不得为了缩短时间中途改变并发、Prompt、解码或 epoch 协议。

## 8. 测试与验收

- Qwen2.5 initial predictions 必须为 fresh，五个 root 的 request spec model 均为 Qwen2.5；
- ErrorSignatures 的 decisive-wrong IDs 必须来自同 epoch Qwen2.5 predictions；
- Qwen3 signature/cluster/candidate/history reuse 数必须为0；
- Manager 仍为397B，Global Memory 与 failure history 同时注入；
- Split/Refine trigger、竞争、locked-child 和同步提交与 Phase17 完全一致；
- Dev150 不生成 ErrorSignature、不进入 Manager、不影响接受/早停/checkpoint；
- discovery 阶段无法读取 heldout 与 VL-RewardBench；
- 中断恢复保持 epoch snapshot、request identity 和已完成缓存不变；
- heldout/VL-RB 只接受 formal Final hash；
- Initial、Transferred E4、Specific Final 使用同一 Qwen2.5 Prompt v2 和 K=3 schedule 比较；
- Native 不重复运行，历史结果只读复用；
- focused unittest、完整 unittest、compileall 和 diff check 通过。

## 9. Evidence Boundary

- 单轨迹、seed=42，不能估计完整演化的跨 seed 方差；
- Discovery100 只有100条，仍可能产生 Manager 与候选选择方差；
- Dev150 是诊断集，不是新的模型选择集；
- heldout-500 与 VL-RewardBench 已被多次查看，结果属于 exploratory development evidence；
- 只比较 Qwen2.5 与 Qwen3 两个相近模型系列，不能直接宣称对任意 VLM model-agnostic；
- 若需要论文级确认，必须在新 Worker、独立外部偏好集或多 seed 轨迹上复核。

## 10. Final Checklist

- [ ] 主比较冻结为 Qwen2.5-specific Final vs Qwen2.5 Initial
- [ ] transferred Phase17 E4 作为第二冻结对照
- [ ] 所有 Qwen2.5 evolution evidence fresh
- [ ] Dev/heldout/VL-RewardBench 均不参与接受与选择
- [ ] Prompt v2、Split/Refine、Manager、seed 与 Phase17一致
- [ ] VL-RewardBench 报告 Overall/Macro/Coverage/Strict 与 paired McNemar
- [ ] 位置偏置、解析失败、请求成本和 root/node 诊断完整
- [ ] 结论严格区分通用迁移与 Worker-specific 共适配

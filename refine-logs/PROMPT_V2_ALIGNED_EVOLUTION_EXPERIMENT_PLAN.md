# Prompt v2 Aligned Evolution 实验计划

## 1. 实验问题

此前已经得到两个重要结果：

- Phase 10 的 `Five-root Locked-Split + Role-aware Refine` Rubric 是使用旧 Pairwise Worker Prompt 演化得到的；在旧 Prompt 下，heldout-500 M1 ACC 为 **71.4%**。
- 保持该 Rubric 不变，仅在推理阶段切换到 Pairwise Worker Prompt v2，heldout-500 M1 ACC 提升到 **76.6%**。

这说明 Prompt v2 能改善既有 Rubric 的执行，但尚未回答：

> 如果 Prompt v2 从初始化反馈开始就参与 Split、Refine、竞争和失败历史，是否能演化出一个比“旧 Prompt 演化 + Prompt v2 推理”更好的新 Rubric，并迁移到 VL-RewardBench？

本实验只改变 Pairwise Worker Prompt，不同时引入 Gate Worker、Root Boundary Pre-Refine、权重搜索或新的接受条件。

## 2. 核心假设与因果对照

记：

- \(R_{v1}\)：现有 Phase 10、由旧 Prompt 演化得到的 Rubric；
- \(R_{v2}\)：本实验、由 Prompt v2 全链路演化得到的新 Rubric；
- \(P_{v1}\)、\(P_{v2}\)：旧、新 Pairwise Worker Prompt。

主要对照为：

| 系统 | Rubric 来源 | 执行 Prompt | 作用 |
|---|---|---|---|
| \(R_{v1},P_{v1}\) | Phase 10 | v1 | 历史参考，heldout ACC 71.4% |
| \(R_{v1},P_{v2}\) | Phase 10 | v2 | **主要 Control**，heldout ACC 76.6% |
| \(R_{v2},P_{v2}\) | 本实验 | v2 | **Treatment** |

主要效应定义为：

\[
\Delta_{\text{aligned evolution}}
=
\operatorname{Acc}(R_{v2},P_{v2})
-
\operatorname{Acc}(R_{v1},P_{v2}).
\]

这样可将“Prompt v2 在执行阶段本身带来的收益”与“Prompt v2 参与演化带来的额外收益”分开。

### Claim map

| Claim | 最低支持证据 | 反证/替代解释 |
|---|---|---|
| C1：Prompt-v2-aligned evolution 能产生更优 Rubric | heldout-500 上 \((R_{v2},P_{v2})\) 高于 76.6%，且 corrected > harmed | 若只高于 71.4% 而不高于 76.6%，收益可能全部来自执行 Prompt |
| C2：改进可迁移到外部奖励模型基准 | VL-RewardBench 上 Treatment 的 OverallAcc、MacroAcc 高于 \((R_{v1},P_{v2})\)，paired net corrected > 0 | 若 heldout 提升而 VL-RB 下降，则可能过拟合 RLHF-V 幻觉分布 |
| C3：变化来自反馈与选择信号对齐，而非算法改动 | Split/Refine、聚合、epoch、Manager、seed 均冻结，仅 Worker Prompt 改变 | 节点数和请求数变化仍可能构成 test-time compute 差异，必须报告 |

本实验使用已多次访问的 discovery-90、heldout-500 和 VL-RewardBench，因此属于 **exploratory paired study**，不表述为新的无偏 confirmatory test。

## 3. 冻结协议

### 3.1 起点与算法

- 从原始五个 initial roots、零 children 重新开始。
- 不继承 Phase 10 的 children、Refine descriptions、竞争决定或失败历史。
- 不使用 Root Boundary Pre-Refine。
- 不使用 Gate Worker；五个 root 仍按既有等权 M1 聚合。
- 不使用 Visual Grounding 0.4 的后验权重。
- Manager 统一使用 `Qwen/Qwen3.5-397B-A17B`，seed=42，默认启用 `global_rubric_v1`。
- discovery 演化最少 3 epoch、最多 5 epoch，沿用同步 epoch commit。

### 3.2 Pairwise Worker Prompt v2

所有 discovery Pairwise 判断统一使用：

```text
model: Qwen/Qwen3-VL-8B-Instruct
prompt: STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1_1
prompt_version: 1.1.0-cache-pilot
temperature: 0.5
max_tokens: 2048
replicate: P05, single replicate
endpoints: vllm-8000 + vllm-8001
scheduler: available-slot dynamic scheduling
endpoint concurrency: 20 + 20
global concurrency: 40
```

Prompt v2 的用户消息顺序冻结为：

```text
image → question → Candidate A → Candidate B → criterion → final output instruction
```

`available-slot dynamic scheduling` 表示每当任一服务器释放并发槽，就立即把下一条请求发给该服务器，而不是预先固定两端各自的请求队列。

### 3.3 Prompt v2 必须贯穿整个演化闭环

```text
Prompt v2 初始 root predictions
→ feedback / decisive-wrong IDs
→ 397B ErrorSignatures
→ 397B clustering / child generation
→ Prompt v2 child Pairwise evaluation
→ Split competition / failure history
→ Prompt v2 Refine self-competition
→ synchronous commit
→ 下一 epoch 的 Prompt v2 feedback
```

不能只在最终推理阶段替换 Prompt。ErrorSignatures、聚类和 children 虽由 Manager 生成，但它们所读取的错误集合必须来自 Prompt v2 predictions。

### 3.4 Split 与 Refine 保持 Phase 10 定义

Split：

- 只调度 initial roots；Phase16-v2 使用 `ACC < 0.75` 且 `Coverage > 0.80`，使五个 Prompt-v2 初始 roots 都先获得一次 Split 尝试。该阈值仅属于本实验，不修改冻结的 Split v1 全局协议。
- children 集合在固定 parent scope 上计算 Specialized Accuracy。
- 接受条件保持 `specialized_accuracy >= parent_accuracy`。
- Split 失败后保留完整失败历史和自然语言归因。
- strong child 的锁定条件、hash-identical description 复用与 Pairwise prediction 复用均保持 Phase 10 协议；不改变部分接受规则。

Refine：

- root：`0.55 < ACC < 0.80`、`Coverage <= 0.80`、`support >= 15`；
- child：`0.5 < ACC < 0.80`、`support >= 15`、`wrong >= 5`；
- 接受条件保持新 description 的自身 conditional ACC 严格高于旧 description，且 support 不低于 15；
- subtree 和完整 M1 只作诊断，不改变 Refine 接受决定。

## 4. Artifact 复用与隔离

允许复用：

- 若五个 initial roots 的 criterion、Prompt v2 hash、解码参数、模型、数据和 endpoint request identity 完全一致，可从既有 Prompt v2 ablation 中投影复用初始 root predictions。
- \((R_{v1},P_{v2})\) 的 heldout-500 和 VL-RewardBench 结果作为只读 Control。
- 对最终新 Rubric 中与 Control 完全同名且 description/hash/request identity 完全一致的 criterion，可复用底层 Pairwise cache。

禁止复用：

- Phase 10 的 Prompt-v1 ErrorSignatures；
- Phase 10 的 clustering、child proposals、candidate Pairwise、accept/reject 决定和 failure histories；
- 仅凭 sample ID 或 criterion name 复用预测；
- heldout 或 VL-RewardBench 的 gold、指标或错误样本进入 discovery Manager 上下文。

原因是 Prompt 改变后，parent predictions 和 decisive-wrong IDs 可能变化，旧 ErrorSignatures 不再代表同一个错误分布。只有 request identity、parent predictions 和 decisive-wrong IDs 全部一致时，才允许逐项复用 signature。

## 5. 实验阶段

### E0：Freeze 与 offline audit

建议新增 stages：

```text
prompt-v2-evolution-freeze
prompt-v2-evolution-audit
```

冻结：

- initial rubric/data/hash；
- Phase 10 Control rubric/hash；
- Prompt v2 模板/hash和 request spec；
- Split/Refine 阈值、epoch、seed、Manager 和 aggregation；
- heldout-500 与 VL-RewardBench 只允许在对应 final stage 访问；
- Control 报告和可复用 prediction 的逐项来源/hash。

Audit 必须证明旧 Prompt cache 不会命中新 Prompt 请求，并报告 initial-root Prompt v2 预测的 generated/reused 数量。

### E1：单 root smoke

```text
prompt-v2-evolution-smoke
```

- 选择一个满足 Split 条件的 initial root；
- 跑通 Prompt v2 feedback → ErrorSignature → clustering → children → Pairwise → competition；
- 禁止访问 heldout；
- 验证两个 endpoint 都收到请求、JSON valid rate、prompt version、max_tokens 和 Manager provenance；
- smoke 不提交到正式演化历史。

### E2：Prompt-v2-aligned full evolution

```text
prompt-v2-evolution-run
prompt-v2-evolution-report
```

输出目录：

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase16_prompt_v2_locked_split_refine_v2/
```

每个 epoch 保存 rubric snapshot、Prompt v2 predictions、feedback、Split/Refine attempts、失败历史、同步提交结果、完整 M1 和 provenance。最终 report 后冻结唯一 final rubric hash；不得根据 heldout 或 VL-RewardBench 选择 epoch。

### E3：heldout-500 paired evaluation

```text
prompt-v2-evolution-heldout
prompt-v2-evolution-final-report
```

三个系统统一使用 Prompt v2 评估：

| 系统 | 作用 |
|---|---|
| Initial five-root + P2 | 初始化参考 |
| Phase 10 \(R_{v1}\) + P2 | 主要 Control，现有 76.6% |
| New \(R_{v2}\) + P2 | Treatment |

报告：ACC、Coverage、正确数、Wilson 95% CI、corrected/harmed、net corrected、exact McNemar、每个 root/subtree 的 ACC 和支持数、节点数与请求成本。

### E4：VL-RewardBench 外部迁移

无论 E3 提升、持平还是下降，最终冻结的新 Rubric 都必须运行 VL-RewardBench，避免结果依赖的选择性外测。

建议新增 stages：

```text
vlrb-prompt-v2-evolved-freeze
vlrb-prompt-v2-evolved-audit
vlrb-prompt-v2-evolved-smoke
vlrb-prompt-v2-evolved-run
vlrb-prompt-v2-evolved-retry
vlrb-prompt-v2-evolved-report
```

输出目录：

```text
output/evolving_structured_rubrics/
  vl_rewardbench_phase16_prompt_v2_evolved_v2/
```

冻结并复用既有 VL-RewardBench Prompt v2 协议：

- 1,247 个 preference pairs；
- K=3，固定 seed=42 的 balanced A/B swap schedule；
- 每个 replicate 独立推理，映射回原始 A/B 后多数投票；
- Prompt v2、temperature=0.5、max_tokens=2048；
- `vllm-8000 + vllm-8001` available-slot pool；
- 解析或技术失败最多重试 10 次，不根据 gold 触发重试；
- 使用结构化 Rubric 推理，不使用 benchmark 的通用 judge prompt；
- 等权五-root M1，不使用 Gate Worker、oracle routing 或后验权重。

VL-RewardBench 主对照：

| 系统 | OverallAcc | MacroAcc | 角色 |
|---|---:|---:|---|
| Initial five-root + P2 | 58.12% | 54.60% | 现有初始化参考 |
| Phase 10 \(R_{v1}\) + P2 | 69.53% | 63.37% | 主要 Control |
| New \(R_{v2}\) + P2 | 待测 | 待测 | Treatment |

最终报告同时给出 General、Hallucination、Reasoning 三类准确率、Coverage、严格正确数、paired corrected/harmed、exact McNemar、K=3 位置一致性、解析率、重试数和总请求量。

## 6. 成功标准与结果解释

### 主要成功标准

- 所有正式接受的 Split/Refine 均满足冻结的 discovery 接受条件；
- heldout-500：Treatment 高于 **76.6%**，且相对 \((R_{v1},P_{v2})\) corrected > harmed；
- VL-RewardBench：Treatment 的 OverallAcc 和 MacroAcc 均高于 **69.53% / 63.37%**，且 Coverage 不出现实质下降；
- 请求身份、final rubric hash、预测和报告可 replay；技术失败最终可审计。

### 结果解释

| 结果 | 解释 |
|---|---|
| heldout 与 VL-RB 都提升 | 支持 Prompt v2 的执行语义与演化反馈对齐能产生更可泛化的 Rubric |
| heldout 提升、VL-RB 下降 | 演化更适配 RLHF-V 幻觉分布，但外部迁移不足 |
| heldout 不超过 76.6%、VL-RB 提升 | Prompt-v2-aligned evolution 主要改善跨分布结构，而非当前 heldout |
| 两者均不提升 | Prompt v2 的已有收益主要来自推理期行为改变，未证明需要重新演化 |

McNemar 不显著时，只表述为 pilot/exploratory evidence。不得将节点更多或推理请求更多直接解释为算法质量提升，必须同时报告 Rubric 节点数与推理成本。

## 7. 测试与验收门槛

- Prompt v2 在 initial roots、children、Refine old/new 和最终 evaluation 中均实际生效。
- Prompt v1 与 Prompt v2 cache key 隔离；Prompt-v1 artifact 不得误命中。
- Prompt v2 prediction 改变后，ErrorSignatures 必须重新生成；精确不变时才允许逐项复用。
- Split v1、locked strong-child retry、Role-aware Refine 的阈值与接受逻辑不变。
- 同一 epoch 所有 candidates 使用同一个 epoch-start rubric memory，下一 epoch 才看到本轮提交结果。
- discovery 期间无法读取 heldout-500 或 VL-RewardBench 数据、gold、预测和指标。
- final rubric 在 E3/E4 前冻结，heldout/VL-RB 后禁止选择 epoch、children 或 description。
- VL-RewardBench K=3 的三次推理是独立请求，A/B swap 映射和多数投票正确。
- retry 只处理技术/解析失败，最多 10 次，且保存原始失败输出。
- 两个 endpoints 的模型、版本和关键生成配置必须验证；checkpoint 路径等部署字段只记录，不作为语义身份。
- 中断恢复只生成缺失请求，已完成且身份一致的结果不得重跑。
- focused tests、Split/Refine regression、VL-RB regression、full `unittest` 和 `compileall` 全部通过。

## 8. 预计成本与运行顺序

推荐顺序：

```text
E0 freeze/audit
→ E1 smoke
→ E2 full discovery evolution
→ 冻结 final rubric
→ E3 heldout-500
→ E4 VL-RewardBench（必跑）
```

成本按既有实验吞吐做区间估计：

- discovery 演化：约 2,000–5,000 次新增 Worker 请求，另有 397B Manager 请求；
- heldout-500：约 `最终新增/改写节点数 × 500` 次请求；
- VL-RewardBench：最坏为 `节点数 × 1,247 × 3`，22 节点约 82,302 次逻辑请求；相同 description/request identity 可减少实际新增请求；
- VL-RewardBench 仍是主要耗时阶段，建议预留约 6–10 小时并使用双服务器。

所有耗时、token、cache reuse 和 endpoint call counts 以实际报告为准，不以预计值替代结果。

# Evolving Structured Rubrics Implementation Plan

> 状态：Phase22、本地27B Manager演化及其Epoch 5完整候选评测已完成；完整候选显示迁移收益，但接受机制尚未证明能稳定选出泛化更好的Rubric。
>
> 更新日期：2026-09-03
>
> 本文按时间记录设计与结果；早期章节中的“当前/下一阶段”属于当时语境。当前入口见[实验索引](experiments/README.md)，实际运行状态以产物为准。
>
> 对应 Idea：[Evolving Structured Rubrics from Multimodal Preferences.md](Evolving%20Structured%20Rubrics%20from%20Multimodal%20Preferences.md)

---

## 1. 目标与当前正式路线

本项目研究如何让自然语言评价准则从扁平列表逐步演化为 Structured Rubric Forest。Forest 可以包含多个 roots、共享 parent 和多个 children，并通过条件边决定每个样本实际执行的路径。

当前正式执行路线为：

```text
Multimodal Preference Pair
    ↓
Root Router（仅 M2）
    ↓
StructuredRubric 中选中的 roots
    ↓
Pairwise Vote Worker ──→ A / B / abstain，唯一聚合投票来源
Gate State Worker    ──→ applicable/status，仅控制 status-dependent edges
    ↓
DualCascadeExecutor
    ↓
Flat 或 Hierarchical Aggregation
    ↓
A / B / Tie + Dual Trace v2
```

Phase 0–4 已完成表示、执行、缓存、重放和实验验证。下一阶段的目标是利用 Pairwise、Gate、Router 和 cascade trace 形成的反馈，演化 criteria 文本和 Rubric 结构。

### 1.1 核心主张

- **结构化执行**：在共享相同 Pairwise outputs 的条件下，比较 routing 和 hierarchical aggregation 是否改善准确率或效率。
- **结构演化**：后续每个 Rubric edit 必须通过完整 cascade 的修改前后对照决定接受或回退，不能只看单节点指标。
- **可归因实验**：模型输出、路由、结构和聚合的影响必须分开，accuracy 只从 offline shared-output replay 计算。

### 1.2 兼容边界

早期 Structured Worker v1 将 applicability、A/B status、pair preference 和 evidence 放在同一次生成中，造成 criteria 执行语义漂移。其 `NodeJudgement`、`StructuredCascadeExecutor` 和 Trace v1 只保留用于历史 artifact replay，不再作为正式实验或后续 evolution loop 的输入。

---

## 2. Phase 0：冻结共享结构语义

### 2.1 Vote 与失败语义

正式聚合只读取 Pairwise Worker 的本地投票：

```text
Vote.A
Vote.B
Vote.ABSTAIN
```

模型明确输出 None/U 才属于有效 abstain。Parse failure、非法 answer 和模型 abstain 必须保留不同 provenance；parse failure 不得被解释为 nondecisive。

### 2.2 Forest 不变量

- Forest 至少包含一个 root；
- 每个 node 最多一个 parent，第一版不支持 DAG；
- 一个 parent 可以有多个 children；
- 多个 children 可以同时激活；
- 每个 node 只属于一棵 root subtree；
- 环、悬空边、重复 node/criterion 和多 parent 明确失败；
- 多前提依赖暂存于 lineage，不强行构造伪链。

### 2.3 EdgeCondition

| Condition | 正式判断来源 | 匹配规则 |
|---|---|---|
| `ALWAYS` | 无 | 直接进入 child |
| `PARENT_NONDECISIVE` | Pairwise output | parse 成功且模型明确 abstain |
| `PARENT_BOTH_PASS` | Gate output | Gate valid、applicable=yes，且 A/B 都 pass |
| `PARENT_BOTH_FAIL` | Gate output | Gate valid、applicable=yes，且 A/B 都 fail |

Gate parse/consistency invalid 或 `applicable!=yes` 时，只阻止依赖 Gate status 的 outgoing edges。它不会覆盖 Pairwise vote，也不会把当前节点重新标注为 abstain。

### 2.4 Traversal 与聚合

- All-nodes traversal 忽略 edge conditions，执行全部 nodes；
- Conditional traversal 先执行 parent，再根据上表决定 eligible children；
- 同一共享 parent 每个样本最多执行一次；
- 每个 eligible child subtree 向上最多返回一票；
- child votes 有唯一多数时采用多数，否则回退 parent vote；
- 每棵 selected root subtree 最多贡献一票；
- selected roots 最终采用 uniform aggregation；无唯一多数返回 Tie。

Root Router 输出 invalid、空选择、重复或未知 root 时，可追踪地回退 all roots。Historical weighting 仅保留为非正式消融工具，不属于当前 core 方法。

### 2.5 完成状态

- [x] Vote、EdgeCondition、TraversalPolicy 和 aggregation 语义冻结
- [x] 多 roots、多 children、共享 prefix 和 subtree 一票语义冻结
- [x] Root Router invalid fallback 语义冻结
- [x] 纯语义单元测试覆盖关键真值表与失败类型

---

## 3. Phase 1：实现 Pairwise/Gate 双通道 Worker

### 3.1 Pairwise Vote Worker

Pairwise Worker 逐字复用 exp4 的判断 prompt 语义，并保存：

```json
{
  "thought": "...",
  "answer": "A | B | None"
}
```

- `answer` 映射为 A/B/abstain，是 aggregation 的唯一权威 vote；
- `thought` 随 artifact 保存，供后续 Manager reflection 使用；
- schema/字段错误触发重试；
- JSON 可解析但 answer token 非法时保留旧 evaluator 行为：answer invalid、投 abstain，但不伪装成 parse failure；
- P05（temperature=0.5）是主配置，P00（temperature=0）是解码消融。

### 3.2 Gate State Worker

Gate Worker 只输出：

```json
{
  "applicable": "yes | no | uncertain",
  "status_a": "pass | fail | uncertain",
  "status_b": "pass | fail | uncertain"
}
```

Gate 不输出 preference 或 evidence，也不参与 aggregation。`applicable!=yes` 时两个 status 必须为 uncertain。Gate 固定 temperature=0，只为具有 `PARENT_BOTH_PASS/FAIL` outgoing edges 的 parent 生成稀疏 artifact。

### 3.3 Artifact 与 cache

- `PairwisePredictionOutput` 覆盖全部 sample×criterion，并保存 flat answers；
- `GatePredictionOutput` 只覆盖 status-dependent parent criteria；
- artifact 保存 sample fingerprint、criterion 原文、request identity 和协议版本；
- Pairwise/Gate 使用独立 cache namespace；
- cache key 对 A/B 顺序、图片内容、criterion 文本、prompt、backend 和 decoding config 敏感；
- corruption 明确失败或进入 quarantine，不静默当作普通 miss。

### 3.4 完成状态

- [x] Pairwise/Gate parser、retry、artifact 和 cache 已实现
- [x] Pairwise parser 保存 `thought` 并区分 abstain/invalid/parse failure
- [x] Gate consistency 独立于 Pairwise preference
- [x] 原 CritiQ-V evaluator 行为未被破坏

---

## 4. Phase 2：实现 Rubric Tree/Forest

### 4.1 数据结构

```text
RubricCriterionSnapshot
RubricNode
RubricEdge(parent_id, child_id, EdgeCondition)
StructuredRubric(nodes, edges, root_ids)
```

Rubric、nodes、edges、examples 和 lineage 均为递归不可变对象。`node_id` 与 `criterion.name` 相互独立，但二者都必须在 Forest 内全局唯一。

### 4.2 校验与复现

- 构造时自动校验 Forest 不变量；
- children 按 child ID、roots 按声明顺序稳定遍历；
- JSON loader 严格校验字段、schema/semantics version 和 canonical SHA-256；
- hash 覆盖 criterion、score、topology、metadata、版本和 root 顺序；
- 不支持旧 Rubric schema 的静默迁移。

正式 offline backends 执行以下兼容检查：

- Pairwise artifact 必须与 Rubric 的全部 criterion name/description 完全一致；
- Gate artifact 中的稀疏 criteria 必须与对应 Rubric nodes 完全一致；
- Router artifact 必须匹配 sample fingerprint、ordered roots 和 Rubric hash；
- offline replay 不创建 Agent，也不访问网络。

### 4.3 完成状态

- [x] 单 parent、多 children、多 roots 的不可变 Forest 已实现
- [x] 查询顺序、序列化和 canonical hash 可复现
- [x] Pairwise/Gate/Router artifact 与 Rubric 的离线匹配已实现

---

## 5. Phase 3：实现 Root Router 与 Dual Cascade Executor

### 5.1 五种执行系统

| Variant | Roots | Traversal | Aggregation | Gate | Router |
|---|---|---|---|---|---|
| B1 Flat | all | all nodes | flat | no | no |
| H1 Hierarchical-only | all | all nodes | hierarchical | no | no |
| G1 Gating-only | all | conditional | flat | status edges only | no |
| M1 All-roots Cascade | all | conditional | hierarchical | status edges only | no |
| M2 Routed-roots Cascade | selected/fallback | conditional | hierarchical | status edges only | yes |

同一温度下，五个系统必须共享同一 Pairwise artifact。G1/M1/M2 额外共享同一 Gate artifact；只有 M2 读取 Root Router artifact。

### 5.2 执行流程

1. M2 联合路由 roots；其他系统直接使用 all roots；
2. 对每个实际访问的 node 请求或读取一次 Pairwise vote；
3. Conditional traversal 仅在 node 存在 status-dependent outgoing edges 时读取 Gate；
4. 根据 EdgeCondition 执行全部 eligible children；
5. Flat 模式聚合所有 visited local votes；
6. Hierarchical 模式递归聚合 child subtrees，再聚合 selected roots。

被 edge 跳过的 child 和未选择的 root 不产生 Pairwise/Gate 调用或 cache lookup。并发可以跨 samples 和 siblings，但每个样本内部仍遵守 parent→child 依赖。

### 5.3 Root Router

Root Router 一次联合选择一个或多个 roots，不判断 A/B preference。Valid selection 按 Rubric root 顺序规范化；invalid decision 回退 all roots并保留原因。Router request、artifact 和 cache identity 覆盖 ordered root descriptions 与 Rubric hash。

### 5.4 Dual Trace v2

每个 trace 分开记录：

- Pairwise output、vote、metrics 和 cache provenance；
- 可选 Gate output、metrics 和 edge condition source；
- Router output、resolved roots 和 fallback reason；
- visited/avoided nodes、edge matched/skipped、subtree/root votes；
- Pairwise、Gate、Router 的真实 API attempts、tokens 和 latency。

Replay 必须从保存的 outputs 重建 traversal、aggregation 和成本汇总；结构、edge、vote、selected roots 或 metrics 被篡改时明确失败。

### 5.5 完成状态

- [x] `DualCascadeExecutor` 和 Dual Trace v2 已实现
- [x] 五种系统语义、共享 prefix 和 Gate sparse calling 已测试
- [x] offline replay、online lazy execution、cache 和成本 telemetry 已验证
- [x] Root Router 多选与 fallback 已验证

---

## 6. Phase 4：使用已有 Criteria 运行静态 Cascade

### 6.1 冻结输入与结构

Phase 4 使用 exp4 最终 17 条 criteria 原文和 heldout-500 工程评估集。heldout-500 已被用于多轮工程诊断，不再解释为 unseen test。

正式结构为 `strict_v0`：

```text
17 nodes / 15 roots / 2 edges

multimodal_alignment
├── PARENT_BOTH_PASS → visual_coherence
└── PARENT_BOTH_PASS → semantic_specificity
```

`projected_v0` 仅是早期多前提投影诊断结构，由 factory 保留以复现历史，不参与当前正式结果。

### 6.2 Shared-output 实验流程

唯一正式入口为：

```powershell
python -m experiments.evolving_structured_rubrics.run_shared_output_pool --help
```

执行阶段：

```text
freeze
→ endpoint-check
→ pairwise-p05 / pairwise-p00
→ gate
→ router
→ reserve-p05（三次独立运行）
→ offline
→ report
```

- endpoint-check 验证两个 vLLM endpoints 的身份、协议和语义一致性，不读取 gold；
- P05/P00 分别生成 heldout-500 全节点 Pairwise artifacts；
- Gate t0 和 Router t0 各生成一份 heldout-500 + reserve-50 的 eval550 artifact；
- reserve-50 的 P05 重复三次，用于估计随机方差；
- 只有 offline stage 读取 gold 并计算 accuracy；
- B1 必须逐样本等于 Pairwise artifact 的 flat answers。

### 6.3 最终结果

| Pairwise | B1 | H1 | G1 | M1 | M2 |
|---|---:|---:|---:|---:|---:|
| P05 | 0.688 | **0.692** | 0.690 | **0.692** | 0.674 |
| P00 | **0.682** | 0.680 | 0.678 | 0.674 | 0.664 |

P05 的 B1 恢复到接近 exp4 的 0.696，说明 Pairwise/Gate 解耦解决了主要 Worker 语义漂移。H1/M1 仅比 B1 高 0.4 percentage point，尚不能证明手工静态 Forest 有稳定收益；M2 低于 M1/B1，说明当前 Root Router 会遗漏有用 roots。

reserve-50 的三次 P05 运行显示不同系统 accuracy 的样本标准差约为 0.012–0.040，小幅单次增益需要结合重复实验或置信区间解释。

最终结论：

```text
PASS_BASELINE_RECOVERY
REVISE_CASCADE
REVISE_ROOT_ROUTER
```

旧 Structured Worker 曾使 B1 从 0.696 降至 0.606，这一失败结果只用于解释双通道架构的来源；详细实验身份、结果和 artifact hash 见 [Shared-output 实验总结](experiment-results/shared_output_pool_v1_summary.md)。

### 6.4 Phase 4 完成状态

- [x] 两 endpoint equivalence check 通过
- [x] P05/P00 Pairwise、Gate t0、Router t0 artifacts 完成
- [x] B1/H1/G1/M1/M2 shared-output replay 完成
- [x] reserve-50 三次 P05 随机方差完成
- [x] 完整 artifacts 已移到仓库外，Git 只保留精简结果与 hash
- [x] 当前基础设施足以为 Rubric evolution 提供反馈

---

## 7. Phase 0–4 Core 完成标准

- [x] 原 CritiQ-V evaluator 行为未被破坏
- [x] Pairwise vote 与 Gate state 完全解耦
- [x] Forest 支持单 parent、多 children、多 roots 和共享 prefix
- [x] 多个 eligible children 可以同时执行
- [x] 每棵 selected root subtree 最多贡献一票
- [x] Root Router valid 时只进入 selected roots，invalid 时回退 all roots
- [x] Pairwise/Gate/Router artifacts 可严格保存、加载和重放
- [x] Dual Trace v2 可重建 traversal、aggregation 和成本
- [x] skipped child 和 unselected root 不产生模型请求
- [x] 同温度五系统严格共享 Pairwise outputs
- [x] Pairwise P05 baseline recovery 达到门槛
- [x] 静态 Cascade 和 Root Router 的当前问题已被量化
- [x] 自动演化算子尚未实现，符合阶段边界

当前稳定模块：

```text
critiq/structured/                    # Rubric、Router、Executor、Trace、Cache
critiq/dual_evaluator.py              # Pairwise/Gate 多模态 evaluator
critiq/dual_worker_prompts.py         # 双通道 prompts
experiments/evolving_structured_rubrics/
└── run_shared_output_pool.py         # 唯一正式实验入口
```

---

## 8. Phase 5：Evolution Core 与 Init Baseline

Phase 5 先建立可审计的 `propose → apply → refresh → evaluate → accept/reject` 语义层，不实现真实演化算子。Rubric 为不可变对象；reject 时继续使用 `R_before`，不执行危险的逆向 rollback。

### 8.1 数据边界

```text
discovery_train_90_pair.jsonl
  → 生成全节点 Pairwise P05 artifact
  → 提取反馈、生成/筛选候选、决定接受或拒绝

heldout_validation_500_pair.jsonl
  → 只评估 Init Rubric 和最终 Evolved Rubric
  → 中间 operator 与 evolution rounds 禁止读取
```

当前 discovery-90 是探索集，允许方法对其适配；heldout-500 的中间访问由 CLI stage 和 manifest 阻止。以后增加 reserve-100 后，再将 candidate acceptance 从 discovery-90 移到 reserve-100。

### 8.2 Multi-Crit Init Rubric

初始 Rubric 使用 [Multi-Crit(CVPR2026)](https://arxiv.org/abs/2511.21662) 为 Open-ended Generation 人工定义的五条准则原文。当前 RLHF-V discovery 数据主要是视觉问答、详细描述和图像内容解释，因此第一版不混入 Verifiable Reasoning 的另一套五条准则。

```text
Completeness and Coverage
  Address the full scope of the task in the user’s query, covering all major
  elements specified in the prompt as well as relevant visual aspects and
  contextual cues.

Visual Grounding and Details
  Reference observable elements in the image such as objects, spatial
  relationships, colors, or text, and bases its description or analysis on
  these details.

Factuality / No Hallucination
  Avoid visual or factual errors, ensuring all details and claims are presented
  in the image or reasonably supported by the prompt.

Creativity and Expressiveness
  Demonstrates imagination and originality when appropriate, or precise and
  knowledgeable articulation for analytical tasks, while remaining
  contextually appropriate.

Clarity and Coherence
  Communicates ideas clearly and logically, with fluent language,
  well-organized structure, and smooth flow of information.
```

结构固定为 `5 nodes / 5 roots / 0 edges`，score 均为 `1.0`，examples 初始为空，lineage 保存论文、任务类型、原始顺序和初始化版本。由于每棵 root 都是单节点 subtree，Init Rubric 的 B1 与 M1 必须逐样本完全一致。

### 8.3 Feedback 与 Gap

节点统计只读取 discovery-90 的 all-node Pairwise artifact：

```text
support       = valid decisive A/B 数
accuracy      = decisive votes 中与 gold 一致的比例
coverage      = support / 90
wrong         = valid decisive 但与 gold 不一致
abstain       = valid None/U
answer_invalid / parse_invalid 分开统计
```

Gap 定义为：对样本 `s`，当前所有 node 都没有产生与 gold 相同的有效 decisive vote。必须区分：

- `rubric_gap`：all-node outputs 中没有正确 decisive vote；
- `cascade_failure`：M1 final 为 wrong/Tie；
- `aggregation_conflict`：存在正确 decisive node，但 M1 final 仍为 wrong/Tie。

旧版 Fitness 的长度与 coverage 惩罚仅保留为历史诊断。当前 Split 的硬接受标准不再使用加权 Fitness，而是在父准则固定适用域上直接比较 parent 与完整 children 集合的 Specialized Accuracy。长度交给后续 `Refine` 算子优化；coverage 过低由生存条件中的最小支持数处理。

父准则的固定适用域定义为其产生有效 decisive A/B 的样本集合：

$$
\mathcal S(c_p)=\{x:c_p(x)\in\{A,B\}\}.
$$

对每个 $x\in\mathcal S(c_p)$，children 先投票，`None` 不参与。A/B 不平票时采用 children 多数结果；平票或全部输出 `None` 时回退父准则：

$$
\hat y_{\mathrm{spec}}(x)=
\begin{cases}
A,&N_A(x)>N_B(x),\\
B,&N_B(x)>N_A(x),\\
c_p(x),&N_A(x)=N_B(x).
\end{cases}
$$

parent 与 specialized 必须使用同一个分母 $|\mathcal S(c_p)|$：

$$
\operatorname{Acc}_{\mathrm{parent}}
=
\frac{\sum_{x\in\mathcal S(c_p)}\mathbf 1[c_p(x)=y(x)]}
{|\mathcal S(c_p)|},
$$

$$
\operatorname{Acc}_{\mathrm{spec}}
=
\frac{\sum_{x\in\mathcal S(c_p)}\mathbf 1[\hat y_{\mathrm{spec}}(x)=y(x)]}
{|\mathcal S(c_p)|}.
$$

children 是一次共同生成的完整拆分方案，不逐个筛除。接受条件为

$$
\boxed{\operatorname{Acc}_{\mathrm{spec}}\ge\operatorname{Acc}_{\mathrm{parent}}}.
$$

每个 child 的 support、accuracy、coverage、cluster accuracy、非目标激活和 leave-one-out 影响继续记录，但只用于诊断与后续 `Refine`；全部 discovery 样本上的 M1 ACC、Coverage 也不参与 Split 的硬接受判定。

### 8.4 Patch 与 ArtifactRefreshPlan

Evolution Core 提供版本化 candidate、patch、diff、feedback、refresh plan 和 decision。Patch 在应用前校验 base Rubric hash，应用后复用 Phase 2 Forest validation。

Artifact 影响规则：

| 修改 | Pairwise | Gate | Router |
|---|---|---|---|
| 修改 description | 刷新该 node | 若为 status parent 则刷新 | 若为 root 则标记 stale |
| 新增 child | 生成 child | `ALWAYS` 不需要 Gate | 不变 |
| 新增 root | 生成 root | 无 status edge则不需要 | 标记 stale |
| 只修改 edge | 复用 | 新增 status edge 时补齐 | roots 不变则复用 |
| 删除 node | 删除对应输出 | 删除无用输出 | root 集合变化时标记 stale |

未受影响 Pairwise outputs 必须逐项复用，避免把采样差异解释为 Rubric edit 的收益。

### 8.5 阈值校准与阶段门槛

Phase 5 已完成一次 Init baseline。discovery-90 的 M1 accuracy/coverage 为 `0.6556/0.9667`，heldout-500 的 Init 结果为 `0.6500/0.9720`，B1 与 M1 逐样本一致；五个 roots 的 coverage 均超过 `0.91`，节点 accuracy 位于 `0.5632–0.6966`。discovery-90 中有 15 个 rubric gaps、31 个 cascade failures，其中16个属于 aggregation conflicts。

基于该分布，第一版触发与结构阈值冻结为：

```text
tau_acc         = 0.55
tau_split       = 0.70
tau_cov_high    = 0.80
tau_refine      = 0.80
N_min_support   = 15
N_min_wrong     = 15
N_min_cluster   = 5
max_children    = 5
```

这些数值只产生 operator 候选，不直接决定接受。Split 的统计候选需满足 `accuracy < 0.70`、`coverage > 0.80`、`support >= 15`、`wrong >= 15`；Refine 候选需满足 `0.55 < accuracy < 0.80`、`coverage <= 0.80`、`support >= 15`。Pairwise Worker 在第一轮继续使用 P05（`temperature=0.5`）。

接受规则按 operator 区分。Refine 使用同一 node 改写前后的 selective accuracy 自竞争，Create 暂保留 discovery-90 M1 结果门槛；Split 仅比较同一父准则区域上的局部准确率：

$$
\operatorname{Acc}_{\mathrm{spec}}\ge\operatorname{Acc}_{\mathrm{parent}}.
$$

条件成立时保留 parent，并整体接纳本次共同生成的 children；条件不成立时丢弃整组 children，Rubric 保持不变。完整 M1 accuracy、全局 coverage、corrected/harmed、child correction/harm、sibling conflict 和 final-valid rate 继续报告，但不参与 Split 硬接受判定。Rubric/Artifact 合法、数据隔离和 trace replay 仍是工程有效性前提。P05 每个候选只运行一次，不设置重复确认。

trigger、operator-specific acceptance 和单次 P05 execution 共同写入版本化配置及 frozen manifest。Refine 在 old/new 各自 A/B 支持集上比较 node accuracy，要求严格提升且新 support 不低于15；Split 以 $\mathcal S(c_p)$ 上的 Specialized Accuracy 为接受依据。Refine 的 subtree/M1 与 Split 的完整 M1 均只作诊断。

---

## 9. Phase 6：逐个实现 Refine、Split、Create

每个算子必须单独实现、真实运行、review 和提交；一轮不能同时接受多个不可归因的修改。

### 9.1 Refine

- 自动触发条件为 `0.55 < accuracy < 0.80`、`coverage <= 0.80`、`support >= 15`；forced smoke 只能绕过触发器，不能绕过 schema、provenance 或竞争。
- 397B Manager 根据全部 decisive-wrong、最多6条 correct/abstain 边界样本、固定的3/2/1多模态代表样本、历史失败归因和 epoch-start `global_rubric_v1` memory 生成一个 description 候选。
- node ID、criterion name、score、parent、edges、root 顺序和 topology 不变；description 必须包含 `Criterion focus`、`Applicable only when`、`Not applicable when`、`Decision rule` 四段且不超过1800字符。
- Pairwise Worker 只读取新 description；只刷新目标 node 的 discovery-90 outputs，其余节点预测逐项复用。
- old/new criterion 分别在自己的有效 A/B 支持集上计算 selective accuracy。仅当 `Acc(new) > Acc(old)` 且 `support(new) >= 15` 时接受；平局拒绝。subtree、完整 M1、coverage、overlap/conflict 只作诊断。
- 有效候选被拒绝后，必须先完成397B自然语言失败归因才能写入历史；transport failure 暂停并可恢复，连续 schema-invalid 不得伪装成完整失败历史。
- 第一项实验固定为 `verified_existence_over_hallucinated_volume` 的 forced smoke，输出到 `phase7_refine_operator_v1/`；只有 discovery 严格提升且 heldout-500 未明显反向退化，才允许启动 `phase7_split_refine_evolution_v1/`。

### 9.2 Split

触发候选为低 accuracy、高 coverage、具有足够 decisive wrong 且错误能形成至少两个有效语义 cluster 的 parent。

```text
decisive wrong samples
  → 397B 多模态 ErrorSignature
  → 397B Manager 对 signatures 做 2–5 个语义 clusters
  → 397B Manager 为每个有效 cluster 生成一个更窄的 child criterion
  → children 多数投票，平票或全 None 时回退 parent
  → 在固定 parent scope 上比较 Specialized Accuracy 与 Parent Accuracy
```

ErrorSignature 保存 task pattern、visual focus、candidate difference、parent failure 和 suggested subdomain。Cluster 必须引用完整且互斥的 sample IDs，每个 cluster 至少包含5条样本，并说明共同判断失败而不是共同图片主题；每条 wrong sample 必须进入一个 cluster 或显式进入 unclustered。由于 Split 保留 parent，不要求有效 clusters 覆盖固定比例的 wrong samples。

三个 Manager 阶段统一使用 `Qwen/Qwen3.5-397B-A17B`，并分别冻结 backend pool、输入模态和 request identity。ErrorSignature 阶段发送图片并生成稳定的视觉错误签名；语义聚类与 child generation 使用文本输入。child generation 读取代表样本的 question/A/B/gold 与 signatures，但不发送图片；Pairwise Worker 及其 P05 设置保持不变。聚类采用一次完整 partition 输出（temperature=0.2），不做事后 repair、自动合并或小簇降级；child generation 使用 temperature=0.7。freeze 阶段会硬检查三个 Manager profile 的模型集合恰好为 `{Qwen/Qwen3.5-397B-A17B}`，防止实验身份漂移。

Split 保留 parent，children 数量严格等于有效 cluster 数量，不设置目标数量：有效 cluster 少于2个则不触发 Split，存在2–5个则分别生成2–5个 children。`max_children=5` 只是结构上限，不允许为了达到上限强行拆分；超出上限的模式进入 unclustered。每个 child 包含 criterion name、description、1–3个对应 cluster 的代表 examples，以及 cluster lineage；examples 只供 Manager、lineage 和审计，Pairwise Worker 不读取 examples。第一版所有新 edges 固定为 `ALWAYS`，不同时优化 Gate/Child Router。无论 children 数量多少，整棵 root subtree 仍最多贡献一票。

Split 竞争时不再对 child Fitness 求算术平均。完整 children 集合先按 A/B 多数产生 specialized vote；`None` 不参与，children 平票或全部 `None` 时回退 parent vote。Parent Accuracy 与 Specialized Accuracy 都只在 freeze 时确定的父准则区域 $\mathcal{S}(c_p)$ 上计算，且使用相同分母。若 `specialized_accuracy >= parent_accuracy`，保留 parent 并整体接纳 children；否则整体回退。不能因为某个 child 较弱而单独删除它，较弱 child 留给后续 `Refine` 优化。

每次 Split 都写入版本化 history；失败项完整保存 parent、Error Signatures、聚类、children、局部 parent/specialized 指标、父区域逐样本预测、corrected/harmed IDs、结构化失败原因，以及 397B Manager 生成的自然语言失败归因。归因需要指出聚类混杂、child 太宽泛、视觉事实错误、偏好方向写反、children 重复或无关场景激活等具体问题，并给出下一次应避免的做法。同一 parent 下一次 Split 时，语义聚类和 child generation 都读取这些历史；历史只用于避免重复失败方案，不覆盖当前数据上的竞争结果。

首个单算子验证对象固定为 `visual_grounding_and_details`。它在 discovery-90 上的 accuracy/coverage/support/wrong 为 `0.6966/0.9889/89/27`，满足触发条件，并且已有实验观察表明视觉 grounding 错误可以形成多个可解释子域。具体 children 仍必须从这27条真实 wrong samples 的 ErrorSignatures 中归纳，不能预先硬编码类别。

旧 Specialize v1 与既有 8B-signature / 397B-signature 实验结果继续保留为历史基线。当前代码已实现 Split v2 的局部 Specialized Accuracy 接受逻辑、失败归因与历史复用，并统一三个 Manager 阶段为 397B。真实运行仍按 freeze → signatures → cluster review → child proposal → evaluate → report 分阶段进行；在新的 discovery-90 Split v2 实验完成前，不将本项标记为端到端通过。

### 9.3 Create

- 从正式 `rubric_gap` 样本中提取 gap signatures 并聚类；
- 为最佳 cluster 生成两个独立 root 候选；
- 检查与现有 nodes 的 vote agreement，避免明显重复；
- 只增加 root，不修改已有 node，也不创建 child；
- M1 接受阶段只补 Pairwise，Router 标记 stale，最终 M2 诊断前再刷新。

Merge、Drop、删除 parent 的 Split 消融、Child Router、DAG、example-conditioned Worker 和 learned EdgeCondition 均不属于第一版 Phase 6。

---

## 10. Phase 7：接入完整 Evolution Workflow

只有 Refine、Split、Create 分别通过后，才根据三者的触发频率、合法率、接受率、收益、失败模式和调用成本冻结调度顺序。

```text
加载当前 Rubric 与 discovery-90 artifacts
  → 提取 feedback
  → 检测已冻结 trigger
  → 生成候选并局部刷新 artifacts
  → discovery-90 完整 M1 before/after
  → 每轮最多接受一个 edit
  → 原子保存 checkpoint
  → 收敛后仅对最终 Rubric 运行一次 heldout-500
```

最终报告比较 Init Rubric 与 Evolved Rubric。B1/H1/G1/M1/M2 可在 discovery-90 上诊断；heldout-500 只允许 `init-baseline` 和 `final-evaluation` 两个阶段读取。

当前 R007 将 `split-signature-qwen35-heldout-visual` 视为一次性 `final-evaluation`：运行前冻结 parent-only、全部 children 与 discovery 预选的 spatial+direct 三个 Visual 版本，四个新 children 仅通过 8001 推理；报告生成后禁止依据 heldout 选择、改写或调参。

### 10.1 R007：Visual Grounding Split heldout-500 结果

本实验评估由 Qwen/Qwen3.5-397B-A17B Manager 生成的完整 Visual Grounding children 集合能否泛化到 heldout-500。ErrorSignature、语义聚类和 child generation 三个 Manager 阶段统一使用 Qwen/Qwen3.5-397B-A17B；Pairwise Worker 固定为 Qwen3-VL-8B-Instruct、P05（temperature=0.5），所有 heldout 推理仅路由到 8001。四个冻结的 children 为 `visual_factuality_verification`、`spatial_geometric_grounding_accuracy`、`direct_answer_visual_accuracy` 和 `fine_grained_attribute_verification`。

这里的 **Parent only** 特指只保留父准则 `visual_grounding_and_details`，并将它作为唯一 root 独立执行；不挂载任何 children，也不参与原始 M1 中其他四个 root 的投票。父准则输出 A/B 时直接作为该版本的最终判断，输出 abstain 时最终记为 Tie。因而 Parent only 与“全部四个 children”版本的核心差异仅在于：后者在同一个父准则下挂载四个 children，先聚合 children 的多数票，children 平票或全部 abstain 时再回退父准则。

#### 全部 heldout-500 结果

| 版本 | 正确数 | Accuracy | Coverage | Covered Accuracy |
|---|---:|---:|---:|---:|
| Parent only（仅 `visual_grounding_and_details`） | 313 / 500 | 0.626 | 0.940 | 0.6660 |
| 全部四个 children + parent fallback | **352 / 500** | **0.704** | **0.976** | **0.7213** |
| Discovery 预选 Spatial + Direct | 327 / 500 | 0.654 | 0.966 | 0.6770 |
| 原始完整 M1 | 325 / 500 | 0.650 | 0.972 | 0.6687 |

完整 children 集合相对 Parent only 提升 `+7.8` percentage points，相对原始完整 M1 提升 `+5.4` percentage points。配对比较如下：

| 配对比较 | Corrected | Harmed | Net corrected | McNemar exact p |
|---|---:|---:|---:|---:|
| Parent → 全部 children | 66 | 27 | +39 | **0.0000647** |
| Parent → Spatial + Direct | 45 | 31 | +14 | 0.1354 |
| 原始完整 M1 → 全部 children | 72 | 45 | +27 | **0.0159** |
| 原始完整 M1 → Spatial + Direct | 50 | 48 | +2 | 0.9196 |

完整四-child 集合的提升具有统计显著性；discovery-90 预选的 Spatial + Direct 子集没有产生显著提升，说明四个 children 的互补作用不能由 discovery 上的局部筛选稳定替代。

#### 父准则固定覆盖区域上的 Specialized Accuracy

heldout-500 中，父准则 `visual_grounding_and_details` 在 470 个样本上产生有效 A/B，因此 Split 的正式竞争区域固定为：

$$
\mathcal{S}(c_p)=\{x\mid c_p(x)\in\{A,B\}\},\qquad |\mathcal{S}(c_p)|=470.
$$

在该区域内，children 先进行多数投票；children 平票或全部 abstain 时回退父准则。统一分母后的结果如下：

| 版本 | 正确数（父区域） | Specialized Accuracy | 相对 Parent | Corrected / Harmed | McNemar exact p |
|---|---:|---:|---:|---:|---:|
| Parent only | 313 / 470 | 0.6660 | — | — | — |
| 全部四个 children + parent fallback | **340 / 470** | **0.7234** | **+5.74 pp** | 54 / 27 | **0.0036** |
| Spatial + Direct + parent fallback | 319 / 470 | 0.6787 | +1.28 pp | 37 / 31 | 0.5446 |

完整 children 集合满足修正后的 Split 接受条件：

$$
\operatorname{Acc}_{\mathrm{spec}}(c_p,\mathcal{C};\mathcal{S}(c_p))
\ge
\operatorname{Acc}_{\mathrm{parent}}(c_p;\mathcal{S}(c_p)),
$$

因此应当整体保留。旧实现使用 child Fitness 算术平均并加入长度、coverage 惩罚，曾将该候选判为 reject；heldout 结果说明旧规则对本次有效 Split 产生了 false negative，并为“Split 只以局部 Specialized Accuracy 作为优化目标”的修正提供了直接证据。

父准则区域外还有 30 个样本。完整 children 对其中 18 个样本给出有效判断并正确 12 个（covered accuracy `0.6667`）。该覆盖扩张只作为诊断，不进入 Split 的局部接受条件。

#### Child 质量与适用性诊断

| Child | 父区域 Support | 父区域 Coverage | 覆盖内 Accuracy |
|---|---:|---:|---:|
| Visual factuality | 418 | 0.889 | 0.7225 |
| Spatial grounding | 289 | 0.615 | 0.6990 |
| Direct answer | 368 | 0.783 | **0.7283** |
| Fine-grained attributes | 423 | 0.900 | 0.6927 |

| 激活与冲突指标（父区域） | 数量 | 比例 |
|---|---:|---:|
| 至少两个 children 投票 | 433 / 470 | 92.1% |
| 四个 children 全部投票 | 210 / 470 | 44.7% |
| Sibling A/B 冲突 | 137 / 470 | 29.1% |

当前 children 更接近多个高度重叠的视觉评审器，而不是边界清晰、互斥的语义子区域；Pairwise Worker 对 `Applicable only when` 的执行仍然偏宽。该问题不在 Split 阶段通过删除较弱 child 解决，而交由后续 `Refine` 算子收紧描述和适用性。重叠也产生了有效 ensemble 收益：在 137 个 sibling 冲突样本上，Parent Accuracy 为 `0.526`，完整 children 子树为 `0.642`。

#### 结论与边界

> 在 heldout-500 上，由 397B Manager 生成的完整 Visual Grounding children 集合，将父准则固定覆盖区域的准确率从 66.60% 提高到 72.34%（`+5.74 pp`，McNemar exact `p=0.0036`），满足修正后的 Split 接受条件，应整体保留。

需要保留以下结论边界：

- `all_children=0.704` 是“Visual 子树作为唯一 root”的独立结果，不等价于“将 children 接入完整 M1 后”的系统结果；
- 结果说明视觉子树具有较强判别能力，但尚不能单独证明“先判断 Visual Grounding、再执行其他 roots”的层级因果机制；
- children 仍有适用范围过宽和输出偏向 B 的风险，需要在新开发集上由 `Refine` 与位置交换诊断继续验证；
- heldout-500 已作为一次性 final evaluation 使用，不得再依据该报告选择、删除或改写 children；后续完整 M1 集成实验必须使用新的验证划分。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/phase6_split_signature_qwen35_397b/heldout_visual_only/report.json`。



## 10.2 Root、Split-only 多 Epoch 演化实验

Commit：`29b7523 feat: add multi-epoch split evolution`

```mermaid
flowchart TD
    base["冻结基础实验<br/>rubric_evolution_phase5<br/>5个初始Root + discovery-90"]
    endpoint["Pairwise Worker<br/>Qwen3-VL-8B · P05<br/>仅使用8001 · 并发20"]
    manager["Split Manager<br/>Qwen3.5-397B-A17B"]
    freeze["split-evolution-freeze<br/>冻结rubric、预测、数据hash、阈值与API身份"]
    trigger{"逐个检查未锁定Root<br/>ACC < 0.70 且 Coverage > 0.80?"}
    ineligible["not_eligible<br/>本轮不执行Split"]
    signatures["ErrorSignature<br/>仅处理parent decisive-wrong样本<br/>多模态输入 · 同一Root最多并发8"]
    reuse{"Parent预测及错误ID未变化?"}
    cached["复用身份匹配的Error Signatures"]
    generated["397B生成新的Error Signatures"]
    history["读取该Root以前的失败历史<br/>聚类、children、指标、corrected/harmed、自然语言归因"]
    cluster["397B语义聚类<br/>串行1 · 每轮必须重新生成"]
    children["397B生成完整children集合<br/>Root内顺序生成并读取sibling context<br/>不同Root之间最多并发2"]
    pairwise["8001评估全部children × discovery-90<br/>None不投票"]
    specialize["固定parent覆盖区域上的Specialized预测<br/>children多数决<br/>平票或全None回退parent"]
    compare{"Specialized ACC ≥ Parent ACC?"}
    accept["接受完整children集合<br/>保留parent用于回退<br/>Root标记accepted_locked"]
    attribution["397B生成自然语言失败归因<br/>rejection的必要收尾阶段"]
    attr_ok{"归因成功且schema有效?"}
    paused["网络失败：paused<br/>不提交历史、不增加attempt<br/>恢复后继续同一归因"]
    invalid["连续schema-invalid：attribution_invalid<br/>不伪装成rejection<br/>不进入下一轮历史"]
    reject["提交competition_rejected<br/>保存完整结构化历史与自然语言归因<br/>Root标记retryable或exhausted"]
    sync["Epoch同步提交<br/>合并全部accepted patches<br/>重新生成rubric、M1与feedback"]
    epoch{"已完成至少3轮<br/>且没有retryable Root?"}
    maxepoch{"已达到第5轮?"}
    retry["下一Epoch<br/>只调度未锁定且可重试Root"]
    report["split-evolution-report<br/>Discovery演化轨迹与局部/全局指标"]
    treatment{"至少有一个accepted Split?"}
    noheldout["no_treatment<br/>禁止访问heldout"]
    heldout["冻结最终rubric hash后<br/>一次性heldout-500评估<br/>仅为accepted children请求8001"]
    final["最终报告<br/>Init vs Final M1<br/>各Root局部泛化、McNemar、成本与节点增长"]

    base --> freeze
    endpoint -. "评估配置" .-> freeze
    manager -. "Manager配置" .-> freeze
    freeze --> trigger
    trigger -- "否" --> ineligible --> sync
    trigger -- "是" --> reuse
    reuse -- "是" --> cached --> history
    reuse -- "否" --> generated --> history
    generated --> signatures
    cached --> signatures
    signatures --> cluster
    history --> cluster
    cluster --> children --> pairwise --> specialize --> compare
    compare -- "是" --> accept --> sync
    compare -- "否" --> attribution --> attr_ok
    attr_ok -- "网络失败" --> paused
    attr_ok -- "schema无效" --> invalid
    attr_ok -- "成功" --> reject --> sync
    sync --> epoch
    epoch -- "是" --> report
    epoch -- "否" --> maxepoch
    maxepoch -- "否" --> retry --> trigger
    maxepoch -- "是" --> report
    report --> treatment
    treatment -- "否" --> noheldout
    treatment -- "是" --> heldout --> final

    classDef input fill:#ECFDF5,stroke:#10B981,color:#111827,stroke-width:2px;
    classDef manager fill:#EEF2FF,stroke:#4F46E5,color:#111827,stroke-width:2px;
    classDef worker fill:#EFF6FF,stroke:#2563EB,color:#111827,stroke-width:2px;
    classDef decision fill:#FFF7ED,stroke:#EA580C,color:#111827,stroke-width:2px;
    classDef success fill:#F0FDF4,stroke:#16A34A,color:#111827,stroke-width:2px;
    classDef failure fill:#FEF2F2,stroke:#DC2626,color:#111827,stroke-width:2px;
    classDef output fill:#FAF5FF,stroke:#7C3AED,color:#111827,stroke-width:2px;

    class base,freeze input;
    class manager,signatures,generated,cluster,children,history,attribution manager;
    class endpoint,pairwise,specialize worker;
    class trigger,reuse,compare,attr_ok,epoch,maxepoch,treatment decision;
    class accept,reject,sync,retry,heldout success;
    class paused,invalid,noheldout failure;
    class report,final,cached,ineligible output;
```

```shell
$ErrorActionPreference = "Stop"

$python = "C:\Users\wenqx\miniconda3\envs\critiq\python.exe"
$module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
$config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
$output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

function Invoke-EvolutionStage {
    param([string]$Stage)

    Write-Host ""
    Write-Host "===== $Stage =====" -ForegroundColor Cyan

    & $python -m $module `
        --config $config `
        --output-dir $output `
        $Stage

    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

# 确认远程逻辑 endpoint vllm-8001 可用
Invoke-RestMethod "http://10.102.138.0:8000/v1/models" | Out-Null
Write-Host "Pairwise API 8001 is ready." -ForegroundColor Green

# 冻结五个初始 roots、discovery-90、触发条件和实验协议
Invoke-EvolutionStage "split-evolution-freeze"

# 五-root、Split-only、多 Epoch 演化
# 最少3轮、最多5轮；成功 root 锁定，失败 root 带归因历史重试
Invoke-EvolutionStage "split-evolution-run"

# 汇总 discovery-90 演化结果
Invoke-EvolutionStage "split-evolution-report"

# 仅在存在 accepted children 时访问一次 heldout-500
Invoke-EvolutionStage "split-evolution-heldout"

# 生成 discovery + heldout 最终报告
Invoke-EvolutionStage "split-evolution-final-report"

```

### 本次实验结果（`phase6_split_only_evolution_v2`）

实验完成 5 个 epoch、14 次 Split attempt：4/5 个 roots 接受完整 children 集合，`completeness_and_coverage` 连续 5 次局部竞争失败后保持 parent-only；最终 rubric 从 5 个节点增长至 19 个节点（新增 14 个 children）。

| Root | 最终状态 | 接受轮次 | children 数 | discovery 局部 Parent → Specialized ACC |
|---|---|---:|---:|---:|
| Completeness | exhausted | — | 0 | 67.07% → 62.20%（最终失败） |
| Visual Grounding | accepted | 5 | 2 | 69.66% → 69.66% |
| Factuality | accepted | 1 | 5 | 60.24% → 61.45% |
| Creativity | accepted | 1 | 3 | 62.79% → 65.12% |
| Clarity | accepted | 2 | 4 | 56.32% → 64.37% |

| 系统 | discovery-90 ACC / Coverage | heldout-500 ACC / Coverage |
|---|---:|---:|
| 初始五-root M1 | 65.56% / 96.67% | 65.00% / 97.20% |
| 最终等权五-root M1 | 65.56% / 100.00% | 69.40% / 99.40% |
| 最终重加权 M1（Visual Grounding=0.40；其余各=0.15） | — | **70.20% / 99.60%** |

最终等权模型在 heldout-500 上增加 22 个净正确样本（corrected=52，harmed=30，exact McNemar `p=0.0198`）。Visual Grounding 父准则加其 children 单独投票也达到 69.40% ACC；将其权重提升至 0.40 后，冻结预测的后验重聚合达到 351/500（70.20%，较等权 +0.8 pp）。这说明该数据划分可能主要受视觉事实 grounding 驱动。

需要严格区分：discovery 的全局 ACC 未提升，说明当前证据支持 Split 改善 heldout 泛化与覆盖，但尚不足以证明多轮局部优化稳定提升 discovery 全局 M1。

实验产物：[最终报告](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase6_split_only_evolution_v2/final_report.md)、[discovery 报告](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase6_split_only_evolution_v2/final/discovery_report.json)、[heldout 报告](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase6_split_only_evolution_v2/heldout500/report.json)。



---

## 10.3 Manager Global-Rubric Memory Ablation

```
e2051b4 feat: freeze split v1 with global rubric memory
```

验证唯一核心假设：

> 在 Split-only 演化中，给 clustering 与 child-generation Manager 提供每个 epoch 最新的完整 Rubric，能否提高最终五-root 等权 M1 ACC。Control 为 10.2 节的 `phase6_split_only_evolution_v2`，Treatment 为 `phase6_split_only_evolution_global_memory_v1`；两者复用完全相同的 ErrorSignatures，Split 触发、竞争、投票和接受条件保持不变。

```shell
$ErrorActionPreference = "Stop"

$python = (Get-Command python).Source
$module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
$config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
$output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

function Invoke-MemoryStage([string]$stage) {
    Write-Host "`n===== $stage =====" -ForegroundColor Cyan

    & $python -m $module `
        --config $config `
        --output-dir $output `
        $stage

    if ($LASTEXITCODE -ne 0) {
        throw "$stage failed with exit code $LASTEXITCODE"
    }
}

# 先运行冻结和单-root smoke：
Invoke-MemoryStage "split-memory-freeze"
Invoke-MemoryStage "split-memory-smoke"

# 确认 smoke 正常后，运行完整的 5-root、3–5 epoch 演化：
Invoke-MemoryStage "split-memory-run"
Invoke-MemoryStage "split-memory-report"

# 最后运行 heldout-500 和最终 Control–Treatment 对照报告：
Invoke-MemoryStage "split-memory-heldout"
Invoke-MemoryStage "split-memory-final-report"

```

### 本次实验结果（`phase6_split_only_evolution_global_memory_v1`）

实验完成 5 个 epoch、17 次 Split attempt。4/5 个 roots 接受完整 children 集合，`visual_grounding_and_details` 在 5 次失败后保持 parent-only；最终 rubric 从 5 个节点增长至 17 个节点（新增 12 个 children）。Treatment 全程只读复用 Control v2 的 157 条 ErrorSignatures，没有重新生成签名；Rubric memory 按 epoch 从 5 个节点更新到 10、15 个节点，并在最终形成 17 个节点，符合“本轮同步提交、下一轮可见”的冻结协议。

| Root | 最终状态 | 接受轮次 | children 数 | discovery 局部 Parent → Specialized ACC | heldout parent-scope Parent → Specialized ACC |
|---|---|---:|---:|---:|---:|
| Completeness | accepted | 5 | 2 | 67.07% → 69.51% | 65.09% → **72.41%** |
| Visual Grounding | exhausted | — | 0 | 69.66% → 62.92%（最终失败） | 保持 parent-only |
| Factuality | accepted | 1 | 5 | 60.24% → **69.88%** | 66.24% → **70.70%** |
| Creativity | accepted | 3 | 3 | 62.79% → 63.95% | 62.12% → **69.45%** |
| Clarity | accepted | 3 | 2 | 56.32% → 59.77% | 60.21% → **66.88%** |

所有在 discovery-90 上被接受的 subtrees 均在 heldout parent scope 上保持正向增益，说明局部 `Specialized ACC >= Parent ACC` 竞争能够筛出具有泛化价值的 children。失败历史也在自然轨迹中发挥了作用：Completeness 的 children 数量从 4 条逐步收缩到 2 条，局部 delta 从负值改善为 +2.44 pp，并在第 5 轮成功接受。

| 系统 | discovery-90 ACC / Coverage | heldout-500 ACC / Coverage | heldout 正确数 | Final children / nodes |
|---|---:|---:|---:|---:|
| 初始五-root M1 | 65.56% / 96.67% | 65.00% / 97.20% | 325 | 0 / 5 |
| Control v2（无全局 memory, 10.2实验） | 65.56% / 100.00% | 69.40% / 99.40% | 347 | 14 / 19 |
| Global-Rubric Memory | 63.33% / 100.00% | **70.00% / 98.60%** | **350** | **12 / 17** |

Treatment 相对初始五-root 在 heldout-500 上提升 5.0 pp，增加 25 个净正确样本（corrected=44，harmed=19，exact McNemar `p=0.00223`）。相对 Control v2，Treatment 进一步增加 3 个正确样本（+0.6 pp），同时用更小的 rubric 获得最高等权 M1 ACC。两者的 lexical near-duplicate pairs 从 14 对减少到 7 对；考虑 children 总数后的重复率由 15.4% 降至 10.6%，说明全局 Rubric 上下文能够帮助 Manager 感知已有准则边界，减少冗余生成。

该对照只有单条演化轨迹，Treatment 与 Control 的直接配对差异为 corrected=30、harmed=27（McNemar `p=0.791`），且 heldout-500 已被前序实验使用，因此不把 memory 的 +0.6 pp 作为独立研究贡献。这里采用更务实的结论：Global-Rubric Memory 在没有修改 Split 核心机制的情况下取得了数值最优 ACC、减少了最终节点和近重复准则，并为后续 Manager 提供了完整的当前 Rubric 状态，适合作为默认基础设施。

**后续默认设置**：除专门研究 Manager memory 的消融实验外，后续 Split、Refine 及其他 Manager 算子均启用 `global_rubric_v1`，向 Manager 提供每个 epoch 起始时最新的 roots、nodes、edges、criterion name 和 description；不提供 examples、预测、gold、ACC 或 heldout 信息。同一 epoch 内尚未提交的 candidates 仍不可见，accepted children 从下一 epoch 开始进入全局 memory。

**Split v1 冻结结论**：当前触发条件、ErrorSignature/聚类/children 生成流程、固定 parent scope 上的 Specialized Accuracy 竞争、整组接受与 parent 回退、失败归因历史以及 `global_rubric_v1` memory contract 共同构成冻结的 Split v1。后续算子直接复用该协议；若需要修改上述语义，必须提升协议版本并使用新的实验目录，不得覆盖本节结果。

实验产物：[最终报告](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase6_split_only_evolution_global_memory_v1/final_report.md)、[discovery 报告](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase6_split_only_evolution_global_memory_v1/final/discovery_report.json)、[heldout 报告](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase6_split_only_evolution_global_memory_v1/heldout500/report.json)。


---

## 10.4 Refine v1 算子实现与实验

```shell
conda activate critiq

$python = (Get-Command python).Source
$module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
$config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
$output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

function Invoke-RefineStage {
    param([string]$Stage)

    Write-Host "`n===== $Stage =====" -ForegroundColor Cyan

    & $python -m $module `
        --config $config `
        --output-dir $output `
        $Stage

    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

# smoke
# 冻结 source rubric、目标节点、Global Memory 和请求身份
Invoke-RefineStage "refine-freeze"

# 生成一个新 description，并执行 discovery-90 self-competition
Invoke-RefineStage "refine-smoke"

# 对同一个候选执行 heldout-500 diagnostic
Invoke-RefineStage "refine-smoke-heldout"

# 汇总 discovery 与 heldout，生成 go/no-go 结论
Invoke-RefineStage "refine-smoke-report"

$reportPath = Join-Path $output "phase7_refine_operator_v1/report.json"
$report = Get-Content -Raw -Encoding UTF8 $reportPath | ConvertFrom-Json

$report | ConvertTo-Json -Depth 20
Write-Host "`nGo/No-Go: $($report.go_no_go)" -ForegroundColor Yellow

# ---- 完整 Split+Refine 演化： ----
Invoke-RefineStage "split-refine-freeze"
Invoke-RefineStage "split-refine-run"
Invoke-RefineStage "split-refine-report"
Invoke-RefineStage "split-refine-heldout"
Invoke-RefineStage "split-refine-final-report"

```

### 10.4.1 Forced Refine smoke 结果

Smoke 从冻结的 `phase6_split_only_evolution_global_memory_v1` 最终 Rubric 出发，只改写 child `verified_existence_over_hallucinated_volume` 的 description。该节点不满足自动 Refine trigger，因此本实验只验证 Refine 机制，不作为自动调度证据。

| 层级 | Discovery old | Discovery new | Delta | Heldout old | Heldout new | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Target node ACC | 50.62% | 64.86% | +14.25 pp | 63.88% | 70.27% | +6.39 pp |
| Completeness subtree ACC | 70.11% | 75.86% | +5.75 pp | 72.20% | **72.41%** | +0.21 pp |
| Equal-root M1 ACC | 63.33% | 66.67% | +3.33 pp | 70.00% | 70.40% | +0.40 pp |

新 description 将 discovery support 从81降至74，但仍远高于最小支持数；discovery M1 corrected=3、harmed=0。heldout 只作诊断且不反向修改 discovery 决策，最终 `go_no_go=go`。

### 10.4.2 五-root Split+Refine V1 演化结果

完整实验从五个初始 roots 重新开始，而不是在 smoke Rubric 上继续演化。由于 Split clustering 与 children 重新生成，最终 Rubric 不包含 smoke 的目标节点，因此该实验不是对 smoke candidate 的直接复现。

| 系统 | Discovery ACC | Heldout ACC | Heldout correct | Coverage |
|---|---:|---:|---:|---:|
| Initial five roots | 65.56% | 65.00% | 325 / 500 | — |
| Split-only + Global Memory control | 63.33% | 70.00% | 350 / 500 | 98.60% |
| Split+Refine final | 66.67% | 70.00% | 350 / 500 | 98.80% |

相对 initial five roots，最终 heldout corrected=51、harmed=26、净纠正25，exact McNemar `p=0.00587`。相对 Split-only + Global Memory control，最终 ACC 持平；Refine 带来的 discovery 优势没有转化成额外 heldout 提升，因此当前结果支持 Refine 的局部修复能力，但尚不能证明自动 Refine 调度能稳定提高最终泛化 ACC。

演化过程中共有6个不同节点执行21次 Refine attempt：20次形成合法候选，其中4次接受、16次竞争拒绝，另有1次 proposal-invalid。4次接受发生在3个 Factuality children 上；其中只有部分 node-level 改进同时传递到 subtree/M1，符合 v1 将 subtree 与完整 M1仅作为诊断的冻结定义。

Visual Grounding root 连续执行5次 Split 均被拒绝；最好一次 Specialized ACC 为68.54%，相对 parent 69.66%只少1个正确样本。旧 Visual children 在当前固定 parent scope 上可达到71.91%，说明主要问题是 Split proposal 的搜索稳定性，而不是 Visual Grounding 不重要。下一阶段优先研究 child-level Split failure feedback、强 child 保留，以及 root/child 分离的 Refine trigger

### 10.4.3 Role-aware Refine：阈值实验与 checkpoint 诊断

此前统一的 Refine trigger 同时要求中等 ACC 与较低 Coverage，容易漏掉“覆盖较广、但仍有足够错误可修复”的 children，或者是”满足覆盖率要求但是ACC达不到阈值，但是refine之后增益很大“的chirldren。为此，本实验保留 root 的原触发规则，而将 child 改为 role-aware 规则：

$$
0.5 < \operatorname{Acc}(c) < 0.80,\qquad
|\mathcal S(c)|\ge15,\qquad
\operatorname{Wrong}(c)\ge5.
$$
在 Split-only + Global Memory 的初始 Rubric (没有做过refine，来验证refine是否有效) 上，统一阈值仅触发 4 个节点；role-aware 规则触发 11 个节点，其中新增 7 个均为 children。实验运行 3 个 epoch，并冻结 source、epoch 1 和 epoch 3 三个 checkpoint，在同一 heldout-500 上进行探索性配对比较。

| Checkpoint | Heldout correct | Equal-root M1 ACC | Coverage | 相对 source corrected / harmed | Exact McNemar |
|---|---:|---:|---:|---:|---:|
| Source（Split-only + Global Memory） | 350 / 500 | 70.0% | 98.6% | — | — |
| Epoch 1 | **356 / 500** | **71.2%** | **98.8%** | 15 / 9（净 +6） | 0.3075 |
| Epoch 3 | 355 / 500 | 71.0% | 98.6% | 22 / 17（净 +5） | 0.5224 |

Role-aware 阈值在第一轮带来正向结果：discovery M1 从 63.33% 升至 68.89%，heldout M1 从 70.0% 升至 71.2%，且 Coverage 保持稳定。后续两轮的 discovery 轨迹为 `68.89% → 67.78% → 66.67%`，heldout 亦从 71.2% 轻微回落至 71.0%；因此该实验支持“放宽 child 的独立触发条件能发现有价值的局部 Refine”，但不支持在本次轨迹中继续迭代会带来额外的全局 M1 增益。两项 heldout 差异均未显著，且该 heldout 已用于开发诊断，结论仅为 exploratory evidence。

为进一步判断 epoch 1 后的再次 Refine 是否本身有效，固定 epoch-1 的其余节点与预测，只替换 epoch 2--3 中实际改写过的 3 个 child description：

| 后续再次 Refine 的 child | Node ACC（epoch 1 → epoch 3） | Support（epoch 1 → epoch 3） | 固定 epoch-1 ensemble 的 M1 |
|---|---:|---:|---:|
| `verified_existence_over_hallucinated_volume` | 65.09% → 68.60% | 424 → 414 | 71.6%（+2 correct） |
| `visual_attribute_verification` | 75.48% → 77.24% | 367 → 312 | 71.0%（−1 correct） |
| `spatial_and_quantitative_clarity` | 68.81% → 73.29% | 452 → 438 | 72.0%（+4 correct） |
| 三者同时替换 | — | — | **72.4%**（corrected / harmed = 12 / 6，净 +6） |

该对照说明再次 Refine 并非无效：`verified_existence_over_hallucinated_volume` 与 `spatial_and_quantitative_clarity` 的改写在固定 ensemble 中均提高最终判断，三者共同替换可达 72.4%。其中 `visual_attribute_verification` 的 node ACC 上升伴随明显 support 收缩，未转化为整体收益。完整 epoch-3 Rubric 仅为 71.0%，表明额外的局部改写在多数投票中抵消了这些收益；因此，本次结果的核心结论是：**role-aware 阈值能够找到可提升的 child，重复 Refine 也可产生局部增益，但局部 self-competition 的接受不保证多节点聚合后的单调提升。**

checkpoint 诊断采用修复版 `heldout_checkpoint_diagnostic_v2`：预测以 **(node ID, description hash)** 为身份，补跑 epoch-1 的 4 个旧 description，并仅在 description/hash 完全一致时复用输出。旧版按 node ID 复用导致的 epoch-1 `72.6%` 不再采用。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/phase8_refine_role_aware_v2/heldout_checkpoint_diagnostic_v2/`。

### 10.5 Visual Grounding Split-retry v2：锁定强 child

本实验从 `phase7_split_refine_evolution_v1` 中 Visual Grounding 的一次失败 Split 出发。原完整 children 集合在固定父区域上为 `61/89=68.54%`，低于 parent 的 `62/89=69.66%`，因此未被接受。v2 不重跑 ErrorSignature 或聚类：先在该失败集合中识别具有足够支持且净纠正为正的 child，锁定其 description 与 discovery Pairwise 预测；然后只为其余聚类重新生成 children，并沿用原 Split 的 parent 回退、多数投票和 `Specialized ACC >= Parent ACC` 接受规则。

强 child 的冻结判据为 `support >= 15` 且相对 parent 的单 child specialized vote `net_corrected >= 3`。本次唯一被锁定的 child 为 `peripheral_detail_verification_accuracy`：在 discovery 父区域上达到 `68/89=76.40%`，相对 parent 净纠正 `+6`（13 corrected / 7 harmed）。完整 v2 集合达到 `63/89=70.79%`，仅高于 parent 1 个样本，因此按既定规则接受；本次第一次 v2 candidate 即被接受，没有形成 v2 内部“连续失败后再次重试”的自然证据。

| Heldout-500 系统 | 完整五-root M1 | Visual 子树（parent scope） | 相对 parent M1 corrected / harmed | McNemar |
|---|---:|---:|---:|---:|
| Parent-only | 70.0%（350 / 500） | 66.60%（313 / 470） | — | — |
| Locked child only | **71.8%（359 / 500）** | 71.91%（338 / 470） | 12 / 3（净 +9） | **0.0352** |
| Full v2（locked + 3 regenerated children） | 70.8%（354 / 500） | **72.55%（341 / 470）** | 11 / 7（净 +4） | 0.4807 |

锁定 child 的正向收益可泛化：它使完整 M1 增加 `+1.8 pp`，且在配对检验中显著。完整 v2 虽使 Visual 子树额外提高 `+0.64 pp`，但相对 locked-only 的完整 M1 下降 `1.0 pp`（6 corrected / 11 harmed）；这不是锁定策略失败，而是新 siblings 的重叠投票稀释了强 child 的收益。heldout child 诊断也显示，`relational_compositional_priority` 在单 child ACC 上仍低于 parent（61.31% vs 62.62%），并且四个 children 的两两冲突率为 12.3%--32.1%。

**采纳的后续协议。** 当失败 Split 中存在强 child 时，保留其作为**稳定局部专家**；未锁定 children 不随之强制淘汰，而是作为待改进准则交给后续 `Refine`，重点收紧适用条件、减少与 locked child 的冲突。Split 继续以完整 children 对 parent 的局部 Specialized Accuracy 进行接受，不额外把“必须超过 locked-only”设为硬门槛，以保留探索空间；但每次后续操作必须记录 locked child、各 child ACC/support、leave-one-out 影响与 sibling conflict。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/phase8_visual_split_retry_locked_v2/`，其中 discovery 为 `report.json`，heldout 为 `heldout500_diagnostic/report.json`。



---

### 10.6 Visual Grounding Split→Refine 局部演化实验

Visual Grounding Split-retry v2 已接受一个固定的四-child 子树：其中 locked child `peripheral_detail_verification_accuracy` 单独投票在 heldout 上优于 parent-only，但完整四-child 多数投票仍可能受到其余 siblings 的低质量或冲突投票影响。该实验不重新验证 Split，也不改变投票规则；它只检验：**在保留已接受的 Split 结构和强 child 的前提下，role-aware Refine 能否改善现有 children，并提高完整 Visual Grounding 子树的集体判断。**

#### 冻结起点与范围

- 起点固定为 `phase8_visual_split_retry_locked_v2/final/rubric_committed.json`，根节点为 `init_02_visual_grounding_and_details`；冻结其 parent、4 个 children、edge、name、node ID、score 与 examples。
- 禁用 Split：不生成 ErrorSignature、不重新聚类、不生成或替换 child，也不改变 Rubric topology。
- Refine 的唯一允许编辑是改写现有 child 的 description。四个候选 child 为：
  - `peripheral_detail_verification_accuracy`（Split locked）；
  - `visual_premise_validation_and_consistency`；
  - `main_subject_factuality_and_grounding`；
  - `relational_compositional_priority`。
- Split locked 的含义仅是该 child 不会在后续 Split 中被重新生成；它并不豁免 Refine。只要满足 Refine trigger，locked child 与其余 children 一样可被改写和竞争。

#### Refine 协议

child 使用 role-aware trigger：

$$
0.5 < \operatorname{Acc}(c) < 0.80,\qquad
|\mathcal S(c)|\ge15,\qquad
\operatorname{Wrong}(c)\ge5.
$$

在冻结的 discovery-90 起点上，四个 children 均满足该条件。每个 child 在每轮最多生成一个 Refine candidate；仅当该 candidate 在自己的有效 A/B 支持集上严格优于旧 description，且新 support 不低于 15 时接受：

$$
\operatorname{Acc}(c')>\operatorname{Acc}(c),\qquad
|\mathcal S(c')|\ge15.
$$

每个 epoch 的四个 child 都基于同一个 epoch-start Rubric、相同的 `global_rubric_v1` memory 和上一轮已提交的 prediction 独立生成与竞争；本轮任何 candidate 都看不到同轮其他 candidate。所有通过 self-competition 的 description 在 epoch 末尾同步提交，因此新的 siblings 只会在下一轮进入彼此的 Global Memory。有效候选被拒绝时，必须先生成自然语言失败归因，再在下一轮携带该 node 的指标、corrected/harmed、abstain 转移、sibling overlap/conflict、旧 proposal 与失败归因重试。最多运行 3 个 epoch，并在没有 triggered 或 retryable child 时提前停止。

Manager 固定为 `Qwen/Qwen3.5-397B-A17B`，使用 Refine v1 prompt 与 `global_rubric_v1`；Pairwise Worker 固定为 Qwen3-VL-8B-Instruct、P05、单 replicate，经 8000 端口运行。每个候选只刷新其 90 个 discovery Pairwise 预测，所有未变化节点的预测严格复用。

#### 最终 heldout-500 验证

仅在 discovery 结束、final Rubric/hash 冻结后访问 heldout-500。只为 description 实际改变的 children 生成新预测，其余节点复用 Split-retry v2 的 heldout artifact。主要比较为：

| 系统 | 作用 |
|---|---|
| Parent-only | Visual parent 的固定基线 |
| Locked child only（source/final） | 强 child 单独贡献的诊断 |
| Split v2 full children | 本实验的直接起点与主要对照 |
| Split v2 + Refined children | 主要结果 |

报告 discovery 与 heldout 的 child-level ACC、support、coverage、corrected/harmed、`None` 转移、sibling overlap/conflict，以及完整五-root M1 与 Visual 子树（全 500 / parent scope）指标。heldout-500 已用于前序开发，因此该验证明确标记为 exploratory；它不反向选择 Refine candidate、epoch 或投票配置。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/phase9_visual_split_refine_local_v1/`。

#### 实验结果

实验完整运行 3 个 epoch，共调度 10 次 Refine attempt，其中 9 次形成有效候选并进入竞争：3 次接受、6 次拒绝；另有 1 次因 description 超过 1800 字符而 proposal-invalid。最终接受的修改为 Epoch 1 的 `peripheral_detail_verification_accuracy`、Epoch 1 的 `main_subject_factuality_and_grounding`，以及 Epoch 2 携带失败历史重试成功的 `relational_compositional_priority`；`visual_premise_validation_and_consistency` 连续三轮均未通过 self-competition。

Discovery-90 的演化轨迹如下。Visual subtree 与完整 M1 的 ACC 均以全部 90 个样本为分母。

| Epoch | 本轮接受的 Refine | Visual subtree ACC | 完整五-root M1 ACC | M1 Coverage |
|---:|---|---:|---:|---:|
| 0 | — | 71.11% | 66.67% | 97.78% |
| 1 | Peripheral、Main subject | 66.67% | 67.78% | 100.00% |
| 2 | Relational composition | 68.89% | 67.78% | 100.00% |
| 3 | 无 | 68.89% | 67.78% | 100.00% |

最终相对 Split v2 起点，discovery Visual subtree 下降 2.22 个百分点，而完整 M1 提高 1.11 个百分点。Epoch 1 的两个 candidate 均独立通过 self-competition，但同步提交后 subtree 从 71.11% 降至 66.67%，说明多个 child 的局部改进在多数投票中存在非线性交互，单节点通过不保证整组同步更新后仍然单调提升。

下表比较四个 children 的 discovery 与 heldout 变化。这里的 child ACC 均是在该 child 自己输出有效 A/B 的 support 上计算，`Support` 不是 90 或 500 的固定分母。

| Child | Discovery：Source → Final ACC（Support） | Heldout：Source → Final ACC（Support） | 主要变化 |
|---|---:|---:|---|
| `peripheral_detail_verification_accuracy` | 78.08% (73) → 80.30% (66) | 72.87% (376) → 77.59% (348) | 错误票明显减少，但 support 收缩；heldout 正确票 274 → 270 |
| `visual_premise_validation_and_consistency` | 62.03% (79) → 62.03% (79) | 72.69% (432) → 72.69% (432) | 三轮候选均拒绝，description 保持不变 |
| `main_subject_factuality_and_grounding` | 65.75% (73) → 68.25% (63) | 75.38% (394) → 75.79% (347) | ACC 小幅提高主要来自更强选择性；heldout 正确票 297 → 263 |
| `relational_compositional_priority` | 62.07% (58) → 65.67% (67) | 61.31% (305) → 68.39% (367) | 最稳定的实质改进：heldout support +62、正确票 +64 |

`relational_compositional_priority` 是本实验中最明确的 Refine 成功案例：第一次改写被拒绝后，Manager 利用失败归因重新放宽过度收缩的适用域，第二轮候选在 discovery 上被接受，并在 heldout 上同时提高 ACC 与 support。相比之下，Peripheral 与 Main subject 的条件 ACC 提高包含明显的 `None` 选择效应，因此不能只根据 ACC 上升解释为总体正确判断数量增加。

Heldout-500 的系统级结果如下。`Parent scope ACC` 在 Visual parent 有效输出 A/B 的固定 470 个样本上计算；`Visual subtree all500` 与完整 M1 均以全部 500 个样本为分母。表中的 Parent-only 指完整五-root系统中 Visual root 不带本次 children，而不是只运行单个 root。

| 系统 | Visual subtree all500 ACC | Parent scope ACC | 完整五-root M1 ACC | M1 Coverage |
|---|---:|---:|---:|---:|
| Parent-only | 62.60% | 66.60% | 70.00% | 98.80% |
| Locked child（source） | 69.20% | 71.91% | **71.80%** | 99.00% |
| Split v2 full children | 71.00% | 72.55% | 71.00% | 99.00% |
| Locked child（final description） | 70.80% | **73.62%** | 71.00% | 99.20% |
| Split v2 + Refined children | **72.60%** | 73.40% | 71.60% | 99.00% |

主要对照 `Split v2 + Refined children` 与 `Split v2 full children` 的 paired 结果为：

| 比较层级 | Corrected | Harmed | Net corrected | Exact McNemar $p$ |
|---|---:|---:|---:|---:|
| Visual subtree，parent scope | 28 | 24 | +4 | 0.678 |
| 完整五-root M1，all500 | 9 | 6 | +3 | 0.607 |

因此，Refine 在 heldout 上把完整 Visual subtree 从 71.00% 提高到 72.60%，并把完整 M1 从 71.00% 提高到 71.60%；两项增量方向均为正，但未达到统计显著。相对 Parent-only，最终 Refined subtree 在 parent scope 上净纠正 32 个样本（55 corrected、23 harmed，$p=0.00038$），说明主要可靠收益仍来自 Split 所建立的局部专家子树，而 Refine 提供的是其上的进一步小幅增益。

Refine 还使六对 siblings 的总冲突数从 437 降至 357（下降 18.3%），汇总 conflict rate 从 24.9% 降至 20.6%，而 joint-decisive 数仅下降 1.2%。这表明冲突下降不只是由统一扩大 `None` 造成，description 的适用边界确实变得更清楚。

总体而言，本实验支持：**在固定 Split 结构上继续 Refine children 能改善 sibling 边界，并在 heldout 上提高完整子树与 M1 的点估计。** 同时，原始 locked child 的完整 M1 为 71.80%，仍略高于最终四-child系统的 71.60%；加之 discovery 中出现同步提交后 subtree 下降，结果也表明 child self-ACC 不能完整代表其对子树和全局聚合的实际贡献。

---

### 10.7 Five-root Locked-Split + Role-aware Refine

```shell
$python = "python"
$module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
$config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
$output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

function Invoke-FiveRootIntegrationStage([string]$Stage) {
    Write-Host "`n===== $Stage =====" -ForegroundColor Cyan
    & $python -m $module --config $config --output-dir $output $Stage
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

# Discovery-90：冻结、离线审计、五 epoch 以内的完整演化、冻结 discovery 结果
Invoke-FiveRootIntegrationStage "five-root-locked-split-refine-freeze"
Invoke-FiveRootIntegrationStage "five-root-locked-split-refine-audit"
Invoke-FiveRootIntegrationStage "five-root-locked-split-refine-run"
Invoke-FiveRootIntegrationStage "five-root-locked-split-refine-report"

# heldout-500:
Invoke-FiveRootIntegrationStage "five-root-locked-split-refine-heldout"
Invoke-FiveRootIntegrationStage "five-root-locked-split-refine-final-report"

```

#### 实验目的与冻结协议

前序实验分别验证了 Split-retry 和 Role-aware Refine 的局部作用。本实验从五个初始 roots 重新开始，将支持强-child锁定的 Split-retry、Role-aware Refine 与 `global_rubric_v1` 统一到最多 5 个 epoch 的完整流程中，检验其能否自动演化出更高质量的五-root Rubric。Split 只调度初始 roots，Refine 调度已提交的 children；同一轮 candidates 基于相同的 epoch-start Rubric 独立竞争并同步提交。discovery-90 决定算子接受，heldout-500 只评估最终冻结 Rubric，正式主指标仍为五-root等权 M1。

为避免 ACC 分母混淆，本节中完整 M1 和 root-subtree 的 `full-500 ACC` 均以全部 500 个样本为分母，`None/Tie` 计为未正确；单节点 ACC 仍以该节点自己的 A/B support 为分母；Split Specialized ACC 则继续使用固定 parent scope。

#### 演化轨迹

最终 Rubric 从 5 个 roots 增长到 22 个节点，共新增 17 个 children。前三轮主要建立子树结构，后两轮不再增加节点，只继续改写现有 child descriptions。

| Epoch | Nodes | Split scheduled / accepted | Refine scheduled / accepted | Discovery M1 ACC |
|---:|---:|---:|---:|---:|
| 0 | 5 | — | — | 65.56% |
| 1 | 12 | 5 / 2 | 0 / 0 | 70.00% |
| 2 | 18 | 3 / 2 | 7 / 5 | 67.78% |
| 3 | 22 | 1 / 1 | 12 / 9 | 65.56% |
| 4 | 22 | 0 / 0 | 16 / 5 | 71.11% |
| 5 | 22 | 0 / 0 | 15 / 4 | **73.33%** |

M1 轨迹 `65.56% → 70.00% → 67.78% → 65.56% → 71.11% → 73.33%` 并不单调：新增 children 与局部 Refine 一度增加投票冲突，直到后两轮继续收紧适用边界后，全局收益才显现。这说明节点 self-competition 的改善不会立即等价为完整 ensemble 的改善。

| Root | Split 轨迹 | 最终 children | 接受时 Specialized ACC delta |
|---|---|---:|---:|
| Completeness | 第 1 轮接受 | 3 | 0.00 pp |
| Visual Grounding | 拒绝 → 拒绝 → 接受 | 4 | 0.00 pp |
| Factuality | 第 1 轮接受 | 4 | +9.64 pp |
| Creativity | 拒绝 → 接受 | 3 | +5.81 pp |
| Clarity | proposal-invalid → 接受 | 3 | +6.90 pp |

Visual Grounding 的三次 Specialized ACC 为 `65.17% → 64.04% → 69.66%`，第三次仅恢复到与 parent 的 69.66% 持平，但它提供了可继续优化的四-child结构；经过后续 Refine，最终 Visual 子树在 heldout-500 上达到 73.0%。这支持“Split 先产生不退化的专家结构，Refine 再修正弱 children”的算子分工。需要明确的是，本轨迹 `locked_retry_count=0`，两次失败中均没有 child 达到锁定门槛，因此本实验不能作为强-child锁定机制的自然证据。

#### 最终结果

| 系统 | Discovery ACC / Coverage | Heldout ACC / Coverage | Heldout correct |
|---|---:|---:|---:|
| Initial five-root M1 | 65.56% / 96.67% | 65.00% / 97.20% | 325 / 500 |
| Split-only + Global Memory | 63.33% / 100.00% | 70.00% / 98.60% | 350 / 500 |
| Five-root Split+Refine | **73.33% / 100.00%** | **71.40% / 98.60%** | **357 / 500** |

相对 Initial，最终系统在 heldout 上提升 6.4 pp，corrected=67、harmed=35、净增加 32 个正确样本，exact McNemar `p=0.00199`。相对 Split-only + Global Memory，提升为 1.4 pp，corrected=32、harmed=25、净增加 7 个正确样本，但 McNemar `p=0.427`，因此这里只将 Refine 的增量表述为正向 exploratory evidence，而不声称显著优于 Split-only。

Role-aware Refine 共执行 50 次：23 次接受、26 次竞争拒绝、1 次 proposal-invalid。以下五个 children 的重复 Refine 改善最明显；Node ACC 以各自 A/B support 为分母，表中的 support/coverage 同时反映其专家化范围。

| Child criterion | Discovery ACC | Discovery support | ACC delta | Heldout ACC / Coverage |
|---|---:|---:|---:|---:|
| `completeness_via_verified_perception` | 66.7% → **82.1%** | 87 → 56 | +15.5 pp | 77.1% / 65.4% |
| `answer_accuracy_over_descriptive_detail` | 57.1% → **71.2%** | 77 → 52 | +14.1 pp | 68.0% / 55.6% |
| `visual_grounding_accuracy` | 67.1% → **80.9%** | 79 → 47 | +13.8 pp | **79.4% / 49.6%** |
| `presence_and_action_verification` | 61.0% → **73.6%** | 82 → 72 | +12.6 pp | 75.1% / 77.0% |
| `factual_grounding_prerequisite _for_expressiveness` | 55.6% → **65.3%** | 81 → 75 | +9.7 pp | 71.0% / 80.8% |

这些结果表明重复 Refine 能将**宽泛 child 收紧为较高准确率的局部专家**，其中部分提升伴随 support 收缩。但 23 次 accepted Refine 中只有 8 次同步提高 subtree ACC，9 次反而降低 subtree ACC；这再次说明 v1 的 node self-competition 能形成局部专家，却不保证父子树或完整 M1 单调提升。

#### Root 子树与聚合诊断

五个最终子树在固定 500 分母下均优于各自 parent，说明新增 children 整体具有有效信息。

| Root | Parent-only full-500 ACC | Parent + children full-500 ACC |
|---|---:|---:|
| Completeness | 60.4% | 69.6% |
| Visual Grounding | 62.6% | **73.0%** |
| Factuality | 62.4% | 67.2% |
| Creativity | 61.0% | 68.4% |
| Clarity | 57.8% | 70.8% |

Visual Grounding 子树单独使用时达到 365/500=73.0%，高于五-root等权 M1 的 357/500=71.4%。加入另外四个 roots 后，虽然纠正了 27 个 Visual 错误，但同时损伤 35 个原本正确的样本，净损失 8 个，说明当前主要瓶颈已从局部专家生成转向 root aggregation。

采用前序实验预设、未在本次 heldout 上搜索的权重 `Visual Grounding=0.4`、其余四个 roots 各 `0.15`，可进一步得到：

| 聚合方式 | Correct | Full-500 ACC | Coverage |
|---|---:|---:|---:|
| 五-root等权 | 357 | 71.4% | 98.6% |
| Visual Grounding only | 365 | 73.0% | 97.4% |
| Visual=0.4，其余各0.15 | **367** | **73.4%** | **99.4%** |

该权重使两个其他 roots 的合计票重 0.30 不足以轻易覆盖 Visual，但三个 roots 达成一致时 0.45 仍可纠正它。相对等权聚合，weighted M1 corrected=24、harmed=14、净增加 10 个正确样本；相对 Visual-only 净增加 2 个。

#### 结论

该实验支持完整的 Split+Role-aware Refine 演化链条：它在不降低最终 Coverage 的情况下，将 heldout 等权 M1 从 65.0% 提升到 71.4%，并将五个 parent 都扩展为更强的子树；Visual Grounding 的 parity Split 经后续 Refine 达到 73.0%，进一步说明非退化 Split 可以先建立可优化结构。与此同时，本轨迹没有触发强-child锁定，相对 Split-only 的 1.4 pp 增益也未显著，且局部 Refine 不保证全局单调改善。预设加权聚合达到 73.4%，表明当前最明确的剩余问题是如何利用不同 root 的可靠性差异。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/phase10_five_root_locked_split_refine_v1/`，其中 discovery 汇总为 `final/discovery_report.json`，heldout 汇总为 `heldout500/report.json`，最终报告为 `final_report.json`。



---

## 11. 阶段性结果

### 11.1 Split+Refine heldout-500 结果

现在已经形成了一个清晰的阶段性闭环：

| 结果                                      | heldout-500 ACC |
| ----------------------------------------- | --------------- |
| Vanilla 直接偏好判断                      | 59.8%           |
| Initial five-root M1                      | 65.0%           |
| 最终 Split+Refine，五-root 等权           | 71.4%           |
| 最终 Split+Refine，Visual=0.4、其余各0.15 | **73.4%**       |

相对 Vanilla，最终系统提高 **13.6 个百分点**；同时 Split、失败重试、强 child 策略、Global Memory 和 Role-aware Refine 都已有独立实验支撑，适合在这里冻结一个版本。

### 11.2 VL-RewardBench 外部迁移实验

#### 1. 实验设置

本实验检验 Phase10 最终 Rubric 能否从开发期 RLHF-V 风格 heldout-500 迁移到 VL-RewardBench。评测集包含 1,247 个偏好对，Pairwise Worker 固定为 `Qwen3-VL-8B-Instruct`。VL-RewardBench 不参与 checkpoint、criterion、聚合权重或后续演化选择，因此该结果只用于外部迁移评估。

所有方法共享同一请求级协议：每个判断执行 K=3 个 replicates，各 replicate 是否交换 A/B 由预先冻结的 swap schedule 决定，再以三次结果的多数票形成稳定判断。在此基础上，各方法按下表进行系统级聚合：

| 方法 | 判断依据 | 最终系统聚合 |
|---|---|---|
| **Initial five-root M1** | Phase 5 初始化的五条 root 准则，不包含演化 children | 五个 roots 等权聚合 |
| **Native VL-RB prompt** | 不使用结构化 Rubric；按 Accuracy、Completeness、Clarity、Relevance 通用 prompt 直接作整体判断 | 无额外 Rubric 聚合 |
| **Visual Grounding subtree only** | Phase10 最终 Rubric 中的 `visual_grounding_and_details` root 及其 accepted children | 仅执行该子树的 parent/children specialized 聚合 |
| **Final weighted** | Phase10 最终完整 Rubric：5 个 roots、17 个 children | 五棵子树聚合；Visual Grounding 权重为 0.4，其余各为 0.15 |
| **Final five-root equal M1** | 与 Final weighted 相同的完整 Rubric 和 criterion 预测 | 五棵子树等权聚合；这是主要迁移结果 |

其中 Final weighted 的权重来自 heldout-500 上预设的方案，未在 VL-RewardBench 上重新搜索；Final weighted 与 Final equal 只是对同一组 criterion 预测采用不同的离线聚合方式。

指标口径如下：

- `OverallAcc` 是在形成明确最终判断的样本上计算的总体准确率；
- `MacroAcc` 是 General、Hallucination 和 Reasoning 三个官方类别 ACC 的算术平均；`Coverage` 是形成明确判断的样本比例；
- `严格 ACC` 固定以全部 1,247 个样本为分母，未决样本计为不正确。

#### 2. 总体结果

| 方法 | OverallAcc | MacroAcc | Coverage | 严格 ACC | 严格正确数 |
|---|---:|---:|---:|---:|---:|
| Initial five-root M1 | 45.04% | 47.75% | 97.75% | 44.03% | 549 |
| Native VL-RB prompt | 54.52% | 53.56% | 99.28% | 54.13% | 675 |
| Visual Grounding subtree only | 63.19% | 57.69% | 97.59% | 61.67% | 769 |
| Final weighted（Visual=0.4，其余各0.15） | 62.50% | 57.64% | **99.44%** | 62.15% | 775 |
| **Final five-root equal M1** | **64.31%** | **59.05%** | 98.64% | **63.43%** | **791** |

Final five-root equal M1 在 OverallAcc、MacroAcc 和严格 ACC 上均为最佳。此前由 heldout-500 预设的 Visual=0.4 权重没有迁移到 VL-RewardBench；其 OverallAcc 比等权聚合低 1.81 个百分点，因此外部结果以未在该 benchmark 上调优的等权 M1 为主。

#### 3. 相比 Initial five-root

| 指标 | Initial | Final Equal | 提升 |
|---|---:|---:|---:|
| OverallAcc | 45.04% | 64.31% | **+19.27 pp** |
| MacroAcc | 47.75% | 59.05% | **+11.30 pp** |
| Coverage | 97.75% | 98.64% | +0.88 pp |
| 严格 ACC | 44.03% | 63.43% | **+19.41 pp** |
| 严格正确数 | 549 | 791 | **+242** |

逐样本比较中，Final Equal 纠正 275 个 Initial 错误，同时损害 33 个 Initial 正确样本，净增加 242 个正确样本；exact McNemar $p=1.12\times10^{-48}$。这说明演化 Rubric 相对初始五条人工准则的提升并非由 Coverage 或少量样本波动造成。

#### 4. 相比 Native 通用 judge

| 指标 | Native | Final Equal | 提升 |
|---|---:|---:|---:|
| OverallAcc | 54.52% | 64.31% | **+9.79 pp** |
| MacroAcc | 53.56% | 59.05% | **+5.49 pp** |
| 严格 ACC | 54.13% | 63.43% | **+9.30 pp** |
| 严格正确数 | 675 | 791 | **+116** |

逐样本比较中，Final Equal 纠正 200 个 Native 错误，同时损害 84 个 Native 正确样本，净增加 116 个正确样本；exact McNemar $p=4.48\times10^{-12}$。因此增益不仅来自 Qwen3-VL-8B-Instruct 本身，也来自演化 Rubric 对同一 Worker 判断过程的结构化引导。

#### 5. 官方类别 ACC

下表使用 VL-RewardBench 官方口径：每个类别的正确数除以该类别形成明确最终判断的样本数，即 `covered_accuracy`；这三列的算术平均等于 MacroAcc。

| 方法 | General | Hallucination | Reasoning | MacroAcc |
|---|---:|---:|---:|---:|
| Initial five-root | 39.89% | 37.23% | **66.13%** | 47.75% |
| Native prompt | 43.68% | 53.07% | 63.92% | 53.56% |
| Visual only | 42.11% | 68.19% | 62.79% | 57.69% |
| Final weighted | 43.02% | 66.98% | 62.94% | 57.64% |
| **Final equal** | **43.82%** | **69.42%** | 63.90% | **59.05%** |

主要收益来自 Hallucination：Final Equal 相对 Initial 从 37.23% 提高到 69.42%，提升 32.19 个百分点；相对 Native 的 53.07% 提升 16.34 个百分点。General 小幅改善，而 Reasoning 相对 Initial 从 66.13% 降至 63.90%。因此当前演化 Rubric 的外部迁移收益主要表现为更强的视觉事实性和幻觉识别，而不是所有视觉推理能力的同步提升。

#### 6. 分析

**准则诱导的偏好冲突。** 初始五个 roots 并非都缺乏判断能力，而是在同一样本上经常给出相互冲突的 criterion-conditioned preferences。例如，Factuality 可能正确识别视觉错误，但 Completeness、Creativity 或 Clarity 会因回答更详细、更流畅而支持另一候选，最终形成多数票掩盖（majority masking）。Hallucination 类别上的 root/subtree 变化清楚展示了这一点：

| Root / subtree | Initial Hallucination covered ACC | 演化后 Hallucination covered ACC |
|---|---:|---:|
| Completeness | 32.40% | 53.00% |
| Visual Grounding | 39.31% | 68.19% |
| Factuality | **74.28%** | 71.10% |
| Creativity | 22.24% | 65.23% |
| Clarity | 36.63% | 68.05% |
| 五-root最终聚合 | 37.23% | **69.42%** |

初始 Factuality 已经达到 74.28%，但被其余 roots 的冲突票稀释；演化后最强专家并未继续提高，整体性能却显著上升。因此主要收益不是“单个最强准则变得更强”，而是原本容易受文本风格影响的准则被重新校准，减少了对正确视觉事实判断的干扰。

**隐含的层级式偏好。** 初始等权 M1 默认准确性、完整性、清晰度和创造性可以相互补偿，但当前偏好数据更接近非补偿式的优先关系：

```text
先满足视觉事实性
→ 再比较完整性、清晰度和创造性
```

Split 与 Refine 使多个 children 都加入视觉事实前置条件，实质上是在恢复“Visual factuality 是其他质量维度 prerequisite”的隐含偏好结构。这种语义收敛一方面改善了 Hallucination 判断，另一方面说明当前 Rubric 表示缺少共享的全局 prerequisite 或 gate，只能把相同约束重复编码进不同 children。

**改变了模型的注意力和决策优先级** 同一个 Qwen3-VL-8B-Instruct，在不同准则下会对同一图像给出不同判断。这说明 Worker 的问题至少有相当一部分不属于“完全看不见”，而属于：

1. 视觉证据被模型感知到了；
2. 但在长文本推理中没有被放在最高优先级；
3. 流畅性、回答长度、常识先验或语言置信度覆盖了视觉证据；
4. 最终投票与其分析过程中出现的视觉事实不一致。

而演化 Rubric 通过指定待检查事实、限制适用范围、精细判断规则并允许证据不足时输出 `None`，提高了已有视觉表征被正确用于最终偏好判断的概率。这一解释与《[Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Unveiling_the_Ignorance_of_MLLMs_Seeing_Clearly_Answering_Incorrectly_CVPR_2025_paper.html)》观察到的视觉 token 注意力弱于系统和问题 token，以及《[Multi-Modal Hallucination Control by Visual Information Grounding](https://openaccess.thecvf.com/content/CVPR2024/papers/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.pdf)》揭示的生成过程中图像条件依赖下降、语言先验增强现象一致。

后续可以验证一下是否改变了模型内部 attention；需视觉 token attention、图像扰动或 conditioned/unconditioned logits 等机制实验验证。

**Test-time compute 与 ensemble。** Final 系统对每个样本执行 22 个 criteria × 3 replicates，而 Native 只执行 3 次整体判断，因此在缺少 compute-matched 对照时，无法排除额外推理预算和 ensemble 对增益的贡献。不过，计算量并非充分解释：Initial five-root 同样执行多准则判断却只有 45.04% OverallAcc，而仅包含一个 root 及其 children 的 Visual Grounding subtree 已达到 63.19%。当前结果更合理的归因是“有效的 criterion 语义、结构化聚合和额外 test-time compute”共同作用，而不是单纯增加请求数。

**数据分布与权重迁移。** VL-RewardBench 的 1,247 个样本中有 749 个属于 Hallucination，和 discovery-90 的视觉事实性错误高度对齐，因此 Visual-only 表现很强。与此同时，VL-RewardBench 还包含 General 和 Reasoning，解释了 heldout-500 上预设的 Visual=0.4 权重没有迁移成功，以及等权五子树能够在 Visual subtree 错误或弃权时提供补充判断。固定 root 权重不是 Rubric 的固有属性，而是与评测分布相关的校准参数。

#### 7. 本质问题、研究定位与证据边界

当前结果也暴露出四个尚未解决的问题：

1. 多个 children 都加入视觉事实前置条件，可能形成语义同质化和重复推理；更自然的表示可能是共享 prerequisite、条件边或层级 gate。
2. 当前节点和子树基本采用固定聚合，局部 ACC 提升不保证全局 ACC 提升；强专家仍可能被较弱但相关的多数票掩盖。
4. discovery 只有 90 条视觉事实性幻觉样本，未覆盖开放式生成、General preference 和多步视觉推理；

基于这些结果，当前方法的核心叙事可以凝练为：

> 传统奖励学习通常将每条人类偏好视为一个独立监督标签。我们认为，少量偏好样本及其错误反馈中还隐含着可泛化的判断依据。为此，我们将偏好经验抽象为一个结构化、可执行且持续演化的 Rubric，使其显式编码跨样本复用的决策模式、适用边界和偏好优先关系，并作为自然语言奖励程序引导固定的多模态 Worker 作出判断。

这里学习的不是 90 个样本的独立答案，而是“先验证存在性”“流畅性不能补偿视觉错误”“证据不足时弃权”等可复用规则：

$$
\boxed{
\text{Sparse Human Preferences}
\rightarrow
\text{Failure Experience}
\rightarrow
\text{Structured Evolving Rubric}
\rightarrow
\text{Reusable Decision Patterns}
\rightarrow
\text{Reward Judgement}
}
$$
这一定位与《[A Survey of Reinforcement Learning for Large Language Models under Data Scarcity](https://arxiv.org/abs/2604.17312)》关注的稀缺高质量监督相契合：本工作将每条偏好从一次性标签转化为可反复执行的显式奖励规则，可理解为对稀缺偏好监督的语义放大（semantic amplification）。但当前完成的是 sparse-preference reward specification discovery，而不是已经完成强化学习。

《[Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)》为“从错误、归因和重试经验中学习可复用规则”提供了更长远的研究视角；不过当前仍是固定离线偏好上的 evaluative experience，而非智能体与环境长期交互产生的自主经验。《[Discovering State-of-the-art Reinforcement Learning Algorithms](https://www.nature.com/articles/s41586-025-09761-x)》自动发现的是 policy/prediction update rule，本工作自动发现的则是可解释的 reward/evaluation rule：前者研究“如何学习”，后者研究“什么是好”。

进行 DPO 等偏好后训练。只有当这些奖励信号能够在无重叠、覆盖生成与推理的数据上改善被训练模型，才能把结论从“可演化的 judge”进一步扩展为“数据稀缺条件下从经验中发现并扩展奖励信号的方法”。

实验 artifact 位于 `output/evolving_structured_rubrics/vl_rewardbench_phase10_transfer_v2_max2048/`；最终 Native 重试报告为 `native_retry_max10/report.json`，全部系统的最终逐样本投票为 `native_retry_max10/combined/logical_votes.json`。

---

## 12. Prompt 优化

### 12.1 Pairwise Worker Prompt v2 + 动态调度

**Pairwise Worker Prompt v1**

````shell
PAIRWISE_MULTIMODAL_WORKER_PROMPT = """## Instruction
You are judging an RLHF-V visual QA / image-text preference pair under one criterion. You are given the image, the source question, and two candidate answers.

Use the image when the criterion depends on visual evidence. If the criterion is not applicable to this pair, answer None.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}

Which candidate better matches the criterion and is more likely to align with human preference?"""

PAIRWISE_WORKER_PROMPT_POSTFIX = """
Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A based on the given criterion.",
    "analysis_b": "Analyze B based on the given criterion.",
    "thought": "Compare A and B.",
    "answer": "A / B / None"
}
```
Return None if any of the following conditions are met:
- The criterion is not applicable to this pair of data pieces.
- They are of the same quality.
- You are unsure.
"""
````

#### 实验目的与设计

Phase 10 的 Pairwise Worker 将 instruction、question、criterion 和 A/B 全部放在同一条动态 User prompt 中，可变 criterion 位于候选之前，既不利于 vLLM 复用公共前缀，也可能使紧邻最终问题的 Candidate B 获得位置优势。本实验在不修改 Phase 10 Rubric、M1 聚合、模型或解码温度的前提下，将稳定任务说明移入 System prompt，并把 User content 调整为：

````shell
PAIRWISE_MULTIMODAL_WORKER_SYSTEM_PROMPT_V2_CACHE = """## Instruction

You are judging a multimodal image-text preference pair under one criterion. You are given the image, the source instruction or question, and two candidate responses.

Use the image when the criterion depends on visual evidence. If the criterion is not applicable to this pair, answer None.

Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A based on the given criterion.",
    "analysis_b": "Analyze B based on the given criterion.",
    "thought": "Compare A and B.",
    "answer": "A / B / None"
}
```

Return None if any of the following conditions are met:
- The criterion is not applicable to this pair of data pieces.
- They are of the same quality.
- You are unsure.
"""

PAIRWISE_MULTIMODAL_WORKER_USER_PROMPT_V2_CACHE = """## Source Instruction or Question
{question}

## Candidate A
{A}

## Candidate B
{B}

## Criterion
**{criterion}**: {description}

Which candidate better matches the criterion and is more likely to align with human preference?"""
````

比较的主要协议为：

| 系统 | Prompt       | 调度                    | 作用                            |
| ---- | ------------ | ----------------------- | ------------------------------- |
| S0   | 原 Prompt v1 | available-slot 动态调度 | Phase 10 已有结果，作为性能基线 |
| S3   | Prompt v2    | available-slot 动态调度 | 新 Prompt 部署候选              |

`available-slot` 动态调度是指：所有请求进入同一个待处理队列，哪个 API 端点或并发槽位先空闲，就立即领取下一条请求；系统不会预先把某个样本或 criterion 固定绑定到特定端点，因此能够减少快慢请求不均造成的空闲等待。

两个系统使用相同的 Phase 10 最终 Rubric（5 roots、17 children，rubric SHA-256=`17ad7a0...7ef`）、`Qwen3-VL-8B-Instruct`、`temperature=0.5` 和单 replicate。S3 将 `max_tokens` 固定为2048，并继续使用原 available-slot 动态 backend pool。S0 的性能直接引用10.7中已经冻结的 Phase 10 结果，避免将一次带随机采样波动的重跑误作新的基线；另行执行的原 Prompt 全量重跑只用于比较耗时、尾延迟与位置偏置。

#### Discovery-90 结果

| 系统           |     M1 ACC | Coverage |
| -------------- | ---------: | -------: |
| S0（Phase 10） | **73.33%** |  100.00% |
| S3             | **73.33%** |  100.00% |

S3 在 discovery-90 上完整复现 Phase 10 的73.33% ACC，并保持100% Coverage，说明新 Prompt 没有破坏已演化 Rubric 在发现集上的整体决策能力。

#### Heldout-500 结果

| 系统           |       Correct |     M1 ACC |   Coverage |
| -------------- | ------------: | ---------: | ---------: |
| S0（Phase 10） |     357 / 500 |     71.40% |     98.60% |
| S3             | **383 / 500** | **76.60%** | **99.40%** |

S3 相对 Phase 10 的 S0 提高 **5.2 pp**；逐样本比较为58 corrected、32 harmed、净纠正26条，exact McNemar $p=0.0080$。22个节点中，21个节点的 full-500 strict ACC 提高，唯一未提高的 `completeness_via_verified_perception` 仅下降0.2 pp，同时其 covered ACC 从74.4%提高到82.4%，表现为更保守的适用域。

#### 效率与位置偏置分析

| 指标          | 原 Prompt timing |       S3 |       变化 |
| ------------- | ---------------: | -------: | ---------: |
| 总耗时        |         64.6 min | 42.0 min | **-35.0%** |
| 推理次数/分钟 |            170.4 |    262.1 | **+53.8%** |
| P50 latency   |          11.92 s |   8.49 s | **-28.8%** |
| P90 latency   |          18.29 s |  11.07 s | **-39.5%** |
| P99 latency   |          66.34 s |  15.85 s | **-76.1%** |
| 额外模型调用  |              136 |       36 | **-73.5%** |
| 输出 tokens   |           3.30 M |   2.76 M | **-16.2%** |

这里每次推理对应一个 criterion-conditioned Pairwise Worker 请求。S3将推理吞吐从每分钟170.4次提高到262.1次。输入 tokens 仅减少0.3%，说明加速并非来自缩短输入，而主要与输出更简洁、解析重试减少和长尾延迟收敛有关。这里的原 Prompt 全量重跑只作为同规模 timing control，不替代 Phase 10 的71.4%性能基线。实验未重置服务端 prefix cache，且原 Prompt先于S3运行，因此35%的端到端加速不能全部归因于 prefix-cache 命中；但吞吐、重试和尾延迟改善均由真实运行记录支持。

进一步分析发现，原 Prompt 存在明显的 Candidate B 偏置，而 Prompt v2 将A/B表现校准到近似对称。heldout gold 本身基本均衡（A=244、B=256）：

| 系统          | 预测 A / B / Tie | gold-A ACC | gold-B ACC |
| ------------- | ---------------: | ---------: | ---------: |
| 原 Prompt重跑 |    212 / 282 / 6 |     62.70% |     75.78% |
| S3            |    247 / 250 / 3 | **77.05%** | **76.17%** |

S3主要将 gold-A ACC 提高14.35 pp，而gold-B基本保持不变。最合理的解释是：原 Prompt 中 Candidate B 紧邻最终问题，**产生了 recency/position bias**；Prompt v2 将 criterion 放在两个候选之后，使最终判断**重新围绕 criterion**，并降低候选位置不对称。

#### 结论与证据边界

当前结果支持将 **Prompt v2 + available-slot动态调度 + `max_tokens=2048`** 作为后续默认 Pairwise Worker 协议：它没有破坏已演化 Rubric 的语义执行，反而**相对历史 Phase 10 提高5.2 pp**，并显著降低运行时间和长尾不稳定性。更准确的表述不是“输出行为没有漂移”，而是“Prompt v2产生了显著但净收益为正的决策校准”。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/pairwise_worker_cache_prompt_ablation_v1/`；discovery S3报告为 `discovery90/s3_report.json`，heldout报告为 `heldout500/s3_report.json`。

### 12.2 Prompt v2 的 VL-RewardBench 外部迁移验证

#### 实验目的与冻结协议

12.1 已在 discovery-90 和 heldout-500 上验证 Prompt v2 能提高 Phase 10 Rubric 的执行性能。本实验进一步检验该收益能否迁移到外部 VL-RewardBench，而不是只适用于开发期的 RLHF-V 风格数据。

评测集固定为 VL-RewardBench 的1,247个偏好对，包含 General、Hallucination 和 Reasoning 三类。Pairwise Worker 固定为 `Qwen3-VL-8B-Instruct`、`temperature=0.5`、`max_tokens=2048`，使用8000和8001两个端点的 available-slot 动态调度。每个偏好对执行 $K=3$ 次随机 A/B 顺序评测，再按多数投票恢复为原始偏好方向。结构化系统使用同一份 Phase 10 最终 Rubric（5 roots、17 children，共22个节点）；22个节点的预测同时离线构造 Initial five-root、Visual-only、Final weighted 和 Final equal，避免为不同聚合方案重复调用模型。Prompt v1 指标直接读取已经完成的历史实验，Prompt v2 的技术失败最多额外重试10次。

指标口径如下：`OverallAcc` 是形成明确最终判断的样本上的总体准确率；`MacroAcc` 是 General、Hallucination 和 Reasoning 三类 covered accuracy 的算术平均；`Strict ACC` 以全部1,247个样本为分母，将未覆盖样本计为错误。

#### 总体结果

| 方法 | OverallAcc | MacroAcc | Coverage | Strict ACC |
| --- | ---: | ---: | ---: | ---: |
| Native VL-RewardBench Prompt | 54.52% | 53.56% | 99.28% | 54.13% |
| Initial five-root，Prompt v1 | 45.04% | 47.75% | 97.75% | 44.03% |
| Initial five-root，Prompt v2 | 58.12% | 54.60% | 98.24% | 57.10% |
| Phase 10 Final equal，Prompt v1 | 64.31% | 59.05% | 98.64% | 63.43% |
| **Phase 10 Final equal，Prompt v2** | **69.53%** | **63.37%** | **98.96%** | **68.81%** |
| Phase 10 Final weighted，Prompt v2 | 68.66% | 62.25% | 99.28% | 68.16% |
| Phase 10 Visual-only，Prompt v2 | 68.85% | 62.31% | 96.79% | 66.64% |

最终等权系统在1,234个明确判断中正确858个，因此 `OverallAcc=858/1234=69.53%`；以全部样本计分仍有 `Strict ACC=858/1247=68.81%`。其 Coverage 接近99%，结果不是通过大量输出 `None` 获得。Prompt v2 下等权聚合比预设 Visual=0.4 的 weighted 聚合高0.87 pp，但两者的配对差异不显著（净差8条，exact McNemar $p=0.332$），因此本实验不足以判定等权聚合普遍优于加权聚合。

#### Prompt 与 Rubric 演化的独立贡献

| 配对比较 | Corrected | Harmed | Net corrected | Exact McNemar $p$ |
| --- | ---: | ---: | ---: | ---: |
| Initial v2 $\rightarrow$ Phase 10 Final equal v2 | 158 | 12 | **+146** | $1.18\times10^{-33}$ |
| Phase 10 Final equal v1 $\rightarrow$ v2 | 141 | 74 | **+67** | $5.73\times10^{-6}$ |
| Initial five-root v1 $\rightarrow$ v2 | 217 | 54 | **+163** | $2.51\times10^{-24}$ |

固定 Prompt v2 后，Rubric 演化仍将 OverallAcc 从58.12%提高到69.53%，提升11.41 pp；MacroAcc提高8.77 pp，Strict ACC提高11.71 pp。固定 Phase 10 Rubric 后，Prompt v2 相对 Prompt v1 又提高5.22 pp OverallAcc。因而最终性能不是单独由 Prompt 改写或 Rubric 演化造成，而是二者的叠加：Prompt v2 提高单准则执行的稳定性，演化 Rubric 则提供更有效的结构化决策模式。

Prompt v2 对 Initial five-root 的增益（+13.09 pp OverallAcc）大于对最终 Rubric 的增益（+5.22 pp），说明详细的演化 criterion 已经能够部分补偿旧 Prompt 的执行偏差；新 Prompt 对较粗的初始 roots 校准作用更强。

#### 类别结果

| 类别 | Initial v2 | Final equal v2 | Rubric 演化增益 | Final equal v1 $\rightarrow$ v2 |
| --- | ---: | ---: | ---: | ---: |
| General | 37.71% | 46.89% | +9.18 pp | +3.07 pp |
| Hallucination | 59.32% | **75.81%** | **+16.48 pp** | **+6.39 pp** |
| Reasoning | 66.77% | 67.41% | +0.64 pp | +3.51 pp |

Rubric 演化带来的146条净纠正中，General、Hallucination 和 Reasoning 分别贡献17、125和4条，即约85.6%的净收益来自 Hallucination。这与 discovery-90 主要包含视觉事实性和幻觉错误相一致。与此同时，Final equal 的 MacroAcc 仍相对 Prompt v1 提高4.33 pp，说明收益并非只由 Hallucination 样本在 VL-RewardBench 中占比较高造成；但 General 的46.89%和较小的 Reasoning 演化增益也表明，当前 Rubric 的主要能力边界仍是视觉事实性偏好判断。

#### 技术可靠性与证据边界

完整评测包含 $1247\times22\times3=82{,}302$ 次 criterion-conditioned Pairwise Worker 判断。初次运行留下70条技术失败，占0.085%；专门重试新增122次模型请求，最终70/70全部恢复，未解决技术失败为0。完整运行记录的平均吞吐为236.9次推理/分钟；动态调度分别向8000和8001端点分配38,334和44,517次实际模型调用，较快端点自动承担了更多请求。

综合而言，VL-RewardBench 结果支持将 **Prompt v2 + available-slot 动态调度 + `max_tokens=2048`** 作为后续默认 Pairwise Worker 协议：它在外部数据上保留了 Rubric 演化的显著收益，并使 Phase 10 Final equal 相对 Prompt v1 再提高5.22 pp。

实验 artifact：`output/evolving_structured_rubrics/vl_rewardbench_phase10_prompt_v2_transfer_v1/`；最终报告为 `final_report.json` 和 `final_report.md`，技术重试报告为 `retry/report.json`。



## 13. Gate Worker

### 13.1 设计与评估协议

Gate Worker 不直接判断 A/B 偏好，只根据图像、问题、候选回答以及同一 parent 下各 child 的 `Applicable only when` / `Not applicable when` 条件，输出每个 child 的 `applicable` 或 `not_applicable`。Parent 仍参与判断；被激活 child 的偏好票仍由 Pairwise Worker 产生，Gate 只控制哪些 child 进入当前父子树的聚合。

当前实验采用以下冻结设置：

- 暂不使用 Root Gate，五个 roots 始终参与最终 M1；
- 每个 root 独立路由其 direct children，不跨 root 激活；
- Gate prompt 采用与 Pairwise Worker Prompt v2 一致的固定 System Prompt + 动态 User Prompt 组织方式；同一个root的 gate worker 提示词是一样的，推理不同样本的时前面大量的Prompt命中缓存，加速推理
- `temperature=0.2`，`max_tokens=2048`，使用8000和8001两个端点动态调度；
- 非法 JSON 最多重试5次；最终失败记录解析错误并回退 all children，不中断整批实验；
- RLHF-V 实验复用冻结的 Phase 10 Pairwise predictions，仅新增 Gate 判断，从而将性能变化归因于路由。

### 13.2 Visual Grounding Gate-only

第一组实验只对 `visual_grounding_and_details` 的4个 children 进行动态路由，其他 roots 保持不变。实验用于验证：criterion description 中自然形成的适用与不适用条件，能否直接作为可执行 edge contract。

| 数据与系统 | Visual 子树 ACC | 完整五-root M1 ACC |
| --- | ---: | ---: |
| discovery-90：Parent only | 68.89% | 71.11% |
| discovery-90：All children | 71.11% | 73.33% |
| discovery-90：Best fixed subset | **77.78%** | 73.33% |
| discovery-90：Dynamic Gate | 76.67% | **75.56%** |
| discovery-90：Oracle routing | 88.89% | 76.67% |
| heldout-500：Parent only | 75.80% | 76.20% |
| heldout-500：All children | **77.60%** | **76.60%** |
| heldout-500：Best fixed subset | 75.20% | 75.80% |
| heldout-500：Dynamic Gate | 74.20% | 75.40% |
| heldout-500：Oracle routing | 88.80% | 78.20% |

Discovery 上 Dynamic Gate 相对 All children 在 Visual 子树中 corrected 7、harmed 2，并把完整 M1 提高2.22 pp；但 heldout 上该优势没有泛化，Visual 子树和完整 M1 分别下降3.40 pp和1.20 pp。与此同时，Gate 将**平均激活量降至0.76/4**，并将 heldout sibling conflicts 从110次降至6次；该实验说明 Gate 能形成稀疏且有语义的局部路由，但仅依赖现有适用性文本还不足以稳定提升未见数据 ACC。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/phase13_visual_grounding_gate_only_v3/`。

### 13.3 完整 Rubric Child-Gate

第二组实验将同一 Gate 机制扩展至五个 roots：五个 roots 仍全部参与，每个 root 分别选择其 direct children。Phase 10 Rubric 共包含17个 children。

| 数据与系统 | Parent only | All children | Visual-only Gate | Full Child-Gate | Oracle routing |
| --- | ---: | ---: | ---: | ---: | ---: |
| discovery-90 M1 ACC | 72.22% | 73.33% | **75.56%** | 70.00% | 82.22% |
| heldout-500 M1 ACC | 75.80% | **76.60%** | 75.40% | 75.80% | 88.20% |

Heldout 上 Full Child-Gate 相对 All children corrected 17、harmed 21、净差−4，**ACC 只下降0.80 pp**，exact McNemar $p=0.627$；Coverage 从99.4%降至98.6%。在结构和计算量方面，平均激活 children 从17个降至3.566个，**减少79.0%**；sibling conflicts 从530次降至13次，减少97.5%；2,500次 Gate 路由的解析成功率为100%。按请求数量估计，完整执行由每样本22次 Pairwise 判断变为约 $5\text{ roots}+5\text{ Gates}+3.566\text{ children}=13.566$ 次请求，对应约1.62倍的理论请求级加速；实际端到端加速仍需在线短路执行验证。

该结果表明 Full Child-Gate 在 RLHF-V heldout 上形成了较好的**精度—效率折中**，但它没有提高 ACC。Oracle 的88.20%同时说明，children 之间存在很大的动态路由上限，主要瓶颈是 Gate 的选择质量，而不是稀疏路由本身。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/phase14_full_rubric_child_gate_v1/`。

### 13.4 VL-RewardBench 外部迁移

第三组实验在 VL-RewardBench 1,247条样本上评估完整 Child-Gate。Pairwise predictions 与第12章 Prompt v2 的 K=3 结果完全冻结；Gate 同样独立运行3次后聚合，因此差异只来自 child routing。

| 系统 | OverallAcc | MacroAcc | Coverage | Strict ACC |
| --- | ---: | ---: | ---: | ---: |
| Parent only | 58.12% | 54.60% | 98.24% | 57.10% |
| All children | **69.53%** | **63.37%** | 98.96% | **68.81%** |
| Full Child-Gate | 60.99% | 56.71% | 98.48% | 60.06% |

Full Child-Gate 相对 All children 的 OverallAcc 下降8.54 pp：corrected 17、harmed 126、净差−109，exact McNemar $p=9.45\times10^{-22}$。下降主要集中在 Hallucination（75.81%→63.48%）和 General（46.89%→39.66%），Reasoning 基本持平（67.41%→66.99%）。解析不是主要问题：18,705次逻辑路由中只有4次最终失败，解析率为99.979%。

Gate 仍把平均激活 children 从17个降至3.297个，减少80.61%，并把 sibling conflicts 从4,597次降至200次。**单独只对一个 root 使用 Gate 时 OverallAcc 仍为67.69%–69.29%**，但五个 roots 同时 Gate 后降至60.99%，说明多个局部剪枝在最终多数投票中发生了非线性叠加。当前 Gate 学习的是“criterion 在语义上是否适用”，尚未建模 child 对最终 M1 投票方向和票差的边际贡献；五个 roots 又始终无条件参与，因此跨-root 冲突也没有得到处理。当前零样本 edge contract 尚不能跨数据分布稳定替代 All children。

因此，现阶段证据支持将 Gate 定位为有效的稀疏化机制，而不是已经完成的精度优化机制。后续 Discovery 应保留全部 child 的**反事实预测**，同时记录 Gate 的**误激活和漏激活**，用实际边际贡献反馈优化 `Applicable only when` / `Not applicable when`；Root Gate 则应作为独立变量验证，避免与 Child Gate 的误差叠加后无法归因。

实验 artifact：`output/evolving_structured_rubrics/vl_rewardbench_phase14_full_child_gate_v1/`；最终报告为 `final_report.json` 和 `final_report.md`。

---

## 14. Root Boundary Pre-Refine 探索实验（暂停）

### 14.1 实验动机与设计

该实验尝试在正式 Split+Refine 之前，先对五个初始 roots 执行 Refine，使每个 root 的 description 显式包含 `Criterion focus`、`Applicable only when`、`Not applicable when` 和 `Decision rule`。预期作用是降低五个 roots 接近全覆盖所造成的语义冗余和投票冲突，再以更新后的 roots 作为后续 Split 起点。

实验产物位于 `output/evolving_structured_rubrics/rubric_evolution_phase5/phase15_root_boundary_pre_refine_split_refine_v3/root_pre_refine/`，冻结协议如下：

- 数据为 discovery-90，候选选择阶段不访问 heldout-500；
- 397B Refine Manager 使用 epoch-start `global_rubric_v1`、当前 root 的错误/正确/弃权证据和完整失败历史，每次只生成一个新 description；
- Pairwise Worker 使用 Qwen3-VL-8B-Instruct，`temperature=0.5`、`max_tokens=2048`，通过8000和8001 available-slot pool执行；
- 同一 epoch 的所有 root 基于同一个起始 Rubric 独立评估，接受项在 epoch 末同步提交；
- 接受条件保持 Refine v1 不变：新 criterion 在自身 A/B 支持集上的条件 ACC 严格提高，且 support 不低于15；Coverage、subtree ACC 和完整 M1 只作诊断；
- 接受 root 立即锁定；拒绝 root 携带自然语言失败归因重试，最多3个 epoch。

### 14.2 三轮演化结果

实验共形成11次候选：Epoch 1 调度全部5个 roots，接受 Factuality 和 Creativity；其余3个 roots 在 Epoch 2–3继续重试但全部失败，最终状态为2个 accepted、3个 exhausted。

| Root | 初始 ACC / Coverage | 最终 ACC / Coverage | 尝试次数 | 最终状态 |
| --- | ---: | ---: | ---: | --- |
| Completeness and Coverage | 68.29% / 91.11% | 68.29% / 91.11% | 3 | exhausted，保留原文 |
| Visual Grounding and Details | 67.42% / 98.89% | 67.42% / 98.89% | 3 | exhausted，保留原文 |
| Factuality, No Hallucination | 59.26% / 90.00% | **62.35% / 94.44%** | 1 | accepted |
| Creativity and Expressiveness | 57.30% / 98.89% | **60.00% / 61.11%** | 1 | accepted |
| Clarity and Coherence | 56.82% / 97.78% | 56.82% / 97.78% | 3 | exhausted，保留原文 |

Factuality 是本次最明确的正向结果：正确数由48/81提高到53/85，节点 corrected/harmed 为7/2；单独替换该 root 时，完整五-root M1 从62.22%提高到65.56%，M1 corrected/harmed 为3/0。Creativity 则主要通过收窄适用域提高条件 ACC：正确数由51/89降至33/55，corrected/harmed 为3/21；单独替换时完整 M1 保持62.22%不变。两项同步提交后的最终 discovery M1 如下。

| 指标 | 初始五-root | Root Pre-Refine 最终 | 变化 |
| --- | ---: | ---: | ---: |
| M1 ACC | 62.22%（56/90） | **64.44%（58/90）** | +2.22 pp |
| M1 Coverage | 97.78% | 95.56% | -2.22 pp |
| Covered ACC | 63.64% | **67.44%** | +3.80 pp |
| 平均激活 roots | 4.77 | 4.43 | -0.34 |
| 五个 roots 全部激活 | 73/90 | 49/90 | -24 |
| 存在 root 冲突的样本 | 44/90 | 45/90 | +1 |

三个失败 root 的条件 ACC 轨迹如下；失败历史没有产生逐轮改善。

| Root | 原 ACC | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: | ---: |
| Completeness | **68.29%** | 63.64% | 60.24% | 58.46% |
| Visual Grounding | **67.42%** | 62.03% | 60.24% | 62.20% |
| Clarity | **56.82%** | 43.14% | 38.89% | 41.03% |

### 14.3 分析与阶段决策

本实验说明 Root Refine 可以产生局部收益，但尚不能稳定优化所有 roots。Factuality 与当前视觉事实/幻觉数据分布高度匹配，因此获得了 ACC、Coverage 和 M1 同向提升；Creativity 的适用域虽然明显专业化，但其条件 ACC 提升主要来自大量弃权，暴露出仅以条件 ACC 接受 root 候选可能奖励 coverage collapse。Completeness、Visual Grounding 和 Clarity 的重试则持续出现过度 `None`、偏好方向反转以及与 Factuality 的边界混淆。

结构上，平均激活量和两两激活重叠有所下降，但改善主要由 Creativity 的 Coverage 从98.89%降至61.11%贡献；冲突样本没有减少，说明实验尚未实现五个 roots 的全面职责分离。更根本的限制是 discovery-90 主要由视觉事实性幻觉样本构成，能够为 Factuality/Visual Grounding 提供反馈，却缺少足够的 Clarity、Creativity、Completeness 和推理类偏好证据。强行让所有 roots 从同一错误分布学习边界，容易把“非目标错误”误写成排除条件。

因此，**Root Boundary Pre-Refine 暂停，不纳入当前主方法，也不将本实验接受的两个 root patches 带入后续主实验**。现阶段主链路继续采用已验证的 Global Memory、Locked-Child Split 和 Role-aware Child Refine。本实验保留为负向探索证据：Root-level 边界学习需要与各 root 匹配的多领域 discovery 数据，以及能够区分真实专业化与选择性弃权的竞争指标；满足这些条件后再单独恢复验证。本阶段不继续执行其后的 Split+Refine，也不据此声称 heldout 泛化收益。



---

## 15. Prompt v2 全链路 Split+Refine 演化（Phase16）

### 15.1 实验动机与冻结对照

第12章说明：保持 Phase10 Rubric 不变，仅将 Pairwise Worker 切换为 Prompt v2，heldout-500 M1 已由71.4%提升至76.6%。Phase16 进一步检验的不是 Prompt v2 的**执行端**收益，而是：让它从 discovery 的初始 Pairwise、错误样本、Split/Refine 竞争和失败历史开始全程参与后，能否演化出更好的 Rubric。

| 系统 | Rubric 来源 | Pairwise Worker Prompt | 作用 |
| --- | --- | --- | --- |
| Initial five-root | 初始五 roots | Prompt v2 | 初始化基线 |
| Phase10 + Prompt v2 | 旧 Prompt 演化的 Phase10 Rubric | Prompt v2 | 主 Control |
| **Phase16 + Prompt v2** | Prompt v2 全链路演化 | Prompt v2 | Treatment |

Phase16 保持 Global Memory、Locked-Child Split、Role-aware Child Refine、局部 Specialized Accuracy 接受规则以及 397B Manager 不变；不引入 Gate、Root Boundary Pre-Refine、权重搜索或新的聚合规则。为使五个初始 roots 都能够被检验，Split 触发阈值设为 `ACC < 0.75` 且 `Coverage > 0.80`。Pairwise Worker 使用 Prompt v2、`temperature=0.5`、`max_tokens=2048`，通过8000和8001 available-slot pool 执行。

### 15.2 五轮演化轨迹

Epoch 1 的五个 roots 全部进入 Split：Completeness、Visual Grounding 和 Clarity 直接接受；Factuality 与 Creativity 在拒绝后携带失败历史重试，并于 Epoch 2 接受，其中 Creativity 保留了已识别的强 child。随后只调度满足 role-aware 条件的 children Refine。Rubric 节点数由5增长到23。

| Epoch | Split 调度 / 接受 | Refine 调度 / 接受 | 提交后节点数 | Discovery M1 ACC / Coverage |
| --- | ---: | ---: | ---: | ---: |
| 0（初始） | – | – | 5 | 72.22% / 97.78% |
| 1 | 5 / 3 | 0 / 0 | 15 | 74.44% / 100.00% |
| 2 | 2 / 2 | 10 / 5 | 23 | 73.33% / 100.00% |
| 3 | 0 / 0 | 14 / 6 | 23 | 73.33% / 98.89% |
| 4 | 0 / 0 | 14 / 1 | 23 | 73.33% / 98.89% |
| 5（最终） | 0 / 0 | 13 / 2 | 23 | 71.11% / 98.89% |

这条轨迹表明 Prompt v2 能支持局部 Split 和 Refine 的接受，但 discovery 最终值并未单调上升：后期重复 Refine 的接受率由 Epoch 2–3 的5/10、6/14下降至 Epoch 4–5 的1/14、2/13，反映当前90条 discovery 反馈不足以稳定选择更深的局部改写。

### 15.3 Heldout-500 checkpoint 诊断

heldout 仅在各 epoch Rubric 固定后进行探索性、后验诊断，不参与任何候选选择。Epoch 1–4 中仅对相对于最终 Rubric 改变的14个 unique description 补充预测，其余节点预测严格复用。

| Checkpoint | Heldout ACC | 正确数 | Coverage |
| --- | ---: | ---: | ---: |
| Epoch 0：Initial five-root Prompt v2 | 75.80% | 379 / 500 | 98.80% |
| Epoch 1 | 76.00% | 380 / 500 | 99.00% |
| Epoch 2 | 76.00% | 380 / 500 | 99.20% |
| **Epoch 3** | **78.20%** | **391 / 500** | **99.80%** |
| **Epoch 4** | **78.20%** | **391 / 500** | **99.80%** |
| Epoch 5：最终 Rubric | 78.00% | 390 / 500 | 99.80% |
| Phase10 + Prompt v2 Control | 76.60% | 383 / 500 | 99.40% |

因此，Prompt v2 全链路演化相对 Prompt v2 Control 在最终 epoch 有+1.4 pp heldout 增益；最佳观察值出现在 Epoch 3–4（+1.6 pp）。但这是在已被多次访问的同一 heldout-500 上进行的 exploratory checkpoint 分析，只能说明该轨迹中的局部收益，不能据此重新选择“正式”epoch。

### 15.4 VL-RewardBench 外部迁移与 checkpoint 对照

为检查 heldout 峰值是否是更普适的改进，使用相同的 Prompt v2、K=3 counterbalanced A/B 顺序和完整1247对 VL-RewardBench，比较初始、Phase10 Control，以及 Phase16 的 Epoch 2、3、5。`OverallAcc`和`MacroAcc`均在有明确最终判断的样本上按该 benchmark 协议计算；Coverage 另行报告。

| 系统 | OverallAcc | MacroAcc | Strict ACC | Coverage | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial five-root Prompt v2 | 58.12% | 54.60% | 57.10% | 98.24% | 712 |
| Phase10 + Prompt v2 Control | **69.53%** | **63.37%** | **68.81%** | 98.96% | **858** |
| Phase16 Epoch 2 | 66.02% | **60.65%** | 65.44% | 99.12% | 816 |
| Phase16 Epoch 3 | 66.29% | 60.48% | 65.60% | 98.96% | 818 |
| Phase16 Epoch 5 | 66.50% | 60.58% | 65.92% | **99.12%** | 822 |

Phase16 相比 Initial Prompt v2 在外部基准上仍有明显正向迁移（Epoch 5 OverallAcc +8.38 pp，110个额外正确样本），说明全链路演化并未失效；但其仍低于“Phase10 Rubric + Prompt v2 推理”Control 3.03 pp。Epoch 3 的 heldout 峰值也没有转化为 VL-RewardBench 最优点：E2、E3、E5 的 OverallAcc 仅在66.02%–66.50%之间变化。该不一致与 discovery-90 主要覆盖视觉事实/幻觉，而 VL-RewardBench 同时包含 General、Hallucination 与 Reasoning 的分布差异一致。

### 15.5 Phase16 E5 的 Root Router + Child Gate 推理诊断

本节进一步固定 Phase16 E5 的23-node Rubric，测试其在外部部署时能否通过两层路由稀疏化：Root Router 从五个 roots 中选择应参与聚合的模块；Child Gate 再在已选 root 内选择 direct children。Pairwise Worker 的 Prompt v2、K=3 counterbalanced A/B 顺序和全部节点预测完全复用；仅新生成路由，因此系统差异只来自路由，而不是 Rubric 或 Pairwise 判断。

| 系统 | OverallAcc | MacroAcc | Strict ACC | Coverage | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All roots + all children | **66.50%** | **60.58%** | **65.92%** | **99.12%** | **822** |
| Root Router only | 66.21% | 60.34% | 62.23% | 93.99% | 776 |
| Child Gate only | 60.59% | 56.94% | 59.66% | 98.48% | 744 |
| Root Router + Child Gate | 62.22% | 58.13% | 57.98% | 93.18% | 723 |

Root Router 将平均 active roots 由5降至3.207（−35.9%）；联合路由后平均 active children 为3.044/18。若按路由结果再执行 Pairwise，每样本理论上仅需 $3.207+3.044=6.251$ 个节点判断，而非23个，具有72.8%的节点级稀疏化潜力；本实验复用了全量 Pairwise artifact，故不将其表述为实测端到端加速。

但当前路由会丢失全量投票中的交叉补偿：相对全量集成，Root Router only 的 corrected/harmed 为9/55（净−46），Child Gate only 为13/91（净−78），联合路由为14/113（净−99）。22,446次逻辑路由中只有5次 Child Gate 在重试后仍不合法（99.978%有效，按协议回退 all children），因此退化并非主要由解析失败造成。该结果将 Gate/Router 定位为 E5 的稀疏化证据，而不是可直接替代全量聚合的精度方案。

实验 artifact：`output/evolving_structured_rubrics/vl_rewardbench_phase16_e5_root_child_router_v1/`；最终报告为 `final_report.json` 和 `final_report.md`。

**阶段结论。** Prompt v2 不仅提升既有 Rubric 的执行，也能够支撑从错误反馈出发的 Split+Refine 全链路演化，并在 heldout-500 上得到75.8%→78.0%的提升；但在现有单一视觉幻觉 discovery 分布下，更多 epoch 的局部优化尚未超过 Phase10 Control 的外部泛化。后续应优先扩展 discovery 的生成、推理和通用偏好覆盖，而不是把 Epoch 3–4 的 heldout 峰值直接作为新的正式模型选择依据。

---

## 16. Discovery-v2 Prompt v2 Split+Refine 演化（Phase17）

### 16.1 实验动机与数据集

Phase16 证明 Prompt v2 全链路演化能够提高 RLHF-V heldout-500，但其 discovery-90 几乎全部来自视觉事实性幻觉，难以为通用偏好、指令遵循和多模态推理提供充分反馈。Phase17 因此不再修改算子或聚合规则，而是单独检验：**当 discovery 数据覆盖更多偏好类型后，同一套 Prompt v2、Locked-Child Split 和 Role-aware Refine 能否获得更好的外部迁移。**

本实验使用新构造的 `Discovery-v2 demo v3`：

| 子集 | 文件 | 数量与分布 | 在演化中的用途 |
| --- | --- | --- | --- |
| Discovery100 | `data/discovery_v2_demo_v3/discovery_100.jsonl` | 100条；Visual/Reasoning/General 为34/33/33；A/B gold 为50/50 | ErrorSignature、Split/Refine 生成与候选接受 |
| Dev150 | `data/discovery_v2_demo_v3/dev_150.jsonl` | 150条；三个领域各50条；六个来源各25条；A/B gold 为75/75 | 每个 epoch 的只读泛化诊断 |

数据来自 RLHF-V、MM-RLHF、ViLReward-73K、MMPR-v1.2、VisionArena-Battle 和 MMIF-23K v2 六个 source family。原始记录经过图片可恢复性、答案完整性、长度异常、近相同回答过滤，并按 source ID、image、question 和 unordered answer pair 与 VL-RewardBench 去重。Discovery100 由70条 coverage core 与30条 hard core 组成，以兼顾分布覆盖和具有判别价值的困难偏好；Dev150 仅按数据属性分层冻结，不使用 Worker 错误选样。数据来源、版本、下载、清洗、去重、筛选和质量分析详见 [Discovery-v2 数据构造文档](<Discovery-v2 Data Collection.md>)。

需要特别区分：Dev150 对 Manager 不可见，不生成 ErrorSignature，不参与候选接受、早停或 checkpoint 选择；它只用于观察演化轨迹是否跨样本泛化。

### 16.2 冻结协议与演化过程

Phase17 从相同的五个初始 roots 独立开始，保持 Phase16 的核心算法不变：

- Pairwise Worker 为 Qwen3-VL-8B-Instruct，使用 Prompt v2 `1.1.0-cache-pilot`、`temperature=0.5`、`max_tokens=2048`，通过8000与8001 available-slot pool 执行；
- Manager 为 Qwen3.5-397B-A17B，并默认使用 `global_rubric_v1`；
- Split 只调度初始 roots，失败后可锁定强 child 并继续优化其他 children；Refine 使用 role-aware child trigger；
- 所有候选只由 Discovery100 决定接受，epoch 内同步提交，固定运行3–5轮；
- 不使用 Gate、Root Router、Root Boundary Pre-Refine、权重搜索或 VL-RewardBench 反馈。

五个 roots 均完成 Split 接受：Visual Grounding 在 Epoch 1 接受；Factuality、Creativity 和 Clarity 在 Epoch 2 接受；Completeness 在 Epoch 4 接受。最终 Rubric 从5个 nodes 增长到27个 nodes，其中22个为 children。共发生11次 Split attempts（5 accepted、5 rejected、1 proposal-invalid）和61次 Refine attempts（21 accepted、38 rejected、2 invalid）。

| Epoch | Nodes | Discovery100 M1 ACC / Coverage | Dev150 M1 ACC / Coverage |
| --- | ---: | ---: | ---: |
| 0：Initial five-root | 5 | 64.00% / 98.00% | 72.67% / 97.33% |
| 1 | 10 | 62.00% / 97.00% | 72.00% / 98.67% |
| **2** | **23** | 65.00% / 99.00% | **75.33% / 100.00%** |
| 3 | 23 | **66.00% / 99.00%** | 74.00% / 99.33% |
| 4 | 27 | 65.00% / 98.00% | 73.33% / 99.33% |
| 5：Final | 27 | 65.00% / 98.00% | 70.67% / 98.67% |

Discovery100 从 Epoch 0 到最终仅提高1.0 pp，且过程不单调；Dev150 在 Epoch 2 达到75.33%后连续下降，最终低于初始2.0 pp。由于协议明确禁止用 Dev150 选择 checkpoint，正式输出仍为 Epoch 5。该轨迹构成后期局部 Refine 可能过度适配 Discovery100 的内部预警，但不能单独证明外部分布也同步退化；16.6 节进一步在 VL-RewardBench 上比较中间 checkpoint。

### 16.3 Heldout-500 探索性回归

RLHF-V heldout-500 仅用于与既有实验做 exploratory regression，不参与演化或 checkpoint 选择。Discovery-v2 与该 heldout 存在少量已知重叠（Discovery100 为2条、Dev150 为6条），因此本节不是新的无偏测试。

| 系统 | Strict ACC | 正确数 | Coverage | Covered ACC |
| --- | ---: | ---: | ---: | ---: |
| Initial five-root Prompt v2 | 75.80% | 379 / 500 | 98.80% | 76.72% |
| Phase10 + Prompt v2 | 76.60% | 383 / 500 | 99.40% | 77.06% |
| Phase16 E5 | **78.00%** | **390 / 500** | **99.80%** | **78.16%** |
| Phase17 Final | 74.00% | 370 / 500 | 99.40% | 74.45% |

Phase17 相对 Initial、Phase10 和 Phase16 E5 分别净减少9、13和20个正确样本；其中相对 Phase16 的 paired corrected/harmed 为11/31，exact McNemar `p=0.0029`。这与 Discovery-v2 不再专门围绕 RLHF-V 视觉幻觉分布构造相符：数据覆盖扩展改善了外部 benchmark 迁移，但牺牲了同域 heldout 表现。

### 16.4 VL-RewardBench 主要外部结果

主要外部测试使用完整 VL-RewardBench 1247对样本、Prompt v2 和 K=3 counterbalanced A/B 顺序。`OverallAcc`是在产生明确最终 A/B 判断的覆盖范围内计算，`MacroAcc`为 General、Hallucination、Reasoning 三类 covered accuracy 的宏平均；`Strict ACC`以全部1247对为分母。

| 系统 | OverallAcc | MacroAcc | Coverage | Strict ACC | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial five-root Prompt v2 | 58.12% | 54.60% | 98.24% | 57.10% | 712 |
| Phase16 E5 | 66.50% | 60.58% | 99.12% | 65.92% | 822 |
| Phase10 + Prompt v2 Control | 69.53% | 63.37% | 98.96% | 68.81% | 858 |
| **Phase17 Discovery-v2 Final** | **69.91%** | **64.06%** | **99.68%** | **69.69%** | **869** |

Phase17 相对 Phase10 Control 的 OverallAcc、MacroAcc 和 Strict ACC 分别提高0.38、0.69和0.88 pp，多正确11条；paired corrected/harmed 为52/41，`p=0.300`，因此应表述为正向 pilot evidence，而不是显著优于 Control。相对 Initial five-root，Phase17 净纠正157条（182 corrected / 25 harmed，`p<10^-29`）；相对 Phase16 E5 净纠正47条（79/32，`p<10^-5`）。

| 类别 | Phase10 Control | Phase17 Final | 变化 |
| --- | ---: | ---: | ---: |
| General | 46.89% | **50.00%** | **+3.11 pp** |
| Hallucination | 75.81% | **76.47%** | +0.66 pp |
| Reasoning | **67.41%** | 65.71% | -1.70 pp |

主要增益来自 General 和 Hallucination，Reasoning 尚未改善。按来源看，POVID 提高约2.97 pp、WildVision 提高约2.67 pp，而 RLHF 子集下降约8.52 pp；因此“分布更宽”已经带来外部总体收益，但100条数据仍不足以稳定覆盖所有推理与偏好子域。

### 16.5 Root 子树与聚合诊断

固定 Phase17 Final Rubric，在 VL-RewardBench 上分别执行每个 root 及其 children，可得到：

| Root subtree | OverallAcc | MacroAcc | Coverage | Strict ACC |
| --- | ---: | ---: | ---: | ---: |
| Completeness | 71.27% | 65.17% | 98.80% | 70.41% |
| Visual Grounding | 63.88% | 59.85% | 97.92% | 62.55% |
| Factuality | 69.95% | 62.77% | 98.48% | 68.89% |
| Creativity | **71.59%** | **66.07%** | **99.36%** | **71.13%** |
| Clarity | 67.88% | 62.95% | 99.12% | 67.28% |

Creativity 和 Completeness 子树单独执行时均高于完整五-root 的 OverallAcc，说明当前等权 root 聚合仍存在错误投票稀释。为理解这一现象，额外进行了不改变任何模型预测的 post-hoc 组合诊断：

| 聚合诊断 | OverallAcc | MacroAcc | Coverage | Strict ACC | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整五 roots | 69.91% | 64.06% | 99.68% | 69.69% | 869 |
| 去掉 Visual Grounding | 72.13% | 65.44% | 92.94% | 67.04% | 836 |
| 去掉 Visual Grounding + Clarity | **71.28%** | **64.85%** | 99.12% | **70.65%** | **881** |

仅去掉 Visual Grounding 时 OverallAcc 虽提高2.22 pp，但四个 roots 容易产生2:2平票，Coverage 降低6.74 pp、总正确数减少33，因此这是选择性弃权造成的表面提升。保留 Completeness、Factuality 和 Creativity 三个 roots 时 Coverage 基本保持，Strict ACC 相对完整五 roots 提高0.96 pp、增加12个正确样本（23 corrected / 11 harmed，`p=0.058`）。不过该组合是在 VL-RewardBench 结果可见后选择的，只能作为“聚合仍有优化空间”的诊断，不能替代 Phase17 完整五-root 主结果。

VL-RewardBench 共执行101,007个逻辑请求；技术重试后仅剩2个 unresolved predictions，最终 Coverage 为99.68%。主要 artifact 位于：

- `output/evolving_structured_rubrics/rubric_evolution_phase5/phase17_discovery_v2_prompt_v2_split_refine_v1/`
- `output/evolving_structured_rubrics/vl_rewardbench_phase17_discovery_v2_prompt_v2_v1/`

### 16.6 Phase17 checkpoint 的 VL-RewardBench 外部迁移诊断

#### 实验动机与设计

Dev150 在 Epoch 2 达到内部最高点，Discovery100 在 Epoch 3 达到最高点，而 Epoch 4 才首次完成全部五个 roots 的 Split。为判断“内部最佳”“结构完成”和“最终 checkpoint”中哪一种更能预测外部迁移，额外冻结并比较以下 Rubric：

| Checkpoint | Nodes | 冻结理由 |
| --- | ---: | --- |
| E2 | 23 | Dev150 最高点，75.33% |
| E3 | 23 | Discovery100 最高点，66.00% |
| E4 | 27 | Completeness 接受 Split，五个 roots 首次全部完成 Split |
| E5 | 27 | 协议规定的正式最终输出，直接复用16.4节结果 |
| Phase10 | 22 | 既有强 Control，直接复用12.2节结果 |

外部评测继续固定为 VL-RewardBench 1,247对样本、Prompt v2、`K=3` counterbalanced A/B 顺序、等权五-root M1。E5 与 Phase10 的逻辑投票完全复用；E2–E4 只为相对 E5 发生 description 变化的18个唯一 criterion 生成缺失预测。所有 checkpoint 在查看本次 benchmark 结果前冻结，结果仅作 exploratory 轨迹诊断，禁止根据 VL-RewardBench 反向改变既有演化接受决定。

#### 总体与配对结果

| Rubric | OverallAcc | MacroAcc | Coverage | Strict ACC | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial five-root Prompt v2 | 58.12% | 54.60% | 98.24% | 57.10% | 712 |
| Phase10 Control | 69.53% | 63.37% | 98.96% | 68.81% | 858 |
| Phase17 E2 | 67.45% | 62.32% | 99.28% | 66.96% | 835 |
| Phase17 E3 | 67.74% | 61.88% | 99.20% | 67.20% | 838 |
| **Phase17 E4** | **70.29%** | **64.42%** | 99.60% | **70.01%** | **873** |
| Phase17 E5 | 69.91% | 64.06% | **99.68%** | 69.69% | 869 |

E2、E3 相对 E5 分别净减少34和31个正确样本，exact McNemar `p=0.00032` 和 `p=0.00019`，说明二者的外部退化不是 Coverage 波动造成的。E4 相对 E5 corrected/harmed 为19/15，净增加4个正确样本，但 `p=0.608`；因此只能认为 E4 与 E5 基本持平、E4 点估计略高，不能宣称显著优于 E5。

| 相对 Phase10 | Corrected | Harmed | Net corrected | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: |
| E2 | 28 | 51 | -23 | 0.0128 |
| E3 | 37 | 57 | -20 | 0.0495 |
| **E4** | **53** | **38** | **+15** | 0.142 |
| E5 | 52 | 41 | +11 | 0.300 |

E4 相对 Phase10 的 OverallAcc、MacroAcc 和 Strict ACC 分别提高0.76、1.05和1.20 pp，是本轨迹观察到的最高完整五-root结果；但配对差异仍未显著，只能作为正向 pilot evidence。

#### 性能跃升来源

E3 到 E4 的主要结构变化是 Completeness 第四次 Split 尝试被接受。该 root subtree 的 VL-RewardBench OverallAcc 从52.04%跃升到70.23%，同时 Factuality 和 Clarity 分别提高0.79和0.95 pp；完整系统正确数因此从838增加到873。

| Root subtree | E2 | E3 | E4 |
| --- | ---: | ---: | ---: |
| Completeness | 52.04% | 52.04% | **70.23%** |
| Visual Grounding | 63.75% | 64.16% | 64.16% |
| Factuality | **70.55%** | 69.51% | 70.30% |
| Creativity | **72.64%** | 71.59% | 71.59% |
| Clarity | 65.18% | 68.12% | **69.07%** |

按官方类别的 covered accuracy，E4 相对 Phase10 在 General 和 Hallucination 上分别提高3.11和0.90 pp，但 Reasoning 下降0.85 pp。E4 到 E5 又接受6次 Refine 后，General 保持50.00%，Hallucination 从76.71%微降至76.47%，Reasoning 从66.56%降至65.71%，最终少4个正确样本。这提示后期 Refine 没有继续转化为外部收益，并出现轻微推理回落；但变化不显著，不能据此概括为稳定的后期过拟合规律。

#### 结论与证据边界

本实验否定了“Dev150 或 Discovery100 的单一内部最高点必然对应最佳外部 checkpoint”：E2、E3 均明显弱于 E4/E5。当前轨迹中，**五个 roots 全部完成 Split 的结构里程碑比内部 M1 峰值更能预测 VL-RewardBench 表现**，其中 Completeness 子树是 E4 跃升的主要来源。不过，这一判断只来自单条演化轨迹，E4 又是在同一 VL-RewardBench 上观察到的最高点，不能据此事后替换协议规定的 E5 正式输出。若将“全部 roots 完成 Split 后才进入 checkpoint 候选”作为未来规则，需要在新轨迹或独立外部数据上预先冻结后验证。

技术重试共处理64个失败请求，恢复63个，仅剩1个 criterion-level prediction 未恢复；E4 系统 Coverage 仍为99.60%。主要 artifact 位于：

- `output/evolving_structured_rubrics/vl_rewardbench_phase17_checkpoint_transfer_v1/`

### 16.7 Qwen2.5-VL Worker 跨模型迁移

#### 实验目的与冻结对照

Phase17 的 Rubric 由 Qwen3-VL-8B-Instruct Worker 演化得到。为判断其中的偏好规则是否只适配原 Worker，本实验保持数据、Rubric、Prompt 和聚合协议不变，仅将 Pairwise Worker 替换为 `Qwen/Qwen2.5-VL-7B-Instruct`。主比较为 **Qwen2.5 Initial five-root vs. Qwen2.5 Phase17 E4**；Native VL-RewardBench Prompt 仅作为辅助基线。

- 测试集固定为 VL-RewardBench 1,247 对样本，Rubric 固定为 Phase17 E4 的27个 nodes；
- Structured Worker 使用 Prompt v2 `1.1.0-cache-pilot`、`temperature=0.5`、`max_tokens=2048`；
- Initial 与 E4 使用相同的 `K=3` counterbalanced A/B schedule，ABA/BAB 数量为624/623；
- 两个 Qwen2.5-VL 服务通过8000与8001 available-slot pool 执行，全局并发40；
- Native 输出按“规则解析 → 397B解析 → 同 Prompt 最多重试10次”的冻结链路恢复；
- Qwen3 对照直接复用相同 benchmark、Prompt v2 和 checkpoint 的既有冻结结果，不根据本次结果重新选择 Rubric。

`OverallAcc` 是覆盖范围内的 ACC；`MacroAcc` 是 General、Hallucination 和 Reasoning 三类 covered accuracy 的宏平均；`Strict ACC` 以全部1,247对样本为分母。

#### 总体与配对结果

| Worker / 方法 | OverallAcc | MacroAcc | Coverage | Strict ACC | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 Native | 41.30% | 44.73% | 100.00% | 41.30% | 515 |
| Qwen2.5 Initial five-root | 43.62% | 44.98% | 94.87% | 41.38% | 516 |
| **Qwen2.5 Phase17 E4** | **57.07%** | **52.95%** | **99.76%** | **56.94%** | **710** |
| Qwen3 Native | 54.52% | 53.56% | 99.28% | 54.13% | 675 |
| Qwen3 Initial five-root | 58.12% | 54.60% | 98.24% | 57.10% | 712 |
| Qwen3 Phase17 E4 | 70.29% | 64.42% | 99.60% | 70.01% | 873 |

Qwen2.5 使用 E4 Rubric 后，OverallAcc、MacroAcc、Coverage 和 Strict ACC 相对同模型 Initial 分别提高 **13.46、7.96、4.89和15.56 pp**，正确数从516增加到710。配对 corrected/harmed 为231/37，净纠正194条，exact McNemar `p=1.86e-35`。因此，E4 的收益并不依赖 Qwen3 Worker：同一组结构化偏好准则可以显著改变另一较弱 VLM 的最终判断。

作为参照，Qwen3 的 Initial→E4 OverallAcc 增益为12.17 pp。两种 Worker 都获得约12–13 pp 的 Rubric 增益，构成明确的跨 Worker 正向证据；Qwen2.5 E4 的 Strict ACC 56.94%也已接近 Qwen3 Initial 的57.10%。但相同 E4 Rubric 下，Qwen2.5 的 OverallAcc 仍比 Qwen3 低13.22 pp（81 corrected / 244 harmed，净−163，`p=3.97e-20`），说明 Rubric 能改善视觉证据的使用和决策优先级，但不能完全补足底层视觉识别与推理能力差距。

#### 类别、子树与聚合诊断

| 类别 | 样本数 | Qwen2.5 Initial Strict ACC | Qwen2.5 E4 Strict ACC | 变化 | 正确数增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| General | 181 | 33.15% | 35.91% | +2.76 pp | +5 |
| Hallucination | 749 | 35.25% | **59.28%** | **+24.03 pp** | **+180** |
| Reasoning | 317 | 60.57% | 63.41% | +2.84 pp | +9 |

194条净新增正确样本中有180条来自 Hallucination，占92.8%。这与 Phase17 的准则内容一致：其最强迁移能力仍集中在视觉真实性、事实验证和证据优先级；General 与 Reasoning 虽有正向变化，但幅度有限。

在加入任何演化 children 之前，Qwen2.5 对五个初始 roots 的单节点判断如下：

| 初始 root 单节点 | Covered ACC | Coverage | Strict ACC |
| --- | ---: | ---: | ---: |
| Completeness | 36.17% | 90.46% | 32.72% |
| Visual Grounding | 42.99% | 93.83% | 40.34% |
| **Factuality** | **63.31%** | 90.06% | **57.02%** |
| Creativity | 33.25% | 90.94% | 30.23% |
| Clarity | 46.08% | **96.07%** | 44.27% |
| 五-root Initial 等权 M1 | 43.62% | 94.87% | 41.38% |

Factuality 是唯一在 Qwen2.5 上具有较强独立判断能力的初始 root；Completeness 和 Creativity 的覆盖率均超过90%，但 Covered ACC 只有36.17%和33.25%，说明问题不是不出手，而是宽泛准则在弱 Worker 上容易产生错误投票。五-root Initial 等权 M1 的 Strict ACC 仅41.38%，也说明多个高覆盖但低准确率 roots 会稀释 Factuality 的正确信号。

| Qwen2.5 E4 Root subtree | OverallAcc | MacroAcc | Coverage | Strict ACC |
| --- | ---: | ---: | ---: | ---: |
| Completeness | 57.10% | 53.19% | 98.88% | 56.46% |
| Visual Grounding | 53.46% | 50.72% | 99.76% | 53.33% |
| Factuality | 57.29% | 52.86% | 98.96% | 56.70% |
| **Creativity** | **62.63%** | **57.87%** | 98.72% | **61.83%** |
| Clarity | 55.20% | 52.14% | 99.52% | 54.93% |

Creativity 原 root 单节点的 Strict ACC 只有30.23%，加入演化 children 后提高到61.83%；Completeness、Visual Grounding 和 Clarity 子树也分别较 root 单节点提高23.74、12.99和10.67 pp，说明跨模型收益主要由具体 children 提供，而不是来自五个宽泛初始 roots。另一方面，最佳 Creativity 子树的 Strict ACC 比五-root 等权 M1 高4.89 pp，表明 root 间错误投票稀释在 Qwen2.5 上同样存在；该比较只作 post-hoc 聚合诊断，不用于替换正式五-root结果。

按单节点的覆盖内 ACC 排序，表现最强的五个 criterion 为：

| 最强单节点 | Covered ACC | Coverage | Strict ACC |
| --- | ---: | ---: | ---: |
| `factual_directness_over_stylistic_flourish` | 69.77% | **95.51%** | **66.64%** |
| `instruction_compliant_visual_grounding` | 68.92% | 84.36% | 58.14% |
| `grounded_coverage_over_hallucinated_volume` | 66.77% | 78.43% | 52.37% |
| `constraint_adherence_over_expressive_detail` | 65.37% | 74.34% | 48.60% |
| `factual_accuracy_over_response_volume` | **70.37%** | 83.64% | 58.86% |

覆盖内 ACC 最低的五个 criterion 为：

| 最弱单节点 | Covered ACC | Coverage | Strict ACC |
| --- | ---: | ---: | ---: |
| `creativity_and_expressiveness`（root） | **33.25%** | 90.94% | 30.23% |
| `completeness_and_coverage`（root） | 36.17% | 90.46% | 32.72% |
| `diagnostic_visual_specificity` | 39.45% | 88.21% | 34.80% |
| `visual_grounding_and_details`（root） | 42.99% | 93.83% | 40.34% |
| `domain_specific_visual_interpretation` | 43.97% | 46.51% | **20.45%** |

最强组全部是演化出的具体 children，而最弱组包含三个宽泛初始 roots，进一步说明跨 Worker 增益来自可操作的局部偏好规则。单节点排名必须结合 Coverage 阅读：例如 `domain_specific_visual_interpretation` 的 covered accuracy 并非最低，但只覆盖46.51%的样本，因此 Strict ACC 最低；相反，`factual_directness_over_stylistic_flourish` 同时保持69.77%的覆盖内准确率和95.51%的覆盖率，是本次迁移中最稳定的单节点。

#### 技术可靠性与证据边界

Structured Worker 共执行101,007个逻辑节点请求，初始无效2,562个（2.54%）；技术重试恢复2,298个，最终仍有264个 node-level 输出无效，仅占全部请求0.261%。这些局部失败大多被其他节点和 `K=3` 聚合吸收，E4 最终只有3个样本未覆盖。Native 初始64个规则解析失败全部通过397B解析或同 Prompt 重试恢复，最终 Coverage 为100%。

Qwen2.5 仍表现出明显位置敏感性：3,741次展示级预测中67.9%选择A、31.6%选择B；当 gold 显示为A/B时准确率分别为73.53%和37.33%，三个 replicate 的原始方向完全一致率为52.12%。由于 Initial 与 E4 使用同一平衡 schedule，主比较的13.46 pp增益不能简单归因于位置分配；但该偏置会增加单样本方差，限制绝对分数的稳定性。

本实验只验证了一个额外 Worker、一个冻结 Rubric 和同一个已多次使用的 VL-RewardBench，因此应表述为 exploratory cross-worker transfer evidence，不能直接推广为完全 model-agnostic。主要 artifact 位于：

- `output/evolving_structured_rubrics/vl_rewardbench_qwen25_phase17_e4_transfer_v1/`

### 16.8 Qwen2.5 从 Initial 开始的模型专属演化（Phase18）

#### 实验目的与设计

16.7只验证了“Qwen3 演化出的 Rubric 能否直接迁移给 Qwen2.5”。Phase18 进一步固定 Manager、数据、Prompt v2、Split/Refine、Global Memory 和等权五-root M1协议，仅将 Pairwise Worker 全程替换为 `Qwen/Qwen2.5-VL-7B-Instruct`，从相同的五个 Initial roots 重新生成 ErrorSignature、Split children 和 Refine description。实验检验：**较弱 Worker 能否从自己的错误经验中演化出有效 Rubric，以及模型专属演化是否优于直接复用 Qwen3 演化结果。**

- 演化数据固定为 Discovery100；Dev150 每个 epoch 只做独立诊断，不对 Manager 可见，不参与算子接受、早停或 checkpoint 选择；
- Pairwise Worker 使用 Prompt v2、`max_tokens=2048`，8000与8001 available-slot pool；Manager 仍为 `Qwen/Qwen3.5-397B-A17B`；
- Split 阈值、Locked-Child retry、Role-aware Refine、同步 epoch commit 和3–5轮协议与 Phase17 保持一致；
- 最终 Rubric 由5个 roots 扩展到25个 nodes：E1接受 Visual Grounding Split，E2接受其余4个 root Split，E3–E5分别接受6、5和4次 Refine；
- RLHF-V heldout-500 与 VL-RewardBench 均为 exploratory 外部诊断，不反向改变 E5 正式输出。

主要 artifact 位于：

- `output/evolving_structured_rubrics/rubric_evolution_phase5/phase18_qwen25_discovery_v2_prompt_v2_split_refine_v1/`
- `output/evolving_structured_rubrics/vl_rewardbench_qwen25_phase18_evolved_v1/`

#### Discovery100 与 Dev150 演化轨迹

下表中的 ACC 以各数据集全部样本为分母，Tie/abstain 不计为正确；Covered ACC 另按有效 A/B 支持集计算。Dev150 始终为 `diagnostic_only`、`selection_forbidden=true`。

| Epoch | Nodes | 本轮接受操作 | Discovery ACC / Coverage | Dev ACC / Coverage |
| --- | ---: | --- | ---: | ---: |
| E0 | 5 | Initial | 58.00% / 95.00% | **72.67%** / 96.67% |
| E1 | 10 | Split ×1 | 57.00% / 93.00% | 72.00% / 96.00% |
| **E2** | 25 | Split ×4 | **63.00% / 100.00%** | 64.67% / 98.67% |
| E3 | 25 | Refine ×6 | 62.00% / 99.00% | 68.00% / 98.67% |
| **E4** | 25 | Refine ×5 | 58.00% / 96.00% | **69.33% / 98.67%** |
| E5 | 25 | Refine ×4 | 62.00% / 99.00% | 68.67% / 98.00% |

E2 是 Discovery100 最高点，但其 Dev150 ACC 只有64.67%，相对 E0 下降8.00 pp。E2→E4 期间，Discovery ACC 从63.00%下降到58.00%，而 Dev ACC 从64.67%恢复到69.33%；对应的 Dev covered ACC 从65.54%提高到70.27%。因此后续 Refine 确实修复了部分跨样本行为，但 E4 仍比 E0 Dev 低3.33 pp，尚不能称为改善了 Initial 的 Dev 泛化。

这一轨迹也暴露出算子局部 Fitness 与完整 M1 的错位：Split/Refine 按 root Specialized ACC 或 node self-competition 接受，并不要求完整五-root M1同步提升，所以局部准则改善可能通过多数投票产生全局退化。Discovery 最优 E2、Dev 最优 E0以及 E2–E4 中 Dev 最优 E4并不一致，说明不能仅依据单一内部分数推断外部 checkpoint。

#### RLHF-V heldout-500

| Qwen2.5 系统 | Strict ACC | Coverage | Covered ACC | 正确数 |
| --- | ---: | ---: | ---: | ---: |
| Initial five-root | 60.80% | 93.80% | 64.82% | 304 / 500 |
| Phase17 E4 直接迁移 | 64.40% | **99.60%** | 64.66% | 322 / 500 |
| **Phase18 E5 模型专属演化** | **66.40%** | 98.80% | **67.21%** | **332 / 500** |

Phase18 E5 相对 Qwen2.5 Initial 净增加28个正确样本（53 corrected / 25 harmed，exact McNemar `p=0.0020`），相对直接迁移的 Phase17 E4 净增加10个（30/20，`p=0.203`）。这说明模型专属演化在同域 heldout 上有明确正向结果，但相对 transferred E4 的优势尚不显著。

#### VL-RewardBench 外部结果

评测继续固定为1,247对样本、Prompt v2、`K=3` counterbalanced A/B schedule 和等权五-root M1；`OverallAcc` 与 `MacroAcc` 均按覆盖内结果计算，`Strict ACC` 以全部样本为分母。

| Qwen2.5 Rubric | OverallAcc | MacroAcc | Coverage | Strict ACC | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial five-root | 43.62% | 44.98% | 94.87% | 41.38% | 516 |
| **Phase17 E4 直接迁移** | **57.07%** | 52.95% | **99.76%** | **56.94%** | **710** |
| Phase18 E5 模型专属演化 | 56.06% | **53.07%** | 98.56% | 55.25% | 689 |

Phase18 E5 相对 Initial 的 OverallAcc、MacroAcc、Coverage 和 Strict ACC 分别提高12.44、8.09、3.69和13.87 pp，corrected/harmed 为209/36，净纠正173条，exact McNemar `p=7.84e-31`。因此，Qwen2.5 能够从自己的 Discovery100 错误经验中完成有效演化，收益并不要求先由 Qwen3 生成准则。

但模型专属 E5 并未超过 Phase17 E4 的直接迁移：OverallAcc 低1.01 pp、Strict ACC 低1.68 pp、少21个正确样本；paired corrected/harmed 为50/71，`p=0.0686`。Phase18 E5 的 MacroAcc 略高0.13 pp，类别上 Reasoning 更高（64.67% vs. 63.41%），但 Hallucination 更低（55.81% vs. 59.28%）。这说明专属演化学习到了一些更适配 Qwen2.5 的推理规则，却没有复制 Phase17 E4 在幻觉类数据上的强优势。

#### Checkpoint 分化与待完成诊断

Phase18 的内部排序存在明显分歧：E2 是 Discovery100 最佳 checkpoint，E4 是 E2–E4 中 Dev150 最佳 checkpoint，而 E5 是协议规定的最终输出。为判断 Discovery、Dev 或演化结构里程碑中哪个更能预测外部迁移，后续应在完全相同的 Qwen2.5 VL-RewardBench 协议下补测 E2、E3、E4；E5、Initial 和 Phase17 E4 直接复用现有结果。该 checkpoint 比较属于预先记录的 exploratory 轨迹诊断，不能根据结果反向修改已经完成的 Phase18 算子接受记录。

**Phase18 结论。** Qwen2.5 从 Initial 开始独立演化后，在 Discovery100、RLHF-V heldout-500 和 VL-RewardBench 上均明显优于同模型 Initial，证明 Split+Refine 链路并非只对 Qwen3 Worker 有效。不过，模型专属 E5 在 VL-RewardBench 上仍略弱于直接迁移的 Phase17 E4，且 Discovery/Dev checkpoint 排名不一致。当前证据更支持“结构化 Rubric 可以跨模型迁移，也可以由目标模型自行演化”，尚不支持“目标模型专属演化必然优于强模型产生的可迁移 Rubric”。

**阶段结论。** Discovery-v2 将少量偏好经验从单一视觉幻觉分布扩展到视觉、推理和通用偏好后，Phase17 正式 E5 在 VL-RewardBench 上达到 OverallAcc 69.91%、MacroAcc 64.06%、Strict ACC 69.69%；探索性 checkpoint 诊断进一步发现，五个 roots 首次全部完成 Split 的 E4 达到当前 Qwen3 Worker 最高点70.29% / 64.42% / 70.01%。固定 E4 Rubric 迁移到 Qwen2.5-VL-7B 后，相对其 Initial five-root 提高13.46 pp OverallAcc；Qwen2.5 从 Initial 独立演化的 Phase18 E5 也提高12.44 pp，但外部结果略低于直接迁移的 E4。整体上，结构化 Rubric 编码的偏好判断模式具有明确的跨 Worker 可复用性和弱 Worker 自主演化能力；General/Reasoning 数据覆盖、位置稳定性、局部 Fitness 与全局聚合的一致性仍是主要限制。

---

## 17. 聚合机制探索：从节点投票到子树分析与最终决策

### 17.1 问题与统一评测协议

Phase17 的 root 子树诊断表明，多个局部专家同时参与时，正确判断仍可能被其他 root 的错误票稀释。为检验“结构化 Rubric 的价值是否应通过简单多数票实现”，本阶段固定 **Phase17 E4 Rubric** 和 `Qwen3-VL-8B-Instruct`，不再演化节点或 description，只改变推理单元、Prompt 中的 Rubric 组织方式和最终聚合机制。评测统一使用 VL-RewardBench 的1,247个偏好对、`temperature=0.5`、`max_tokens=2048` 和冻结的 `K=3` counterbalanced A/B schedule；VL-RewardBench 不用于选择聚合变体。

保留的关键系统如下：

| 系统 | 每个 replicate 的推理与聚合方式 |
| --- | --- |
| **S0 Explicit Recursive** | 27个节点分别判断，再按既有 parent/children 与五-root 等权规则递归聚合 |
| **S3 Unified Subtree** | 每个 root 及其 children 一次性输入模型，视为一条统一决策策略；五个子树结果等权聚合 |
| **S4 Global Arbiter** | 在 S3 的五份同-replicate子树报告上增加一次全局仲裁；Prompt 明确允许 `None` |
| **Clean S5-v2** | 输入与 S4 相同，但 Prompt 要求尽量给出 A/B；parser 仍把原生 `None` 记为合法弃权，仅技术失败使用同一 Prompt 重试，不使用 tie-break/rescue Prompt |
| **S6 Unified Full-Rubric** | 把五个 root、全部 children 和层级关系一次性交给单个 Worker，由模型隐式选择相关准则、解决冲突并直接输出偏好 |

Clean S5-v2 复用已冻结的18,705份 S3 子树报告，只重新生成3,741次 Arbiter 判断。它使用全新 protocol、输出目录和 cache namespace，确保不会复用旧 Arbiter 输出；旧 S5 的100% Coverage结果仅作为只读协议参照。

### 17.2 端到端推理流程

Clean S5-v2 由两个串联模块组成：Unified-Subtree Worker 先将五棵 Rubric 子树分别转化为结构化证据报告，Global Arbiter 再综合五份报告得出一次偏好。它不会先把五棵树投成一个多数结果再交给 Arbiter。对于 `K=3`，整个“子树分析 → 全局仲裁”流程独立执行三次，最后只聚合三次 Arbiter 结论：

```text
图像 + 问题 + Candidate A/B
              │
              └─ 对 r ∈ {1, 2, 3} 独立执行（使用 replicate r 的冻结 A/B 顺序）：
                    ├─ Completeness root + children     → 子树报告 R1
                    ├─ Visual Grounding root + children → 子树报告 R2
                    ├─ Factuality root + children       → 子树报告 R3
                    ├─ Creativity root + children       → 子树报告 R4
                    └─ Clarity root + children          → 子树报告 R5
                                │
                                ▼
                      Global Arbiter 综合 R1–R5
                                │
                                ▼
                      replicate 结论 y_r ∈ {A, B, None}
                              │
                              └─ {y_1, y_2, y_3} 按 K=3 多数聚合 → 最终偏好
```

这里的 `R1–R5` 是完整分析报告，不只是五个 A/B/None 标签。因而 Global Arbiter 可以检查不同子树的事实依据、识别不适用或错误报告，并在准则冲突时重新判断证据强弱。三次结论会先映射回原始 Candidate 顺序；只有至少两次一致选择同一个 Candidate 时才输出最终 A/B，否则最终结果为 `None`。因此 A/B/None 各一票，或只有一次 A/B 判断而另外两次均为 `None` 时，都不会被强行排序。

### 17.3 Unified-Subtree Worker

#### 17.3.1 作用与调用粒度

传统 S0 对 Rubric 中的27个节点分别调用 Pairwise Worker，再通过递归多数投票得到结果。Unified-Subtree Worker 改变的是**推理单元**：一次调用接收一个 root 及其全部 children，把整棵子树作为统一决策策略进行一次综合判断。children 是针对不同情形的专业判断指导，不会各自产生一张独立选票，也不需要额外的 Gate 或 Router。

对一个样本的一个 replicate，系统分别调用五次 Unified-Subtree Worker，得到五份子树报告；`K=3` 时，每个样本共生成15份子树报告。

| 项目 | Unified-Subtree Worker 的内容 |
| --- | --- |
| 多模态输入 | 图像、原始问题或指令、Candidate A、Candidate B |
| Rubric 输入 | 一个 root 的 name/description，以及该 root 下按冻结顺序排列的全部 children name/description |
| 不可见信息 | 其他 root、其他子树的判断、gold、最终多数结果 |
| 推理方式 | 将 root 的总体目标与 children 的专业边界合并为一条决策策略 |
| 输出 | 一份包含 A/B 分析、综合比较和 A/B/None 结论的子树报告 |

#### 17.3.2 System Prompt

System Prompt 固定 Unified-Subtree Worker 的角色和推理规则，核心模板如下：

````text
UNIFIED_SUBTREE_SYSTEM_PROMPT = """## Instruction

You are judging a multimodal image-text preference pair under one structured rubric subtree. You are given the image, the source instruction or question, two candidate responses, and the rubric subtree.

The root defines the broad criterion, while its children provide specialized guidance for particular cases. Treat the entire subtree as one unified decision policy, not as a set of independent votes.

Use the image when the rubric depends on visual evidence. If the rubric subtree as a whole is not applicable to this pair, answer None.

Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A based on the given criterion.",
    "analysis_b": "Analyze B based on the given criterion.",
    "thought": "Compare A and B.",
    "answer": "A / B / None"
}
```

Return None if any of the following conditions are met:
- The rubric subtree as a whole is not applicable to this pair of data pieces.
- They are of the same quality.
- You are unsure.
"""
````

这段 Prompt 的关键约束不是“在 children 中选择一个节点”，而是要求模型同时理解 root 与 children。root 确定总体评价目标，children 补充更细粒度的适用情形和决策边界，模型最终只生成一份整棵子树的判断。

#### 17.3.3 User Prompt 与输出

图像通过多模态消息传入；User Prompt 保存每个样本和每棵子树的动态内容，顺序如下：

```text
## Question
<question>

## Candidate A
<candidate_a>

## Candidate B
<candidate_b>

## Structured Rubric Subtree

### Root Criterion
**<root_name>**: <root_description>

### Specialized Child Criteria
1. **<child_1_name>**
<child_1_description>
...

Which candidate better follows this structured rubric subtree and is
more likely to align with human preference?
```

Worker 返回的完整报告为：

```json
{
  "analysis_a": "Analyze Candidate A under the entire subtree.",
  "analysis_b": "Analyze Candidate B under the entire subtree.",
  "thought": "Compare A and B by integrating the root and children.",
  "answer": "A / B / None"
}
```

四个字段都会被保存并传给 Global Arbiter。`None` 是合法语义弃权，表示整棵子树不适用、两者质量相同或证据不足；它不是解析失败。

### 17.4 Global Arbiter

#### 17.4.1 输入与决策规则

Global Arbiter 在同一个 replicate 内读取原始图像、问题、A/B回答，以及五份 Unified-Subtree 报告。它把报告视为**相关证据**而不是五张独立选票：不能直接统计五个 `answer` 标签，而要重新核对原始多模态证据，并允许判断某份报告错误或不适用。以下 Prompt 描述专指最终采用的 Clean S5-v2；S4 使用相同输入，但其 Prompt 明确允许 `None`，因此两者的 Coverage 不可混为一谈。

其 System Prompt 固定以下证据优先级：

1. 可验证的视觉与事实正确性；
2. 在事实成立之后比较完整性；
3. 清晰度与创造性只用于区分其他方面可接受的回答，不能补偿事实错误。

Prompt 明确要求为每个 pair 返回相对偏好且不要弃权；即使两个回答都不完美，也应选择证据更强、错误更轻的 A 或 B。

````text
GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT = """## Instruction

You are the final decision arbiter for one structured multimodal preference system. You are given the image, the source instruction or question, two candidate responses, and five subtree assessments produced under complementary rubric dimensions.

Treat the subtree assessments as correlated evidence, not independent votes. Do not decide by counting their A/B labels. Independently verify the image, question, and candidate responses. A subtree assessment may be incorrect or inapplicable.

Prioritize verifiable visual and factual correctness. Consider completeness after factual validity; clarity and creativity may distinguish otherwise acceptable responses but cannot compensate for factual errors.

Return a relative preference for every pair. If both responses are imperfect or the evidence is limited, select the response with stronger support and the less severe error. Do not abstain.

Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A using the image and subtree evidence.",
    "analysis_b": "Analyze B using the image and subtree evidence.",
    "thought": "Integrate the evidence into one relative preference.",
    "answer": "A / B"
}
```
"""
````

#### 17.4.2 User Prompt、输出与重试

Arbiter 的 User Prompt 把原始样本放在前面、五份报告放在后面，保持稳定布局：

```text
## Source Instruction or Question
<question>

## Candidate A
<candidate_a>

## Candidate B
<candidate_b>

## Subtree Assessments
### <root_1_name>
{"analysis_a": ..., "analysis_b": ..., "thought": ..., "answer": ...}
...
### <root_5_name>
{"analysis_a": ..., "analysis_b": ..., "thought": ..., "answer": ...}

Integrate the image evidence and all subtree assessments into one final preference.
```

Arbiter 使用与子树报告相同的四字段 JSON 输出模板，但 parser 只强制校验 `answer`。虽然 Prompt 明令输出 A/B、不要弃权，模型原生输出的 `None` 仍被记录为合法语义弃权。只有 transport、空响应、非法 JSON 或非法标签属于技术失败；这些请求使用完全相同的 Prompt 重试，不存在第二套 tie-break 或 rescue Prompt。固定 System Prompt、将样本和报告放入 User Prompt，也使跨请求共享前缀可以被 vLLM 缓存。

### 17.5 VL-RewardBench 主要结果

| 系统 | Strict ACC | OverallAcc | MacroAcc | Coverage | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 Explicit Recursive | 70.01% | 70.29% | 64.42% | 99.60% | 873 |
| S3 Unified Subtree | 66.72% | 68.42% | 62.97% | 97.51% | 832 |
| S4 Global Arbiter（允许 `None`） | 67.44% | **71.64%** | 66.11% | 94.15% | 841 |
| **Clean S5-v2（A/B-preferred，原生 `None`）** | **71.13%** | 71.59% | **66.24%** | **99.36%** | **887** |
| S6 Unified Full-Rubric | 62.07% | 62.07% | 59.51% | 100.00% | 774 |

S3 比 S0 低3.29 pp Strict ACC，说明“把每棵树隐式压缩为一次判断”本身不能解决聚合问题。S4 的覆盖内准确率达到71.64%，但73个最终弃权使 Strict ACC 只有67.44%；它表现出较好的选择性，却不适合作为需要为每个 pair 给出排序的主系统。Clean S5-v2 将 Coverage 恢复到99.36%，相对 S3 corrected/harmed 为74/19，净增加55条（exact McNemar `p=7.72e-9`）；相对 S4 为54/8，净增加46条（`p=1.71e-9`）。相对 S0 虽净增加14条、Strict ACC 提高1.12 pp，但 corrected/harmed 为62/48，`p=0.215`，因此当前只能报告更高的点估计，不能声称显著优于 S0。S6 将整个 Rubric 压入一次调用后的显著退化及其原因在17.9节单独分析。

### 17.6 为什么 Global Arbiter 有效

Global Arbiter 接收图像、问题、A/B回答和五份同-replicate子树报告，将这些报告视为相关证据而非五张独立选票。逐 replicate 审计 Arbiter 是否推翻五个子树的多数意见，结果如下：


| 子树判断格局 | Replicate 数 | 推翻次数 | 推翻后修正 | 推翻后损害 | 中性转移 | 净修正 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5–0 一致 | 2,115 | 0 | 0 | 0 | 0 | 0 |
| 4–1 强多数 | 817 | 71 | 50 | 17 | 4 | +33 |
| 3–2 弱多数 | 579 | 160 | 113 | 43 | 4 | +70 |
| 平局 / 稀疏 / 大量 `None` | 230 | 105 | 58 | 9 | 38 | +49 |
| **合计** | **3,741** | **336** | **221** | **69** | **46** | **+152** |

**“推翻”**表示 Global Arbiter 的结论和上述五棵子树的多数/聚合结论不同。比如说五个子树多数投票是 A (A / A / A / B / B)，但是Global Arbiter结果是 B，这就是“推翻”，如果 gold 是 B，就是**“推翻后修正”**

五个子树完全一致时，Arbiter 从不推翻；随着子树分歧增加，观测到的推翻比例也随之提高。336次推翻中有221次修正、69次损害，另有46次没有改变 Strict correctness，主要对应错误答案与 `None` 之间的转移。4–1强多数、3–2弱多数以及平局/稀疏状态分别净修正33、70和49次。这一模式与“**重新审查冲突证据**”机制一致，说明 Arbiter 并非简单复刻子树多数，但该审计本身不能完全排除模型间接利用多数信号。

不同 VL-RewardBench 类别上的结果进一步表明，这一收益并非完全均匀。下表以 S0 Explicit Recursive 为基线；Strict ACC 的分母是该类别全部样本，`None` 计为未答对，`OverallAcc` 则只在 Coverage 内计算：

| 类别 | 样本数 | S0 Strict ACC | Clean S5-v2 Strict ACC | Strict 变化 | Clean S5-v2 Coverage | Clean S5-v2 OverallAcc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| General | 181 | 49.17% | **54.14%** | +4.97 pp | 98.34% | 55.06% |
| Hallucination | 749 | 76.50% | **77.84%** | +1.34 pp | 99.73% | 78.05% |
| Reasoning | 317 | **66.56%** | 64.98% | -1.58 pp | 99.05% | 65.61% |

按 Strict ACC 的百分点增幅看，Global Arbiter 在 General 上提升最大，Hallucination 也有小幅改善；Reasoning 则下降1.58 pp。因而总体提升不能解释为所有类别都同步受益，当前聚合策略在复杂推理证据的取舍上仍有改进空间。

在 S4 的73个最终 `None` 样本上，Clean S5-v2 给出39个正确、27个错误并保留7个 `None`：该子集 Strict ACC 为53.42%（39/73），覆盖内准确率为59.09%（39/66）。因此 A/B-preferred Prompt 没有提高全局 OverallAcc，而是把略高于随机的可判别信息转化为正式排序：S4 与 Clean S5-v2 的 OverallAcc 几乎相同（71.64% vs. 71.59%），但 Strict ACC 提高3.69 pp。

### 17.7 效率与结论边界

下表统一按 VL-RewardBench 的1,247个样本和 `K=3` 计算逻辑请求量，不包含技术重试：

| 系统 | 每个 replicate 的模型调用 | 每样本 `K=3` 调用数 | 完整评测请求量 | 相对 S0 减少 | 主 run 分段时间 |
| --- | --- | ---: | ---: | ---: | ---: |
| S0 Explicit Recursive | 27个 node Worker | 81 | 101,007 | — | 约435.6分钟（7.26小时）* |
| S3 Unified Subtree | 5个 Unified-Subtree Worker | 15 | 18,705 | 81.48% | 136.6分钟（2.28小时） |
| S4 Global Arbiter | 5个 Unified-Subtree Worker + 1个 Global Arbiter | 18 | 22,446 | 77.78% | 约170.7分钟（2.84小时）** |
| Clean S5-v2 | 5个 Unified-Subtree Worker + 1个 Global Arbiter | 18 | 22,446 | 77.78% | 约171.4分钟（2.86小时）** |

S3 作为对照系统时，五个子树 `None` 不计票，A/B 中唯一多数获胜；A/B 平票或没有决定性子树则输出 `None`。S4 与 Clean S5-v2 不使用这一步子树多数，直接把五份报告交给 Arbiter。

S4 与 Clean S5-v2 的主 run 时间由实际分段运行求和得到：先运行 S3 生成18,705份子树报告，再运行3,741次 Arbiter。各阶段的具体计时如下，均不包含 freeze、report、最终缓存扫描和单独 retry stage：

| 实测阶段 | 模型请求 | 主运行耗时 | 推理吞吐 | 说明 |
| --- | ---: | ---: | ---: | --- |
| S0 Explicit Recursive 历史计时 | 101,007 | 26,134.4秒（435.57分钟） | 231.89次/分钟 | 相同27-node、`K=3`协议的 Phase17 计时参考* |
| S3 Unified-Subtree Worker | 18,705 | 8,196.7秒（136.61分钟） | 136.92次/分钟 | 完整生成五子树报告 |
| S4 Global Arbiter 增量阶段 | 3,741 | 2,043.7秒（34.06分钟） | 109.83次/分钟 | 复用 S3 子树报告 |
| Clean S5-v2 Global Arbiter 增量阶段 | 3,741 | 2,085.0秒（34.75分钟） | 107.66次/分钟 | 复用 S3；初次7次失败由后续同 Prompt retry 恢复，retry 耗时未计入 |

\* S0 的来源报告标记 `fresh_timing_valid=false`，因为运行包含中断恢复和部分缓存复用；435.6分钟只能作为历史量级参考，不能视为严格受控计时。

\** S4/S5-v2 没有单独执行一次无缓存端到端 run，170.7/171.4分钟分别由 S3 主 run 加对应 Arbiter 主 run 得到，且未计入单独 retry stage，因而只能视为主 run 的近似下界。以此分段计时对比 S0 历史参考，Clean S5-v2 耗时约缩短60.7%，即约2.54×加速、耗时约为参考值的39.3%；该比较不是严格受控测速。请求量减少77.78%由固定协议直接计算，不受运行中断或缓存状态影响。

补充的 K=1、单顺序内部诊断没有显示一致收益：Discovery100 上 Arbiter 与显式聚合同为65.0% Strict ACC，Dev150 从73.33%降至62.67%，RLHF-V heldout-500 为73.40%，略低于可用的 Phase17 E5 显式参考74.00%（该 heldout 对照并非相同 E4 Rubric，只能作诊断）。因此现有证据支持的是：**在 VL-RewardBench 的冻结 K=3 协议下，Global Arbiter 最终决策能以更少请求达到比显式递归投票更高的点估计；其 Strict ACC 显著优于统一子树多数和高弃权 Arbiter，但 OverallAcc 与高弃权 Arbiter基本持平。**

### 17.8 Clean S5-v2 收益来源：完整报告与 factuality-first 顺序消融

#### 17.8.1 研究问题与实验设计

Clean S5-v2 同时引入了两个可能带来收益的因素：Global Arbiter 能读取五份完整子树报告，而其 System Prompt 还显式规定了 factuality-first 决策优先级。为区分二者的作用，本实验固定 Phase17 E4 Rubric、1,247条 VL-RewardBench 样本、Qwen3-VL-8B-Instruct、`K=3` counterbalanced A/B schedule、`temperature=0.5`、`max_tokens=2048`、parser、`None` 语义和评估脚本，只构造以下顺序消融：

实验计划见 `refine-logs/GLOBAL_ARBITER_EVIDENCE_PRIORITY_ABLATION_PLAN.md`，机器可读结果与报告位于 `output/evolving_structured_rubrics/vl_rewardbench_global_arbiter_evidence_priority_ablation_v1/`。

| 变体 | Arbiter 接收的子树证据 | Arbiter 决策规则 | 是否新增推理 |
| --- | --- | --- | ---: |
| V0 `label_only_neutral` | 五个 root 名称及各自 `A/B/None` 标签 | Neutral | 是 |
| V1 `full_report_neutral` | 五份完整 `analysis_a/analysis_b/thought/answer` 报告 | Neutral | 是 |
| V2 `full_report_factuality_first` | 与 V1 相同的五份完整报告 | Clean S5-v2 factuality-first | 否，严格复用 |

V0 与 V1 使用逐字相同的 neutral System Prompt；唯一差异是 User Prompt 中每棵树只保留 `answer`，还是保留完整报告。V1 的 System Prompt 则由 V2 正式 Prompt 精确删除以下一段得到，其余文字和 JSON schema 不变：

```text
Prioritize verifiable visual and factual correctness. Consider completeness
after factual validity; clarity and creativity may distinguish otherwise
acceptable responses but cannot compensate for factual errors.
```

实验复用 S3 已冻结的18,705份子树报告。V0、V1各新增 `1,247 × 3 = 3,741` 次 Arbiter 请求；V2 严格复用现有3,741次正式结果，不重新生成。评估 gold 始终读取数据集的 `preferred_original_index`。该设计只支持两个有顺序条件的比较：V1−V0 衡量 neutral 条件下完整报告的收益，V2−V1 衡量 full-report 条件下显式 factuality-first 的额外收益。由于没有 `label_only_factuality_first`，不能把两个差值解释为彼此完全独立的因果贡献。

#### 17.8.2 总体与配对结果

| 系统 | Strict ACC | OverallAcc | MacroAcc | Coverage | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0 Label-only Neutral | 65.76% | 65.76% | 60.86% | **100.00%** | 820 |
| **V1 Full-report Neutral** | **71.13%** | 71.42% | 66.17% | 99.60% | **887** |
| V2 Full-report Factuality-first | **71.13%** | **71.59%** | **66.24%** | 99.36% | **887** |

| 条件比较 | Strict 变化 | Corrected / Harmed |
| --- | ---: | ---: |
| V1−V0：Neutral 下完整报告 vs. 标签 | **+5.37 pp** | 104 / 37 |
| V2−V1：Full-report 下 factuality-first vs. Neutral | **0.00 pp** | 19 / 19 |

完整报告使 Strict ACC 从65.76%提高到71.13%，净增加67个正确样本；置信区间完全大于零，且 Coverage 只下降0.40 pp，因此增益不是依靠大量弃权获得的。相反，factuality-first 虽然改变了43条最终判断，却恰好修正19条、损害19条，Strict 正确数保持887不变。V2 的 OverallAcc 比 V1 高0.17 pp，只是因为它少覆盖3条样本；这不是全样本准确率收益，也再次说明本阶段应以 `None` 计错的 Strict ACC 为主指标。

#### 17.8.3 类别、来源与位置稳定性

| 类别 | 样本数 | V0 Strict | V1 Strict | V1−V0 | V2 Strict | V2−V1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| General | 181 | 45.30% | **55.80%** | **+10.50 pp** | 54.14% | -1.66 pp |
| Hallucination | 749 | 70.09% | **77.84%** | **+7.74 pp** | 77.84% | 0.00 pp |
| Reasoning | 317 | **67.19%** | 64.04% | **-3.15 pp** | 64.98% | +0.95 pp |

V1 相对 V0 在 General 上 corrected/harmed 为22/3，在 Hallucination 上为73/15，但在 Reasoning 上为9/19。来源级趋势一致：PoVID、RLHF-V、WildVision 分别提高8.71、10.29和10.53 pp，Reasoning Tasks 则下降3.15 pp。完整报告明显增强了视觉事实、幻觉与开放偏好判断，但并非对所有任务无条件有益；在复杂推理中，较长且相关的子树报告可能引入错误锚定，或让视觉事实证据压过任务本身的推理结构。

完整报告还显著降低了 A/B 顺序敏感性：

| 系统 | Gold 显示为 A 时 ACC | Gold 显示为 B 时 ACC | 位置差距 |
| --- | ---: | ---: | ---: |
| V0 Label-only Neutral | 59.34% | 69.26% | **9.92 pp** |
| V1 Full-report Neutral | 69.32% | 70.33% | **1.01 pp** |
| V2 Full-report Factuality-first | 69.00% | 70.65% | 1.65 pp |

该结果说明，只有标签时 Arbiter 更容易依赖位置或表层模式，而完整报告提供的具体证据能稳定其跨顺序决策。

#### 17.8.4 效率、技术可靠性与结论

| 系统 | Arbiter 逻辑请求 | Input tokens | Output tokens | 完整耗时 | 吞吐 | P50 / P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 Label-only Neutral | 3,741 | 4.42M | 1.12M | 18.85分钟 | 198.46次/分钟 | 7.88s / 17.19s |
| V1 Full-report Neutral | 3,741 | 9.81M | 1.29M | 36.19分钟 | 103.38次/分钟 | 16.69s / 27.36s |
| V2 Full-report Factuality-first | 3,741 | 9.94M | 1.30M | 34.75分钟* | 107.66次/分钟* | 历史冻结参考 |

V1 相对 V0 的输入 token 增加约122%、耗时增加约92%、吞吐下降约47.9%，说明 +5.37 pp 收益包含更多结构化证据和更多 test-time compute，不能被表述为纯提示语义贡献。V1 与 V2 的输入量每请求只相差约33 tokens，基本对应被删除的 factuality-first 段落，支持二者是近似干净的 Prompt 规则比较。V0 首轮8个失败样本、V1首轮5个失败样本均通过相同 Prompt 的定向重试恢复，最终 parse valid rate 均为100%；修复后的计时累计 smoke、initial full 和真实失败重试，不包含全量缓存扫描。

\* V2 为历史冻结运行；相同 temperature 下不设置 generation seed，因此 V1/V2 仍存在运行时间不同带来的随机性残余。现有结果支持的结论是：**在 neutral arbitration 下，五份完整子树报告相对五个离散标签带来显著的准确率与位置稳定性收益；在 full-report 条件下，额外的 factuality-first 段落没有观测到总体 Strict ACC 增益。**这并不意味着视觉事实优先原则在系统中不存在作用：子树报告、演化后的 criterion 和 neutral Arbiter 的独立核验要求已经隐式编码了大量事实优先信息。当前收益应主要归因于“保留并综合完整结构化证据”，而不应归因于单独的一段优先级指令。

### 17.9 Unified Full-Rubric Worker：一次调用整合全部准则的消融

#### 17.9.1 实验问题与方法

S0、S3 和 Clean S5-v2 都保留了某种显式分解：分别以 node、root 子树或“子树报告 + Arbiter”为推理单元。本消融进一步检验一个更激进的假设：**结构化 Rubric 是否只需作为一段完整上下文交给模型，而不必显式执行其节点和子树结构。**实验固定 Phase17 E4 的5个 roots、27个 nodes、全部 descriptions 和拓扑，不重新演化 Rubric，只改变推理方式：

```text
图像 + 问题 + Candidate A/B
              │
              ▼
完整 Phase17 E4 Rubric
5个 roots + 22个 children + 层级关系
              │
              ▼
一个 Unified Full-Rubric Worker
隐式选择相关准则并解决准则冲突
              │
              ▼
直接输出 A / B（原生 None 仍按语义弃权统计）
```

该系统记为 **S6 Unified Full-Rubric**。每个 sample/replicate 只调用一次模型，不运行 node Worker、Unified-Subtree Worker、Gate、Root Router、root 多数投票或 Global Arbiter，也不使用 fallback 和第二套 rescue Prompt。System Prompt 的静态部分包含完整 Rubric，并要求把层级结构视为一个统一决策策略；User Prompt 只包含图像、问题和 A/B 回答。核心输出仍与 Pairwise Worker Prompt v2 对齐：

```json
{
  "analysis_a": "Analyze A using the relevant rubric evidence.",
  "analysis_b": "Analyze B using the relevant rubric evidence.",
  "thought": "Compare A and B and resolve any criterion conflicts.",
  "answer": "A / B"
}
```

模型为 `Qwen3-VL-8B-Instruct`，`temperature=0.5`、`max_tokens=2048`，不设置 generation seed，并使用两个 endpoint 的 available-slot pool。Discovery100、Dev150 和 RLHF-V heldout-500 各运行一次（`K=1`、不交换）；VL-RewardBench 严格复用冻结的1,247个样本、`K=3` counterbalanced A/B schedule，每个 replicate 独立生成后再做三次结论的多数聚合。Prompt 要求 A/B，但 parser 对模型原生 `None` 保持语义透明；只有 transport、空响应、非法 JSON、缺失字段或非法标签使用同一 Prompt 最多重试10次。

#### 17.9.2 内部数据结果

| 数据集 | 显式递归 Strict ACC | S6 Strict ACC | Strict 变化 | 显式 Coverage | S6 Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discovery100 | 65.00% | 63.00% | -2.00 pp | 98.00% | 100.00% |
| Dev150 | 73.33% | 72.67% | -0.67 pp | 99.33% | 100.00% |
| RLHF-V heldout-500 | 74.00%* | 72.20% | -1.80 pp | 99.40%* | 100.00% |

Discovery100 上 corrected/harmed 为9/11，Dev150 为16/17，heldout-500 为34/43，三组净变化分别为-2、-1和-9条，均未达到显著差异。S6 在内部数据上没有发生覆盖坍缩，但也没有表现出准确率收益。需要注意，Discovery100 和 Dev150 的对照是同一 Phase17 E4 Rubric；现有 heldout 对照来自 Phase17 E5，因为没有冻结的 E4 heldout prediction，因此带 `*` 的 heldout 数字只能作为历史诊断，不能视为严格的同 Rubric 聚合消融。

Dev150 的平均变化较小，但不同领域相互抵消：General 提高2 pp、Visual 提高10 pp，Reasoning 下降14 pp；按来源看，MM-RLHF 提高12 pp，而 ViLReward-73K 下降20 pp。这说明“总分接近”并不表示单次整合在各类偏好上具有相同决策行为。

#### 17.9.3 VL-RewardBench 结果

| 系统 | Strict ACC | OverallAcc | MacroAcc | Coverage | 相对 S6 的 Strict 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 Explicit Recursive | 70.01% | 70.29% | 64.42% | 99.60% | +7.94 pp |
| S3 Unified Subtree | 66.72% | 68.42% | 62.97% | 97.51% | +4.65 pp |
| Clean S5-v2 Global Arbiter | **71.13%** | **71.59%** | **66.24%** | 99.36% | +9.06 pp |
| **S6 Unified Full-Rubric** | 62.07% | 62.07% | 59.51% | **100.00%** | — |

S6 虽然覆盖全部1,247个样本，但只答对774条。相对 S0 corrected/harmed 为68/167，净减少99条（exact McNemar `p=8.62e-11`）；相对 S3 为80/138，净减少58条（`p=1.04e-4`）；相对 Clean S5-v2 为69/182，净减少113条（`p=6.31e-13`）。因此下降不是由少量随机波动或 `None` 引起，而是单次整合系统性地改变了偏好判断。

类别结果定位了主要退化来源：

| 系统 | General Strict ACC | Hallucination Strict ACC | Reasoning Strict ACC |
| --- | ---: | ---: | ---: |
| S0 Explicit Recursive | 49.17% | 76.50% | 66.56% |
| S3 Unified Subtree | 46.96% | 72.36% | 64.67% |
| Clean S5-v2 | **54.14%** | **77.84%** | 64.98% |
| S6 Unified Full-Rubric | 48.07% | 63.28% | **67.19%** |

S6 的 Reasoning 比 S0 高0.63 pp，但 Hallucination 低13.22 pp；相对 S0 的净损失99条几乎全部来自 Hallucination。PoVid 从 S0 的89.06%降到70.31%，是最明显的来源级退化。进一步检查发现，S6 在 Hallucination 子集选择更长回答的比例为53.27%，而 S0 和 Clean S5-v2 分别为44.98%和44.04%；在 S6 相对 S0 的受损样本中，错误选中的回答有67.66%是更长回答。这与完整 Prompt 中 completeness、specificity 和 presentation 等准则对视觉事实性产生**证据稀释**的解释一致：模型容易把更丰富但未经视觉验证的细节当成质量优势。

#### 17.9.4 效率、技术可靠性与结论

| 系统 | 每 replicate 调用数 | VL-RewardBench 逻辑请求 | 相对 S0 请求减少 |
| --- | ---: | ---: | ---: |
| S0 Explicit Recursive | 27 | 101,007 | — |
| S3 Unified Subtree | 5 | 18,705 | 81.48% |
| Clean S5-v2 | 6 | 22,446 | 77.78% |
| **S6 Unified Full-Rubric** | **1** | **3,741** | **96.30%** |

S6 的 VL-RewardBench 主 run 耗时为1,486.1秒（24.77分钟），吞吐为151.04次请求/分钟、50.35个样本的三-replicate bundle/分钟；但一次请求平均约含7,314个输入 token，是 S3 子树请求平均长度的约3.09倍，因此调用数下降不会等比例转化为 prefill 和墙钟时间下降。S6 共处理4,491个逻辑请求，初次有38个技术失败，同 Prompt 重试后全部恢复，最终 parse valid rate 为100%、未解决技术失败为0、语义 `None` 为0；技术链路不是准确率下降的原因。

该消融否定了“只要把完整 Rubric 文本提供给模型，就能替代结构化执行”的假设。它同时说明两点：第一，S6 确实把 VL-RewardBench 请求量降到最低，证明统一调用在工程上可行；第二，**Rubric 的收益不仅来自 description 中包含了哪些知识，也来自如何隔离局部证据、保留子树分析并在最终阶段显式解决冲突。**当前结果支持继续采用“5份 Unified-Subtree 报告 → Clean Global Arbiter”的两阶段系统，而不是把全部27个准则压缩进一次判断。S6 因而作为关键负消融保留：它把“结构化执行”与“仅提供结构化文本”清楚地区分开。

### 17.10 Clean S5-v2 的 Qwen2.5 跨模型迁移

#### 17.10.1 实验问题与控制变量

前述结果均以 Qwen3-VL-8B-Instruct 为 Worker。本实验进一步检验：**“五棵子树分别形成证据报告，再由 Global Arbiter 综合决策”的 Clean S5-v2 聚合机制，是否也能迁移到较弱的 Qwen2.5-VL-7B-Instruct。**实验冻结 Phase17 E4 Rubric、五棵子树、Unified-Subtree Prompt、Global Arbiter Prompt、parser、A/B 顺序和聚合协议，只替换 Worker 模型；因此它与16.7节的“Phase17 E4 Rubric 跨模型迁移”不同，本节关注的是聚合机制本身的跨模型行为。

- Discovery100、Dev150 和 RLHF-V heldout-500 各执行一次完整的“5份子树报告 → 1次 Arbiter”流程，即每个样本6次调用、`K=1`、不交换；
- VL-RewardBench 固定1,247个样本和原有 `K=3` counterbalanced schedule，每个 replicate 独立生成5份子树报告和1次 Arbiter 判断，最后只聚合三次 Arbiter 结论；
- Prompt 要求优先输出 A/B，但 parser 对模型原生 `None` 保持语义透明；只有 transport、空响应、非法 JSON、缺失字段或非法标签使用同一 Prompt 重试，最多10次；
- 两个 Qwen2.5-VL endpoint 采用 available-slot pool；不重新演化 Rubric，也不修改任何 description、权重或路由。

本节以全样本 `Strict ACC` 为主要指标：最终 `None` 计错。`OverallAcc` 只在有明确 A/B 判断的覆盖集合上计算，必须与 Coverage 一起解释。

#### 17.10.2 内部数据结果

| 数据集 | Qwen3 Clean S5-v2 Strict ACC | Qwen2.5 Clean S5-v2 Strict ACC | Strict 变化 | Qwen3 Coverage | Qwen2.5 Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discovery100 | 65.00% | 51.00% | -14.00 pp | 96.00% | 92.00% |
| Dev150 | 62.67% | 58.67% | -4.00 pp | 99.33% | 92.67% |
| RLHF-V heldout-500 | 73.40% | 62.60% | -10.80 pp | 99.20% | 96.60% |

Dev150 上 Qwen2.5 的覆盖内 `OverallAcc` 为63.31%，略高于 Qwen3 的63.09%，但它只覆盖92.67%的样本，而 Qwen3 覆盖99.33%；全样本 Strict ACC 仍低4.00 pp。因此该 `OverallAcc` 不能解释为 Qwen2.5 整体更强，而是更多弃权筛掉困难样本后的选择性结果。

Qwen2.5 还表现出明显的展示位置偏置。Discovery100 和 Dev150 的 gold 分别严格平衡为50/50和75/75，heldout-500也接近均衡（A/B 为244/256），但模型输出如下：

| 数据集 | Gold A/B | Qwen3 输出 A/B/None | Qwen2.5 输出 A/B/None |
| --- | ---: | ---: | ---: |
| Discovery100 | 50 / 50 | 41 / 55 / 4 | **65 / 27 / 8** |
| Dev150 | 75 / 75 | 68 / 81 / 1 | **105 / 34 / 11** |
| RLHF-V heldout-500 | 244 / 256 | 265 / 231 / 4 | **333 / 150 / 17** |

这说明内部数据上的下降不能只归因于 Rubric 不适配：Qwen2.5 在综合长子树报告时更偏向当前展示的 Candidate A，同时更容易在证据冲突时弃权。由于内部协议是 `K=1`，该结果只能诊断位置敏感性，不能把位置偏置与跨子树综合能力完全分离。

#### 17.10.3 VL-RewardBench 主要结果

| Worker / 聚合方式 | Strict ACC | OverallAcc | MacroAcc | Coverage | 正确数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 Explicit Recursive E4 | 70.01% | 70.29% | 64.42% | 99.60% | 873 |
| **Qwen3 Clean S5-v2** | **71.13%** | **71.59%** | **66.24%** | 99.36% | **887** |
| Qwen2.5 Explicit Recursive E4 | 56.94% | 57.07% | 52.95% | **99.76%** | 710 |
| **Qwen2.5 Clean S5-v2** | **58.06%** | **59.69%** | **54.72%** | 97.27% | **724** |

在 Qwen2.5 上，Clean S5-v2 相对同模型 Explicit Recursive 将 Strict ACC 从56.94%提高到58.06%，增加 **1.12 pp / 14个正确样本**；MacroAcc 提高1.77 pp。覆盖内 OverallAcc 提高2.62 pp，但 Coverage 同时下降2.49 pp，因此不能把2.62 pp当作全局收益。配对 corrected/harmed 为112/98，exact McNemar `p=0.370`：方向为正，但尚未达到统计显著。

Qwen3 和 Qwen2.5 在本次冻结评测中都恰好由 Clean S5-v2 净增加14个正确样本，即 Strict ACC 均提高1.12 pp；这个数值一致说明聚合机制存在方向一致的迁移迹象，但只是单次运行中的数值巧合，不能据此声称模型无关的不变增益。相同 Clean S5-v2 下，Qwen2.5 仍比 Qwen3 低13.07 pp，paired corrected/harmed 为92/255、`p=7.56e-19`，说明底层 Worker 的视觉理解和跨报告推理能力仍是绝对性能的主要决定因素。

类别结果显示迁移收益并不均匀：

| 类别 | Qwen2.5 Explicit Strict ACC | Qwen2.5 Clean S5-v2 Strict ACC | 变化 |
| --- | ---: | ---: | ---: |
| General | 35.91% | 37.57% | +1.66 pp |
| Hallucination | 59.28% | 62.62% | **+3.34 pp** |
| Reasoning | **63.41%** | 58.99% | **-4.42 pp** |

来源级结果与该趋势一致：RLAIF-V 和 RLHF-V 分别提高8.15和10.29 pp，WildVision 提高1.17 pp，PoVid基本持平；Reasoning Tasks 则下降4.42 pp。Clean S5-v2 能帮助 Qwen2.5 统一视觉事实和幻觉证据，但较弱 Worker 在综合五份报告完成复杂推理时更容易损失已有的 reasoning 判断。

#### 17.10.4 Coverage 为什么下降

内部数据最终 `None` 的来源如下：

| 数据集 | 语义 `None` | 技术失败阻断 Arbiter | 最终未覆盖 |
| --- | ---: | ---: | ---: |
| Discovery100 | 7 | 1 | 8 |
| Dev150 | 9 | 2 | 11 |
| RLHF-V heldout-500 | 17 | 0 | 17 |

因此 heldout-500 的覆盖下降完全来自模型主动输出 `None`，不是解析或服务故障。VL-RewardBench 的3,741次 Arbiter 调用中有52次语义 `None`、22次因未解决的子树技术失败而阻断，单次非 A/B 比例约1.98%；但 `K=3` 要求至少两个 replicate 给出相同的A或B，少量弃权会与顺序敏感造成的 A/B 分歧共同放大为最终未覆盖：

| 三次 Arbiter 结论 | 最终未覆盖样本数 | 原因 |
| --- | ---: | --- |
| A / B / None | 26 | A/B各一票，没有两票同向多数 |
| A / None / None | 3 | 只有一票A |
| B / None / None | 2 | 只有一票B |
| None / None / None | 3 | 三次均未形成A/B |
| **合计** | **34** | Coverage = 1,213 / 1,247 = 97.27% |

所以 Coverage 下降的核心机制是：**Qwen2.5 更保守的语义弃权 + 更强的 A/B 位置敏感性 + `K=3` 对分歧的放大**。技术失败有次要影响，但不是13 pp模型差距的主要来源。

#### 17.10.5 Arbiter 行为、效率与结论

逐 replicate 检查显示，Qwen2.5 Arbiter 从不推翻五棵子树的5–0一致判断；对4–1强多数的30次推翻净损害3次，对3–2弱多数的146次推翻净修正46次，对平局、稀疏或大量 `None` 情况的332次推翻净修正158次。这说明 Global Arbiter 的主要价值仍来自处理弱多数和证据不完整情形，而不是重新解释已经一致的五棵子树。上述是 replicate 级诊断，经过 `K=3` 聚合后，最终样本级收益仍是14条，二者不能直接等同。

本实验共执行26,946个逻辑请求，其中内部数据4,500个、VL-RewardBench 22,446个。主 run 分别耗时1,330.3秒和5,793.7秒，合计约118.7分钟；技术重试额外约24.7分钟。重试后仍有61个未解决的调用级技术失败，约占全部请求0.23%，不足以解释模型间主要性能差距。两个 endpoint 的调用占比约51.2%/48.8%，available-slot 调度基本平衡。

**本实验结论。** Clean S5-v2 在 Qwen2.5 上相对显式递归取得与 Qwen3 同方向的 +1.12 pp Strict ACC 点估计，说明“子树证据报告 → Global Arbiter”不是完全依赖单一 Worker 的偶然机制；但该增益未达到统计显著，且伴随 Coverage 下降和 Reasoning 退化。现有证据支持将 Clean S5-v2 描述为**具有有限跨模型可迁移性的聚合机制**，不支持声称它能消除弱 Worker 的能力差距。Qwen2.5 的 A 位置偏置、语义弃权和跨报告推理不足，是后续若继续使用较弱 Worker 时必须单独处理的问题。



---

## 18. 对齐演化与反事实系统审计

### 18.1 问题与共同实验设置

第17章已经确定最终推理采用 Clean S5：五个 Unified-Subtree Workers 分别把一棵完整子树转化为分析报告，再由 Global Arbiter 综合五份报告输出最终偏好。但此前的 Split/Refine 主要根据 node-level Pairwise ACC 接受，演化优化的对象与最终系统实际使用的对象并不一致。

这一阶段不增加 Create、Merge 等新算子，先比较两种接受目标，再由 Phase22 扩展局部竞争的评价范围（见18.7节）：

- **Phase19：最终系统级接受。** 候选必须让 Discovery100 上的 Global-Arbiter Strict ACC 上升。
- **Phase21：root-local 接受。** 候选必须改善所属 root 的 Unified-Subtree 判断，再把通过的 roots 同步提交。

共同数据设置如下：

| 数据集 | 样本数 | 用途 |
| --- | ---: | --- |
| Discovery100 | 100 | 生成错误经验、提出候选并执行接受判断 |
| Dev150 | 150 | 每轮独立诊断，不反馈 Manager、不选择 checkpoint |
| RLHF-V heldout-500 | 500 | 演化结束后的内部泛化测试 |
| VL-RewardBench | 1,247 | 最终外部评测，采用 K=3 Clean S5 |

所有接受比较均使用 Strict ACC：输出 None 也计错。逐样本收益定义为：

$$
\operatorname{Net}(C)
=
N_{\mathrm{corrected}}(C)
-
N_{\mathrm{harmed}}(C),
$$

其中 corrected 表示基线错而候选对，harmed 表示基线对而候选错。

<a id="phase19"></a>

### 18.2 Phase19：直接用最终系统收益接受候选

#### 候选如何产生

Phase19 的错误筛选分两步。第一步先为每个 root 定义系统归因集合：

$$
\mathcal A_r
=
\{x\mid \text{Global Arbiter 判错，且 root }r\text{ 的 Unified report 也判错}\}.
$$

$\mathcal A_r$ 只是第二步筛选使用的 allowlist，并不是最终进入算子的错误样本。随后再与对应节点的 Pairwise 明确错误取交集：

| 算子 | 最终进入算子的错误样本 | Manager 如何使用 |
| --- | --- | --- |
| Split | $\mathcal E_r^{\mathrm{split}}=\mathcal A_r\cap\mathcal P_r$，其中 $\mathcal P_r$ 是 root Pairwise 明确投错的样本 | 逐样本生成 ErrorSignatures，再聚类并生成 children |
| Refine | $\mathcal E_c^{\mathrm{refine}}=\mathcal A_r\cap\mathcal P_c$，其中 $\mathcal P_c$ 是目标 child Pairwise 明确投错的样本 | 不生成 ErrorSignatures；连同 correct/abstain boundary cases 直接交给 Refine Manager 改写一个 child description |

只有 Split 使用 ErrorSignatures。该过滤只保留已经传导为系统错误的 root 问题，也会排除“root 自己判断错误、但被其他 roots 补救”的样本。

#### 候选如何竞争和提交

对每个候选，系统只重新生成发生变化的 root report，另外四份 epoch-start reports 直接复用；随后重新调用 Global Arbiter，并与本轮共同基线逐样本比较。只有 $\operatorname{Net}(C)>0$ 的候选才有资格提交。

如果同一 root 有多个正收益 Refine，只保留系统收益最高的一个。不同 roots 都有正收益时，先把各自的候选 reports 合并，再运行一次联合 Arbiter：

- 联合系统仍提高，则一起提交；
- 联合系统不提高，则只提交最佳 singleton；
- 提交后的 reports 直接成为下一轮基线，不重新生成。

因此，Phase19 的接受语义与最终推理完全一致，但接受观测来自 Discovery100 上一次 K=1 Arbiter realization。

#### 结果与问题

Phase19 五轮共比较27个候选，只接受2个 Split：Completeness 和 Creativity。Rubric 从5个 roots 增长到10个 nodes。

| 数据集 | Initial Strict ACC | Phase19 final | 变化 |
| --- | ---: | ---: | ---: |
| Discovery100 | 69.00% | 72.00% | +3.00 pp |
| Dev150 | 70.67% | 71.33% | +0.67 pp |
| Heldout-500 | 75.40% | 74.80% | -0.60 pp |
| VL-RewardBench | 68.08% | 70.57% | +2.49 pp |

VL-RewardBench 表中的初始值对应 Initial five-root；若与主要强 Control Phase17 E4 比较，Phase19 final 从**71.13%降至70.57%**，相差-0.56 pp。Discovery 提升没有稳定迁移。候选生成阶段使用“系统归因样本、root/child Pairwise 错误”的交集，使 Manager 能看到的**错误样本明显减少**，并排除了被其他 roots 补救的局部错误，限制了错误经验覆盖和候选多样性。候选选择阶段又让 Discovery100 同时参与错误发现和接受判断； Phase19 因而可能生成不充分的局部专家候选，并把同数据选择和采样波动误认为候选的真实系统收益。

<a id="phase21"></a>

### 18.3 Phase21：用完整 root subtree 做局部原子竞争

Phase21 不再让 Pairwise、Specialized ACC 和 Global Arbiter 共同定义局部演化，而是把 Unified-Subtree Worker 贯穿整个 root 内闭环。

#### 一轮演化如何运行

每轮开始时，五个 roots 分别运行 Unified-Subtree baseline。对 root $r$：

1. baseline 输出 A/B 的样本构成冻结 scope；baseline None 不进入 scope；
2. scope 内判断与 gold 不一致的样本构成 Unified mismatches；
3. 这些 mismatches 生成该 root 本轮的 ErrorSignatures；
4. Split 或 Bundle Refine 根据同一批错误经验提出完整 root-subtree 候选；
5. 候选再次运行 Unified-Subtree Worker，并在相同 frozen scope 上与 baseline 配对比较；
6. 只有 corrected 多于 harmed 才原子接受整个 subtree 修改。

候选不能通过输出更多 None 来缩小本轮评价范围：scope 始终由 epoch-start baseline 冻结，候选在该 scope 内输出 None 时按错误处理。Coverage 仍会单独报告，但不能改变接受分母。

两个算子的单位也随之改变：

- **Split bundle** 一次生成整套 children；不锁定强孩子、不允许部分接纳，所有 children 共同接受或拒绝。但是失败后不会重新聚类。
- **Bundle Refine** 读取 root 和全部 children，由 Manager 选择真正需要小修的部分 descriptions；ID、criterion name 和树结构保持不变，所有 edits 共同接受或拒绝。

同一 root 每轮最多一个候选。不同 roots 分别通过后全部同步提交；accepted root 直接复用候选竞争阶段的 report，未修改 root 复用 baseline report。Global Arbiter 只在提交后计算系统诊断，不能改变或回滚局部决定。Specialized ACC 同样只作只读诊断。

#### 结果与问题

Phase21 连续5轮为五个 roots 各产生一个 Split bundle，共25个候选。所有候选在各自 frozen scope 上都满足 $\operatorname{Net}(C)\le 0$，因此全部拒绝：

- accepted Split：0；
- accepted Bundle Refine：0；
- 因为没有任何 Split 成功，正式全量运行没有进入 Bundle Refine；
- final Rubric 与 initial Rubric 的相同，仍为5 roots、0 children。

Phase21 的候选接受只比较单棵 root 最终输出的 A/B/None 标签是否正确，相当于把每棵子树当成独立分类器。本次25个候选的 root-local 净收益均未大于0，因此全部被拒绝，最终 Rubric 没有发生变化。然而，正式系统并不是对五个 root 标签做多数投票，而是让 Global Arbiter 阅读五份完整报告后再决策；报告中的视觉证据、答案冲突和判断理由同样会影响最终结果。后续反事实审计发现，25个局部拒绝候选中有23个反而提高了 Discovery100 上的系统 Strict ACC。这说明 root-local ACC 只是局部代理指标，不能等同于候选对最终系统的真实效用。

### 18.4 两种接受目标暴露的不同问题

| 协议 | 接受目标 | 优点 | 主要问题 |
| --- | --- | --- | --- |
| Phase19 | 同一 Discovery100 上的 Global-Arbiter 系统收益 | 与最终推理语义一致 | K=1 噪声；候选生成和接受共用数据，容易选择过拟合 |
| Phase21 | frozen scope 上的 root-local Unified 收益 | scope、错误经验、候选评价完全一致 | 优化的是局部分类正确率，不是报告对 Arbiter 的证据价值 |

这说明不能简单地在“局部指标”和“同数据系统指标”之间二选一。为判断两类指标究竟错在哪里，后续反事实审计分别检查候选进入完整系统后的效用、K=1 结果的稳定性、跨-root组合效应和独立数据迁移。

<a id="counterfactual-audit"></a>

### 18.5 反事实审计：局部收益能否转化为系统泛化

#### 18.5.1 为什么可以做缓存级反事实实验

虽然 Phase21 拒绝了全部25个 Split bundles，但每个候选在 Discovery100 上的完整 Unified-Subtree reports 已经缓存。因而可以构造反事实系统：

    候选所属 epoch 的五份 baseline reports
            │
            ├─ 目标 root：替换为 candidate report
            └─ 其他四个 roots：保持 baseline report
            │
            ▼
    同一个 Clean S5 Global Arbiter
            │
            ▼
    与该 epoch baseline 逐样本比较

这个实验不重新调用 Unified-Subtree Worker，不重新生成 candidates，也不修改正式 Rubric。唯一新增变量是“Global Arbiter 是否看到该候选 root report”。因此它能够直接测量候选报告的系统效用。

#### 18.5.2 K=1 rejected-candidate audit

第一阶段先测试预先冻结的6个正向、边界和负向候选，确认整条反事实管线；随后补齐全部25个候选。每个候选只替换一棵 root，并重新运行100条 Discovery 样本的 K=1 Arbiter，共新增2,500次 Arbiter 请求。

共同系统 baseline 为62.00%。25个局部拒绝候选中，23个反事实系统 Strict ACC 上升，2个下降，最大单候选增益为+9 pp。但“局部选择性效用”和“系统 Strict ACC 增量”的相关性很弱：

| root-local 指标 | Pearson | Spearman |
| --- | ---: | ---: |
| frozen-scope selective utility | 0.194 | 0.164 |
| formal root-scope net corrected | 0.244 | 0.183 |
| covered accuracy delta | 0.204 | 0.193 |

结果说明 Phase21 的局部门槛会拒绝一些可能帮助最终 Arbiter 的 reports。不过 K=1 仍可能夸大正收益，因此不能根据这25个结果重新选择 checkpoint。

#### 18.5.3 K=3 singleton 与 coalition audit

第二阶段先把共同 baseline 和25个 singleton 全部升级到 K=3：每个系统运行三次 Arbiter，以 A/B 多数作为最终结果，没有多数则为 None。K=1 与 K=3 的候选收益 Pearson 为0.874、Spearman 为0.830，收益符号一致率为72%；K=3 下仍有16个正收益、4个持平、5个负收益候选。说明 K=1 有噪声，但局部拒绝与系统收益错位并非完全由随机性造成。

提高到 K=3 后，root-local selective utility 与系统收益的 Pearson 仍只有0.082、Spearman 只有0.054，基本没有预测能力。

随后在 Epoch 1 和 Epoch 5 分别穷举五个 root 候选的全部 $2^5=32$ 个 subsets。空集是共同 baseline，单元素是 singleton，其余是多个 root reports 的联合替换。

| Epoch | K=3 baseline | 最佳 singleton | 正 singleton 联合 | 五候选全部提交 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64.00% | Completeness：70.00%（+6） | +3 | -1 |
| 5 | 64.00% | Completeness：67.00%（+3） | +2 | -5 |

多个单独正收益候选合并后没有获得收益之和，全部提交反而下降。这表明 roots 之间存在明显交互：不同 reports 可能重复、冲突，或改变 Arbiter 对证据优先级的理解。因此，同一轮多个 roots 通过时不能按 singleton 收益机械同步提交，必须实际评价 coalition。

#### 18.5.4 Dev150 transfer audit

K=3 coalition audit 仍在 Discovery100 上评价和比较系统，无法排除同数据选择过拟合。为此，在查看 Dev150 结果前冻结9个系统：

- 共同 baseline；
- Epoch 1 的 Completeness、Completeness+Visual Grounding+Factuality、正 singleton 联合和全部候选；
- Epoch 5 的 Completeness、Clarity、正 singleton 联合和全部候选。

这些系统共享10个唯一 candidate roots。每个 candidate root 只在 Dev150 上生成一次 Unified reports，再由不同系统组合复用；每个冻结系统运行 K=3 Global Arbiter。Dev150 只做验证，不反向选择新的组合。

| 冻结系统 | Discovery 中的角色 | Dev150 Strict ACC | 相对72.00% baseline |
| --- | --- | ---: | ---: |
| E1 Completeness | E1 最佳 singleton | 68.00% | -4.00 pp |
| E1 C+V+F | E1 强 coalition | 71.33% | -0.67 pp |
| E1 positive union | E1 正 singleton 联合 | 70.00% | -2.00 pp |
| E1 all | E1 全候选 | 65.33% | -6.67 pp |
| E5 Completeness | E5 最佳 singleton | **73.33%** | **+1.33 pp** |
| E5 Clarity | E5 并列强 singleton | 71.33% | -0.67 pp |
| E5 positive union | E5 正 singleton 联合 | 72.00% | 0.00 pp |
| E5 all | E5 全候选 | 70.67% | -1.33 pp |

Discovery 收益与 Dev 收益的 Pearson 为0.173、Spearman 为0.093，收益符号只在3/8个系统上一致。唯一保持正增益的 E5 Completeness 为5 corrected、3 harmed，但95% bootstrap CI 为[-2.00,+4.67] pp，McNemar exact $p=0.727$，尚不能认为它稳定优于 baseline。

### 18.6 主要问题与证据边界

Phase19、Phase21 和三次反事实审计共同暴露了五个问题：

1. **局部优化目标错位。** root-local A/B/None 正确率衡量的是子树作为独立分类器的表现，不能表示完整分析报告对 Global Arbiter 的证据价值。
2. **系统接受观测不稳定。** Discovery100 上 K=1、无 seed 的单次 Arbiter 判断存在明显波动，1–2条样本翻转就可能改变小增益候选的接受结论。
3. **候选生成与接受共用数据。** Phase19 在 Discovery100 上发现错误、生成候选并接受候选，观察到的系统增益包含选择后偏差；其 Discovery 提升没有在 heldout 和 VLRB 上复现。
4. **跨-root效用非加性。** singleton 候选单独为正，不代表多个候选联合提交仍为正；全部候选组合在 Epoch 1 和 Epoch 5 都出现下降。
5. **Discovery 系统收益缺乏迁移性。** Discovery 与 Dev150 收益的 Pearson 只有0.173、Spearman 只有0.093，收益符号仅3/8一致。

这些结果不能支持“Phase19 或 Phase21 优于 Phase17 E4”，也不能支持某个被拒候选已经具有稳定的泛化收益。当前能够得到的结论仅限于：**node-level、root-local 和同数据 system-level 三种接受信号都存在各自的失真来源，尚没有一种信号被证明能够稳定选择出在独立数据上更好的 Rubric。**

---

<a id="phase22"></a>

### 18.7 Phase22：全样本 root-subtree 原子竞争（含条件重聚类设计）

#### 实验设计

Phase22 暂以“每棵树的局部改善可能带来系统改善”为工作假设，不把它当作已证明的结论。整体沿用 Phase21，只将接受范围从 baseline 输出 A/B 的 frozen scope 扩展到全部 Discovery100，并允许 Split 失败后按归因决定是否重新聚类。

当前已提交的 root subtree 与候选 subtree，都以 Unified-Subtree Worker 在全部100条样本上的预测进行比较；基线复用已保存的报告，不每轮重新采样。设 $p_i^0$、$p_i^1$ 分别为修改前后的预测，$y_i$ 为 gold，则：

$$
g_i=\mathbf{1}[p_i^1=y_i]-\mathbf{1}[p_i^0=y_i],
\qquad G=\sum_{i=1}^{100}g_i.
$$

即错→对记+1，对→错记−1，其他记0；**只有 $G>0$ 才接受，持平也拒绝。** `None` 按错误处理；API/解析失败单独暂停处理，不作为科学负收益。

- **候选生成**：仍沿用 Phase21 的触发条件，错误经验来自 Unified-Subtree 的明确 A/B 错判，而非 node-level Pairwise；baseline `None` 纳入竞争，但不因此自动纳入错误签名生成。
- **原子操作**：Split 整套 children 共同接受或拒绝；Refine 可修改 root 与 children 的部分描述，但全部 edits 共同竞争。不同 roots 独立通过后同步提交，Global Arbiter 仅作提交后诊断，不参与接受或回滚。
- **失败重试**：归因为 `cluster_or_decomposition_error` 时重新聚类；其他科学失败复用聚类并重新生成整套 children。不锁定强 child，不部分接受。
- **冻结设置**：从5个初始 roots 开始，最多5轮；Worker 为 Qwen3-VL-8B-Instruct，Manager 为 Qwen3.5-397B-A17B。内部推理 $K=1$、temperature=0.5、max_tokens=2048、不设 generation seed；VLRB 为 $K=3$。Dev150 只诊断，heldout-500 与 VLRB 不参与选择。本节不是 Qwen3.5-27B Worker 的结果。

#### 演化过程

表中数字为每个候选相对当轮已提交 root 的全样本净收益；除标明 Refine 外均为 Split。

| Epoch | 完整性 | 视觉细节 | 事实性 | 创造性 | 清晰性 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1 | −6 | −1 | **Split +1，接受** | −8 | −9 |
| 2 | −7 | −2 | **Refine +1，接受** | −3 | −9 |
| 3 | −7 | −1 | Refine −1 | −6 | −5 |
| 4 | −7 | −4 | Refine −2 | −6 | −12 |
| 5 | −10 | −2 | Refine −4 | −16 | −6 |

25次候选竞争仅接受2次（1次 Split、1次 Refine），都发生在事实性 root；节点数从5增至9。Epoch 2之后已提交的树不再变化，不能把后续负收益候选解释成最终树持续退化。

第一次接受修正7条、损害6条；第二次修正6条、损害5条，均净增加1条。进一步分解，两次都是“原 A/B scope 内净损失1条，原 `None` 范围内新增2条正确判断”。因此，新指标确实改变了这两个候选的接受决定，但收益并非原判断范围内的准确率提高。

#### 最终结果

以下均为 **Clean S5 最终系统（Global Arbiter）的 Strict ACC**，不是单棵 root 的准确率；`None` 计错，净变化是 Phase22 比对照多或少判断正确的样本数。

| 数据集 | 对照 | 对照 ACC | Phase22 ACC | 净变化 |
| --- | --- | ---: | ---: | ---: |
| Discovery100 | Initial | 68.00% | 69.00% | +1 |
| Heldout-500 | Initial | 76.80% | 76.00% | −4 |
| VLRB（1,247条） | Initial | 68.08% | 67.92% | −2 |
| VLRB（1,247条） | Phase21 final | 68.48% | 67.92% | −7 |
| VLRB（1,247条） | Phase17 E4 | 71.13% | 67.92% | −40 |

VLRB 相对 Initial、Phase21 的 McNemar exact $p$ 分别为0.927、0.600，未检测到显著差异，不等于证明等效。相对 Phase17 E4，修正46条、损害86条，下降3.21 pp，$p=0.000631$，配对 bootstrap 95% CI 为[−5.05, −1.44] pp。**71.13%是 Phase17 E4 强基线，不是 Phase22 的演化起点。** Phase21 final 与 Initial 的树相同，单次预测结果仍可因随机推理而不同。

VLRB 相对 Initial 的类别净变化为 General +1、Hallucination −5、Reasoning +2；事实性 root 的局部提高没有体现为 Hallucination 类别的外部收益。最终未解决的技术失败为0；首轮162个技术失败已由 retry 清除。

#### 主要发现与下一步

1. **局部小收益尚未转化为可靠泛化收益。** 第一次 Split 使事实性 root 从55%升到56%，Discovery 系统却从68%降到65%；第二次 Refine 后系统回到69%。这提示局部标签收益不等于报告的系统价值，但不足以否定“局部最优可能有利于全局”的一般假设，本次也未证明达到局部最优。
2. **单次小增益需要稳定性检查。** 同一最终树在 Epoch 2–5 的 Dev150 ACC 为74.67%、71.33%、72.00%、74.00%；这不是结构变化带来的轨迹。两次接受都只有+1/100，不能直接视为可复现提升。
3. **本次没有实际测试重聚类收益。** 16次后续 Split 重试全部复用原聚类；23次失败归因为18次过度修正、5次兄弟边界冲突，均未触发重新聚类。结果不能单独归因于“重聚类有效或无效”。
4. **反思能指出问题，但修正闭环仍弱。** 当前失败归因最多展开6个 harmed 和3个 corrected 案例，未逐条分析候选的全部新错误。后续可保持净收益指标不变，比较“逐错误反思→汇总建议→整树 revise”，并检查合并/重新划分建议能否真正触发结构操作；在新实验中预先冻结方案，不用已查看的 VLRB 挑选候选。

**结论：Phase22 的接受规则挡住了大量负收益候选，但只产生两次微小局部改进，没有检测到相对初始树的外部收益，且明显落后于 Phase17 E4。优先改进候选与反思质量、验证小收益稳定性，而不是仅增加演化轮数。**

结果来源：[演化与 heldout 报告](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase22_all_sample_subtree_adaptive_recluster_evolution_v1/final_report.json)、[Dev150 轨迹](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase22_all_sample_subtree_adaptive_recluster_evolution_v1/dev150_trajectory.json)、[VLRB 报告](../output/evolving_structured_rubrics/vl_rewardbench_all_sample_adaptive_recluster_evolution_v1/final_report.json)。

---

## 19. 本地 Qwen3.5-27B Manager 演化与完整候选诊断

### 19.1 实验设置与演化结果

目的：降低 Manager 成本的同时，检查小参数量 Manager 模型生成的准则是否有效。沿用 Phase17 的 Discovery100、Dev150、五个初始 roots、最多5轮、局部 Split 竞争、Locked-Child retry、Role-aware Refine 和同步提交；Pairwise Worker 仍为 Qwen3-VL-8B-Instruct，Prompt v2、temperature=0.5、max_tokens=2048。Phase17 是**子准则多数投票递归**的聚合方式，所以当前实验默认是这个聚合方式。

本节记录最终完成的 `no_thinking_compact_ids` 运行：Manager 为本地 Qwen3.5-27B，`enable_thinking=false`、`max_completion_tokens=16384`；解析失败最多重试3次，最小 cluster 从5降为2，样本标识改为简单序号并在代码中映射回来 (因为输出过错误样本id，样本的id过于复杂)。**因此不是严格的“只换 Manager”单变量消融，也不与早期 thinking-on 运行混用。** 具体配置和运行命令见[实验计划](experiments/phase17-manager-qwen35-27b/plan.md)。

- Split 共21次：1次接受、18次竞争拒绝、2次提案无效；仅 Factuality 在 Epoch 1 接受。
- Refine 共20次：1次接受、4次竞争拒绝、15次提案无效；Epoch 3 接受 Factuality 下几何推理孩子的修改。
- 最终**仅有5个 roots + 5个孩子，共10节点**；Discovery Strict ACC 从64%升到65%。
- RLHF-V heldout-500 为75.60%（378/500），初始对照为75.80%（379/500）；
- VL-RewardBench Overall ACC 为57.49%、Macro ACC 为54.43%；相对历史 Phase17 E5（69.91%、64.06%，869条正确），分别**下降12.42、9.63个百分点**，正确数从869降至706，净减少163条；最终技术失败为0。

### 19.2 Epoch 5 完整 Rubric：不按竞争结果筛掉孩子

为区分“生成的准则差”与“筛选丢掉了有用准则”，固定上述已完成运行：Completeness、Visual Grounding、Creativity、Clarity 各取 Epoch 5 最后一次候选子树，包含被拒绝的孩子；Factuality 保留最终已接受子树及 Refine。孩子数依次为5、5、5、5、3，共 **28节点**。不累加历史候选，不重新调用 Manager，也不是从第一轮开始全部接受的重新演化实验。

仅替换输入 Rubric，继续使用原节点推理与子树聚合：孩子形成明确多数时采用孩子判断，否则回退父节点；最后五棵子树等权投票。VL-RewardBench 固定1,247条样本、$K=3$ 平衡 A/B 顺序，两台相同 Worker 服务各并发100、合计200。三次遗留的3个技术失败全部恢复。3小时推理完成，一共推理 104,748 次，吞吐为 **575.38 次逻辑推理/分钟**。

| 最终评测方案 | 正确数 / 1247 | Overall ACC | Macro ACC | Strict ACC |
|---|---:|---:|---:|---:|
| 原始 Qwen3-VL-8B（Native VL-RB Prompt，无结构化 Rubric） | 675 | 54.52% | 53.56% | 54.13% |
| 初始五个 roots（历史 Prompt v2 对照） | 712 | 58.12% | 54.60% | 57.10% |
| Phase17 E5（397B Manager，27节点） | 869 | 69.91% | 64.06% | 69.69% |
| 27B 正常竞争版，10节点（上次运行） | 706 | 57.49% | 54.43% | 56.62% |
| **27B Epoch 5 完整候选，28节点** | **886** | **71.39%** | **66.09%** | **71.05%** |

Overall ACC 排除最终平局/弃权，Strict ACC 以全部样本为分母。完整候选相对 **Phase17 E5** 提升1.48个百分点 Overall ACC、2.03个百分点 Macro ACC，多正确17条；相对初始五个 roots 和原始模型分别提升13.27、16.87个百分点 Overall ACC。原始模型使用 Native 通用评判 Prompt，其他方案使用结构化 Prompt v2；

历史基线来源：[Phase17 E5 报告](../output/evolving_structured_rubrics/vl_rewardbench_phase17_discovery_v2_prompt_v2_v1/final_report.json)（含初始五个 roots）、[Native 最终重试报告](../output/evolving_structured_rubrics/vl_rewardbench_phase10_transfer_v2_max2048/native_retry_max10/report.json)。此处 Phase17 指原397B Manager的正式E5结果，不是探索性选择的E4。

### 19.3 子树贡献：父节点与完整子树对比

复用本次 retry 后的同批节点预测，分别计算父节点单独判断和完整子树判断，再按 $K=3$ 多数聚合。以下均为全部1,247条样本上的 **Strict ACC**，平局/弃权计错；“纠正/改错”以父节点判断为对照，不是该子树对五树系统的独立贡献。

| Root | 父节点 ACC | 完整子树 ACC | 纠正 / 改错 | 净收益 |
|---|---:|---:|---:|---:|
| Completeness | 50.76% | **73.70%** | 344 / 58 | **+286** |
| Visual Grounding | 55.25% | **66.24%** | 172 / 35 | **+137** |
| Factuality | **69.21%** | 68.40% | 58 / 68 | −10 |
| Creativity | 51.32% | **72.81%** | 310 / 42 | **+268** |
| Clarity | 53.97% | **70.01%** | 231 / 31 | **+200** |

**补回被拒绝孩子的四棵树全部改善，Completeness 和 Creativity 收益最大。** 不同子树可能纠正同一条样本，因此净收益不能相加。Factuality 净减少10条正确样本，McNemar exact $p=0.423$，不足以认定为稳定退化。

Completeness 子树单独的 Strict ACC 为73.70%，高于五棵树等权聚合后的71.05%，提示聚合可能稀释强子树的判断。不过，这是查看测试结果后的诊断，不能在当前测试集上挑出最佳子树并替代正式五树结果。

### 19.4 分组收益与代表性孩子准则

以下 ACC 均在各组产生明确最终判断的样本上计算；Phase17 E5 为历史397B Manager结果，竞争版本为本次同批预测的离线聚合。

| 类别 | 样本数 | Phase17 E5 ACC | 同批竞争版本 ACC | 完整版本 ACC | 相对同批竞争版正确数变化 |
|---|---:|---:|---:|---:|---:|
| General | 181 | 50.00% | 38.07% | **55.00%** | +32 |
| Hallucination | 749 | 76.47% | 57.35% | **77.88%** | +156 |
| Reasoning | 317 | 65.71% | **66.13%** | 65.40% | −1 |

**相对竞争版本的改善主要来自幻觉判断，其次是 General，没有显示推理能力的普遍提升。** 相比历史 Phase17 E5，完整版本在三类中的正确数变化分别为+9、+9、−1，合计+17。

下面四个孩子强调“事实准确优先于完整、具体、表达丰富或语言流畅”。表中是各单节点三次聚合后的指标，不是整棵子树的指标；

| 孩子准则（简写） | 所属 Root | 覆盖率 | 覆盖范围 ACC |
|---|---|---:|---:|
| 证据依据 vs 虚构完整性 | Completeness（完整性与覆盖） | 85.24% | 80.62% |
| 视觉内容忠实 vs 虚构细节 | Visual Grounding（视觉依据与细节） | 82.04% | 80.84% |
| 视觉事实性 vs 风格性幻觉 | Creativity（创造性与表现力） | 87.01% | 78.16% |
| 视觉事实准确 vs 表达流畅 | Clarity（清晰性与连贯性） | 81.64% | 76.92% |

不同孩子的覆盖样本不同，不能把覆盖范围 ACC 当作全样本准确率直接比较。结合准则内容与分组结果，一个可能的解释是：孩子不仅细化父准则，还纠正了 Completeness、Creativity、Clarity 对丰富、完整或流畅表达的偏好，将判断拉回事实依据。多个 roots 同时发生这种变化，也可能强化系统的事实性偏好。**这仍是待逐样本验证的解释，不是已经证明的因果结论。**

### 19.5 主要发现

1. **27B 能生成有迁移价值的准则。** 子树对比显示，被拒绝的候选中仍包含有效准则。多个孩子强调“事实依据优先于完整、丰富或流畅表达”，与幻觉类收益一致。
2. **竞争指标和数据分布是否影响演化的结果**。最后只有一个子树竞争成功，但是其他子树并不是没有效益，在bench上测试的收益也不少，但是在演化过程中不断被拒绝，然后继续修改再拒绝的迭代模式中。

**结论：这次瓶颈不只是准则生成能力，还包括能否识别并保留已经生成的有效准则。**

结果来源：[27B演化与heldout](../output/evolving_structured_rubrics/phase17_manager_qwen35_27b_no_thinking_compact_ids/rubric_evolution_phase5/phase17_discovery_v2_prompt_v2_split_refine_v1/final_report.json)、[正常竞争版VLRB](../output/evolving_structured_rubrics/vl_rewardbench_phase17_manager_qwen35_27b_no_thinking_v1/final_report.json)、[完整候选VLRB](../output/evolving_structured_rubrics/vlrb_27b_full/final_report.json)、[完整候选最终预测](../output/evolving_structured_rubrics/vlrb_27b_full/retry/combined/logical_votes.json)。本地原始产物保留用于复核，不随文档提交。

---

### 19.6 完整 Rubric + Clean S5-v2：8B 与 27B Worker

#### 设置与结果口径

固定19.2节的 Epoch 5 完整候选（5个 roots、23个孩子，共28节点），不重新演化。复用第17章 **Clean S5-v2（A/B-preferred，原生 `None`）**：五个 Unified-Subtree Workers 各读取一棵完整子树，Global Arbiter 综合五份报告；每个样本运行三次，再按既有多数规则聚合。两组使用相同的1,247条 VL-RewardBench 样本、$K=3$ 平衡 A/B 顺序、提示词和 Rubric，`temperature=0.5`、常规 `max_tokens=2048`。Rubric SHA-256 为 `fa5286cbe828af4527ddd239bf9a522ad96d83451ff1a1c24338d5056474fc7f`。

- **实验A：8B Clean S5-v2**，子树 Worker 和 Arbiter 均为 Qwen3-VL-8B-Instruct。
- **实验B：27B Clean S5-v2**，两者均换为 Qwen3.5-27B，服务端关闭 thinking，两服务各并发20、合计40。不是只换 Arbiter；包含非法转义解析恢复及两条4096输出上限补救，恢复记录见19.7节末链接。

下表同时加入的**标准 VL-RewardBench 仓库原生评测**。其 $K=5$、与本地 Clean S5-v2 每样本18次逻辑调用（共22,446次，不含重试）并非等预算。

| 模型与方案 | K | 正确数 / 1247 | Overall ACC | Macro ACC | Strict ACC | 覆盖率 |
|---|---:|---:|---:|---:|---:|---:|
| 8B 原生 | 5 | 662 | 54.26% | 53.20% | 53.09% | 97.83% |
| 8B + 完整 Rubric，显式递归 | 3 | 886 | 71.39% | 66.09% | 71.05% | 99.52% |
| **8B + 完整 Rubric + Clean S5-v2** | **3** | **879** | **72.11%** | **65.36%** | **70.49%** | **97.75%** |
| 27B 原生 | 5 | 950 | 76.37% | 72.54% | 76.18% | 99.76% |
| **27B + 完整 Rubric + Clean S5-v2** | **3** | **1,031** | **83.08%** | **78.79%** | **82.68%** | **99.52%** |

Overall ACC 排除最终平票/弃权；Strict ACC 以全部1,247条为分母。两组 Clean S5-v2 最终技术失败均为0，最终平票/弃权分别为28条、6条，不属于技术失败。

#### 配对收益与重复评测

- **27B Clean S5-v2 相对8B Clean S5-v2**：纠正216条、改错64条，净增加152条；Strict ACC 提升12.19个百分点，配对 bootstrap 95%区间为 **[+9.70, +14.68]个百分点**（10,000次、seed=42），McNemar exact $p=1.92\times10^{-20}$。Overall、Macro 分别提高10.97、13.44个百分点。
- **8B Clean S5-v2 相对8B显式递归**：纠正66条、改错73条，净减少7条，$p=0.611$。Overall略升，但Strict略降，**没有证据表明此次 Clean S5-v2 显著优于显式递归**。
- 相对各自原生基线，8B和27B的 Clean S5-v2 分别多正确217条、81条，**Strict提升17.40、6.50个百分点**。27B未判对样本从297条降到216条，减少约27.3%。

| 单次评测 Strict ACC | 第1次 | 第2次 | 第3次 | 三次聚合后 |
|---|---:|---:|---:|---:|
| 8B Clean S5-v2 | 68.89% | 69.05% | 69.53% | 70.49% |
| 27B Clean S5-v2 | 81.96% | 82.12% | 81.96% | 82.68% |

27B在三次中均领先；报告中的重复/顺序不一致计数从382降至196，但它混合了采样与A/B顺序影响，不能直接视为纯位置偏差。三次评测用于同一次运行的聚合，不是三次独立实验。

### 19.7 收益来自哪里：类别与子树

#### 分类结果

下表为各类别**覆盖范围 ACC**，与Overall一致；各组覆盖样本可能不同。

| 类别 | 样本数 | 8B原生 | 8B Clean S5-v2 | 27B原生 | 27B Clean S5-v2 |
|---|---:|---:|---:|---:|---:|
| General | 181 | 41.18% | 52.30% | 55.80% | **63.33%** |
| Hallucination | 749 | 52.17% | 80.54% | 78.37% | **86.29%** |
| Reasoning | 317 | 66.24% | 63.23% | 83.44% | **86.75%** |

为排除覆盖率差异，两组 Clean S5-v2 的全样本结果如下：

| 类别 | 8B Strict ACC（正确数） | 27B Strict ACC（正确数） | 正确数变化 |
|---|---:|---:|---:|
| General | 50.28%（91） | 62.98%（114） | +23 |
| Hallucination | 79.04%（592） | 85.71%（642） | +50 |
| Reasoning | 61.83%（196） | 86.75%（275） | **+79** |

需要区分两个结论：

1. **固定 Clean S5-v2，从8B换到27B**，最大净收益来自Reasoning：贡献79/152条，约52%，而该类别仅占约25%的样本。
2. **固定27B，从原生换到结构化流程**，覆盖范围ACC的额外提升主要来自Hallucination（+7.92个百分点）和General（+7.53），Reasoning为+3.31。8B的Reasoning反而下降3.01个百分点，说明复杂流程不保证弱模型推理表现提高。

一个与结果一致的解释是：更强模型提供了推理基础，而完整准则帮助系统优先考虑事实证据，抑制对丰富、流畅但缺乏依据的回答的偏好；这与19.4节孩子准则内容一致，但尚非因果证明。General仍是27B系统的相对短板。

#### 五棵子树的判断质量

每棵子树单独取三次输出，按同一多数规则计算全部1,247条上的Strict ACC。这是 **Unified-Subtree直接判断**，不同于19.3节的节点递归聚合；不是子树对最终系统的独立贡献。

| Root | 8B Strict ACC | 27B Strict ACC | 提升（百分点） |
|---|---:|---:|---:|
| Completeness | 69.29% | 79.79% | +10.51 |
| Visual Grounding | 60.71% | 76.10% | **+15.40** |
| Factuality | 63.35% | 77.87% | **+14.51** |
| Creativity | 69.13% | 79.55% | +10.43 |
| Clarity | 62.47% | 73.46% | +10.99 |

**五棵子树全部改善，提升并非只发生在最后的Arbiter。** 27B最终系统82.68%也高于其最佳单子树79.79%；8B对应为70.49%与69.29%。但子树和Arbiter同时换模型，不能据此分离各自贡献，也不能在测试集上选择最佳子树替换正式系统。

**结论与限制：** 当前支持“完整Rubric + Clean S5-v2在27B上获得明显更高的系统性能，并在其原生基线上增加81条正确判断”；

结果来源：[8B最终报告](../output/evolving_structured_rubrics/vlrb_27b_full_s5/final_report.json)、[27B最终报告](../output/evolving_structured_rubrics/vlrb_27b_full_s5_qwen35/final_report.json)、[27B最终预测](../output/evolving_structured_rubrics/vlrb_27b_full_s5_qwen35/predictions/full.json)、[最后JSON恢复记录](../output/evolving_structured_rubrics/vlrb_27b_full_s5_qwen35/rescue_4096/final_json_recovery.json)。

---

### 19.8 27B + 完整 Rubric + 递归投票：与 Clean S5-v2 对照

#### 设置与最终结果（2026-09-07）

复用19.2节的节点推理和递归算法，只将 Worker 换为 Qwen3.5-27B，并发40，关闭 thinking。固定同一28节点 Rubric（哈希见19.6节）、1,247条样本、$K=3$ 和 A/B 顺序，`temperature=0.5`、常规 `max_tokens=2048`。孩子形成明确多数时覆盖父判断，否则回退父节点，最后五根等权投票。主要对照是 **27B Clean S5-v2**，不是原入口报告中沿用的历史 Phase10。

| 模型与完整 Rubric 推理方式 | 正确数 / 1247 | Overall ACC | Macro ACC | Strict ACC |
|---|---:|---:|---:|---:|
| 8B 递归投票 | 886 | 71.39% | 66.09% | 71.05% |
| 8B Clean S5-v2 | 879 | 72.11% | 65.36% | 70.49% |
| 27B 原生 | 950 | 76.37% | 72.54% | 76.18% |
| **27B 递归投票（本次）** | **988** | **80.00%** | **75.23%** | **79.23%** |
| **27B Clean S5-v2** | **1,031** | **83.08%** | **78.79%** | **82.68%** |

递归版相对27B S5少正确43条，Overall、Macro、Strict分别下降3.08、3.56、3.45个百分点。“S5 → 递归”：纠正22条、改错65条

三次单独评测的Strict分别为79.15%、78.83%、78.75%，S5对应81.96%、82.12%、81.96%，每次均领先；这些是同一次运行的重复/顺序评测，不是三个独立seed实验。原生27B参考结果为950条正确、Strict 76.18%：本次多正确38条，但原生的提示词、$K=5$及解析流程不同，不是等预算对照。

#### 分类差异：Reasoning 降幅最大

以下统一为全部组内样本上的 **Strict ACC**；纠正/改错仍以27B S5为对照。

| 类别 | 样本数 | 27B S5 | 27B 递归 | 纠正 / 改错 | 净变化 |
|---|---:|---:|---:|---:|---:|
| General | 181 | 62.98% | 58.56% | 6 / 14 | −8 |
| Hallucination | 749 | 85.71% | 83.31% | 11 / 29 | −18 |
| Reasoning | 317 | 86.75% | 81.39% | 5 / 22 | −17 |

三个类别均下降；Hallucination损失数量最多，但Reasoning降幅最大（5.36个百分点）。差距并非来自最后一个解析失败，也不局限于单一类别。

#### 孩子仍然有价值，但 Factuality 被孩子多数拖累

复用本次同批节点预测，比较父准则单独判断与完整子树递归判断，均按$K=3$聚合、以全部1,247条计算Strict。不是子树对五树系统的独立贡献。

| Root | 父准则 Strict | 完整子树 Strict | 纠正 / 改错 | 净收益 |
|---|---:|---:|---:|---:|
| Completeness | 67.76% | 79.87% | 203 / 52 | +151 |
| Visual Grounding | 70.81% | 77.47% | 112 / 29 | +83 |
| Factuality | 80.51% | 77.87% | 30 / 63 | **−33** |
| Creativity | 67.84% | 79.47% | 182 / 37 | +145 |
| Clarity | 71.13% | 78.11% | 126 / 39 | +87 |

四棵子树改善，净收益不能跨树相加。同批初始五根等权投票正确913条（Strict 73.22%），完整递归正确988条（79.23%），增加75条，说明孩子准则整体有用。Factuality父准则独自正确1,004条，子树仅971条：当前规则不是父子共同投票，而是孩子有多数便覆盖父判断。这是需要检查的具体退化来源，不能据测试结果直接删除孩子。

#### 离线诊断：差距不全来自最后的 Arbiter

取已生成的27B S5五份子树报告，仅使用其答案做五根等权投票，再沿用原$K=3$聚合；不重新推理、不改变正式结果。

| 推理与汇总方式 | 正确数 | Strict ACC | Overall ACC | 最终覆盖数 |
|---|---:|---:|---:|---:|
| 独立节点 → 递归投票 | 988 | 79.23% | 80.00% | 1,235 |
| S5子树报告 → 五根多数投票（离线诊断） | 1,010 | 80.99% | 83.20% | 1,214 |
| S5子树报告 → Arbiter | 1,031 | 82.68% | 83.08% | 1,241 |

即使不使用Arbiter，S5报告多数投票也比递归多正确22条；在同一批报告上，Arbiter又纠正40条、改错19条，净增加21条（McNemar exact $p=0.00864$），同时扩大覆盖。多数投票的Overall略高，但少正确21条，不能忽略覆盖差异而认为它更优。

#### 推理成本与结论

| 已记录成本 | 27B 递归 | 27B S5 | 递归 / S5 |
|---|---:|---:|---:|
| 逻辑调用 | 104,748 | 22,446 | 4.67倍 |
| API尝试 | 106,667 | 24,563 | 4.34倍 |
| 输入tokens | 158,464,958 | 79,752,044 | 1.99倍 |
| 输出tokens | 39,408,017 | 9,935,301 | 3.97倍 |
| 总tokens | 197,872,975 | 89,687,345 | **2.21倍** |

递归主运行三次累计约23小时13分钟，吞吐75.17次逻辑推理/分钟。

**结论：完整Rubric有用，但在当前27B实验中，递归投票使用约2.21倍tokens，仍比Clean S5-v2少正确43条。** 可能原因包括准则错误相关、多数投票丢失判断理由，以及强父判断被孩子多数覆盖；这些是待验证解释。8B中S5未明显优于递归、27B中却明显领先，也提示推理方式与模型能力可能存在交互，而非某种聚合规则普遍更好。

复现与证据：[运行计划](experiments/phase17-manager-qwen35-27b/plan.md)、[最终报告](../output/evolving_structured_rubrics/vlrb_27b_full_recursive_qwen35/final_report.json)、[配对及成本](../output/evolving_structured_rubrics/vlrb_27b_full_recursive_qwen35/comparison_clean_s5.json)、[节点及子树聚合结果](../output/evolving_structured_rubrics/vlrb_27b_full_recursive_qwen35/retry/combined/logical_votes.json)、[历史输出恢复记录](../output/evolving_structured_rubrics/vlrb_27b_full_recursive_qwen35/history_recovery.json)。离线诊断复用这些结果与19.7节链接的S5最终预测；原始产物不随文档提交。

---

## 20. 后续候选

- **Merge / Drop**：在前三个算子稳定后再处理节点重挂接和历史生存状态；
- **删除 parent 的 Split 消融**：与正式的“保留 parent 并挂载 children”Split 分开；
- **Child Router**：替代固定 EdgeCondition 选择 children，但不得改变 Pairwise 权威 vote；
- **DAG / learned edge / examples 进入 Worker**：分别作为后续独立扩展，不与第一版演化闭环混合。

---

## 21. Review Checklist

- [x] Phase5–18 的传统 Split/Refine 由 node-level Pairwise/local 指标驱动；Phase19 明确作为独立 aligned protocol，保留 Pairwise 反馈用于触发、候选生成与诊断，但以 Unified-Subtree + Global-Arbiter Strict ACC 决定正式提交
- [x] Gate 只控制 status-dependent edges，不覆盖 Pairwise vote
- [x] 第一版采用单 parent Forest，多前提依赖保存在 lineage
- [x] discovery-90 用于反馈、筛选和接受；heldout-500 不参与选择。Init/Final 之外的 checkpoint 结果均标记为 exploratory 后验诊断
- [x] Init Rubric 使用 Multi-Crit Open-ended 五条原文，结构为 5 roots / 0 edges
- [x] Gap 从 all-node Pairwise artifact 计算，不受 traversal 隐藏节点影响
- [x] Split 使用 ErrorSignature → Cluster → Child，并保留 parent
- [x] 第一版 Split edges 全部使用 `ALWAYS`
- [x] Split children examples 不进入 Pairwise Worker
- [x] Split 使用固定 parent scope 上的 Specialized Accuracy，children 平票/全 None 回退 parent
- [ ] Refine forced smoke 通过后才启动 Split+Refine；Create 单独通过后才加入完整 workflow
- [x] Phase 5 Init baseline 与 feedback report 完成
- [x] 根据 Phase 5 实测分布冻结 trigger 与结构阈值
- [x] 冻结精简的 candidate acceptance 与单次 P05 execution 规则
- [x] Split 竞争成功时整体接纳 children，失败时完整回退并记录结构化历史与自然语言归因
- [ ] 三个单算子分别完成端到端验证
- [ ] Phase 7 调度顺序完成 review 并冻结

---

## 22. Rationale Matters 的 VL-RewardBench 对照结果

下表转录自 *Rationale Matters: Learning Transferable Rubrics via Proxy-Guided Critique for VLM Reward Models* 中展示的 VL-RewardBench 结果。数值单位为百分比；粗体保留原图中的重点标记。

| 模型 | Proxy Agent | Data Size | OverallAcc | MacroAcc |
| --- | --- | ---: | ---: | ---: |
| **Proprietary Models** |  |  |  |  |
| GPT-4o (2024-08-06) | None | - | 65.80 | 62.40 |
| Claude-3.5-Sonnet (2024-06-22) | None | - | 55.30 | 53.60 |
| Claude-3.7-Sonnet | None | - | 66.31 | 66.53 |
| **Open-source Models** |  |  |  |  |
| VITA-1.5 | None | - | 16.48 | 16.53 |
| SliME | None | - | 19.04 | 17.64 |
| NVLM-D-72B | None | - | 40.10 | 44.10 |
| Llama-3.2-90B | None | - | 56.20 | 53.90 |
| Qwen2-VL-72B | None | - | 39.50 | 43.00 |
| IXC-2.5-Reward | None | >200k | 65.80 | 70.00 |
| R1-Reward | None | >200k | 71.92 | 71.44 |
| Unified-Reward-SFT | None | >200k | 66.10 | 66.50 |
| Unified-Reward-Think | None | >200k | **73.80** | 72.30 |
| **Our Baselines** |  |  |  |  |
| Proxy-GRM-SFT | None | 10k | 69.53 | 69.18 |
| Proxy-GRM-RL | None | 45k | 72.17 | 71.18 |
| Proxy-GRM-RL | Proxy-SFT | 50k | **75.22** | **73.93** |
| Proxy-GRM-RL | Proxy-RL | 60k | 73.38 | **72.38** |

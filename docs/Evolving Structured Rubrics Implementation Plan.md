# Evolving Structured Rubrics Implementation Plan

> 状态：Phase 0–4 基础设施完成，下一阶段进入 Rubric evolution loop
>
> 更新日期：2026-08-03
>
> 当前分支：`research/evolving-structured-rubrics`
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

初始 Rubric 使用 [Multi-Crit](https://arxiv.org/abs/2511.21662) 为 Open-ended Generation 人工定义的五条准则原文。当前 RLHF-V discovery 数据主要是视觉问答、详细描述和图像内容解释，因此第一版不混入 Verifiable Reasoning 的另一套五条准则。

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

Fitness 使用 `alpha=2.0`、`lambda=0.1`、`len_max=2000`。Phase 5 不用 Fitness 自动触发 operator；Phase 6 Split 将 Fitness 作为候选竞争与接受依据。

对父标准 `c` 产生的 `m` 个候选子标准 `C = {c_1, ..., c_m}`，第一版集体适应度定义为各子标准适应度的算术平均：

$$
\operatorname{Fitness}(C)
= \frac{1}{m}\sum_{i=1}^{m}\operatorname{Fitness}(c_i).
$$

每个子标准的 accuracy 与 coverage 都限制在父标准适用域 $\mathcal{S}(c)$ 上计算：

$$
\operatorname{Acc}_{\mathcal{S}(c)}(c_i)
= \frac{|\{s \in \mathcal{S}(c) : c_i(s) \neq \mathrm{NA} \land c_i(s)=p_s\}|}
{|\{s \in \mathcal{S}(c) : c_i(s) \neq \mathrm{NA}\}|},
$$

$$
\operatorname{Cov}_{\mathcal{S}(c)}(c_i)
= \frac{|\{s \in \mathcal{S}(c) : c_i(s) \neq \mathrm{NA}\}|}{|\mathcal{S}(c)|}.
$$

若某个 child 在 $\mathcal{S}(c)$ 上 support 为0，则该 Split candidate 无法形成有效集体适应度并直接拒绝。第一版算术平均暂不校正 children 之间的 coverage 差异、适用域重叠或 vote 冲突；这些作为后续 collective-fitness 消融与改进项。

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

接受规则按 operator 区分。Refine 与 Create 继续使用 discovery-90 M1 结果门槛；Split 的主要接受条件改为：

$$
\operatorname{Fitness}(\{c_1,\ldots,c_m\}) > \operatorname{Fitness}(c).
$$

Split 竞争成功时保留 parent 并整体接纳当前 children；竞争失败时丢弃整组候选 children，Rubric 保持不变，并记录失败历史。完整 M1 accuracy、coverage、corrected/harmed、child correction/harm、sibling conflict 和 final-valid rate 继续报告，但第一版不参与 Split 硬接受判定。Rubric/Artifact 合法、数据隔离和 trace replay 仍是工程有效性前提。P05 每个候选只运行一次，不设置重复确认。

trigger、operator-specific acceptance 和单次 P05 execution 共同写入版本化配置及 frozen manifest。Refine/Create 以 discovery-90 M1 为接受依据；Split 以 $\mathcal{S}(c)$ 上的集体 Fitness 为接受依据，完整 M1 仅作为诊断。

---

## 9. Phase 6：逐个实现 Refine、Split、Create

每个算子必须单独实现、真实运行、review 和提交；一轮不能同时接受多个不可归因的修改。

### 9.1 Refine

- 根据 node 的 wrong、abstain、Pairwise thought 和代表多模态样本，让 CritiQ-V Manager 先反思再生成两个 description 候选；
- node ID、criterion name、root 顺序和 topology 不变；
- 只刷新目标 node Pairwise outputs；
- discovery-90 完整 M1 before/after 是唯一接受依据。

### 9.2 Split

触发候选为低 accuracy、高 coverage、具有足够 decisive wrong 且错误能形成至少两个有效语义 cluster 的 parent。

```text
decisive wrong samples
  → 逐样本 ErrorSignature
  → Manager 对 signatures 做 2–5 个语义 clusters
  → 每个有效 cluster 生成一个更窄的 child criterion
  → 计算 children 的算术平均集体 Fitness
  → 与 parent Fitness 竞争
```

ErrorSignature 保存 task pattern、visual focus、candidate difference、parent failure 和 suggested subdomain。Cluster 必须引用完整且互斥的 sample IDs，每个 cluster 至少包含5条样本，并说明共同判断失败而不是共同图片主题；每条 wrong sample 必须进入一个 cluster 或显式进入 unclustered。由于 Split 保留 parent，不要求有效 clusters 覆盖固定比例的 wrong samples。

三个 Manager 阶段分别冻结 model、backend pool、输入模态和 request identity。第一轮由本地 Qwen3-VL-8B 生成多模态 ErrorSignature；语义聚类与 child 生成使用硅基流动 Qwen3.5-397B-A17B 的思考模式。后两个阶段使用文本输入，child 阶段读取代表样本的 question/A/B/gold 与 signatures，不发送图片。聚类采用 v1 的一次完整 partition 输出（temperature=0.2），不做事后 repair、自动合并或小簇降级；child 生成使用 temperature=0.7。这样聚类失败保持可见，同时允许仅通过配置替换 Manager，而不改变算子语义。

Split 保留 parent，children 数量严格等于有效 cluster 数量，不设置目标数量：有效 cluster 少于2个则不触发 Split，存在2–5个则分别生成2–5个 children。`max_children=5` 只是结构上限，不允许为了达到上限强行拆分；超出上限的模式进入 unclustered。每个 child 包含 criterion name、description、1–3个对应 cluster 的代表 examples，以及 cluster lineage；examples 只供 Manager、lineage 和审计，Pairwise Worker 不读取 examples。第一版所有新 edges 固定为 `ALWAYS`，不同时优化 Gate/Child Router。无论 children 数量多少，整棵 root subtree 仍最多贡献一票。

Split 竞争时，每个 child 的 accuracy、coverage 和 Fitness 都在 parent 的 $\mathcal{S}(c)$ 上计算，再对所有 child Fitness 求算术平均。集体 Fitness 严格高于 parent Fitness 时，保留 parent 并整体接纳 children；否则整体回退。children 之间 coverage 不均、适用域重叠和 vote 冲突只进入诊断报告，暂不改变第一版算术平均公式。

每次 Split 无论成功或失败都写入版本化 history。记录至少包含 parent ID 与 Rubric hash、trigger statistics、cluster 与 child identities、各 child 的 support/accuracy/coverage/Fitness、集体 Fitness、parent Fitness、decision 和 reasons。失败记录中的原因、cluster 划分和 child 描述作为同一 parent 下一次 Split 的 Manager 参考输入，使重新 Split 能避开已验证失败的划分或描述；历史只提供上下文，不覆盖当前数据上的竞争结果。

首个单算子验证对象固定为 `visual_grounding_and_details`。它在 discovery-90 上的 accuracy/coverage/support/wrong 为 `0.6966/0.9889/89/27`，满足触发条件，并且已有实验观察表明视觉 grounding 错误可以形成多个可解释子域。具体 children 仍必须从这27条真实 wrong samples 的 ErrorSignatures 中归纳，不能预先硬编码类别。

当前已实现的旧 Specialize v1 结果保留为历史基线，其 Manager 协议、trigger、ErrorSignature/cluster/child schema、确定性 Patch、局部 Pairwise artifact 拼装和机制诊断继续复用。下一版将术语统一为 Split，并把旧的完整 M1 accuracy-delta 接受逻辑改为上述 $\mathcal{S}(c)$ 上的算术平均集体 Fitness 竞争。真实运行仍按 signatures → cluster review → child proposal → evaluate 分阶段进行；在新的 discovery-90 Split 实验完成前，不将本项标记为端到端通过。

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

当前 children 更接近多个高度重叠的视觉评审器，而不是边界清晰、互斥的语义子区域；Pairwise Worker 对 `Applicable only when` 的执行仍然偏宽。该问题不在 Split 阶段通过删除较弱 child 解决，而交由后续 `define` 算子收紧描述和适用性。重叠也产生了有效 ensemble 收益：在 137 个 sibling 冲突样本上，Parent Accuracy 为 `0.526`，完整 children 子树为 `0.642`。

#### 结论与边界

> 在 heldout-500 上，由 397B Manager 生成的完整 Visual Grounding children 集合，将父准则固定覆盖区域的准确率从 66.60% 提高到 72.34%（`+5.74 pp`，McNemar exact `p=0.0036`），满足修正后的 Split 接受条件，应整体保留。

需要保留以下结论边界：

- `all_children=0.704` 是“Visual 子树作为唯一 root”的独立结果，不等价于“将 children 接入完整 M1 后”的系统结果；
- 结果说明视觉子树具有较强判别能力，但尚不能单独证明“先判断 Visual Grounding、再执行其他 roots”的层级因果机制；
- children 仍有适用范围过宽和输出偏向 B 的风险，需要在新开发集上由 `define` 与位置交换诊断继续验证；
- heldout-500 已作为一次性 final evaluation 使用，不得再依据该报告选择、删除或改写 children；后续完整 M1 集成实验必须使用新的验证划分。

实验 artifact：`output/evolving_structured_rubrics/rubric_evolution_phase5/phase6_split_signature_qwen35_397b/heldout_visual_only/report.json`。

---

## 11. 后续候选

- **Merge / Drop**：在前三个算子稳定后再处理节点重挂接和历史生存状态；
- **删除 parent 的 Split 消融**：与正式的“保留 parent 并挂载 children”Split 分开；
- **Child Router**：替代固定 EdgeCondition 选择 children，但不得改变 Pairwise 权威 vote；
- **DAG / learned edge / examples 进入 Worker**：分别作为后续独立扩展，不与第一版演化闭环混合。

---

## 12. Review Checklist

- [x] 正式聚合 vote 只来自 Pairwise Worker
- [x] Gate 只控制 status-dependent edges，不覆盖 Pairwise vote
- [x] 第一版采用单 parent Forest，多前提依赖保存在 lineage
- [x] discovery-90 用于反馈、筛选和接受；heldout-500 只用于 Init/Final
- [x] Init Rubric 使用 Multi-Crit Open-ended 五条原文，结构为 5 roots / 0 edges
- [x] Gap 从 all-node Pairwise artifact 计算，不受 traversal 隐藏节点影响
- [x] Split 使用 ErrorSignature → Cluster → Child，并保留 parent
- [x] 第一版 Split edges 全部使用 `ALWAYS`
- [x] Split children examples 不进入 Pairwise Worker
- [x] 第一版 Split 集体 Fitness 使用 child Fitness 算术平均
- [x] Refine、Split、Create 单独通过后才组合 workflow
- [x] Phase 5 Init baseline 与 feedback report 完成
- [x] 根据 Phase 5 实测分布冻结 trigger 与结构阈值
- [x] 冻结精简的 candidate acceptance 与单次 P05 execution 规则
- [ ] Split 竞争成功时整体接纳 children，失败时完整回退并记录可供下一轮参考的 history
- [ ] 三个单算子分别完成端到端验证
- [ ] Phase 7 调度顺序完成 review 并冻结

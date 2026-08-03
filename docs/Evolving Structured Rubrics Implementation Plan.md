# Evolving Structured Rubrics Implementation Plan

> 状态：Phase 0–4 基础设施完成，下一阶段进入 Rubric evolution loop
> 日期：2026-08-03
> 目标集成分支：`research/evolving-structured-rubrics`
> 对应 Idea：[Evolving Structured Rubrics from Multimodal Preferences.md](Evolving%20Structured%20Rubrics%20from%20Multimodal%20Preferences.md)

---

## 1. 实现目标

本项目不是简单地把已有 criteria 排成若干互不相关的线性链，而是要实现一个真正的 Structured Rubric Forest：

```text
Structured Rubric Forest
├── Root 1
│   ├── Child 1.1
│   │   ├── Leaf 1.1.1
│   │   └── Leaf 1.1.2
│   └── Child 1.2
└── Root 2
    └── Child 2.1
```

对每个多模态偏好样本：

1. Root Router 对全部 roots 做一次联合路由，选择一个或多个需要进入的 roots；
2. 只执行 selected roots；禁用 router 的消融系统才执行所有 roots；
3. 每个共享 prefix node 只执行一次；
4. 根据 parent judgement 和 edge condition 判断哪些 children 可以继续；
5. child 再判断自身是否适用；
6. 多个满足条件且适用的 children 可以同时执行；
7. 每棵 selected root subtree 最终最多贡献一票；
8. 多棵 selected root subtrees 聚合为最终 A/B preference。

本计划将原 Idea 中含义冲突的 `Split` 明确拆成两种不同操作：

- `Specialize`：保留 parent，并在其下新增更具体的 children；parent 继续作为共享前缀和运行时 gate。它是算子阶段首个 split-family 算子，但不属于 core v1。
- `SplitReplace`：用 children 替换 parent，parent 退出可执行 forest，仅保留在 lineage 中。core v1 和首轮算子实现均不包含它，后续可作为独立算子或消融实验评估。

`Specialize` 应产生真实的 parent-child structure，而不是把共享 parent 复制到多条独立 chain 中：

```text
错误：
Visual Grounding → Spatial
Visual Grounding → Completeness

正确：
Visual Grounding
├── Spatial
└── Completeness
```

---

## 2. 需要验证的核心主张

### C1：结构化执行是否有效

在 criterion 内容和 worker outputs 完全相同的情况下：

> Cascade execution 是否比 flat voting 具有更好的准确率，或更好的准确率—调用成本权衡？

最低证据：

- 离线 replay 中，四个 2×2 系统使用同一批 node judgements；
- 分别隔离 routing 和 hierarchical aggregation 的贡献；
- 通过 All-Roots Full Cascade 与 Routed-Roots Full Cascade 的对照，单独测量 Root Router 的贡献；
- 在线 lazy execution 中，Cascade 的真实调用量明显减少且准确率基本不下降，或准确率—真实成本权衡优于 Flat；
- 能解释哪些 child 修正了 parent，哪些 child 造成了伤害。

### C2：结构演化是否有效

加入演化算子后：

> Create/Specialize/Refine/Merge/Drop 是否能够改善完整 cascade，而不仅是改善单个 criterion？

最低证据：

- 每个 accepted edit 都有修改前后的完整 cascade 对照；
- proposal data 用于发现问题和生成候选；
- validation data 用于接受或回退；
- 不能只凭新节点自身 accuracy 接受修改。

---

## 3. 为什么不能直接使用旧 A/B/U 缓存进行正式 Cascade

当前 CritiQ-V worker 输出：

```text
A / B / U
```

其中 `U` 同时可能表示：

- criterion 不适用；
- A/B 都通过；
- A/B 都失败；
- 无法区分；
- worker 不确定。

但 cascade routing 需要明确：

- parent 是否适用；
- A/B 是否分别通过 parent；
- 是否满足进入 child 的前置条件。

例如，`visual_coherence` 只应在 A/B 都通过 `multimodal_alignment` 后执行。旧 `U` 无法区分“都通过”和“不适用”。

因此：

- 旧 A/B/U 缓存只用于复现 Original CritiQ-V；
- Structured Flat 和 Structured Cascade 必须共享新的 structured node judgement；
- 关键机制对比采用 routing × aggregation 的 2×2，而不是直接把新 structured prompt 与旧 prompt 混为同一机制比较。

---

## 4. 总体实现路线

```text
Phase 0：冻结结构语义
    ↓
Phase 1：实现 Structured Node Judgement
    ↓
Phase 2：实现 Rubric Tree/Forest 数据结构
    ↓
Phase 3：实现 Root Router 与共享前缀 Cascade Executor
    ↓
Phase 4：使用已有 Criteria 运行静态 Cascade 实验
    ↓
Phase 5：建立统一 Operator 接口
    ↓
Phase 6：逐个实现 Refine / Specialize / Create / Merge / Drop
    ↓
Phase 7：接入完整 Evolution Workflow
```

当前 `feat/evolving-structured-rubrics-core` 分支只完成 Phase 0–4。Phase 5–7 在 core 验收后分别实现。

---

## 5. Phase 0：冻结结构语义

- [x] 定义 node judgement
- [x] 定义 pairwise preference 与 pass/fail status 的关系
- [x] 定义 edge continuation condition
- [x] 定义多 child 行为
- [x] 定义 subtree aggregation
- [x] 定义 Root Router 的输入、输出与失败回退
- [x] 定义多 roots 聚合
- [x] 定义 abstain、tie 和 parse failure
- [x] 使用纯单元测试覆盖所有状态组合

### 5.1 Node Judgement

建议结构：

```python
@dataclass
class NodeJudgement:
    applicable: Literal["yes", "no", "uncertain"]
    status_a: Literal["pass", "fail", "uncertain"]
    status_b: Literal["pass", "fail", "uncertain"]
    pair_preference: Literal['A', 'B', 'tie', 'uncertain']
    evidence_a: str
    evidence_b: str
    parse_ok: bool
```

`consistency_ok` 不属于模型输出或构造参数，而是代码根据上述字段计算得到的只读属性。

三个判断承担不同职责：

- `applicable`：该 criterion 是否适用于当前样本；
- `status_a/status_b`：A 和 B 是否分别满足该 criterion，主要用于 edge routing；
- `pair_preference`：在该 criterion 下更偏好 A、B、平局还是无法确定，直接决定节点的 local vote。

`pair_preference` 不能只由 `status_a/status_b` 推导。即使 A/B 都 pass 或都 fail，worker 仍可能根据满足程度给出 A 或 B；因此必须显式保留 pairwise preference。

Local vote 和 traversal eligibility 使用以下确定性规则：

| parse_ok | applicable | pair_preference | local vote | 是否检查 outgoing edges |
|---|---|---|---|---|
| false | 任意 | 任意 | invalid（不等同 abstain） | 否，终止该分支 |
| true | no | 任意 | inapplicable / abstain | 否，终止该分支 |
| true | uncertain | 任意 | applicability-uncertain / abstain | 否，终止该分支 |
| true | yes | A | A | 是 |
| true | yes | B | B | 是 |
| true | yes | tie / uncertain | abstain | 是 |

还需要进行跨字段一致性校验：

- `status_a=pass && status_b=fail` 时，`pair_preference` 只能为 A；
- `status_a=fail && status_b=pass` 时，`pair_preference` 只能为 B；
- both-pass、both-fail 或含 uncertain status 时，允许 A、B、tie 或 uncertain；
- `applicable!=yes` 时，`status_a=status_b=uncertain` 且 `pair_preference=uncertain`；
- 违反规则的输出记为 invalid，不静默改写为 abstain。

因此在 **Conditional traversal（G1/M1/M2）** 中，只有 `parse_ok=true && applicable=yes` 且通过一致性校验的节点才允许检查 outgoing edges。Parse failure、schema/consistency invalid、inapplicable 和普通 abstain 分开记录。H1 是专门隔离 hierarchical aggregation 的 All-nodes 系统，不使用该终止规则：ancestor invalid/inapplicable 只使该节点 local vote 为 abstain，descendant subtrees 仍独立执行和聚合。

### 5.2 Edge Condition

第一版使用有限枚举：

```python
class EdgeCondition:
    ALWAYS
    PARENT_NONDECISIVE
    PARENT_BOTH_PASS
    PARENT_BOTH_FAIL
```

示例：

```text
multimodal_alignment
    └── visual_coherence
        condition = PARENT_BOTH_PASS
```

Specialize 产生的 residual specialist 可以使用 `PARENT_NONDECISIVE`。第一版不允许 LLM 生成任意执行代码。

在 Conditional traversal 中，Edge truth table 只在 `parse_ok=true && applicable=yes` 时计算：

| EdgeCondition | 精确定义 |
|---|---|
| `ALWAYS` | parent 合法且 applicable 时恒为 true |
| `PARENT_NONDECISIVE` | parent `pair_preference` 为 `tie` 或 `uncertain` |
| `PARENT_BOTH_PASS` | `status_a=pass && status_b=pass` |
| `PARENT_BOTH_FAIL` | `status_a=fail && status_b=fail` |

若 parent parse failure、一致性校验失败、`applicable=no` 或 `applicable=uncertain`，所有 outgoing edges 都视为 false。注意：both-pass 或 both-fail 不再自动等价于 non-decisive，是否有 local preference 由 `pair_preference` 决定。

### 5.3 多 Children

当一个 parent 有多个 children：

1. parent 每个样本只执行一次；
2. 对每条 edge 检查 parent condition；
3. condition 不满足的 child 不调用；
4. condition 满足时执行 child；
5. child 通过自身 `applicable` 决定是否进入该分支；
6. 多个 children 可以同时适用，因此一个样本可以激活多个分支。

该行为已经冻结：第一版不强制选择唯一 child。只要多个 edge conditions 同时满足、对应 children 自身也 applicable，就允许它们同时执行；这些分支最终仍在同一 root subtree 内聚合为最多一票。

### 5.4 Subtree Aggregation

为避免不平衡树中的“最深节点”歧义，executor 使用递归的“一棵 child subtree 一票”算法。每个节点的 `evaluate_subtree(node)` 只返回 A、B 或 abstain：

1. 计算当前节点的 `local_vote`；
2. 对每个满足 edge condition 的 child，递归计算一个 `child_subtree_vote`；
3. 丢弃 abstaining child votes；
4. 若没有 decisive child vote，返回当前节点的 local vote；
5. 若存在 decisive child votes，对这些 child-subtree votes 做等权 A/B 投票；
6. child votes 有唯一多数时返回该多数；
7. child votes 平票时，只回退一次到当前节点的 local vote；当前节点也 abstain 时返回 abstain。

示例：

```text
Root local vote = A
├── Child subtree 1 vote = B
├── Child subtree 2 vote = B
└── Child subtree 3 vote = abstain

Root subtree vote = B
```

即使不同 child branches 深度不同，每个直接 child subtree 也只返回一票。Shared parent 只在当前递归层回退一次，不会因多个 abstaining branches 被重复计票。

### 5.5 Root Router

Root Router 在进入 forest 前进行一次联合多标签路由。它接收：

- 当前样本的 image、question、response A 和 response B；
- 当前 StructuredRubric 的全部 root IDs 与简洁 criterion descriptions；
- root-router prompt/schema version。

建议输出：

```python
@dataclass
class RootRoutingDecision:
    selected_root_ids: tuple[str, ...]
    rationale_by_root: dict[str, str]
    parse_ok: bool
```

与 node judgement 相同，router 的一致性由代码校验，不接受调用方直接设置 `consistency_ok`。

语义约束：

1. 对一个样本只进行一次联合 router 调用，不逐个 root 单独询问；
2. `selected_root_ids` 是多选集合，可以同时包含一个或多个 roots；
3. Router 只判断 criterion 是否值得进入，不得输出或参与 A/B preference；
4. 选择结果必须是当前 `root_ids` 的无重复非空子集；
5. valid decision 只执行 selected roots；
6. parse failure、未知/重复 root ID 或空选择均记为 router invalid，并回退到 all roots，避免因路由失败丢失整个样本；
7. fallback 必须写入 trace，并将额外调用和全 roots 执行计入真实成本；
8. 禁用 Root Router 的基线直接使用 all roots，不产生 router call。

`rationale_by_root` 的键必须恰好等于 `selected_root_ids`，每个值是非空字符串；缺键、多键、空 rationale 或 selected IDs 与 rationale keys 不一致时，均判为 consistency invalid 并触发 all-roots fallback。Schema 禁止输出 A/B preference 或未声明字段。

Router v1 只以 `selected_root_ids` 作为执行选择的唯一真值来源，不输出 `confidence_by_root`。原因是当前尚未冻结 confidence threshold，同时保留 selected IDs 与 confidence 会产生两个可能矛盾的路由信号。若后续研究 confidence-threshold routing，应在新的 semantics version 中加入并作为独立消融，而不是改变 v1 cache/trace 的解释。

Router 能看到 A/B 是为了判断某些 criterion 是否与候选内容相关，但 prompt 只允许输出适用 root IDs。需要用 A/B position swap 检查 root selection consistency，避免 Router 暗中承担 preference prediction。主 M2 推理按给定 ordering 只调用一次 Router；swap audit 是独立的成对诊断运行，其额外 router 与 worker calls/tokens/latency 单独计费和报告，不并入主 M2 单次推理成本。

### 5.6 Root Aggregation

只执行 Root Router 选中的 roots；每棵 selected root subtree 最多一票：

```text
Root 1 subtree → A
Root 2 subtree → B
Root 3 subtree → A

Final → A
```

Router-disabled 消融执行所有 roots。Phase 0 v1 只冻结 `uniform` aggregation：每个 decisive root subtree 的权重为 1；忽略 abstaining subtrees，A 票多则输出 A，B 票多则输出 B，票数相等或没有 decisive selected root 时输出 Tie。Tie 在主 accuracy 中按错误计，并单独报告 tie rate。

`historical_accuracy` weighting 不属于 Phase 0。它延迟到 Phase 4，届时作为次要消融实现；权重只能在 proposal split 上预先冻结，不得用 validation/test 更新。加入该模式时必须补充零权重、非法权重、加权平票和版本兼容测试。

### 5.7 预计代码

新增：

```text
critiq/structured/__init__.py
critiq/structured/judgement.py
critiq/structured/semantics.py
critiq/structured/aggregation.py
tests/__init__.py
tests/structured/__init__.py
tests/structured/test_semantics.py
```

Phase 0 冻结并公开：

```python
STRUCTURED_SEMANTICS_VERSION = "1.0.0"
```

Phase 1–4 新增的 judgement cache、router cache、trace 和 experiment config 必须记录该版本并纳入 cache key。缺失版本或 major version 不一致时禁止静默 replay；任何会改变 edge、traversal、consistency 或 aggregation 执行结果的修改都必须提升 major version。

统一测试命令使用 Conda `critiq` 环境：

```powershell
C:\Users\wenqx\miniconda3\envs\critiq\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

顶层 `tests/__init__.py` 保证 `python -m unittest discover` 也能够递归发现测试，避免零测试假绿。

### 5.8 验收标准

- [x] 每种 parent state 都有确定行为
- [x] both-pass/both-fail 时仍能保留 A/B pairwise preference
- [x] 不一致的 status 与 pair_preference 会被识别为 invalid
- [x] subtree 聚合接口只接收一次 parent local vote，并在 child 平票/全 abstain 时只回退一次
- [x] 每条满足条件的 child edge 独立返回 true，允许多个 children 同时执行
- [x] 每个 child subtree 和每棵 selected root subtree 最多向上一层贡献一票
- [x] uniform root aggregation 只统计 selected roots，unselected roots 不影响结果
- [x] RootRoutingDecision 支持选择一个或多个 roots，并按 rubric root 顺序规范化
- [x] valid routing 只返回 selected roots；router invalid 时可追踪地回退到 all roots
- [x] Root Router decision schema 不包含 A/B preference
- [x] Router v1 只使用 selected_root_ids，不保留无执行语义的 confidence
- [x] selected-root aggregation 拒绝空选择、字符串输入和非法 root ID
- [x] semantics version 已冻结并公开
- [x] 默认 unittest discovery 能发现并运行 structured tests
- [x] tie、abstain、inapplicable、applicability uncertain、consistency invalid 与 parse failure 语义分离

以上是 Phase 0 的纯语义/API 验收。真实 executor 中“parent 每个样本只调用一次”和“共享前缀只执行一次”的调用计数测试仍属于 Phase 3，不在 Phase 0 中提前实现。

---

## 6. Phase 1：实现 Structured Worker Output

- [x] 新增 StructuredMultiModalPairEvaluator
- [x] 保留原 MultiModalPairEvaluator 行为
- [x] 设计 structured worker prompt
- [x] 解析 NodeJudgement
- [x] 实现 deterministic local vote
- [x] 保存节点级原始输出
- [x] 测试 A/B position swap

### 6.1 实现思路

保留现有：

```python
MultiModalPairEvaluator
```

新增：

```python
StructuredMultiModalPairEvaluator
```

Worker 输出：

```json
{
  "applicable": "yes",
  "status_a": "pass",
  "status_b": "fail",
  "pair_preference": "A",
  "evidence_a": "...",
  "evidence_b": "..."
}
```

代码使用 `pair_preference` 生成 A/B/abstain；`status_a/status_b` 用于 cascade routing。dataclass、worker JSON、cache 与 trace 统一使用 snake_case 小写字段名。解析器同时执行 schema 和跨字段一致性校验，不允许用 pass/fail 状态覆盖 worker 明确给出的合法 pairwise preference。

Phase 1 的 schema parser 只负责 JSON 结构、字段类型和 enum token；跨字段冲突仍保留为 `parse_ok=true && consistency_ok=false`。重试耗尽后，若至少得到过一个 schema-合法输出，则保留最后一个 consistency-invalid judgement；只有从未得到 schema-合法输出时才生成 `parse_ok=false` 的全 uncertain sentinel。两种失败都保存原始响应和尝试次数。

Structured 输出 artifact 冻结以下独立版本：

```python
STRUCTURED_SEMANTICS_VERSION = "1.0.0"
STRUCTURED_WORKER_SCHEMA_VERSION = "1.1.0"
STRUCTURED_WORKER_PROMPT_VERSION = "1.0.0"
STRUCTURED_WORKER_PARSER_VERSION = "1.0.0"
```

artifact 保存 criteria snapshot、sample ID、ordered input fingerprint、每个 node 的原始响应和 judgement，以及 Flat-uniform sanity answer。加载时严格检查版本和矩阵完整性；replay 只从保存的 local decisions 重算结果，不创建 Agent。

artifact 通过 `StructuredWorkerRequestSpec` 冻结真实请求身份：完整 prompt SHA-256、model、显式且不含敏感信息的 backend/checkpoint ID、decoding config、截断长度、图片传输模式和数据字段名。每个样本的 fingerprint 使用实际发送的截断后 A/B；嵌入式本地图片使用文件内容 SHA-256，而不再只哈希路径。多个实验系统共享输出前，调用方必须执行 `assert_request_compatible()`。
artifact 构造时会防御性复制并递归冻结 node-output mappings 和 decoding metadata；`to_dict()` 与 `save_json()` 在序列化前再次校验完整矩阵。
### 6.2 实验隔离

后续保留两类输出协议：

| 系统族 | Worker output | 用途 |
|---|---|---|
| Original CritiQ-V | A/B/U | 旧系统参考 |
| Structured 2×2 | 相同 NodeJudgement | 比较 Flat、Hierarchical-only、Gating-only 与 Full Cascade |

四个 Structured 系统共享同一输出协议。offline replay 还必须共享同一批节点输出，以便分别隔离 conditional routing 与 hierarchical aggregation；`Original CritiQ-V` 只作旧系统参考。

### 6.3 预计代码修改

```text
critiq/evaluator.py
critiq/structured_prompts.py
critiq/structured/version.py
critiq/structured/worker_output.py
历史 Structured Worker v1 只保留协议与 artifact replay 兼容代码，不再保留独立 smoke CLI。
tests/structured/test_evaluator.py
tests/structured/test_worker_output.py
tests/structured/test_smoke.py
```

### 6.4 验收标准

- [x] Structured output 能区分 applicability 与 A/B status
- [x] Structured output 显式保留 pairwise preference
- [x] status 与 pair_preference 的跨字段一致性可校验
- [x] Structured 2×2 系统可以共享同一版本化输出 artifact
- [x] 原 MultiModalPairEvaluator 的输出与测试保持不变
- [x] 解析失败不会静默转换为 abstain
- [x] 输出可保存、加载和无模型重放
- [x] Flat-uniform sanity 输出 accuracy、coverage 和 criterion coverage
- [x] 默认 unittest discovery 运行全部 Phase 0/1 测试

- [x] 使用显式 worker backend ID 重新生成 Schema v1.1.0 真实 VLM smoke artifact
### 6.5 Phase 1 验证结果

- Conda `critiq` 环境：Phase 1 完成时 52 个 `unittest` 全部通过；
- `compileall`、`import critiq`、`import critiq.structured` 和顶层 Structured evaluator 导入通过；
- Fake Agent 覆盖 schema/consistency retry、并发乱序恢复、artifact round-trip、无模型 replay 和旧 A/B/U evaluator 回归；
- 真实 VLM smoke 使用 discovery 样本 `rlhfv-000769`、`visual_grounding` criterion 和 temperature 0，对原顺序与 A/B swap 各调用一次；
- 最终两次输出均为 schema-valid、consistency-valid、`both-pass + tie`，`ab_swap_consistent=true`；
- Schema v1.1.0 smoke 使用服务端可重复核验的 backend identity：`vllm@0.11.0|model=Qwen/Qwen3-VL-8B-Instruct|root=/media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct|max_model_len=25800`；
- artifact 严格加载、`assert_request_compatible()` 和无模型 Flat replay 均通过，且输出中不包含 API key、base URL 或不稳定的 `created` 字段。
- 第一版具体 JSON 示例曾导致模型复制示例 evidence 且两次都选择 A；已在 v1 commit 前移除具体答案示例，改为字段级 enum 约束，避免示例锚定。

Schema v1.1.0 真实 smoke artifact：

```text
output/structured_worker_smoke/predictions_schema_v1_1.json
  SHA256: CDB253FD8118AE39E9D95FA58D8298E8B643CDD3BB8113D6F446174BE000AE61
output/structured_worker_smoke/predictions_schema_v1_1_report.json
  SHA256: C0F1B2379AF95514D5D451F747C2BB35DCFDB6559CD2BE339DCC5BCB7752CFA5
```

### 6.6 10 样本批量诊断

使用 discovery 前 10 个样本、同一条 `visual_grounding` criterion 和 temperature 0；每个样本分别执行原序与 A/B swap，共 20 次真实 VLM 调用。

- schema success：10/10；单次 judgement consistency：10/10；
- applicability swap consistency：10/10；status swap consistency：4/10；pair-preference swap consistency：4/10；严格完整 judgement swap consistency：3/10；
- 原序 decision coverage 为 0.50，tie 计错的 overall accuracy 为 0.40，decisive 样本上的 selective accuracy 为 0.80；
- swap 结果恢复到原始候选方向后，decision coverage 为 0.50，overall accuracy 为 0.30，selective accuracy 为 0.60；
- 原序 preference 分布为 `tie=1, uncertain=4, B=5`，所有 decisive 输出都选择 B，提示需要进一步检查位置偏置和候选内容分布；
- 去除 “candidate A/B” 标签后，evidence 相似度不低于 0.9 的样本为 1/10，即 `rlhfv-000769`。大多数样本不再复用近乎相同的 evidence；
- 这里的 accuracy 使用整对 RLHF-V 人类偏好标签，而不是 criterion-specific gold，只能作为 sanity 指标，不能单独解释为 `visual_grounding` 的真实准确率。

结论：structured schema 与失败追踪已经稳定，但 worker 的 status 和 pair preference 对 A/B 顺序仍不够稳健。单样本 smoke 通过只证明协议可运行，不能证明语义判断已经有效；进入更大规模效果实验前应先迭代 prompt 或推理协议，并继续用 swap consistency 作为门槛。

批量诊断 artifact：

```text
output/structured_worker_smoke/predictions_batch_10_schema_v1_1.json
  SHA256: 6538B9F61EBA2BFAED4F6FECF89F7AB8A01B5E5A61BA144A74FC9D535C0D50B7
output/structured_worker_smoke/predictions_batch_10_schema_v1_1_report.json
  SHA256: C23A08CDD18E0C08B5DB0BD9792EC07CA3E86111AA994B1312BF0800550DEDC4
```

### 6.7 Prompt 优化探索与回退

在相同 10 个 discovery 样本上，对 v1、v2、v2.1 都进行了 original/swap 测试；v2 和 v2.1 还增加了 exact repeat。结果简表如下：

| Prompt | Schema | Repeat 标签一致 | Strict swap consistency | Coverage |
| --- | ---: | ---: | ---: | ---: |
| v1 | 20/20 | 未测 | 3/10 | 0.50 |
| v2 | 29/30 | 8/10 | 4/10 | 0.20 |
| v2.1 | 30/30 | 10/10 | 3/10 | 0.50 |

v2/v2.1 改善了部分输出纪律和同顺序复现性，但没有稳定提高 swap invariance；v2.1 还把 visual-grounding 专用判断细节写入了全局模板，不适合作为后续多 criteria 共用的泛化 prompt。因此当前 active prompt 回退到通用 v1，v2/v2.1 原文与 artifact 仅作为探索记录保留。

开放问题：通用 structured prompt 如何保持 criterion-agnostic、如何区分 backend 重复波动与位置偏差、以及如何提高 swap consistency 而不依赖大量 tie。它们留到后续独立 prompt/swap 实验处理，不在当前结构与执行器实现中继续展开。

实验报告：

```text
output/structured_worker_smoke/predictions_batch_10_schema_v1_1_report.json
output/structured_worker_smoke/predictions_prompt_v2_batch_10_schema_v1_1_report.json
output/structured_worker_smoke/predictions_prompt_v2_1_batch_10_schema_v1_1_report.json
```

---

## 7. Phase 2：实现 Rubric Tree/Forest

- [x] 实现 RubricNode
- [x] 实现 RubricEdge
- [x] 实现 StructuredRubric
- [x] 支持共享前缀
- [x] 支持多 children
- [x] 支持多个 roots
- [x] 保存 lineage
- [x] 实现序列化与结构校验

### 7.1 数据结构

```python
@dataclass(frozen=True)
class RubricCriterionSnapshot:
    name: str
    description: str
    score: float

@dataclass(frozen=True)
class RubricNode:
    node_id: str
    criterion: RubricCriterionSnapshot
    examples: tuple[Mapping[str, JSONValue], ...]
    lineage: Mapping[str, JSONValue] | None
```

```python
@dataclass(frozen=True)
class RubricEdge:
    parent_id: str
    child_id: str
    condition: EdgeCondition
```

```python
@dataclass(frozen=True)
class StructuredRubric:
    nodes: Mapping[str, RubricNode]
    edges: tuple[RubricEdge, ...]
    root_ids: tuple[str, ...]
    schema_version: str
    semantics_version: str
```

### 7.2 第一版结构限制

- 一个 node 最多一个 parent；
- 一个 parent 可以有多个 children；
- 支持多个 roots；
- 不允许环；
- 不允许跨 root 共享 child；
- 暂不实现多 parent DAG。

这是第一版有意保留的简化：先用 tree/forest 验证单父节点 root-to-leaf cascade 是否有效，不在基础实现中同时引入通用 DAG、合取 gate 或复杂依赖求值。

结构构建必须遵守以下约束：

- 只为“一个 child 明确依赖一个 parent”的关系建立 edge；
- 如果 criterion 同时要求多个前提，不得为了适配 forest 而把这些前提任意串成一条伪链；
- 第一版将这类关系记录为 `unsupported_multi_parent_dependency`，对应 criterion 默认保留为独立 root；
- 在查看 heldout-500 结果前冻结哪些节点作为 root、哪些单父依赖建立 edge。

当静态实验显示多前提 criterion 经常被错误激活、造成明显 child harm，或无法表达的依赖占比较高时，再评估升级为 multi-parent DAG 或 synthetic conjunction gate。该升级属于后续改进，不作为 core 分支的阻塞项。

### 7.3 预计代码

```text
critiq/structured/
├── __init__.py
├── judgement.py
├── schema.py
├── validation.py
└── version.py

tests/structured/
├── test_schema.py
└── test_validation.py
```

### 7.4 验收标准

- [x] 能表示共享 root 的多分支结构
- [x] 共享 parent 只保存一次
- [x] JSON round-trip 无损
- [x] 环、悬空 child、重复 ID 会明确失败
- [x] 原 Criterion checkpoint 不受影响

### 7.5 Phase 2 验证结果

- Rubric schema 使用独立的 `STRUCTURED_RUBRIC_SCHEMA_VERSION = "1.0.0"`，并同时冻结 Phase 0 semantics version；
- `node_id` 与 `criterion.name` 相互独立但均全局唯一，提供双向稳定映射；
- node、edge、forest、criterion snapshot、examples 与 lineage 均为防御性复制后的不可变对象；
- forest 构造时严格拒绝环、悬空边、重复 edge、多 parent、错误 roots、跨 root 共享 child 和重复 criterion name；
- JSON loader 严格验证字段、版本、拓扑和 canonical SHA-256；node/edge 存储顺序不改变 hash，root 顺序和实际 rubric 内容会改变 hash；
- Phase 1 prediction artifact 可按完整 criterion name/description 集合进行无模型兼容性校验；
- Conda `critiq` 环境共 71 个 `unittest` 全部通过，其中新增 19 个 Phase 2 测试；原 Phase 0/1 测试全部保持通过；
- 本阶段未修改 evaluator、workflow、router 或旧 `Criterion` checkpoint 协议，也未调用模型、图片或网络。

---

## 8. Phase 3：实现 Root Router 与 Cascade Executor

- [x] 实现一次联合调用、多选 roots 的 Root Router
- [x] valid routing 只从 selected roots 开始
- [x] router invalid 时回退到 all roots 并保存原因
- [x] parent 每个样本只执行一次
- [x] 根据 EdgeCondition 展开 children
- [x] 根据 child applicability 过滤分支
- [x] 支持多分支递归
- [x] 聚合 subtree vote
- [x] 聚合 final vote
- [x] 保存完整 execution trace
- [x] 支持 offline replay 与 online lazy execution
- [x] 分开统计反事实访问节点数与真实模型成本

### 8.1 执行流程

```text
if root_router_enabled:
    routing = root_router.route(sample, rubric.roots)
    selected_roots = routing.selected_root_ids
    if routing is invalid:
        selected_roots = rubric.root_ids
else:
    selected_roots = rubric.root_ids

function evaluate_subtree(node, backend):
    judgement = backend.evaluate_once(node)

    if judgement.parse_ok == false:
        return abstain
    if judgement.consistency_ok == false:
        return abstain
    if judgement.applicable != yes:
        return abstain

    local_vote = deterministic_vote(judgement)
    child_votes = []

    for each outgoing edge:
        if edge_condition_matches(judgement, edge.condition):
            child_vote = evaluate_subtree(edge.child, backend)
            if child_vote is A or B:
                child_votes.append(child_vote)

    if child_votes is empty:
        return local_vote

    if child_votes have a unique A/B majority:
        return that majority

    return local_vote if local_vote is A or B else abstain

for each root in selected_roots:
    root_vote = evaluate_subtree(root, backend)

final_vote = aggregate_root_votes(root_votes)
```

该伪代码与 §5.4–§5.6 完全一致：Root Router 只选择起始 roots，不产生 A/B vote；每个直接 child subtree 递归返回一票，不再使用“全局最深节点”规则。相同 executor 通过不同 backend 分别执行离线 replay 和在线 lazy inference。

伪代码中的 `return abstain` 只表示“该节点向聚合层投影为不投票”，不代表把 invalid 重新标注成普通 abstain。节点 trace 必须用 `outcome_reason` 保留原始原因，至少区分 `decisive`、`tie`、`preference_uncertain`、`inapplicable`、`applicability_uncertain`、`parse_failure` 和 `consistency_invalid`。

上面的伪代码描述 Conditional executor（G1/M1/M2）。H1 使用单独的 All-nodes hierarchical replay：

```text
function evaluate_all_nodes_subtree(node, backend):
    judgement = backend.evaluate_once(node)
    local_vote = project_to_vote_or_abstain(judgement)

    child_votes = []
    for each child, ignoring edge condition:
        child_vote = evaluate_all_nodes_subtree(child, backend)
        if child_vote is A or B:
            child_votes.append(child_vote)

    return aggregate_children_with_parent_fallback(child_votes, local_vote)
```

因此 H1 的 ancestor parse failure、consistency invalid 或 inapplicable 不阻止 descendants 执行；它只让 ancestor local vote 为 abstain。这样 B1 vs H1 只改变 flat vs hierarchical aggregation，不引入 conditional stopping。

### 8.2 Trace

每个样本至少保存：

```json
{
  "sample_id": "...",
  "root_routing": {
    "enabled": true,
    "selected_root_ids": ["visual_grounding", "multimodal_alignment"],
    "fallback_to_all_roots": false,
    "outcome_reason": "valid"
  },
  "roots": [
    {
      "root_id": "visual_grounding",
      "visited_nodes": [
        {
          "node_id": "visual_grounding",
          "depth": 0,
          "applicable": "yes",
          "status_a": "pass",
          "status_b": "pass",
          "pair_preference": "tie",
          "local_vote": "abstain",
          "outcome_reason": "tie"
        },
        {
          "node_id": "spatial",
          "depth": 1,
          "applicable": "yes",
          "status_a": "pass",
          "status_b": "fail",
          "pair_preference": "A",
          "local_vote": "A",
          "outcome_reason": "decisive"
        }
      ],
      "subtree_vote": "A"
    }
  ],
  "final_vote": "A"
}
```

### 8.3 两种执行模式

第一阶段使用 **offline replay** 验证算法机制：

- 预先得到所有样本—节点的 structured judgements；
- Flat、Hierarchical-only、Gating-only 和 Full Cascade 共享完全相同的 node outputs；
- 对 Routed-Roots Full Cascade 预先得到并冻结 RootRoutingDecision；
- executor 只决定哪些缓存结果会被访问，以及如何聚合；
- 此时的 visited nodes / avoided nodes 是反事实调用量，只能证明算法层面的访问差异，不能作为真实 token、latency 或 API cost。

第二阶段使用 **online lazy execution** 验证真实效率：

- 不预先填满所有节点输出，只在 traversal 实际访问节点时调用 worker；
- Root Router 先进行一次真实联合调用，只为 selected roots 启动 subtree execution；
- 被 gate 跳过的节点不得产生模型请求；
- Router 和 node worker 分别保存 model calls、input/output tokens、latency、估算费用、cache hit/miss 和 retries，端到端指标包含两者；
- 使用与 Flat 相同的模型、prompt、decoding config 和缓存策略；
- 在线结果用于支持真实效率结论，离线结果用于隔离 routing/aggregation 机制。

### 8.4 缓存键

至少包含：

```text
sample_id
ordered A/B hashes
node description hash
structured output schema version
worker model/checkpoint
prompt version
decoding config
parser version
```

同一个共享 prefix node 对同一样本只执行一次。

Root Router 使用独立缓存键，至少包含：

```text
sample_id
image/question/ordered A/B hashes
ordered root IDs and descriptions hash
rubric schema/version
router model/checkpoint
root-router prompt/schema version
decoding config
parser version
```

### 8.5 预计代码

```text
critiq/structured/
├── root_router.py
├── root_router_prompts.py
├── executor.py
├── aggregation.py
├── cache.py
└── trace.py

tests/structured/
├── test_root_router.py
├── test_executor.py
├── test_branching.py
├── test_aggregation.py
└── test_cache.py
```

### 8.6 验收标准

- [x] 共享 parent 不重复执行
- [x] Router valid 时不调用 unselected root subtrees
- [x] Router invalid 时回退到 all roots，并区分失败原因
- [x] Router 支持一次选择多个 roots
- [x] 多 children 能根据条件进入或跳过
- [x] 多个 eligible children 可同时执行
- [x] 不适用 child 不贡献判断
- [x] 一个 root subtree 最多一票
- [x] 多 roots 聚合可重现
- [x] trace 能重建每一步决定
- [x] fake evaluator 覆盖全部执行路径
- [x] H1 在 ancestor invalid/inapplicable 时仍执行 descendants
- [x] offline 2×2 与 M2 可共享完全相同的 node outputs
- [x] online lazy execution 不调用被 gate 跳过的节点
- [x] online 指标包含 Root Router 与 node workers 的真实成本
- [x] 反事实 visited-node counts 与真实 calls/tokens/latency 分开记录

### 8.7 Phase 3 验证结果

- 新增独立 Structured Root Router、B1/H1/G1/M1/M2 executor、online/offline backend、版本化 JSON cache、调用计量以及可严格重放的 execution trace；
- Conda `critiq` 环境共 98 个 `unittest` 全部通过，覆盖 Router 解析与重试、五种系统、并发上限、cache 冷暖/刷新/损坏、offline artifact 共享、成本统计、trace 篡改拒绝及 smoke rubric profiles；
- discovery 样本 `rlhfv-000769` 的真实 VLM smoke 通过：M1 实际访问 2/3 nodes；M2 冷运行包含 1 次 Router 和 2 次 Worker 调用，共 3126 tokens；暖缓存重放结果相同，3 次 cache hit、0 次新增 API 调用；
- M1/M2 均依据 parent judgement 跳过 `visual_error_diagnosis`，未访问 child 没有产生 Worker 请求；M2 Router 输出 schema/consistency valid，original/swap 均选择全部两个 roots，因此本次真实 smoke 未触发 root pruning，该行为只由离线/Fake backend 测试覆盖；
- smoke trace 均可在不创建 Agent、不读取 cache 的情况下重建为相同 final preference；artifact 未保存 API key、base URL 或 gold answer。
- Phase 3 review 后补充：Offline Router 强制校验样本 fingerprint；trace 加载和 replay 均核验 node 成本汇总，并按 backend source 核验 Router/Node 当前成本与 generation provenance；Root Router request metadata 递归不可变；真实 smoke 的 root pruning 改为 `exercised/result` 三态报告。
- 单一 smoke 入口新增 `branching_v1` profile：使用 `R1/C1/G1` 等结构 node ID 表示 3 roots、10 nodes、7 条 `PARENT_NONDECISIVE` edges，并保留 `minimal_v1` 用于旧结构复现；该 profile 的真实 API 调用尚未执行。

---

## 9. Phase 4：使用已有 Criteria 运行静态 Cascade

- [x] 固定使用 exp4 final heldout 实际评估的 17 条 criteria
- [x] 根据 description 中的前置依赖人工构造 forest
- [x] 在查看测试结果前冻结结构与 Root Router prompt/config
- [x] 使用 Structured Evaluator 重新推理
- [x] 使用相同 NodeJudgement 运行 routing × aggregation 2×2
- [x] 增加 All-Roots Full Cascade vs Routed-Roots Full Cascade
- [x] 先完成 offline replay，再运行 online lazy execution
- [x] 输出 accuracy、coverage、反事实访问量、真实成本和 path statistics
- [x] 分析 child correction 与 child harm

### 9.1 为什么需要重新推理

旧缓存只能复现 Original CritiQ-V，不能可靠判断 `PARENT_BOTH_PASS` 等条件。静态 Cascade 使用已有 criterion 文本，但通过新 structured prompt 重新获得：

```text
applicable
status_a
status_b
pair_preference
```

这一阶段不生成、不改写 criterion。

实际实验验证了“重新推理”不仅改变输出格式，也可能改变 criteria 的执行语义。exp4 的 17 条 criteria 是**在旧 A/B/U Worker 下演化并筛选出来的**；Structured Worker 同时判断 applicability、两个独立 status、pair preference 和 evidence 后，B1 accuracy 从 exp4 的 0.696 降至 0.606。因此，**重新推理虽然为 EdgeCondition 提供了必要状态，却引入了尚未解决的 Worker/criteria 语义漂移**。这个问题是 Phase 4 的主要研究结论，而不是普通实现错误。

当前暂时将 heldout-500 作为 Phase 4 的阶段性测试集，不在 core 阶段重新划分数据。manual forest、structured prompt、Root Router prompt、edge conditions、聚合规则和实验配置必须在首次查看该阶段结果前冻结。

如果首次结果被用于修改 prompt、topology、edge conditions、aggregation 或阈值，那么 heldout-500 后续只承担工程迭代与诊断用途；每次运行必须保存版本化配置和结果，不能继续把后续结果解释为严格 unseen test。当前不额外划分最终测试集，但在形成论文级最终结论前需要重新确定独立评估协议。

### 9.2 静态结构来源

第一版固定使用 exp4 final heldout 实际评估的 17 条 criteria。唯一机器可读来源为：

```text
output/rlhfv_exp4_dis90_val100_n10_wp-final-heldout500_e10/
└── final_heldout_raw_prediction.json
    └── criteria  # exactly 17
```

中文核对表为同目录下的 `final_criteria_table_zh.md`。构建脚本必须从上述 JSON 的 `criteria` 数组读取 name、description 和 score，不得改用 `epoch_final.json.current_criteria`。17 个 criterion 全部保留为同一 node pool，再根据下面的规则人工组织为单父节点 root-to-leaf forest。

只根据 criterion description 中明确存在、且可表示为单父关系的依赖语言建立 edges。例如：

```text
Visual Coherence is evaluated only after
Multimodal Alignment is confirmed.
```

可以构建：

```text
Multimodal Alignment
    ↓ PARENT_BOTH_PASS
Visual Coherence
```

若描述为“Completeness 同时依赖 Visual Grounding 与 Factual Consistency”，第一版不把它强行线性化为 `Visual Grounding → Factual Consistency → Completeness`。该依赖记录为 `unsupported_multi_parent_dependency`，Completeness 暂时作为独立 root；后续根据实验结果决定是否升级为 DAG 或 conjunction gate。

共享前缀结构：

```text
Multimodal Alignment
├── Visual Coherence
└── Semantic Specificity
```

`Ambiguity Resolution` 等多前提节点只在 projected_v0 诊断结构中投影到该 parent，不属于 strict_v0 的正式 edges。

实际冻结了两个版本：

- `strict_v0`：17 nodes、15 roots、2 edges，只保留 description 中可明确表示为单父关系的依赖，作为正式 Gate 拓扑；
- `projected_v0`：17 nodes、9 roots、8 edges，将多前提关系投影到一个主要 parent，仅作为离线诊断消融。

两者使用完全相同、未简写的 17 条 criterion name、description 和 score；projected_v0 不参与 Phase 4 正式通过判定。

### 9.3 实验系统

核心实验先禁用 Root Router，从 all roots 开始，采用 internal routing × aggregation 的 2×2。所有系统均使用相同的 Structured NodeJudgement；区别只在是否根据 parent condition 条件执行 child nodes，以及是否按 forest 层级聚合。

| ID | Routing | Aggregation | 精确定义 |
|---|---|---|---|
| B1 Flat | All nodes | Flat | 执行全部 nodes，对所有 applicable 且 decisive 的 local votes 等权投票 |
| H1 Hierarchical-only | All nodes | Hierarchical | 执行全部 nodes；ancestor invalid/inapplicable 不终止 descendants，聚合时忽略 parent edge condition，每个 child subtree 递归返回一票 |
| G1 Gating-only | Conditional | Flat | parent edge condition 决定是否调用 child，child applicability 决定是否贡献并继续；对所有 visited 且 decisive 的 local votes 等权投票 |
| M1 All-Roots Full Cascade | Conditional | Hierarchical | 从 all roots 开始，根据 parent condition 条件访问 children，并使用 subtree/root 层级聚合 |

成对比较的解释：

- B1 vs H1：隔离 hierarchical aggregation；
- B1 vs G1：隔离 conditional routing；
- H1 vs M1：在 hierarchical aggregation 下测 routing；
- G1 vs M1：在 conditional routing 下测 hierarchical aggregation；
- B1 vs M1：只作为完整系统差异，不能单独归因于某一个机制。

在 2×2 之外增加 Root Router 对照：

| ID | Root start | Internal routing | Aggregation | 作用 |
|---|---|---|---|---|
| M1 All-Roots Full Cascade | All roots | Conditional | Hierarchical | 不使用 Root Router 的完整内部 cascade |
| M2 Routed-Roots Full Cascade | Selected roots | Conditional | Hierarchical | 使用一次联合多选 Root Router 的最终 core 系统 |

M1 vs M2 只隔离 Root Router 的贡献。M2 的端到端成本必须包含一次 router call；Router invalid 回退到 all roots 的样本仍属于 M2，并单独报告。

补充参考与消融：

| ID | 系统 | 作用 |
|---|---|---|
| B0 | Original CritiQ-V：旧 A/B/U + Flat | 原系统参考，不进入 2×2 机制归因 |
| A1 | Full Cascade without parent status conditions | 只依赖 child applicability，检验 edge condition 的额外作用 |

offline replay 中，B1/H1/G1/M1/M2 必须共享完全相同的 node outputs，M2 另外使用预先冻结的 RootRoutingDecision。online lazy execution 至少运行 B1、M1 和 M2：B1 提供全节点成本参考，M1 测内部 cascade，M2 测包含 Root Router 的最终端到端系统；三者使用同一 node worker 配置。

### 9.4 指标

主指标：

- preference accuracy；
- program coverage；
- tie rate；
- A/B position consistency。

离线执行指标（反事实，不作为真实成本）：

- 平均访问节点数；
- 反事实 avoided-node calls；
- 平均最大深度；
- root 直接终止比例；
- edge 激活比例；
- leaf 使用率。

在线执行指标（真实成本）：

- 实际 worker calls；
- Root Router calls 与 node worker calls；
- input/output tokens；
- wall-clock latency；
- 可选的估算 API cost（本地 vLLM 实验记为 `null`）；
- cache hit/miss；
- retries 与 parse failures。

Root Router 指标：

- 平均与分位数 selected-root count；
- 每个 root 的 selection rate；
- router invalid / empty / unknown-ID / fallback rate；
- A/B swap 下的 root-selection consistency；
- M1 vs M2 的 accuracy、coverage、calls、tokens 与 latency 差异。

结构诊断：

- child corrects parent；
- child harms parent；
- child 被执行但 inapplicable；
- parent condition 阻止正确 child；
- parent condition 阻止错误 child；
- subtree 内部冲突。

### 9.5 Go 条件

工程正确性条件必须全部满足：

- [x] offline B1/H1/G1/M1/M2 使用完全相同的 node outputs
- [x] offline/online M2 使用已冻结的同版本 Root Router prompt/config
- [x] shared prefix 每个样本只执行一次
- [x] 四个 2×2 系统重放完全可复现
- [x] M1/M2 除 Root Router 与 selected roots 外保持相同
- [x] trace 可以重建 final vote
- [x] online M1/M2 不调用未访问节点，并记录 router + workers 的真实 tokens/latency；本地 vLLM 不估算美元 cost
- [x] structured-output final-valid rate ≥ 95%

在当前 heldout-500 阶段性测试集上冻结以下两个工程 Gate；满足任一项表示 cascade 机制达到继续研究的最低条件：

1. **Accuracy Gate**：`Acc(M2)-Acc(B1) >= +1.0` percentage point；
2. **Efficiency Gate**：online lazy execution 中，M2 相对 B1 的实际总 calls 或 tokens 减少 ≥ 20%，且 `Acc(M2)-Acc(B1) >= -1.0` percentage point。

上述 Gate 已在首次 heldout-500 推理前写入 frozen manifest，不再根据结果调整。

工程 Gate 是必要条件而非充分条件。如果 Structured B1 明显弱于 exp4 B0，仍必须先解决 Worker/criteria 语义对齐，不能直接进入 Operator 阶段。

同时必须满足：

- [x] 至少存在 10 个可解释的 child-corrects-parent 样本（实际 53 个）
- [x] 至少一个非 root node 的 activation rate ≥ 5%（两个 strict child 均为 48%）
- [x] 报告 paired bootstrap 95% CI 和逐样本差异；本阶段不以显著性作为工程合并门槛

若两个 Gate 都不满足，状态记为 **REVISE**，优先检查 prompt、edge conditions 和 aggregation，不进入自动 Operator 实现。

#### 9.5.1 实际实验结果

最终实验版本为 `phase4_static_v2_cross_sample10_global30`，使用 heldout-500 和 strict_v0 共享 NodeJudgement：

| Variant | Accuracy | Coverage | Tie rate |
|---|---:|---:|---:|
| B1 Flat | 0.606 | 0.926 | 0.074 |
| H1 Hierarchical-only | 0.612 | 0.928 | 0.072 |
| G1 Gating-only | 0.600 | 0.924 | 0.076 |
| M1 All-roots Cascade | 0.604 | 0.922 | 0.078 |
| M2 Routed-roots Cascade | 0.622 | 0.916 | 0.084 |

M2 相对 B1 提升 1.6 percentage points，paired bootstrap 95% CI 为 `[-0.6, +3.8]` percentage points。M2 含 Router 共 4,680 次 logical calls，相对 B1 的 8,500 次减少 44.94%；Worker calls 减少 50.82%；总 tokens 从 14,879,803 降至 10,146,617，减少 31.81%。Accuracy、efficiency、child correction 和 child activation 四项 Gate 均通过，工程状态为 `PASS_BOTH`。

50 样本 swap audit 也已完成：criterion-level 平均一致性约 61.3%，系统级一致性为 B1 56%、H1/G1 52%、M1/M2 54%，Router exact-selection consistency 为 70%。这些结果只作为位置敏感性风险记录，不作为 Phase 4 硬 Gate。

这里的 `PASS_BOTH` 只表示 M2 相对**同一 Structured Worker 产生的 B1**具有更好的准确率—成本权衡，不表示当前完整方法优于 exp4。

### 9.6 失败解释

以下诊断一旦用于修改系统，后续 heldout-500 运行均记为工程迭代，不再视为首次冻结测试：

- B1 已明显弱于 B0：先修正 structured worker prompt；
- B1 正常但 H1 下降：hierarchical aggregation 可能引入错误权重或错误回退；
- B1 正常但 G1 下降：conditional routing 可能错误跳过有效节点；
- H1/G1 正常但 M1 下降：routing 与 hierarchical aggregation 可能存在负向交互；
- M1 正常但 M2 下降：Root Router 可能遗漏有效 roots 或受到 A/B position 影响；
- M2 selected roots 很少且 coverage 下降：Root Router 过度过滤；
- M2 selected roots 接近 all roots：Root Router 缺少实际筛选能力；
- 很少进入 child：parent condition 过严；
- 几乎执行所有 nodes：applicability/edge 没有起作用；
- 多 branch 冲突严重：subtree aggregation 需要调整；
- offline avoided calls 较多、online 成本不下降：检查真实 lazy execution、缓存与额外控制开销；
- M1 有成本收益但 M2 没有：检查 Router 自身 tokens/latency 是否抵消节点节省；
- 只降低真实 calls/tokens、不提升 accuracy：结构可能主要提供效率价值。

本次实验实际命中了第一种情况：B1=0.606，明显低于 exp4 B0=0.696。当前最合理的解释是 Structured Worker 同时承担 applicability、status、pair preference 和 evidence，改变了旧 criteria 在 A/B/U Worker 下形成的判别边界。M2 虽然在这批 Structured outputs 上把 accuracy 提升到 0.622，但无法弥补 Worker 协议造成的 9.0-point 基线下降。

因此 Phase 4 的结论分为两层：

- **执行机制层面**：Root Router 和 Cascade Executor 已跑通；M2 相对 Structured B1 有小幅 accuracy 改善和显著真实成本下降。
- **方法有效性层面**：Structured Worker 尚未与 exp4 对齐，当前结果不能支持直接进入 Operator 阶段。下一步必须先做同样本、同 criteria、同模型配置下的旧 A/B/U Worker vs Structured Worker 对照。

#### 9.6.1 基线恢复：投票与 Gate 双通道解耦

离线对齐报告进一步确认，下降在 Cascade 之前已经发生：Structured Worker 的 node decisive rate 从 62.3% 降至 49.2%，与 exp4 node vote 的完全一致率只有 60.1%，并且 Structured B1 的 Gold-B accuracy 从 73.4% 降至 50.8%。主要原因是旧 criteria 与新 schema 的执行语义冲突、both-pass/both-fail tiebreaker 引入噪声、一次生成同时承担五类判断，以及明显的 A 位置偏置；最终 parse/consistency failure 约 0.1%，不是主因。

修复采用两个互不覆盖的通道：

- **Pairwise Vote Worker**：逐字复用 exp4 prompt/postfix，只输出并解析 A/B/None；该结果是 aggregation 的唯一 local vote。
- **Gate State Worker**：只在 Conditional traversal 遇到 `PARENT_BOTH_PASS/FAIL` 的已访问 parent 时调用，只输出 applicability、status_a、status_b，不输出 winner 或 evidence。
- `PARENT_NONDECISIVE` 只读取 Pairwise Worker 明确返回的 None/U；parse failure 不视为 nondecisive。
- B1/H1 不调用 Gate；skipped child 和 unselected root 不产生 Pairwise 或 Gate 请求。
- 新执行产物使用独立 Trace v2，旧 Structured Worker v1.1、Phase 4 artifact、Trace v1 和 cache 继续只读兼容。

后续可将 **Child Router** 作为 internal routing 的改进策略：由模型根据当前样本、parent criterion、parent pairwise vote 和直接 children 的 descriptions，一次联合输出一个或多个 `selected_child_ids`，直接决定进入哪些 children，而不再通过 Gate status 匹配手工 `EdgeCondition`。第一版暂不采用该方案，继续使用更简单、可解释的 `Gate State Worker + EdgeCondition` 作为规则基线；后续可对比 `M1/M2-Rule` 与 `M1/M2-ChildRouter` 的 accuracy、路由稳定性和真实成本。Child Router 必须支持多 child 同时选择，invalid 输出应可追踪地回退 all children，并且不得修改 Pairwise Worker 的权威 A/B/None 投票。

当前代码已完成协议、evaluator、离线/在线 backend、独立 cache、Dual Cascade Executor、Trace v2 和 shared-output 实验入口。离线 `worker_alignment_report` 已复现 exp4 accuracy=0.696、Structured accuracy=0.606、node agreement=60.071%；随后 exact legacy Pairwise P05 在统一 backend pool 上得到 B1=0.688，达到预设的 baseline recovery 门槛。由此确认主要下降来自 Structured Worker 与旧 criteria 的执行语义漂移，而不是 dataset、criteria 文本或 Cascade executor。

在正式运行前进一步冻结以下工程约束：Pairwise parser 必须像 exp4 一样同时解析 `answer` 和 `thought`，缺少任一字段才触发结构重试；JSON 合法但 answer token 非 A/B/None 时保留旧行为——投票无效但不重试。`thought` 随 Pairwise artifact 保存，供后续 manager reflection 使用。真实阶段由 `stage_status.json` 强制执行 `freeze → pairwise-pilot → pairwise-500 → gate-500 → offline → B1/M1/M2 online → finalize`，失败或风险状态不能被静默跳过。可恢复 cold-online 成本从独立 cache namespace 的唯一 `generation_metrics` 重建，并与当前恢复进程新增成本分开报告，避免“已写 cache、未写 sample trace”的调用丢失。Trace v2 replay 必须重新验证完整 edge 数量、child 访问、canonical nodes、root visited/avoided、Router resolved、counterfactual metrics、aggregation 和成本汇总。

### 9.7 当前正式代码与实验结论

```text
experiments/evolving_structured_rubrics/
├── analysis.py
├── experiment_utils.py
├── rubric_factory.py
├── run_shared_output_pool.py
├── worker_alignment.py
└── configs/
    └── shared_output_pool.example.json
```

实现与验证状态：

- [x] 唯一正式 CLI 支持 `freeze`、endpoint equivalence、P05/P00 Pairwise、Gate、Router、reserve repeats、单节点修复、offline replay 和 report；
- [x] frozen manifest 校验 rubric、prompt、模型与 backend pool identity；
- [x] 两个 endpoint 采用 first-available slot 调度，并分别限制 endpoint 与全局并发；
- [x] 生成阶段支持 per-node cache、可恢复执行、进度日志与 ETA；
- [x] Pairwise Vote 与 Gate State 协议、cache、backend、Dual Executor 和 Trace v2 已实现，旧 Trace v1 保持可加载；
- [x] 离线 Worker drift 报告复现 0.696 → 0.606 和 60.071% node agreement；
- [x] endpoint equivalence check 通过：Pairwise agreement 93.33%，Gate/Router agreement 100%；
- [x] P05 baseline recovery 通过：B1=0.688；P00 B1=0.682；
- [x] 同温度的 B1/H1/G1/M1/M2 严格共享相同 Pairwise artifact，accuracy 只来自 offline replay；
- [x] Gate v2 与 Root Router 各生成一份 temperature=0 的 eval550 artifact；
- [x] reserve-50 的 P05 三次独立重复已完成；
- [x] 完整 raw artifact 已移到仓库外归档，Git 只保留精简结果与 hash；
- [x] Phase 4 结论为 `PASS_BASELINE_RECOVERY / REVISE_CASCADE / REVISE_ROOT_ROUTER`。

---

## 10. 当前 Core 分支完成标准

`feat/evolving-structured-rubrics-core` 合并前必须满足：

- [x] 原 CritiQ-V evaluator 未被破坏
- [x] Structured evaluator 区分 applicability、A/B status 与 pairwise preference
- [x] Structured output 通过 schema 与跨字段一致性校验
- [x] 支持一个 parent 多个 children
- [x] 多个 eligible children 可以同时执行
- [x] 第一版每个 node 最多一个 parent，多前提依赖不被强行线性化
- [x] 共享 parent 每个样本只执行一次
- [x] 支持多个 roots
- [x] Root Router 一次联合调用可选择一个或多个 roots
- [x] Router valid 时只进入 selected roots，invalid 时可追踪地回退到 all roots
- [x] 一棵 root subtree 最多贡献一票
- [x] 支持有限枚举 EdgeCondition
- [x] 支持完整 trace 和 prediction cache
- [x] 支持 offline replay 与真实 online lazy execution
- [x] 使用 exp4 final heldout 的固定 17 条 criteria 构造静态 root-to-leaf forest
- [x] offline B1/H1/G1/M1/M2 使用同一 node outputs
- [x] online B1/M1/M2 得到包含 Router 成本的真实 calls/tokens/latency 对比
- [x] 分别得到可解释的 internal routing、aggregation、Root Router 与 accuracy/cost 对比
- [x] 尚未实现自动演化算子（按阶段边界有意保留）
- [x] Pairwise P05 在统一 backend pool 上恢复 B1=0.688，达到预设 baseline recovery 门槛
- [x] aggregation vote 已与 Structured status 解耦，Pairwise Worker 固定复用 exp4 prompt
- [x] exact legacy parser 保存 `thought`，阶段状态机、可恢复 cold-cost provenance 和 Trace v2 严格重放已闭环

预计新增：

```text
critiq/structured_prompts.py

critiq/structured/
├── __init__.py
├── judgement.py
├── schema.py
├── validation.py
├── root_router.py
├── root_router_prompts.py
├── executor.py
├── aggregation.py
├── cache.py
└── trace.py

tests/structured/
├── test_semantics.py
├── test_judgement.py
├── test_schema.py
├── test_validation.py
├── test_evaluator.py
├── test_root_router.py
├── test_executor.py
├── test_branching.py
├── test_aggregation.py
└── test_cache.py

experiments/evolving_structured_rubrics/
├── experiment_utils.py
├── rubric_factory.py
├── analysis.py
├── run_shared_output_pool.py
└── configs/
    └── shared_output_pool.example.json
```

预计修改：

```text
critiq/evaluator.py
critiq/__init__.py
pyproject.toml
```

当前阶段不修改：

```text
critiq/workflow.py
critiq/router.py
```

Root Router 是 Evolving Structured Rubrics 的新组件，实现在 `critiq/structured/root_router.py`，不复用或修改现有 CritiQ-V 的 `critiq/router.py`，避免改变旧系统行为。

Phase 0–4 的推理与验证基础设施已经闭环。下一阶段可以进入 Rubric evolution loop，但应把当前静态 Cascade 与 Root Router 结果作为需要改进的反馈，而不是已经成立的方法结论。Child Router 继续作为后续候选优化，本轮不实现。

---

## 11. Phase 5：统一 Operator 接口

静态 Cascade 验证后，再建立算子框架：

```python
class EvolutionOperator:
    def should_trigger(context) -> bool:
        ...

    def propose(context) -> list[EditCandidate]:
        ...

    def apply(rubric, candidate) -> StructuredRubric:
        ...

    def rollback(rubric, candidate) -> StructuredRubric:
        ...
```

每个候选都产生：

```text
R_before
R_after
```

节点指标用于触发和诊断，完整 Cascade 指标用于接受或回退：

$$
\Delta_{cascade}
=
Acc(R_{after})-Acc(R_{before})
$$

---

## 12. Phase 6：逐个实现演化算子

版本范围需要明确区分：

- core v1 只完成 Phase 0–4，不实现任何自动演化算子；
- 算子阶段先实现 Refine，用于打通 propose/apply/evaluate/rollback 闭环；
- Specialize 是随后实现的首个 split-family 结构算子；
- SplitReplace 暂不实现。

### 12.1 Refine

- [ ] 从 node 错误样本生成新描述
- [ ] 保持 node_id 和 edges 不变
- [ ] 比较 old/new node metrics
- [ ] 比较完整 cascade
- [ ] 支持 rollback

先实现 Refine，用它验证 operator 的 propose/apply/evaluate/rollback 闭环。

### 12.2 Specialize

- [ ] 聚类 parent 的错误或 abstain 样本
- [ ] 生成多个 children
- [ ] 为 child 生成 applicability
- [ ] 生成有限枚举 EdgeCondition
- [ ] 保留 parent 作为共享 prefix 与运行时 gate
- [ ] 将 children 接到共享 parent 下
- [ ] 比较修改前后的完整 cascade
- [ ] 保存 lineage 和 specialize failure history

其结构变换为：

```text
before: parent

after:  parent
        ├── child 1
        └── child 2
```

parent 不退出可执行 forest，children 也不替换 parent。重点检查共享 prefix、children residual coverage、correction/harm 和碎片化。

`SplitReplace` 不属于 core v1 或首轮算子实现：如果后续需要测试“删除 parent、只保留 children”是否更好，应另建算子和消融，不能与 `Specialize` 共用同一语义。

### 12.3 Create

- [ ] 将 gap 定义为完整 cascade 错误或全弃权样本
- [ ] 从 gap cluster 生成新 root
- [ ] 独立检查新 root support
- [ ] 加入 forest 后验证完整 cascade

不再使用原文中有问题的 `Acc_s(c)`。

### 12.4 Merge

- [ ] 用 node judgement agreement 找候选
- [ ] 检查共同适用域大小
- [ ] 生成 merged node
- [ ] 处理 children 重挂接
- [ ] 验证完整 cascade

### 12.5 Drop

- [ ] 找到低质量 node 或 subtree
- [ ] 明确删除后的 child 处理方式
- [ ] 暂时删除并重放 cascade
- [ ] 完整性能不下降才接受

---

## 13. Phase 7：接入完整 Evolution Workflow

只有单算子分别通过后，才修改 `critiq/workflow.py`：

```text
评估当前 StructuredRubric
    ↓
计算 node-level metrics
    ↓
计算 cascade-level metrics
    ↓
检测 operator trigger
    ↓
每次选择一个 operator/candidate
    ↓
proposal split 生成修改
    ↓
validation split 比较完整 cascade
    ↓
接受或回退
    ↓
保存 structure、lineage、trace 和 metrics
```

一轮最多接受一个结构修改，避免多个算子同时变化后无法归因。

---

## 14. 推荐提交顺序

```text
1. test: 定义结构化量规的节点与级联语义
2. feat: 添加结构化多模态节点判断
3. feat: 添加Rubric Tree/Forest数据结构
4. feat: 添加一次联合多选的Root Router
5. feat: 实现共享前缀的级联执行器
6. feat: 添加subtree与selected-root聚合
7. exp: 使用exp4最终17条criteria运行offline结构实验
8. exp: 运行包含Root Router成本的online lazy inference
9. docs: 记录静态cascade结果与下一阶段决策
```

---

## 15. Review Checklist

实现前需要确认以下设计决定：

- [x] Root Router 对所有 roots 做一次联合多选，只执行 selected roots；router invalid 时回退到 all roots
- [x] 多个 children 可以同时 applicable，不提前强制选择唯一分支
- [x] 每棵 selected root subtree 最终只产生一票，避免共享 prefix 重复赋权
- [x] `PARENT_NONDECISIVE` 按 `pair_preference in {tie, uncertain}` 判断
- [x] 新增 structured worker output，并显式保存 pairwise preference；不使用旧 A/B/U 直接承担 cascade routing
- [x] Core 分支只实现静态 cascade 与 Root Router，operators 后续逐个实现
- [x] 第一版采用 forest，每个 node 最多一个 parent，暂不支持通用 DAG
- [x] 多前提依赖不强行改写成单链，暂记为 unsupported 并将 criterion 保留为独立 root
- [x] 原 Split 在首个 split-family 算子中明确为保留 parent 的 Specialize；SplitReplace 暂不实现
- [x] Phase 4 使用当前 heldout-500 作为阶段性测试集；首次查看前冻结配置，结果驱动修改后转为工程迭代用途
- [x] offline replay 只验证算法机制，真实效率必须由 online lazy execution 验证
- [x] online lazy execution 属于 core 合并条件，成本包含 Root Router 与 node workers
- [x] routing × aggregation 使用 Flat / Hierarchical-only / Gating-only / Full Cascade 的 2×2
- [x] M1 All-Roots Full Cascade vs M2 Routed-Roots Full Cascade 单独隔离 Root Router
- [x] 静态 forest 固定使用 exp4 final heldout 实际评估的 17 条 criteria
- [x] Phase 0 只实现 Uniform root voting；historical accuracy weighting 延迟到 Phase 4 作为消融

完成 Review 后，再从 Phase 0 开始实现并逐项更新本文件中的 checklist。

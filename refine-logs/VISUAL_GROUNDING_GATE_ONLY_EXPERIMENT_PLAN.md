# Visual Grounding Gate-only Experiment Plan

**Problem**: Visual Grounding 子树的四个 children 在所有样本上同时参与投票，语义不适用的 child 可能制造 sibling conflict，并稀释正确的局部判断。

**Method Thesis**: 将 criterion description 中的 `Criterion focus`、`Applicable only when` 和 `Not applicable when` 解释为可执行的隐式边，由一个同-root Gate Worker 在推理时选择适用 children，可以在不修改 Rubric、criterion 投票或聚合规则的情况下减少错误投票干扰。

**Date**: 2026-08-16

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：同-root Gate 能改善 Visual Grounding 子树的选择性聚合 | 直接验证当前“所有 children 同时投票”的聚合瓶颈 | 在相同 Pairwise votes 上，Gate 相对 all-children 在 discovery-90 与 heldout-500 均获得正的 corrected-harmed，并保持覆盖率 | B1、B3 |
| C2：收益来自样本级语义路由，而不是重新推理或选择固定强子集 | 排除新 Pairwise 推理、后验挑选 children 等混杂因素 | 所有 criterion votes 逐项复用；Gate 优于 discovery 选定的 best fixed subset，且交换 A/B 后路由基本不变 | B2、B3 |

**Anti-claim to rule out**：Gate 只是一个隐式偏好 judge，借由看出 A/B 胜者来挑选会投正确票的 child，而不是真正判断 criterion applicability。

## Frozen Protocol

### Source artifacts

- Rubric：`phase10_five_root_locked_split_refine_v1/final/rubric.json`，冻结其 hash、22 个 nodes、5 个 roots 与全部 edges。
- Pairwise Worker：复用 Prompt v2 + available-slot 动态调度产生的 discovery-90 和 heldout-500 predictions；不生成任何新的 parent/child A/B vote。
- 目标子树：`init_02_visual_grounding_and_details` 及其四个 direct children。
- 其余四个 roots：Rubric、votes 和 subtree aggregation 全部保持不变。
- discovery 阶段不得读取 heldout 数据、预测或指标。heldout-500 已被前序实验访问，因此本实验明确标记为 exploratory paired diagnostic。

### Gate scope

Gate 只能激活 Visual Grounding root 下的四个 direct children：

1. `peripheral_and_subtle_detail_grounding`
2. `visual_evidence_priority_over_assumptions`
3. `presence_and_action_verification`
4. `compositional_structure_grounding`

description 中指向其他 root/node 的文字只写入 `outside_local_scope` 诊断字段，绝不跨 root 激活。

### Compact Gate prompt

同一 parent 的固定 System Prompt 包含：

- 简短的 applicability-router 角色约束；
- parent ID、name 和 focus；
- 按 Rubric edge 顺序排列的四个 children；
- 每个 child 的 name、ID、focus、apply 和 exclude；
- 固定 JSON schema。

路由字段直接从 description 的三个固定段落确定性提取，不通过 LLM 摘要；`Decision rule`、examples、gold、历史指标、Pairwise votes 和 Worker thought 均不进入 Gate prompt。

动态 User Prompt 只包含 image、question、Candidate A/B，以及一句“仅执行路由，不判断 A/B”。候选回答被声明为不可信数据，不得作为指令执行。

Gate 输出：

```json
{
  "decisions": {
    "<child_id>": {
      "status": "applicable | not_applicable | uncertain",
      "reason": "brief routing reason"
    }
  },
  "outside_local_scope": []
}
```

每个 direct child 必须恰好出现一次。`reason` 保留用于首轮行为审计，限制为一句短理由。Gate 不得输出 A/B preference。

### Gate execution

- 模型：Qwen3-VL-8B-Instruct。
- temperature：0.2；每个样本单次采样，不进行多 replicate 投票。
- 每个样本一次 Gate 请求；单一冻结 endpoint，避免混入后端差异。
- 输出上限：2048 tokens；每个 `reason` 只能是一句不超过 30 个英文词的短理由，输出完整 JSON 后立即结束，不得重复 JSON 或继续分析。
- 优先使用 JSON constrained decoding，失败时最多重试 3 次。
- `applicable` 和 `uncertain` 均激活；`not_applicable` 不激活。
- 合法空集合：直接回退 parent。
- Gate transport/parse 最终失败：退化为激活全部四个 children，使技术失败等价于 Control，而不是人为提高结果。
- request identity 冻结 prompt version、Rubric/contract hashes、child order、model、temperature、max tokens、endpoint identity 和 schema version。

### Fixed aggregation

对样本 \(x\)，令 Gate 激活集合为 \(G(x)\)。只统计其中有效的 A/B child votes：

$$
\hat y_{G}(x)=
\begin{cases}
A,&N_A(G(x))>N_B(G(x)),\\
B,&N_B(G(x))>N_A(G(x)),\\
c_p(x),&N_A(G(x))=N_B(G(x)).
\end{cases}
$$

平票、全部 active children 输出 None，或者 Gate 合法地激活空集合时均回退 Visual Grounding parent。该 Visual subtree vote 随后按现有等权五-root M1 规则与其他四个 root 聚合。

## Paper Storyline

- Main paper must prove：语义边驱动的局部路由能减少同 root children 的错误投票稀释，并改善最终偏好判断。
- Appendix can support：Gate 理由、outside-local-scope 引用、位置一致性、吞吐与缓存日志。
- Intentionally cut：跨-root routing、root-level virtual parent、Gate 与 Rubric 联合演化、修改 criterion description、重新生成 Pairwise votes。

## Experiment Blocks

### B0：Offline freeze 与路由合同审计

- **Claim tested**：实验只改变 child activation。
- **Why**：防止错误复用不同 Prompt、Rubric 或 dataset 的 predictions。
- **Checks**：Rubric hash；四个 direct children；三个段落可确定性提取；Prompt v2 prediction 的 dataset/criterion/request identity；discovery/heldout 隔离。
- **Success criterion**：90 与 500 样本的 parent/child votes 完整且 hash 匹配；Gate contract 不含 Decision rule、gold、votes 或指标。
- **Priority**：MUST-RUN。

### B1：Discovery-90 Gate-only 主实验

- **Claim tested**：C1。
- **Compared systems**：
  1. Parent-only；
  2. All-children：现有四 child 全激活；
  3. Best-fixed-subset：在 discovery-90 穷举 \(2^4\) 个固定 child subsets；
  4. Dynamic Gate：每个样本由 Gate 选择 subset；
  5. Oracle dynamic routing：使用 gold 穷举每样本 subsets，仅作理论上界，不属于可部署系统。
- **Primary metrics**：Visual subtree ACC（分母固定为全部 90）、corrected、harmed、net corrected。
- **Secondary metrics**：Coverage、covered ACC、完整五-root M1 ACC、平均激活 child 数、空路由率、uncertain 率、sibling conflict、每个 child 的 activation/support/accuracy。
- **Success criterion**：Gate valid rate 至少 99%；Gate 相对 all-children 的 net corrected > 0；Coverage 下降不超过 1pp；完整 M1 不退化。
- **Failure interpretation**：若 Gate 接近 all-children，边条件缺少选择性；若低于 parent/all-children，Gate 误读边或 `uncertain` 策略过宽；若低于 best fixed subset，动态路由的额外复杂度尚无必要。
- **Priority**：MUST-RUN。

### B2：Gate 是否真正做 applicability routing

- **Claim tested**：C2 与 anti-claim。
- **Position audit**：按冻结 sample order 选择 20 个 discovery 样本，将 A/B 文本交换后重新调用 Gate；交换前后 child statuses 理论上应保持不变。
- **Metrics**：active-set exact match、per-child status agreement、Jaccard、含 A/B preference 的违规输出数。
- **Qualitative audit**：各检查存在/动作、细粒度背景、外部假设、构图四类至少 3 个样本，并检查 exclusion 是否优先于 broad focus。
- **Success criterion**：active-set exact match 至少 90%，不存在直接 A/B preference 字段，定性样本未出现系统性的 winner-driven routing。
- **Failure interpretation**：若交换敏感，应先修 Gate prompt/解码，不访问 heldout。
- **Priority**：MUST-RUN。

### B3：Heldout-500 一次性泛化

- **Claim tested**：C1、C2。
- **Prerequisite**：B0 完整通过；B1 获得正 net corrected；B2 通过位置审计；冻结 Gate prompt、best fixed subset、fallback 和全部 hashes。
- **Compared systems**：Parent-only、All-children、discovery-frozen Best-fixed-subset、Dynamic Gate、Oracle upper bound。
- **Primary metrics**：Visual subtree ACC 和完整五-root M1 ACC，均以 500 为分母。
- **Paired metrics**：Gate vs All、Gate vs Best-fixed 的 corrected/harmed、net corrected、exact McNemar；Wilson 95% CI。
- **Secondary metrics**：Coverage、conflict reduction、activation distribution、parse fallback、延迟和请求吞吐。
- **Success criterion**：Dynamic Gate 相对 All-children 的 heldout net corrected > 0、Coverage 降幅不超过 1pp；若同时优于 frozen Best-fixed-subset，才支持“样本级动态 Gate 必要”。McNemar 不显著时只表述为 exploratory evidence。
- **Failure interpretation**：discovery 提升而 heldout 下降表示 Gate 过拟合 90 条幻觉数据的边界；优于 All 但不及 Best-fixed 表示选择性有用，但动态路由尚未证明必要。
- **Priority**：MUST-RUN after gate。

### B4：效率与缓存诊断

- **Claim tested**：紧凑 System routing contract 不引入不可接受的推理成本。
- **Metrics**：请求数、总时间、requests/min、P50/P90/P99 latency、input/output tokens、retry/parse-failure；vLLM prefix-cache hit rate 由服务端日志人工记录。
- **Interpretation**：该块只报告工程代价，不参与 Gate 性能 pass/fail。
- **Priority**：NICE-TO-HAVE，但运行时自动产出基础指标。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Main Risk |
|---|---|---|---|---|---|
| M0 | 冻结与离线重放 | freeze + audit | hashes、90/500 votes、四个 contracts 全匹配 | 无模型请求 | artifact identity 混用 |
| M1 | 端到端 smoke | 20 discovery Gate calls | 20/20 schema valid，无法跨 root 激活 | 约数分钟 | schema/多模态请求错误 |
| M2 | discovery 主实验 | 90 primary + 20 swapped | B1、B2 均通过才开放 heldout | 110 个短输出请求 | Gate 偷做偏好判断 |
| M3 | discovery 报告与 freeze | offline aggregation | 冻结 prompt、best subset、rubric/prediction hashes | 无模型请求 | 后验调参 |
| M4 | heldout 一次性测试 | 500 Gate calls | 结果生成后禁止改 prompt/route policy | 约 5–20 分钟，受尾部请求影响 | exploratory heldout 已重复使用 |
| M5 | 最终报告 | offline paired analysis | 所有逐样本系统采用同一 gold/order | 无模型请求 | 指标分母混淆 |

## Artifact Layout

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase13_visual_grounding_gate_only_v3/
    frozen_manifest.json
    routing_contract.json
    offline_audit.json
    smoke/
    discovery90/
      gate_predictions.json
      position_audit.json
      systems.json
      report.json
    heldout500/
      frozen_manifest.json
      gate_predictions.json
      systems.json
      report.json
    final_report.json
    final_report.md
```

建议 CLI stages：

```text
visual-gate-freeze
visual-gate-smoke
visual-gate-discovery
visual-gate-report
visual-gate-heldout
visual-gate-final-report
```

## Tests and Acceptance Gates

- contract extraction 只读取 focus/applicable/not-applicable，排序和 hash 确定。
- Gate 输入不含 gold、Pairwise outputs、Worker thought、metrics、examples 或 Decision rule。
- 输出必须覆盖全部且仅覆盖四个 direct child IDs；状态枚举严格校验。
- cross-root ID 即使出现在文字引用中也不能进入 active set。
- `applicable/uncertain` 激活，合法空集合回退 parent，技术失败回退 all-children。
- Gate aggregation 的 all-children replay 必须逐样本复现冻结 Control。
- 只替换 Visual subtree vote；其他四个 root votes 必须逐项相同。
- Best-fixed-subset 只能在 discovery 选择，并在 heldout freeze 前记录 hash。
- swap audit 只交换 A/B 内容，不改变图像、question、parent 或 child contracts。
- discovery stages 无法读取 heldout；heldout stage 校验所有冻结 hashes。
- ACC 始终报告全样本分母，同时单独报告 Coverage 与 covered ACC。
- unit tests 使用 fake multimodal backend；完整 unittest、compile 和 regression checks 通过。

## Compute and Data Budget

- Discovery：90 次主 Gate + 20 次交换顺序审计；无需 22×90 Pairwise 重跑。
- Heldout：500 次 Gate；无需 22×500 Pairwise 重跑。
- Manager/397B：0 次。
- 最大成本不是 GPU 数量，而是长尾请求和多模态图像预处理；2048-token 输出上限用于降低合法 JSON 被截断的风险，短 reason 和“JSON 后立即结束”约束用于限制异常解码。

## Risks and Mitigations

- **Gate 隐式判断 A/B**：禁止读取 child votes，输出无 A/B 字段，并做 swap consistency audit。
- **自然语言边存在交叉引用**：硬限制 direct children；外部引用仅诊断。
- **`uncertain` 导致接近 all-children**：首版保守激活并报告 uncertain 率；本实验中不后验修改策略。
- **技术失败虚假提高 ACC**：失败强制回退 all-children。
- **fixed-subset 已足够**：将其作为强简单基线；若 Gate 不超过它，不声称动态路由必要。
- **heldout 重复使用**：明确标记 exploratory，未来应在新的跨分布数据上做 confirmatory test。

## Final Checklist

- [ ] Main comparison isolates activation only
- [ ] Pairwise Prompt v2 predictions are hash-frozen and reused
- [ ] Same-root-only routing is enforced
- [ ] Gate cannot observe gold or child votes
- [ ] Fixed subset and oracle bounds are reported
- [ ] Position-invariance audit passes before heldout
- [ ] Full-sample ACC and Coverage are not conflated
- [ ] Heldout remains a one-shot exploratory stage

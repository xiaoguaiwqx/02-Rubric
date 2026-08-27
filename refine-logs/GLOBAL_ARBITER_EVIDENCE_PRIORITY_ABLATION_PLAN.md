# Global Arbiter 证据形式与 factuality-first 顺序消融计划

> 状态：实验 runner、配置、CLI stages 与 focused tests 已实现，尚未运行模型实验。由于规划时当前协作线程无法创建 reviewer 子任务，本计划仍标记为 **[pending external review]**。

## 1. 研究问题与可支持的结论

本实验只回答两个有顺序条件的问题：

1. **V1 − V0**：在相同的 neutral arbitration 下，将五棵子树的 `A/B/None` 标签替换为五份完整报告，带来多少收益？
2. **V2 − V1**：在相同的 full-report 输入下，加入 Arbiter 层显式的 factuality-first 规则，带来多少额外收益？

允许的 claim 仅为：

- “在 neutral arbitration 下，完整报告相对标签输入带来 Δ 指标变化”；
- “在 full-report 条件下，factuality-first 相对 neutral 带来 Δ 指标变化”。

该设计不是完整的 (2\times2) 因子实验，因为缺少 `label_only_factuality_first`。因此禁止把两个差值解释成两个机制彼此独立的因果贡献，也不估计 interaction。

## 2. 真实基线与冻结身份

代码与 artifact 审计确认，正式 V2 应复用：

- 结果目录：`output/evolving_structured_rubrics/vl_rewardbench_global_arbiter_ab_preferred_none_v2`
- protocol：`global-arbiter-ab-preferred-none-tolerant-v2-vlrb-only`
- prompt：`global-arbiter-evidence-synthesis-ab-only-v1`
- 模型：`Qwen/Qwen3-VL-8B-Instruct`
- Rubric SHA-256：`007bd32557007aebc6f952cc63a49135f262648aff8943fbe99addecb4674b4d`
- A/B schedule SHA-256：`d0636fa7e4f409589ec4d36ce9917f20cf7fe1feedd3f37a4522d31579d8ce57`
- system prompt SHA-256：`f6e20d219eaee04be33bbac2bddfe9f7c72c9cad3365c7aa897602911bfee3a2`
- (K=3)，temperature `0.5`，`max_tokens=2048`，generation seed 不设置；parser 接受 `A/B/None`。

现有 V2 冻结结果为 Strict ACC 71.13%、covered OverallAcc 71.59%、Coverage 99.36%、Macro Strict ACC 65.65%；新实验不得覆盖或改写这些 artifact。

历史目录 `vl_rewardbench_global_arbiter_v1` 的 prompt 仍包含 factuality-first 段落，只是允许模型 abstain，**不能作为 neutral V1 复用**。

三组系统冻结为：

| 变体 | 子树证据 | Arbiter 规则 | 是否新增推理 |
|---|---|---|---:|
| V0 `label_only_neutral` | 五个 root 名称及各自 `A/B/None` | neutral | 是 |
| V1 `full_report_neutral` | 五份完整缓存报告 | neutral | 是 |
| V2 `full_report_factuality_first` | 同一批五份完整缓存报告 | 当前正式 factuality-first | 否，严格复用 |

这里“label-only”只限制**子树证据的表示**。图像、问题、Candidate A/B、五个 root 的名称与顺序仍必须提供给 V0；否则 V0 与 V1 的差异将不再只是标签和完整报告。

## 3. 最小 Prompt 差异

### 3.1 V1 相对 V2

V1 必须复用 V2 的完整 system prompt，只删除以下 factuality-first 段落：

```diff
- Prioritize verifiable visual and factual correctness. Consider completeness after factual validity; clarity and creativity may distinguish otherwise acceptable responses but cannot compensate for factual errors.
```

其余文本逐字节不变，包括：

- 将五棵子树看作 correlated evidence，而非独立投票；
- 独立核验图像、问题和 A/B；
- 倾向给出相对偏好并保留当前 “Do not abstain” 指令；
- 相同 JSON schema；
- parser 仍把模型自然输出的 `None` 作为合法 abstention，而非技术失败。

### 3.2 V0 相对 V1

V0 与 V1 使用**完全相同的 neutral system prompt**。唯一差异位于 user prompt 的 `Subtree Assessments` payload：

```text
### <root criterion name>
{"answer":"A"}
```

V1 的对应位置为现有完整报告：

```text
### <root criterion name>
{"analysis_a":"...","analysis_b":"...","thought":"...","answer":"A"}
```

不得增加“只数标签”“多数表决”或其他 V0 专用提示。两组的图像、问题、A/B、root 顺序、结尾 instruction 和 JSON 输出格式完全相同。

## 4. 缓存读取与完整性审计

冻结来源：

- compact 预测：`output/evolving_structured_rubrics/vl_rewardbench_implicit_unified_subtree_prompt_v1/predictions/vlrb_full.json`
- 完整报告 cache：`output/evolving_structured_rubrics/vl_rewardbench_implicit_unified_subtree_prompt_v1/cache/vlrb_full/<prefix>/<cache_key>.json`
- K=3 schedule：`output/evolving_structured_rubrics/vl_rewardbench_phase17_checkpoint_transfer_v1/order_schedule.json`
- V2 正式结果：`output/evolving_structured_rubrics/vl_rewardbench_global_arbiter_ab_preferred_none_v2/`

freeze/audit 必须逐项验证：

1. 1,247 个 sample、顺序和 `sample_id` 与 dataset records 完全一致；
2. 每个 sample 有 3 个 replicate，每个 replicate 有严格按 Rubric root 顺序排列的 5 份报告；
3. 总计 `1,247 × 3 × 5 = 18,705` 份报告均存在、`parse_ok=true`，compact answer 与 raw report answer 一致；
4. 每份 cache 的 protocol、prompt version、request key、sample、replicate、root、A/B order 均匹配；
5. 五份报告包含 `analysis_a/analysis_b/thought/answer`；V0 标签必须直接投影自同一份报告的 `answer`；
6. 评估 gold 只读取 VL-RewardBench record 的 `preferred_original_index`；任何子树或 Arbiter 输出都不能作为 gold；
7. V2 的 model/checkpoint、endpoint identity、decoding、parser、schedule、Rubric 和 prompt hash 全部与上节冻结值一致，否则禁止复用并硬失败；
8. 当前 live endpoint 必须重新提供相同的 Qwen3-VL-8B-Instruct checkpoint。若端口仍运行 Qwen2.5-VL，则 freeze 必须失败，而不是静默继续。

`18,705` 是需要扫描和复用的子树报告数，不是每个新变体的 Arbiter 调用数。每个 Arbiter 变体只新增 `1,247 × 3 = 3,741` 个唯一调用。

## 5. 结果目录与恢复协议

计划新增 runner：

```text
experiments/evolving_structured_rubrics/global_arbiter_evidence_priority_ablation.py
```

配置块名称：

```text
global_arbiter_evidence_priority_ablation_v1_experiment
```

独立结果目录：

```text
output/evolving_structured_rubrics/
  vl_rewardbench_global_arbiter_evidence_priority_ablation_v1/
    frozen_manifest.json
    offline_audit.json
    prompt_specs/
      neutral.json
      factuality_first_reference.json
    subsets/
      smoke_50.json
    cache/
      v0_label_only_neutral/<prefix>/<request_hash>.json
      v1_full_report_neutral/<prefix>/<request_hash>.json
    predictions/
      smoke.json
      full.json
    v2_reuse_manifest.json
    paired_comparison.json
    response_distribution.json
    category_source_analysis.json
    efficiency.json
    final_report.json
    final_report.md
```

每条 call artifact 至少保存 `sample_id`、replicate、display order、root 顺序、source cache keys、variant、raw response、parsed answer、metrics、endpoint、attempts 和 request hash。

cache key 不包含 `smoke/full` stage，只包含冻结的内容身份和 variant。因此 smoke 是 full 的严格子集，后续 full 可直接复用它，且不会因换 stage 重复计费。

## 6. 固定样本与运行顺序

VL-RewardBench 当前类别分布为 General 181、Hallucination 749、Reasoning 317。所有子集在任何模型调用前用 seed=42 和稳定 hash 冻结；不能按预测正确性选样。

### 6.1 Smoke 50

- 配额：General 7、Hallucination 30、Reasoning 13；
- 在类别内继续按 source 分层，并确保六个 source family 至少各出现一次；
- 同一 50 样本对 V0/V1 各运行 K=3，共 300 个 logical calls；
- 只验证缓存、prompt、双端点、A/B 映射、解析和恢复，不看准确率作 go/no-go。

### 6.2 Full 1,247

- 先运行 V0 full，再运行 V1 full；顺序写入 manifest；
- 每个变体 3,741 个 Arbiter calls，两组共 7,482 个唯一新调用；
- 由于 cache 跨 stage 复用，smoke 不增加最终 unique call 数；
- 最后离线接入 V2 现有 3,741 份结果并生成三系统 paired report。

建议顺序：`freeze → audit → smoke → full run → retry → report`。V0/V1 不并行混跑，以便 wall-time 与 endpoint telemetry 可解释；两者都使用相同的 available-slot 双端点池。

## 7. 调用量、token、时间和费用估算

当前正式 V2 的实测参考：

- 3,741 logical calls；
- input tokens：9,935,419，约 2,656/call；
- output tokens：1,300,957，约 348/call；
- 主运行耗时：2,085 秒，约 34.75 分钟；
- 对现有缓存中 500 份子树报告的快速审计显示，每份报告约 320 output tokens，五份约 1,600 tokens。

据此估计：

| 变体 | 新调用 | 预计 input tokens | 预计 output tokens | 预计双端点 wall time |
|---|---:|---:|---:|---:|
| V0 label-only neutral | 3,741 | 约 4.0M–4.8M | 约 1.0M–1.3M | 约 25–35 分钟 |
| V1 full-report neutral | 3,741 | 约 9.7M–9.9M | 约 1.2M–1.4M | 约 34–40 分钟 |
| V2 formal reuse | 0 | 0 | 0 | 0 |
| 合计 | **7,482** | **约 13.7M–14.7M** | **约 2.2M–2.7M** | **约 1.0–1.25 小时** |

这里 V0 的短输入不保证同比缩短 wall time，因为 Arbiter 的自回归输出仍可能是主要瓶颈。项目使用本地 vLLM 时新增 API 货币费用为 0；若迁移到计费 API，只按冻结价格代入：

\[
\mathrm{Cost}=T_{in}P_{in}/10^6+T_{out}P_{out}/10^6.
\]

没有供应商单价时不虚构人民币或美元费用。

## 8. 指标与统计分析

### 8.1 主要指标

- **Strict ACC**：K=3 后的 sample-level 最终判断；`None`/无法形成最终偏好计错。作为论文主指标。
- 两个预注册主差值：
  - `Δ_report = StrictACC(V1) − StrictACC(V0)`；
  - `Δ_priority = StrictACC(V2) − StrictACC(V1)`。

### 8.2 次要指标

- OverallAcc / covered accuracy、Coverage、MacroAcc；
- General / Hallucination / Reasoning 及各 source 分层结果；
- per-replicate 与 K=3 后的 A/B/None 数量和比例；
- None rate、position/order disagreement、K=3 tie/abstain 数；
- V0→V1、V1→V2 的 final flip rate、corrected、harmed、net corrected；
- input/output tokens、latency p50/p95、API attempts、retry、endpoint counts、wall time。

### 8.3 Paired statistics

- 统计单位是同一 `sample_id` 的 K=3 最终判断，不把 3 个 replicate 当独立样本；
- 对两个主比较分别计算 exact two-sided McNemar test；
- 用按类别分层的 paired bootstrap（10,000 次）报告 accuracy delta 的 95% CI；
- 两个主检验使用 Holm 校正；
- 类别/source 分层只作探索性分析，同时给 sample count 和 CI，不作选择性结论；
- A/B/None 与 flip 使用完整转移矩阵，避免只报告净收益。

## 9. 潜在混杂因素与解释边界

1. **信息与计算量不可完全分开。** V1 比 V0 多约 1,600 个缓存报告 tokens/call，所以 V1−V0 是“完整报告表示在 neutral 条件下的系统收益”，包含更多证据和更多 test-time compute，不能写成纯粹的语义信息贡献。
2. **neutral 只表示删除 Arbiter 层显式 priority。** 五份子树报告及其生成 prompt 本身仍可能编码事实优先，故 V2−V1 只测显式 Arbiter factuality-first 段落的增量。
3. **V2 是历史冻结运行。** 相同温度且 seed 未设置时，V0/V1 与历史 V2 的生成时间不同，服务端随机性是残余混杂。K=3 和 paired sample 可降低但不能消除它；这是遵守“不重复运行 V2”的代价。
4. **标签不能匿名。** V0 必须保留 root criterion name 与固定顺序，否则会同时移除标签的语义身份。
5. **相同 parser 不等于相同 None 率。** Prompt 虽偏向 A/B，parser 对三组均接受模型自然产生的 `None`；不得用额外 rescue prompt 改写某一组。
6. **Smoke 不得变成 benchmark 调参。** smoke 只验证技术链路，不能用其准确率修改 prompt 或挑选变体。
7. **V2 复用必须是 exact reuse。** 只要 model/checkpoint、prompt、schedule、Rubric、parser 或 evaluation hash 任一不一致，就不能将历史结果标成 V2。

## 10. 成功标准与停止条件

### 工程成功

- 18,705/18,705 source reports 完整且身份一致；
- V2 exact-reuse audit 全通过；
- smoke 后所有技术失败经相同 Prompt 最多 10 次重试均恢复；
- full 最终 unresolved technical failures 为 0，三组使用相同 sample/order/gold/evaluator；
- full 恰有 1,247 个 sample × 3 replicates × 2 个新变体，且每条保留 `sample_id`。

### 科学解释

- 若 `Δ_report` 的 95% CI 全部大于 0，可支持“neutral 下完整报告带来正增益”；跨 0 则结论为不确定；小于等于 0 则不支持该收益。
- 若 `Δ_priority` 的 95% CI 全部大于 0，可支持“full-report 下 factuality-first 带来额外正增益”；其余情况同理。
- 无论结果正负都必须报告，不能以“达到某个 ACC”作为停止或删结果条件。

立即停止并修复协议的条件：source/V2 hash 漂移、live 模型不是冻结 Qwen3 checkpoint、样本或 schedule 不一致、gold 来源错误、某组使用了不同 parser/解码/重试、无法恢复的技术失败。不得用 smoke 准确率决定是否运行 full。

## 11. 计划接口与完整命令

实现时配置块必须精确冻结为：

```json
{
  "protocol_version": "global-arbiter-evidence-priority-ablation-v1",
  "source_subtree_experiment": "vl_rewardbench_implicit_unified_subtree_prompt_v1",
  "source_v2_experiment": "vl_rewardbench_global_arbiter_ab_preferred_none_v2",
  "source_rubric_experiment": "phase17_discovery_v2_prompt_v2_split_refine_v1",
  "source_epoch": 4,
  "variants": {
    "v0_label_only_neutral": {
      "evidence_mode": "label_only",
      "decision_rule": "neutral",
      "reuse": false
    },
    "v1_full_report_neutral": {
      "evidence_mode": "full_report",
      "decision_rule": "neutral",
      "reuse": false
    },
    "v2_full_report_factuality_first": {
      "evidence_mode": "full_report",
      "decision_rule": "factuality_first",
      "reuse": true
    }
  },
  "k": 3,
  "generation_seed_policy": "unset",
  "temperature": 0.5,
  "max_tokens": 2048,
  "max_parse_retries": 10,
  "smoke_sample_count": 50,
  "subset_seed": 42,
  "endpoint_ids": ["vllm-8000", "vllm-8001"],
  "scheduler": "sample_bundle_available_slot_affinity",
  "semantic_answer_space": ["A", "B", "None"],
  "selection_after_benchmark_forbidden": true
}
```

以下 stage 已注册，可按顺序运行：

```powershell
conda activate critiq

$python = (Get-Command python).Source
$module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
$config = "experiments/evolving_structured_rubrics/configs/rubric_evolution_phase5.example.json"
$output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

function Invoke-ArbiterAblationStage {
    param([string]$Stage)
    Write-Host "`n===== $Stage ====="
    & $python -m $module `
        --config $config `
        --output-dir $output `
        $Stage
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

Invoke-ArbiterAblationStage "vlrb-arbiter-ablation-freeze"
Invoke-ArbiterAblationStage "vlrb-arbiter-ablation-audit"
Invoke-ArbiterAblationStage "vlrb-arbiter-ablation-smoke"
Invoke-ArbiterAblationStage "vlrb-arbiter-ablation-run"
Invoke-ArbiterAblationStage "vlrb-arbiter-ablation-retry"
Invoke-ArbiterAblationStage "vlrb-arbiter-ablation-report"
```

## 12. 不需要增加的消融

本轮不增加：

- `label_only_factuality_first`：只有在要估计完整 (2\times2) interaction 或声称独立因果贡献时才需要；当前 claim 明确禁止这样写；
- 子树多数投票、root 权重搜索、单 root 删除：回答的是聚合结构问题，不是本轮两个顺序问题；
- 重新生成子树报告或重新演化 Rubric：会破坏低成本与证据冻结；
- K=1/K=5、temperature、max_tokens、prompt wording sweep：会引入额外变量；
- Qwen2.5 或其他模型迁移：属于跨模型泛化，不属于本轮组件消融；
- 不同 neutral prompt 多版本：会把 V1 变成 prompt search；
- 专用 A/B rescue、强制 schema 或不同 retry 上限：会改变 parser/失败处理协议。

这三个系统足以支持预定的**顺序分解**，但不支持机制独立性、interaction 或完整因果分解。

# Qwen2.5-VL Clean S5-v2 跨模型聚合实验实施计划

**问题**：Phase17 E4 Rubric 已能迁移到 Qwen2.5-VL-7B-Instruct，但尚不清楚 Clean S5-v2 的“Unified-Subtree 报告 → Global Arbiter”聚合机制是否也能由 Qwen2.5 稳定执行。

**方法主张**：固定 Rubric、Prompt、数据、解码、A/B schedule、parser 和聚合协议，只把完整推理系统的 backbone 从 Qwen3-VL-8B-Instruct 替换为 Qwen2.5-VL-7B-Instruct，可以隔离 Clean S5-v2 的跨模型迁移能力。

**日期**：2026-08-27

## 1. Claim Map

| Claim | 为什么重要 | 最小可信证据 | 对应实验 |
| --- | --- | --- | --- |
| C1：Clean S5-v2 聚合机制能够跨 Worker 模型迁移 | 决定“子树证据 → 全局仲裁”是不是只适用于 Qwen3 | Qwen2.5 Clean S5-v2 相对同模型 E4 Explicit Recursive 的 Strict ACC 增量为正，并报告配对 corrected/harmed | B2、B3 |
| C2：聚合增益和 backbone 能力可以被区分 | 防止把 Qwen3 的绝对优势误写成聚合机制优势 | 构造 `model × aggregation` 2×2 对照，分别报告模型效应、聚合效应和交互项 | B3 |
| Anti-claim：差异来自 Prompt、Rubric、顺序或恢复协议变化 | 保证“变量只有模型”成立 | Prompt hash、Rubric hash、数据 hash、K=3 schedule、parser 和解码逐项冻结；所有 Qwen2.5 报告 fresh 生成 | B0、B1 |

## 2. Paper Storyline

- **主结果必须回答**：Qwen2.5 Clean S5-v2 是否优于 Qwen2.5 E4 Explicit Recursive；其提升或退化发生在哪些 VL-RewardBench 类别。
- **跨模型对照回答**：相同 Clean S5-v2 系统从 Qwen3 换成 Qwen2.5 后下降多少，是否与两模型在 Explicit Recursive 下的基础差距一致。
- **机制诊断回答**：Qwen2.5 Arbiter 是否真正推翻错误的子树多数，还是仅复制子树标签；位置一致性和 `None` 行为是否恶化。
- **本轮不做**：Qwen3/Qwen2.5 混合子树与 Arbiter、Root Gate、Router、Prompt 改写、Phase18 Qwen2.5-specific Rubric、额外 best-of-N 或 rescue Prompt。它们会破坏单变量设计。

## 3. 冻结协议

### 3.1 唯一实验变量

| 组件 | Qwen3 Clean S5-v2 Control | Qwen2.5 Clean S5-v2 Treatment |
| --- | --- | --- |
| Unified-Subtree Worker | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Global Arbiter | `Qwen/Qwen3-VL-8B-Instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Phase17 E4 Rubric | 相同 | 相同 |
| Unified-Subtree Prompt | byte-identical | byte-identical |
| Global Arbiter Prompt | byte-identical | byte-identical |
| 解码、parser、重试 | 相同 | 相同 |
| 数据与 A/B schedule | 相同 | 相同 |

完整推理链中的两个模型角色必须同时使用 Qwen2.5。不能复用 Qwen3 子树报告，也不能让 Qwen3 继续担任 Arbiter；否则变量会变成混合模型系统。

### 3.2 Rubric

- 来源：`phase17_discovery_v2_prompt_v2_split_refine_v1` Epoch 4。
- Rubric SHA256：`007bd32557007aebc6f952cc63a49135f262648aff8943fbe99addecb4674b4d`。
- 结构：5 roots、27 nodes。
- 不使用 Phase18 Qwen2.5-specific Rubric。
- 不读取 ACC、Coverage、gold、Manager evidence 或历史预测作为模型输入。

### 3.3 Prompt 与解析

实现不能复制并手工维护 Prompt 文本，而应直接引用现有常量：

- Unified-Subtree：`internal_global_arbiter_k1.UNIFIED_SUBTREE_SYSTEM_PROMPT`；
- 子树 User Prompt：`internal_global_arbiter_k1.subtree_user_prompt(...)`；
- Global Arbiter：`global_arbiter_ab_only.GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT`；
- Arbiter User Prompt：`_global_arbiter_ab_only_support.global_arbiter_user_prompt(...)`。

冻结以下 hash：

- Unified-Subtree System Prompt SHA256；
- Global Arbiter System Prompt SHA256；
- 至少一个 canonical subtree User Prompt SHA256；
- 至少一个 canonical Arbiter User Prompt SHA256；
- subtree parser 和 arbiter parser 版本。

Prompt 要求 Arbiter 尽量输出 A/B，但 parser 接受模型原生 `None` 并把它记为语义弃权。只有 transport error、空响应、非法 JSON、缺少 `answer` 或非法标签属于技术失败；技术失败使用同一个 Prompt 最多额外重试10次，不使用 tie-break、fallback 或第二套 rescue Prompt。

### 3.4 模型与执行参数

- Worker/Arbiter：`Qwen/Qwen2.5-VL-7B-Instruct`。
- 两个 endpoint 必须返回相同 model 和 checkpoint identity。
- `temperature=0.5`。
- `max_tokens=2048`。
- generation seed 不设置。
- scheduler：沿用 Clean S5-v2 的 sample-bundle available-slot affinity。
- 同一个 sample/replicate 的五个子树和 Arbiter 形成一个 bundle；Arbiter 只能在五份子树报告都形成后调用。
- 应用层 Prompt 保持 byte-identical；Qwen2.5 自身 tokenizer/chat template 属于预期的模型变量。

## 4. 推理流程

每个 sample/replicate 独立执行：

```text
图像 + 问题 + Candidate A/B
              │
              ├─ Completeness 子树      → Qwen2.5 报告 R1
              ├─ Visual Grounding 子树  → Qwen2.5 报告 R2
              ├─ Factuality 子树         → Qwen2.5 报告 R3
              ├─ Creativity 子树         → Qwen2.5 报告 R4
              └─ Clarity 子树            → Qwen2.5 报告 R5
                              │
                              ▼
                    Qwen2.5 Global Arbiter
                              │
                              ▼
                       A / B / None
```

五份报告是相关证据，不先进行 root 多数投票。Global Arbiter 同时读取原图、问题、A/B 和五份报告，重新判断证据优先级。VL-RewardBench 的三个 replicate 各自运行完整链路，最后只聚合三次 Arbiter 结论。

## 5. 数据、调度与预算

### 5.1 内部 K=1

| Split | 样本数 | 每样本调用 | 逻辑请求数 | A/B swap | 用途 |
| --- | ---: | ---: | ---: | --- | --- |
| Discovery100 | 100 | 5子树 + 1 Arbiter | 600 | 否 | 与演化数据的适配诊断 |
| Dev150 | 150 | 5子树 + 1 Arbiter | 900 | 否 | 多源内部泛化诊断 |
| RLHF-V heldout-500 | 500 | 5子树 + 1 Arbiter | 3,000 | 否 | 视觉幻觉历史诊断 |
| **合计** | **750** |  | **4,500** |  |  |

每个 split 只推理一次；结果不得用于修改 Prompt、Rubric 或选择 checkpoint。内部运行不设置基于准确率的 go/no-go，完成技术 smoke 后必须继续全部正式阶段。

### 5.2 VL-RewardBench K=3

- 数据：冻结的1,247个 preference pairs。
- 严格复用已有 K=3 counterbalanced ABA/BAB schedule。
- 每个 replicate 独立生成5份子树报告和1次 Arbiter 判断。
- 每个样本18次逻辑请求，总计：

```text
1,247 × 3 × (5 + 1) = 22,446
```

- 每次结果先映射回原始 Candidate index，再对三个 Arbiter 结论做多数聚合。
- `None` 不参与 A/B 计票；若无法形成至少两票一致的 A 或 B，最终输出 `None`。

### 5.3 总预算

```text
内部 K=1          4,500
VL-RewardBench   22,446
-----------------------
总逻辑请求       26,946
```

技术重试不计入逻辑请求数，但必须单独报告真实 model generation count。

## 6. 对照与因果分解

### 6.1 主 2×2 对照

| Backbone | Explicit Recursive | Clean S5-v2 |
| --- | --- | --- |
| Qwen3-VL-8B | 冻结 S0：Strict 70.01% | 冻结 Control：Strict 71.13% |
| Qwen2.5-VL-7B | 冻结 E4：Strict 56.94% | 本实验 Treatment：待测 |

定义：

```text
Qwen3 aggregation gain  = Qwen3 Clean S5-v2 - Qwen3 Explicit E4
Qwen2.5 aggregation gain = Qwen2.5 Clean S5-v2 - Qwen2.5 Explicit E4
aggregation interaction  = Qwen2.5 gain - Qwen3 gain
```

Qwen3 的已知聚合点估计增益为 `71.13% - 70.01% = +1.12 pp`。主要问题不是要求 Qwen2.5 达到 Qwen3 的绝对分数，而是 Qwen2.5 的 aggregation gain 是否仍为正。

### 6.2 可复用 Control

只读复用并冻结 hash：

- Qwen3 Clean S5-v2：`vl_rewardbench_global_arbiter_ab_preferred_none_v2/final_report.json`；
- Qwen3 E4 Explicit Recursive：S0 frozen control；
- Qwen2.5 E4 Explicit Recursive：`vl_rewardbench_qwen25_phase17_e4_transfer_v1/final_report.json`；
- Qwen3 内部 K=1 Clean S5：`internal_global_arbiter_k1_v1/final_report.json`。

对照仅用于离线指标比较，不能向 Qwen2.5 Treatment 提供报告内容。若 dataset、Rubric、schedule、Prompt 或 metric implementation 任一身份不匹配，则相关配对比较必须标记为不可用，不能静默降级为同身份结果。

## 7. 实验模块

### B0：Freeze 与单变量审计

- **Claim tested**：除 backbone 外所有应用层变量相同。
- **检查**：Rubric/data/schedule/Prompt/parser/decode hash；两个 endpoint 的 Qwen2.5 model/checkpoint identity；Qwen3 cache reuse count=0。
- **成功条件**：审计逐项通过；不是只比较 model 字符串。
- **失败解释**：属于协议漂移，禁止开始正式推理。
- **优先级**：MUST-RUN。

### B1：双角色 smoke

- **数据**：内部每 split 2条 + VL-RewardBench 20条完整 K=3 bundle。
- **目的**：确认五份子树报告、Arbiter 输入、A/B 映射、两个 endpoint 和同 Prompt retry 均可用。
- **指标**：subtree/arbiter parse valid rate、endpoint call counts、latency、bundle completeness。
- **成功条件**：每个有效 bundle 恰好包含5份报告和1次 Arbiter；解析失败可由相同 Prompt 恢复。
- **说明**：smoke 只有技术 gate，不设准确率 gate。
- **优先级**：MUST-RUN。

### B2：内部 K=1 诊断

- **数据**：Discovery100、Dev150、RLHF-V heldout-500。
- **主要指标**：Strict ACC、Coverage、OverallAcc、MacroAcc（适用时）。
- **诊断**：domain/source、A/B/None 分布、Qwen2.5 与 Qwen3 Clean S5 预测一致率、corrected/harmed。
- **成功解释**：三个 split 都接近 Qwen3 与 Qwen2.5 backbone 差距所预期的范围，且没有异常 Coverage 坍缩。
- **失败解释**：若 Dev/heldout 明显比 Discovery 下降，说明 Qwen2.5 的子树报告或仲裁对分布变化更敏感。
- **优先级**：MUST-RUN。

### B3：VL-RewardBench 主实验

- **主要指标**：Strict ACC。
- **次要指标**：OverallAcc、MacroAcc、Coverage、correct count。
- **配对比较**：
  1. Qwen2.5 Clean S5-v2 vs Qwen2.5 E4 Explicit Recursive；
  2. Qwen2.5 Clean S5-v2 vs Qwen3 Clean S5-v2；
  3. Qwen2.5 aggregation gain vs Qwen3 aggregation gain。
- **统计**：corrected/harmed/net corrected、exact McNemar、Wilson 95% CI。
- **类别**：General、Hallucination、Reasoning 以及各 benchmark source。
- **优先级**：MUST-RUN。

### B4：机制与效率诊断

- **子树行为**：每个 root 的 A/B/None 分布、Qwen2.5/Qwen3 报告结论一致率。
- **仲裁行为**：5–0、4–1、3–2、平局/稀疏格局下的推翻数、推翻后 corrected/harmed、净修正。
- **位置稳定性**：逐 replicate Strict ACC、同顺序一致率、swap-mapped agreement、最终 K=3 增益。
- **效率**：subtree 与 arbiter 分段耗时、逻辑请求、generation 数、tokens、请求/分钟、sample bundle/分钟、p50/p95/max latency、endpoint 负载。
- **优先级**：MUST-RUN，全部由正式输出离线计算，不增加请求。

## 8. 实现结构

### 8.1 新增 runner

新增：

```text
experiments/evolving_structured_rubrics/qwen25_clean_s5_transfer.py
```

职责：

- 冻结模型、Prompt、Rubric、dataset 和 schedule；
- 统一调度内部 K=1 与 VL-RewardBench K=3；
- fresh 生成 Qwen2.5 子树报告与 Arbiter 输出；
- 同 Prompt 技术重试；
- 汇总跨模型、同模型聚合、类别、机制和效率结果。

优先复用现有纯函数和 Prompt 常量，不修改旧实验 artifact，不通过运行时覆盖旧模块全局常量来切换实验身份。新协议使用独立 config key、cache namespace、request kind 和 output directory。

### 8.2 配置块

新增：

```json
"qwen25_clean_s5_transfer_experiment": {
  "protocol_version": "qwen25-clean-s5-v2-transfer-v1",
  "worker_model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "source_experiment": "phase17_discovery_v2_prompt_v2_split_refine_v1",
  "source_epoch": 4,
  "rubric_sha256": "007bd32557007aebc6f952cc63a49135f262648aff8943fbe99addecb4674b4d",
  "internal_k": 1,
  "vl_rewardbench_k": 3,
  "generation_seed_policy": "unset",
  "temperature": 0.5,
  "max_tokens": 2048,
  "max_parse_retries": 10,
  "endpoint_ids": ["vllm-8000", "vllm-8001"],
  "scheduler": "sample_bundle_available_slot_affinity",
  "semantic_answer_space": ["A", "B", "None"],
  "selection_after_diagnostics_forbidden": true
}
```

最终实现时配置必须完整列出 dataset counts、smoke counts、Control artifact identities 和 Prompt versions，不能只依赖默认值。

### 8.3 CLI stages

```text
qwen25-clean-s5-freeze
qwen25-clean-s5-audit
qwen25-clean-s5-smoke
qwen25-clean-s5-internal-run
qwen25-clean-s5-vlrb-run
qwen25-clean-s5-retry
qwen25-clean-s5-report
```

运行顺序固定。`internal-run` 的科学结果不能阻止 `vlrb-run`；只有身份审计或请求链路错误可以阻止后续阶段。

### 8.4 输出目录与 artifact

```text
output/evolving_structured_rubrics/qwen25_clean_s5_v2_transfer_v1/
```

主要 artifact：

```text
frozen_manifest.json
prompt_spec.json
offline_audit.json
schedules/internal_k1.json
schedules/vlrb_k3.json
bundles/<split>/...
progress/<stage>.json
parse_failures.json
retry_summary.json
internal_report.json
vlrb_report.json
paired_comparison.json
category_analysis.json
arbiter_override_analysis.json
position_analysis.json
efficiency.json
final_report.json
final_report.md
```

Git 只保留代码、测试、计划、manifest、prompt spec 和紧凑报告；bundles、raw generations、cache、predictions、progress 和 endpoint assignment 默认忽略。

## 9. 恢复与缓存语义

- cache key 必须包含：protocol、model identity、endpoint checkpoint identity、Prompt hashes、dataset/sample ID、replicate、A/B order、root ID、Rubric hash 和 decode settings。
- Arbiter cache key 额外包含五份 source report bundle 的 canonical SHA256。
- Qwen3 cache 与 Qwen2.5 cache 必须物理或逻辑隔离。
- smoke 可以在正式 run 中按完整 request identity 复用，但主运行计时必须区分 cache hit 和 fresh generation。
- 单个 subtree 技术失败时，同一个 sample/replicate 的 Arbiter 暂不调用；retry 恢复齐5份报告后再调用 Arbiter。
- 达到重试上限仍失败时，不回退到 root vote、Qwen3 输出或其他 Prompt；继续完成其他样本，并在报告中计入 unresolved technical failure。
- report 阶段不得因为少量 unresolved failure 整体崩溃，应同时报告 Strict ACC、Coverage 和技术失败率，允许后续再次执行 retry。

## 10. 测试矩阵

必须新增 focused unittest，使用 fake/offline backend：

- Qwen2.5 config、endpoint identity 和 model name 严格冻结；
- Phase17 E4 Rubric hash、5 roots / 27 nodes 正确；
- 两个 System Prompt 与 Clean S5-v2 byte-identical；
- 对 canonical 样本生成的 User Prompt 与旧协议相同；
- 除 model/endpoint/cache namespace 外，normalized request spec 与 Qwen3 Clean S5-v2 一致；
- internal 每条恰好5 subtree + 1 arbiter；
- VL-RewardBench 每条恰好3个独立 bundle，共15 subtree + 3 arbiter；
- 三个 replicate 的 A/B swap 映射和多数聚合正确；
- subtree/arbiter 原生 `None` 是语义输出，不触发技术重试；
- 非法 JSON、空响应和非法 label 才触发同 Prompt retry；
- Arbiter 只能在5份解析成功的报告形成后调用；
- Arbiter cache identity 包含 report bundle hash；
- 不读取 Qwen3 子树 cache 或 Arbiter prediction；
- Control hash 漂移时硬失败；
- unresolved technical failure 不触发 fallback，也不阻止报告生成；
- Discovery/Dev/heldout 不能修改 Prompt、Rubric、schedule 或 checkpoint；
- focused unittest、完整 unittest、compileall 和 `git diff --check` 通过。

## 11. Run Order 与时间预算

| Milestone | 目标 | Stages | 继续条件 | 预计时间 | 风险 |
| --- | --- | --- | --- | ---: | --- |
| M0 | 冻结并证明单变量 | freeze → audit | 所有身份检查通过 | <2分钟 | endpoint/model 漂移 |
| M1 | 验证双角色链路 | smoke | 两端可用，bundle/解析/重试正确 | 5–15分钟 | Qwen2.5 长输出 |
| M2 | 完成内部 K=1 | internal-run | 不设准确率 gate | 30–70分钟 | 长尾请求 |
| M3 | 完成外部 K=3 | vlrb-run | 跑完1,247×3 bundles | 2–4小时 | 22,446次请求 |
| M4 | 技术恢复与报告 | retry → report | 报告可生成，失败透明记录 | 10–30分钟 | 少量顽固解析失败 |

总墙钟时间预计3–5小时。正式运行串行使用同一对 endpoint，避免内部和 VL-RewardBench 相互争抢服务。实际 ETA 由 smoke 和前100个正式 bundle 的滚动延迟更新。

## 12. 结果解释

| 观测 | 结论 |
| --- | --- |
| Qwen2.5 Clean S5 > Qwen2.5 Explicit E4 | 聚合机制在较弱 backbone 上仍有效 |
| Qwen2.5 aggregation gain 接近 Qwen3 的 +1.12 pp | Clean S5-v2 聚合收益具有较好的跨模型一致性 |
| Qwen2.5 Clean S5 ≈ Explicit E4 | 机制可运行，但 Qwen2.5 未充分利用子树报告 |
| Qwen2.5 Clean S5 < Explicit E4 | 子树报告或全局证据综合依赖更强 backbone |
| Coverage 显著下降 | Qwen2.5 更容易把复杂冲突转化为 `None` |
| Arbiter 很少推翻多数 | Qwen2.5 可能只复制子树标签，没有充分仲裁 |
| 推翻很多但 harmed 高 | Qwen2.5 能重新决策，但冲突消解质量不足 |
| Hallucination 保持而 Reasoning 下降 | 视觉事实优先级可迁移，复杂推理综合能力不足 |

Qwen2.5 的绝对分数低于 Qwen3 不构成实验失败；决定聚合迁移结论的是同一 Qwen2.5 内部的 `Clean S5-v2 - Explicit E4` 差值及其配对统计。

## 13. 最终验收清单

- [ ] Phase17 E4 Rubric、四套 Control 和 K=3 schedule 完整冻结
- [ ] Unified-Subtree 与 Arbiter Prompt hash 和 Clean S5-v2 一致
- [ ] 两个角色都使用 Qwen2.5，Qwen3 inference reuse 为0
- [ ] 内部4,500次与 VL-RewardBench 22,446次逻辑请求正确
- [ ] 三个 VL-RewardBench replicate 独立生成并正确映射
- [ ] `None`、技术失败和重试语义保持透明
- [ ] 主表同时报告 Strict、Overall、Macro、Coverage
- [ ] 完成 model × aggregation 2×2 对照与交互项
- [ ] 完成类别、位置、Arbiter 推翻和效率诊断
- [ ] 无结果驱动的早停、Prompt 修改或 checkpoint 选择
- [ ] focused/full unittest、compileall、diff check 全部通过

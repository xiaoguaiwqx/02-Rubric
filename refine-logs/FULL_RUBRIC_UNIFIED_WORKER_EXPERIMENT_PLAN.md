# Unified Full-Rubric Worker v1 实验计划

**问题**：现有 S0 对每个 node 独立推理，S3 对五棵子树分别推理，Clean S5-v2 再用 Global Arbiter 汇总五份子树报告。它们都假设结构化 Rubric 需要显式分解执行。本实验检验：把完整五-root Rubric 一次性交给同一个 Worker，让模型隐式选择适用准则、处理准则冲突并直接输出偏好，是否能够以更少调用保持或提高准确率。

**方法主张**：结构化 Rubric 可以作为一份统一的决策策略使用，不一定需要逐节点投票或多阶段仲裁。

**日期**：2026-08-26

## 1. Claim Map

| Claim | 为什么重要 | 最小可信证据 | 对应实验 |
|---|---|---|---|
| C1：完整 Rubric 可以由单次 Worker 隐式整合 | 判断结构是否必须通过显式多 Worker 执行 | 在相同 Rubric、模型和解码协议下，与 S0、S3、Clean S5-v2 比较 Strict ACC、Coverage 和 VL-RewardBench 指标 | B1、B2 |
| C2：隐式整合能显著降低推理成本 | 验证结构化方法是否可以简化为实用推理系统 | VL-RewardBench 每个 replicate 从 5–27 次调用降至1次，并报告实际耗时、吞吐和 token | B2、B3 |
| Anti-claim：结果只是更高 test-time compute 带来的 | 新方法的模型调用最少 | 新方法只允许每个样本、每个 replicate 一次模型调用，无 Gate、子树 Worker、节点 Worker、Arbiter 或 fallback | B1–B3 |

## 2. 冻结协议

### 2.1 Rubric 与模型

- Rubric：`phase17_discovery_v2_prompt_v2_split_refine_v1` 的 Epoch 4。
- Rubric hash 必须与现有 S0、S3、Clean S5-v2 对照使用的 hash 完全一致。
- Worker：`Qwen/Qwen3-VL-8B-Instruct`。
- Pairwise 风格：与 Prompt v2 对齐的四字段 JSON 推理格式。
- `temperature=0.5`。
- `max_tokens=2048`。
- generation seed 不设置。
- endpoint：配置中的 `vllm-8000 + vllm-8001` available-slot pool。

### 2.2 唯一实验变量

本实验只改变 Rubric 的执行与聚合方式：

```text
图像 + 问题 + Candidate A/B
              │
              ▼
完整五-root Rubric
包含全部 roots、children 与层级关系
              │
              ▼
一个 Unified Full-Rubric Worker
隐式选择相关准则并解决冲突
              │
              ▼
直接输出 A / B
```

禁止使用：

- 独立 node Worker；
- 五个 Unified-Subtree Worker；
- root 多数投票；
- Gate Worker 或 Root Router；
- Global Arbiter；
- parent/root fallback；
- 单独的 tie-break 或 rescue Prompt；
- criterion-level 中间分数和显式权重聚合。

### 2.3 完整 Rubric 序列化

按冻结 Rubric 的 canonical root 顺序写入全部节点：

1. root criterion name 与 description；
2. 该 root 下每个 child 的 criterion name 与 description；
3. 保留 root-child 层级，但不把节点转换成独立选票；
4. 每个节点恰好出现一次；
5. 不写入 gold、历史预测、ACC、Coverage、Manager 反馈或 benchmark 标签。

冻结以下身份：

- source rubric SHA256；
- canonical serialization SHA256；
- node ID 集合及数量；
- root-child edge 集合；
- System/User Prompt SHA256；
-解码参数与 endpoint identity。

## 3. Prompt 设计

### 3.1 System Prompt

System Prompt 的固定前缀包含角色、完整 Rubric 和输出规范。核心语义为：

```text
You are judging a multimodal preference pair using one complete structured rubric.
Treat the full rubric hierarchy as a unified decision policy, not as independent votes.
Use the image when visual evidence is relevant. Apply only criteria that bear on the
actual difference between the responses. Resolve conflicts using direct evidence,
task requirements, error severity, and likely human preference.

[FULL STRUCTURED RUBRIC]

Return one JSON object:
{
  "analysis_a": "Analyze Candidate A using the relevant rubric evidence.",
  "analysis_b": "Analyze Candidate B using the relevant rubric evidence.",
  "thought": "Compare the candidates and resolve any criterion conflicts.",
  "answer": "A / B"
}
```

要求模型作出相对偏好，不要求输出适用节点、逐节点分数或路由过程。完整 Rubric 位于静态 System Prompt 中，以便两个 vLLM endpoint 分别复用固定前缀。

### 3.2 User Prompt

User Prompt 只包含每条请求的动态内容：

```text
## Question
{question}

## Candidate A
{A}

## Candidate B
{B}

Which candidate better follows the complete structured rubric and is more likely
to align with human preference?
```

图像作为该 User message 的多模态输入提供。

### 3.3 解析与重试

- Prompt 明确要求 `A/B`。
- Parser 接受 `A`、`B`；若模型原生输出 `None`，将其记录为合法语义 abstention，而不是技术失败。
- `None` 不重试，进入 Coverage 与 Strict ACC 统计。
- 只对 transport error、空响应、非法 JSON、缺失 answer 或非法 label 使用同一 Prompt 重试。
- 最多额外重试10次；禁止改用其他 rescue Prompt。
- 正式报告保留 parse valid rate、semantic None count 和 unresolved technical failures。

## 4. 数据与运行协议

### 4.1 内部数据：K=1

| Split | 数量 | 推理次数 | A/B swap | 用途 |
|---|---:|---:|---|---|
| Discovery100 | 100 | 每条1次 | 否 | 观察与演化数据的适配程度 |
| Dev150 | 150 | 每条1次 | 否 | 跨来源内部泛化诊断 |
| RLHF-V heldout500 | 500 | 每条1次 | 否 | 与历史视觉幻觉测试对照 |

三套数据总计750次初始模型调用。它们均为诊断，不用于重新选择 Rubric、Prompt 或 checkpoint。

### 4.2 VL-RewardBench：K=3

- 数据：冻结的1,247个 preference pairs。
- 严格复用现有 counterbalanced K=3 A/B schedule。
- 每个 replicate 由 Full-Rubric Worker 独立生成，不复用同一样本的另一次生成结果。
- 每次输出先映射回原始 A/B index，再在三个 replicate 上多数聚合。
- 至少两个 replicate 同意 A 或 B 时输出该答案；否则输出 `None`。
- 新增初始请求数：`1,247 × 3 = 3,741`。

### 4.3 总请求预算

```text
Discovery100       100
Dev150             150
heldout500         500
VL-RewardBench   3,741
----------------------
总计             4,491 次初始请求
```

技术重试不计入逻辑请求数，但单独报告真实 model generation count。

## 5. 对照系统

| 系统 | 执行方式 | VL-RewardBench 每个 replicate 调用数 | 本轮是否重新推理 |
|---|---|---:|---|
| S0 Explicit Recursive | 27个 node Worker 后递归聚合 | 27 | 否，严格校验后复用 |
| S3 Unified Subtree | 5个子树 Worker 后等权多数 | 5 | 否，严格校验后复用 |
| Clean S5-v2 | 5个子树 Worker + 1个 Global Arbiter | 6 | 否，严格校验后复用 |
| S6 Unified Full-Rubric | 1个 Worker 直接判断 | **1** | **是** |

Control 只有在 dataset hash、Rubric hash、模型、K=3 schedule、A/B 映射和指标实现完全一致时才能复用。内部数据优先复用 Phase17 E4 的显式递归结果；缺少严格同身份对照时，报告中标记为 historical reference，不伪装成配对实验。

## 6. 实验模块

### B1：内部 K=1 诊断

- **Claim tested**：完整 Rubric 单次推理是否能在 Discovery、Dev 和 heldout 上稳定工作。
- **指标**：Strict ACC、Coverage、covered ACC、各领域 ACC、A/B/None 分布、parse valid rate。
- **关键对照**：Phase17 E4 Explicit Recursive K=1。
- **解释**：若 Discovery 高而 Dev/heldout 明显下降，说明单次隐式整合容易利用 discovery-specific 线索；若三套数据均接近显式系统，则支持简化执行。
- **优先级**：MUST-RUN。

### B2：VL-RewardBench 主消融

- **Claim tested**：一次 Full-Rubric 判断能否替代显式分解与 Global Arbiter。
- **主要指标**：Strict ACC。
- **次要指标**：OverallAcc、MacroAcc、Coverage、General/Hallucination/Reasoning Strict ACC、每个 replicate 的 ACC/Coverage、order disagreement、paired corrected/harmed 和 exact McNemar。
- **关键对照**：S0、S3、Clean S5-v2。
- **解释**：
  - 达到或超过 Clean S5-v2：完整 Rubric 本身足以支持隐式全局决策；
  - 与 Clean S5-v2 相差不超过1 pp：形成明显更便宜的实用替代方案；
  - 下降1–3 pp：存在准确率—成本权衡；
  - 下降超过3 pp：说明显式子树分析或额外 test-time compute 仍不可替代。
- **优先级**：MUST-RUN。

### B3：效率与失败分析

- **Claim tested**：减少逻辑调用是否转化为真实时间和 token 成本下降。
- **指标**：逻辑请求数、实际 generation 数、输入/输出 token、主运行耗时、请求/分钟、样本/分钟、重试次数、长尾请求分布；vLLM prefix cache hit rate 由服务器日志补充。
- **失败分析**：按长 Prompt 截断、循环输出、非法 JSON、semantic None、General/Reasoning/Hallucination 退化和位置不一致分类。
- **优先级**：MUST-RUN。

## 7. Stages 与 Artifact

建议新增独立 runner：

```text
full-rubric-worker-freeze
full-rubric-worker-audit
full-rubric-worker-smoke
full-rubric-worker-internal-run
full-rubric-worker-vlrb-run
full-rubric-worker-retry
full-rubric-worker-report
```

输出目录：

```text
output/evolving_structured_rubrics/full_rubric_unified_worker_v1/
```

主要 artifact：

```text
frozen_manifest.json
rubric_snapshot.json
rubric_serialization.json
prompt_spec.json
offline_audit.json
schedules/internal_k1.json
schedules/vlrb_k3.json
predictions/internal_discovery100.json
predictions/internal_dev150.json
predictions/internal_heldout500.json
predictions/vlrb_full.json
progress/{stage}.json
parse_failures.json
retry_summary.json
efficiency.json
paired_comparison.json
category_analysis.json
final_report.json
final_report.md
```

逐请求缓存、raw generations、bundles、progress 和大 prediction artifacts 留在本地并加入 `.gitignore`；Git 只保留代码、测试、计划和紧凑报告。

## 8. Run Order

| Milestone | 目标 | 运行 | 判断条件 | 预计成本 | 风险与处理 |
|---|---|---|---|---|---|
| M0 | 冻结身份 | freeze + audit | 全部 hash、节点数、schedule、endpoint identity 通过 | 离线，<1分钟 | Rubric/control 漂移时硬失败 |
| M1 | 验证 Prompt 与解析 | smoke 20条，内部与VLRB均覆盖 | 请求发往两个 endpoint；格式、图像和重试链路可用 | 约1–5分钟 | 长输出时仍保持 `max_tokens=2048` |
| M2 | 内部诊断 | Discovery100 + Dev150 + heldout500 | 无需科学 go gate，完成后继续 | 约15–45分钟 | semantic None 只记入指标 |
| M3 | 外部主实验 | VL-RewardBench K=3 | 跑完全部3,741次逻辑请求 | 约45–150分钟 | 用进度文件估计真实 ETA |
| M4 | 技术恢复与报告 | retry + report | unresolved technical failures 尽量为0；非零也保留并出报告 | 约5–30分钟 | 不用替代 Prompt rescue |

总预计墙钟时间约1–3小时，最终以 smoke 的每请求耗时和长尾生成情况重新估算；整个科学实验不设置基于准确率的提前停止条件。

## 9. 必须覆盖的测试

- 完整 Rubric 序列化包含全部冻结 nodes，每个 node 恰好一次。
- root-child topology 和 canonical 顺序稳定，序列化 hash 可复现。
- System Prompt 静态包含完整 Rubric，User Prompt 不重复 Rubric。
- 请求中不存在 gold、历史预测、指标或 Manager 反馈。
- 每个 internal sample 恰好一个逻辑请求。
- 每个 VL-RewardBench sample 恰好三个独立逻辑请求。
- K=3 A/B swap 映射和多数聚合正确。
- `None` 是语义输出而不是解析失败；非法 JSON 才触发同 Prompt 重试。
- 不存在 Gate、node/subtree Worker、Arbiter 或 fallback 请求。
- 两个 endpoint 使用 available-slot pool，endpoint call count 被记录。
- S0/S3/Clean S5-v2 只有在完整 provenance 校验后才能复用。
- discovery/dev/heldout 结果不能修改 Rubric、Prompt 或 VL-RewardBench schedule。
- focused unittest、完整 unittest、compile 和 `git diff --check` 通过。

## 10. 结果表目标

主表：

| System | Strict ACC | OverallAcc | MacroAcc | Coverage | Calls / replicate | Total VL-RB calls | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 Explicit Recursive | frozen | frozen | frozen | frozen | 27 | 101,007 | historical |
| S3 Unified Subtree | frozen | frozen | frozen | frozen | 5 | 18,705 | historical |
| Clean S5-v2 | frozen | frozen | frozen | frozen | 6 | 22,446 | historical |
| S6 Unified Full-Rubric | TBD | TBD | TBD | TBD | **1** | **3,741** | TBD |

附表报告三个内部 split、类别结果、逐 replicate 稳定性和技术失败。论文叙事以 Strict ACC 和真实请求成本为主，不只比较覆盖内 Accuracy。

## 11. 最终验收清单

- [ ] Phase17 E4 Rubric 与三个对照身份完全冻结
- [ ] 完整 Rubric 没有遗漏、重复或扁平化错误
- [ ] 每个 replicate 只有一次新模型调用
- [ ] 内部 K=1 与 VL-RewardBench K=3 协议正确隔离
- [ ] Prompt 要求 A/B，parser 对原生 None 保持语义透明
- [ ] 不存在第二层 rescue 或后验 fallback
- [ ] 主结果同时报告 Strict ACC、Coverage、OverallAcc 与 MacroAcc
- [ ] 请求数、实际耗时、吞吐和 token 成本可核验
- [ ] 所有准确率结果都在完整运行后统一分析，不设置结果导向的 go/no-go

## 12. 实现状态

截至 2026-08-26，独立 runner、冻结配置、CLI stages、缓存与同 Prompt
技术重试、K=1/K=3 调度、对照复用校验、指标报告和 focused tests 已实现。
正式模型推理仍保持未运行状态，等待用户按 M0→M4 顺序启动。

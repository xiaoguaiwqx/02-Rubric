# VL-RewardBench Phase10 External Transfer Plan

**Problem**: 当前 Split + Role-aware Refine 在开发用 heldout-500 上达到 71.4% 等权 M1；固定 Visual Grounding 权重为 0.4、其余 roots 各 0.15 后达到 73.4%。需要检验这套最终 Rubric 是否能迁移到 VL-RewardBench，而不是只适合当前 RLHF-V 风格数据。

**Method thesis**: 冻结 Phase10 最终 22-node Rubric 和预设 root 权重后，结构化逐准则判断应优于同一 Qwen3-VL-8B-Instruct 的通用 judge prompt 与未演化五-root Rubric。

**Date**: 2026-08-11

**Experiment ID**: `vl_rewardbench_phase10_transfer_v1`

本计划取代旧的 `vl_rewardbench_external_transfer_v1` 主系统选择。旧计划使用 Phase8 epoch-1 的 17-node Rubric；本实验只测试最新 Phase10 Rubric，不继续运行旧 checkpoint。

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1：Rubric 演化具有外部迁移收益 | 这是 Split + Refine 链路的核心研究价值 | Phase10 等权 M1 在 VL-RewardBench 上高于 Initial five-root，paired net corrected > 0 | B1、B2 |
| C2：冻结的 root reliability 权重具有可迁移价值 | 区分 Rubric 内容收益与聚合权重收益 | Phase10 weighted 高于或至少不弱于 Phase10 equal-root，且不降低 Coverage | B2 |
| Anti-claim：收益只是通用 8B judge 本身带来的 | 排除结构化流程只是换一种提示词包装 | Phase10 系统高于同一模型的 benchmark-native judge | B1、B2 |

## Paper Storyline

- 主结果：同一 8B VLM 下，比较 benchmark-native judge、Initial structured M1、Phase10 equal-root 与 Phase10 weighted。
- 机制分解：`Initial -> Phase10 equal` 表示 Rubric 演化收益；`Phase10 equal -> weighted` 表示固定 root 权重收益。
- 附录诊断：各 source/task group、各 root/subtree、Visual-only、顺序偏置和错误转移。
- 明确不做：不在 VL-RewardBench 上重新选择 checkpoint、搜索权重、修改 criterion、运行 Manager 或继续演化。

该 benchmark 含 RLAIF-V、RLHF-V 等来源，而本协议按既定决定不做样本重叠审计。因此结果记为 **exploratory cross-benchmark transfer**，不能表述为严格数据独立的无偏泛化测试。

## Frozen Systems

| System | Rubric / prompt | Aggregation | Role |
|---|---|---|---|
| `native_vlrb_prompt` | `data/VL_RewardBench/prompt.py` | 单次 judge 的 K=3 多数 | 同模型通用 judge baseline |
| `initial_five_root_equal` | Phase-5 deterministic initial Rubric，5 nodes | 五 roots 各 0.2 | 未演化结构化 baseline |
| `phase10_final_equal` | Phase10 final Rubric，22 nodes | 五 roots 各 0.2 | 隔离 Rubric 演化收益 |
| `phase10_final_weighted` | 与上一系统共享全部 node predictions | Visual Grounding=0.4；其余四 roots 各 0.15 | Primary system |

冻结 Phase10 artifact：

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase10_five_root_locked_split_refine_v1/final/rubric.json
```

冻结 Rubric hash：

```text
17ad7a0b6350338b100384b701303008746463911662f4a628b90c4632cdf7ef
```

Phase10 equal 与 weighted 只能进行离线聚合，不得分别调用 Worker。Initial 中与 Phase10 完全同 identity 的 roots 允许按 criterion description hash 复用预测；若任一字段不同则必须单独生成。

## Block B0: Offline Freeze and Metric Audit

- **Claim tested**: 实验身份、标签映射和聚合实现正确。
- **Dataset**: `data/VL_RewardBench/data/test-00000-of-00001.parquet`，预期 1,247 rows。
- **Freeze**:
  - parquet、官方 prompt、Phase10 Rubric、Initial Rubric 和 evaluator hashes；
  - 1,247 个稳定 row IDs，包括九个重复原始 ID 的 row suffix；
  - `human_ranking` 中 0=preferred、1=rejected 的映射；
  - K=3 order schedule、双服务器分片表、模型身份、解码参数和两个 endpoint identities；
  - 四个系统及全部报告指标。
- **Gate**: 任一 hash、row count、image bytes、response count、root weight sum 或标签映射不一致即硬失败。
- **Priority**: MUST-RUN。

## Block B1: Same-model Baselines

- **Claim tested**: 结构化系统不是仅依赖 Qwen3-VL-8B-Instruct 的基础判断能力。
- **Compared systems**:
  - `native_vlrb_prompt`：官方通用 Accuracy/Completeness/Clarity/Relevance prompt；
  - `initial_five_root_equal`：五个初始 root 的结构化 Pairwise + M1。
- **Setup**:
  - Native decoding 固定为官方 `temperature=0.2, top_p=0.2`；
  - Structured Worker 固定为现有 P05 request identity；
  - 两者均使用 Qwen3-VL-8B-Instruct，Manager 不参与；
  - gold、ranking 和 benchmark rationale 不进入任何模型 prompt。
- **Interpretation boundary**: 两种系统的 prompt 与 decoding 均不同，因此这是同模型 system baseline，不是严格 prompt-only ablation。
- **Priority**: MUST-RUN。

## Block B2: Phase10 Transfer and Weight Decomposition

- **Claim tested**: C1、C2。
- **Compared systems**:
  - `initial_five_root_equal`；
  - `phase10_final_equal`；
  - `phase10_final_weighted`。
- **Shared inference**: Phase10 的 22 个 criteria 每个 sample/order 仅推理一次；equal 与 weighted 从同一 root/subtree votes 离线计算。
- **Primary result**: `phase10_final_weighted` K=3 strict Overall ACC。
- **Decisive comparisons**:
  1. Phase10 equal vs Initial：Rubric 演化贡献；
  2. Phase10 weighted vs Phase10 equal：固定权重贡献；
  3. Phase10 weighted vs Native：完整系统贡献。
- **Success criterion**:
  - 最低支持 C1：Phase10 equal 比 Initial 获得正的 paired net corrected；
  - 完整正向结果：Phase10 weighted 同时高于 Initial 与 Native，Coverage 无实质下降；
  - 若 exact McNemar 不显著，只表述为 exploratory evidence。
- **Failure interpretation**:
  - Equal 不高于 Initial：Rubric 演化未迁移；
  - Equal 提升但 weighted 下降：0.4/0.15 权重过拟合开发集；
  - Structured 低于 Native：逐准则分解在该 benchmark 的任务分布上引入了错误聚合。
- **Priority**: MUST-RUN。

## Block B3: Frozen Diagnostics

- **No extra inference**: 全部由 B2 predictions 离线计算。
- **Metrics**:
  - 六个官方 source groups：`povid`、`reasoning_tasks`、`rlaif-v`、`rlhf-v`、`vlfeedback`、`wildvision-battle`；
  - 每个 root/subtree 的 strict ACC、Coverage、covered ACC；
  - Visual-only M1；
  - children support、sibling overlap/conflict；
  - native parser-invalid、structured Tie/None；
  - A/B order consistency 与 directional position gap；
  - corrected、harmed、net corrected、exact McNemar。
- **Use**: 解释迁移收益来自哪些视觉偏好子域；任何诊断不得用于重新选择权重或 Rubric。
- **Priority**: MUST-RUN for report，Visual-only 与 child-level表可放 appendix。

## Counterbalanced K=3 Protocol

对每个 pair 的两个原始响应 `(r0, r1)`，用 seed=42 和稳定 SHA256 顺序决定首个排列 `b`，三次顺序固定为 `(b, 1-b, b)`。全数据中 doubled order 近似严格平衡：624 rows 使用 A/B/A，623 rows 使用 B/A/B。

所有输出先映射回原始 response index，再做三票多数。`Tie`、全部 `None` 或 native parse-invalid 在 strict ACC 中计为错误，并单独进入 Coverage/validity 诊断。K=3 的第三次调用是独立 stochastic replicate，不允许复制第一次输出。

## Dual-endpoint Execution Protocol

实验同时使用：

```text
vllm-8000
vllm-8001
```

两个 endpoint 都必须服务 `Qwen/Qwen3-VL-8B-Instruct`。Freeze 分别记录 endpoint ID、base URL、served model、vLLM version、checkpoint provenance、max model length 和健康检查结果。checkpoint 的机器路径可以不同，但 served model 和本实验的 prompt/token 上限必须兼容；任一 endpoint 模型身份不符时硬失败。

请求不使用运行时随机 round-robin，而是冻结确定性分片：

\[
e(r)=\operatorname{SHA256}(\text{system}|\text{replicate}|\text{sample}|\text{criterion hash})\bmod 2.
\]

- hash=0 固定发送到8000，hash=1固定发送到8001；
- 每个 system、replicate 和大规模 criterion batch 的两端负载应接近1:1；
- resume 必须沿用 `endpoint_schedule.json`，不能重新分配已经冻结的 shard；
- transport retry 仍在原 endpoint 上执行；原 endpoint 不可用时暂停实验，不静默 failover；
- 每条 prediction 保存实际 endpoint ID、attempts、errors、latency 和 token metrics；
- Initial 与 Phase10 的 exact prediction reuse 同时要求 prompt/request identity 相同，不因物理 endpoint 不同而重复生成。

Smoke 额外选择20个已调度的结构化请求，在另一 endpoint 做 parity diagnostic，检查两端均能正确接收图像、解析 schema 并返回有效 vote。由于 P05 是 stochastic decoding，vote 完全一致率只作诊断，不设置相等硬门槛；两端 transport/schema valid rate 必须通过。

## Metrics and Statistical Reporting

主要指标：

1. K=3 strict Overall ACC：分母固定 1,247；
2. 官方六 source groups 的 macro ACC；
3. Coverage 与 covered ACC。

统计报告：

- Overall ACC 的 Wilson 95% CI；
- 三个预注册 paired comparisons 的 corrected/harmed/net corrected；
- exact two-sided McNemar p-value；
- replicate-level ACC 与 order disagreement；
- 不因任何 subgroup 或单次 replicate 结果改变主结论系统。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | Estimated cost | Risk |
|---|---|---|---|---:|---|
| M0 | 冻结数据、Rubric、权重和指标 | offline freeze + audit | hashes、映射和双端身份通过 | <5 min | 旧 manifest 污染；使用新目录 |
| M1 | 验证端到端链路 | 固定20 rows × K=3 × native/22 nodes + parity | 两端均可用，映射正确，valid rate可接受 | 约1,400 requests，约0.25–0.5 h | parser、路径或端点漂移 |
| M2 | 建立 native baseline | 1,247 × K=3，双端确定性分片 | 可恢复完成，无 identity drift | 3,741 requests，约0.5–1.5 h | native 输出过长 |
| M3 | 生成结构化预测 | 22 × 1,247 × K=3，双端确定性分片 | 所有 node/order/endpoint shards 完整 | 82,302 requests，约12–20 h | tail latency、单端服务重启 |
| M4 | 离线聚合与报告 | 四系统 + paired diagnostics | manifest/hash 全部一致 | <10 min | 聚合实现漂移 |

主体实验最多 86,043 个唯一模型请求，另有约20个跨端 parity 请求。Initial 的五个 roots 若与 Phase10 完全同 identity，则直接复用 Phase10 对应预测，不增加请求；equal/weighted 聚合也不增加请求。在两端吞吐近似且无资源竞争时预计约 13–22 小时；若两个 API 实际共享同一推理资源，耗时不会严格减半，应以分端 progress 估算。

## CLI and Artifacts

使用新的 stage 与目录，不能读取或覆盖旧 `vl_rewardbench_external_transfer_v1`：

```text
vlrb-phase10-freeze
vlrb-phase10-audit
vlrb-phase10-smoke
vlrb-phase10-run
vlrb-phase10-report
```

```text
output/evolving_structured_rubrics/vl_rewardbench_phase10_transfer_v1/
  frozen_manifest.json
  dataset_manifest.json
  order_schedule.json
  endpoint_schedule.json
  offline_audit.json
  smoke/
  run/native/
  run/structured/phase10_final/
  run/combined/
  report.json
  report.md
```

Runner 必须以 `(system, replicate, sample, criterion_hash)` 为恢复 shard。网络中断可续跑；schema/identity/hash drift 必须硬失败。最终 report 只能读取已经冻结的 inference artifacts，不得发起请求。

## Risks and Mitigations

- **开发集选择偏差**：Phase10 Rubric 和权重来自反复查看 heldout-500。
  - 报告为 exploratory external transfer；VL-RewardBench 不再参与任何选择。
- **训练/测试来源重叠**：VL-RewardBench 包含 RLHF-V/RLAIF-V 来源，且本协议不做重叠审计。
  - 明确写入 manifest 和论文限制，不宣称严格数据独立。
- **位置偏置**：单次 A/B 判断可能偏向固定位置。
  - 使用确定性、全局平衡的 K=3 双顺序协议并报告 order gap。
- **双端模型或运行配置漂移**：8000与8001可能指向不同 checkpoint 或上下文配置。
  - Freeze 两端实际身份，执行双端 smoke/parity，并把每个 shard 永久绑定到一个 endpoint。
- **单端故障造成样本选择偏差**：动态 failover 会改变已冻结的 endpoint 分布。
  - 失败 shard 在原端重试；持续不可用时暂停并显式升级协议，不静默迁移。
- **权重结果被误读为 Rubric 收益**：0.4/0.15 来自开发结果。
  - 同时报告 Phase10 equal，使内容演化与权重收益可分解。
- **运行时间过长或末尾卡住**：82k structured requests 受 tail latency 影响。
  - 原子 shard、实时 progress.json、有限超时与安全恢复；禁止只依赖 stdout ETA。

## Final Checklist

- [ ] Phase10 Rubric hash 固定为 `17ad7a...df7ef`
- [ ] root 权重在 inference 前冻结且和为1
- [ ] Initial、Phase10 equal、Phase10 weighted 使用明确的同一聚合实现
- [ ] equal/weighted 共享相同 node predictions
- [ ] K=3 每个 row 同时出现两种 response order
- [ ] 8000/8001 身份和健康检查均冻结
- [ ] endpoint schedule 确定、近似1:1且恢复时不漂移
- [ ] 每条输出记录实际 endpoint provenance
- [ ] benchmark gold/ranking 未进入模型 prompt
- [ ] strict ACC 分母固定为1,247
- [ ] VL-RewardBench 未用于 checkpoint、criterion 或权重选择
- [ ] 全量测试、focused tests、compile 与 diff checks 通过

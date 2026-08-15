# Pairwise Worker Cache-oriented Prompt Ablation Plan

**Problem**: 当前 Pairwise Worker 在同一图像上逐 criterion 独立推理，但请求很早就在 question/criterion 处分叉，且相同样本的请求可能被分配到不同服务器并同时冷启动，导致 vLLM prefix cache 利用率偏低。

**Method thesis**: 在不改变 Rubric、Worker 模型、解码参数、独立逐准则判断和聚合逻辑的前提下，将固定规则移入 System Prompt、把 criterion 放到 image/question/A/B 之后，并采用 sample-affinity 的 `seed -> fan-out` 调度，可以提高 prefix cache 命中率和吞吐，同时保持判断质量基本不退化。

**Date**: 2026-08-14

## 1. Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：sample-affinity 与 cache warm-up 能提高缓存利用率 | 先证明低命中不只是 Prompt 文本问题，也来自请求路由和并发冷启动 | 保持旧 Prompt 时，S1 相比 S0 的 prefix-cache hit rate、TTFT 或 wall time 明显改善，且请求数和输出协议不变 | B1 |
| C2：cache-oriented Prompt 能进一步提高效率且不损害 Rubric 判断 | 这是后续所有 Split/Refine/benchmark 推理是否值得迁移到 Prompt v2 的直接依据 | S2 相比 S0/S1 更快；discovery-90 与 heldout-500 的 M1 ACC、Coverage 和 parse-valid rate 满足非退化标准 | B2、B3 |

**Anti-claim to rule out**：速度提升只是减少请求、减少节点、缩短输出上限、改变模型或复用了旧 prediction cache，而不是 prefix layout 与调度产生的。

## 2. Frozen Protocol

### 2.1 Rubric and datasets

- Rubric：冻结 Phase10 最终 22-node Rubric；所有系统使用完全相同的 node name、description、topology 和 root aggregation。
- 数据：
  - smoke：从 discovery-90 冻结选择 10 个样本；
  - primary：完整 discovery-90；
  - external diagnostic：既有 heldout-500，仅作 exploratory evaluation。
- 每个样本使用同一冻结 A/B 顺序；不向 Worker 提供 gold、examples、Manager memory 或错误历史。
- 单 replicate、P05；不以 heldout 结果选择 criterion 或继续演化。

### 2.2 Model and decoding

- Worker：`Qwen3-VL-8B-Instruct`。
- API：同时使用8000和8001；freeze 时验证模型、checkpoint 与关键服务身份。
- 固定 `temperature=0.5`、`max_tokens=2048`、parser、structured retry 上限和图像预处理。
- 所有系统具有完全相同的逻辑请求数：`sample_count × 22`；重试单独统计。
- 每个 variant 使用独立、初始为空的 prediction cache；旧实验输出仅作为审计基线，不计入本次计时。

### 2.3 Prompt v2

System Prompt 只包含数据集无关的固定内容；相对旧 Prompt 仅做最小语义泛化：

```text
## Instruction

You are judging a multimodal image-text preference pair under one
criterion. You are given the image, the source instruction or question,
and two candidate responses.

Use the image when the criterion depends on visual evidence. If the
criterion is not applicable to this pair, answer None.

Your response should be in the following JSON format:
{
    "analysis_a": "Analyze A based on the given criterion.",
    "analysis_b": "Analyze B based on the given criterion.",
    "thought": "Compare A and B.",
    "answer": "A / B / None"
}

Return None if any of the following conditions are met:
- The criterion is not applicable to this pair of data pieces.
- They are of the same quality.
- You are unsure.
```

User content 使用以下顺序：

```text
[IMAGE]

## Source Instruction or Question
{question}

## Candidate A
{A}

## Candidate B
{B}

## Criterion
**{criterion}**: {description}

Which candidate better matches the criterion and is more likely to
align with human preference?
```

所有 node 在同一样本上的请求，只有最后的 criterion name/description 不同。System template、User template、content-block order 和两者 hash 必须进入新的 request identity。不得覆盖或伪装成旧 `PAIRWISE_WORKER_PROMPT_VERSION=1.0.0`；新协议使用独立的 pilot version 和缓存目录。

### 2.4 Sample-affinity scheduler

- 使用冻结的 sample hash 将一个样本的全部22个 node 固定映射到8000或8001；重试不得换端口。
- 同一样本先执行 preorder 中冻结的第一个 criterion 作为 seed。
- seed 成功完成后，立即在同一 endpoint 上 fan-out 其余21个 criteria。
- 每个 endpoint 同时只保持少量 active samples，默认 `active_samples_per_endpoint=2`；全局并发上限保持当前配置，不通过提高并发伪造加速。
- 不允许先对全部样本运行 seed 后再统一 fan-out，以避免 sample prefix 在复用前被淘汰。
- scheduler artifact 保存 sample-to-endpoint、seed criterion、提交/完成时间、retry 和 endpoint call counts。

## 3. Compared Systems

| ID | Prompt | Scheduler | Purpose |
|---|---|---|---|
| S0 Control | 当前冻结 `PAIRWISE_MULTIMODAL_WORKER_PROMPT`；全部内容为现有 User content | 当前请求级路由与并发提交 | 重现实验基线 |
| S1 Scheduler-only | 与 S0 byte-identical | sample-affinity + `seed -> fan-out` | 单独测量调度和缓存预热收益 |
| S2 Prompt-v2 | 固定 System + image/question/A/B + criterion-last User | 与 S1 完全相同 | 测量 Prompt layout 的增量收益 |

S0、S1、S2 的模型、Rubric、数据、A/B 顺序、解码、parser、重试上限和聚合完全相同。

## 4. Experiment Blocks

### B0: Offline Contract and Smoke

- **Claim tested**：实现没有改变判断任务或请求规模。
- **Dataset**：冻结的10个 discovery 样本，共 `10 × 22 = 220` 个逻辑请求/variant。
- **Checks**：
  - S0 prompt/request identity 与当前实现一致；
  - S1 只改变 scheduler identity；
  - S2 的 System/User template、image-first 和 criterion-last 顺序正确；
  - 同一样本全部 nodes 落在同一 endpoint；
  - seed 完成时间早于该样本的 fan-out 提交时间；
  - 三个系统逻辑请求数完全相等；
  - 无 gold、dataset name、examples 或 Manager memory 进入请求。
- **Go gate**：所有请求完成，final-valid rate 至少99%，无 identity drift 或 endpoint-affinity violation。
- **Priority**：MUST-RUN。

### B1: Discovery-90 Scheduling and Prompt Ablation

- **Claim tested**：分别隔离 scheduler 和 Prompt layout 的效率贡献。
- **Dataset**：discovery-90，22 nodes，单 replicate；每个 variant 1,980 个逻辑请求。
- **Compared systems**：S0、S1、S2。
- **Primary efficiency metrics**：
  - vLLM prefix-cache hit rate 及 cache-hit tokens；
  - total prompt tokens、uncached prompt tokens；
  - TTFT 的 median/P90/P99；
  - requests/minute、总 wall time；
  - 每 endpoint 请求量、错误、重试和尾部延迟。
- **Quality metrics**：M1 ACC、Coverage、covered ACC、parse-valid rate、每节点 ACC/Coverage，以及 S2 相对 S0 的 corrected/harmed 和 exact McNemar。
- **Success criteria**：
  - S1 相比 S0 至少满足一项：prefix hit rate `+10pp`、median TTFT `-15%` 或 wall time `-10%`；
  - S2 相比 S0 的 prefix hit rate目标至少 `+15pp`，或 wall time至少降低20%；
  - S2 的 M1 ACC 不低于 S0 超过1.0个百分点，Coverage 不低于1.0个百分点；
  - final-valid rate 不低于 S0，且至少99%。
- **Failure interpretation**：
  - S1 无收益：主要瓶颈不是冷启动/跨端路由，或 vLLM 未缓存当前多模态 prefix；
  - S1 有收益但 S2 无增益：静态/图像 block 没有按预期进入可复用 prefix，需检查 chat template 和多模态 cache key；
  - S2 更快但 ACC 下降：criterion 后置或 System role 改变了 Worker 行为，不能直接作为默认协议。
- **Priority**：MUST-RUN。

### B2: Heldout-500 Efficiency and Generalization Diagnostic

- **Claim tested**：discovery 上的效率提升能否扩展到长运行，并且不破坏既有 heldout 表现。
- **Dataset**：heldout-500，22 nodes，单 replicate；每个 variant 11,000 个逻辑请求。
- **Run policy**：B1 达到 go gate 后运行 S0 与 S2；S1 默认不在 heldout 重跑，除非 B1 需要定位异常。
- **Metrics**：与 B1 相同，额外报告 root/subtree ACC、最终等权 M1、Visual=0.4诊断聚合、paired corrected/harmed 和 exact McNemar。
- **Success criteria**：
  - S2 wall time 相对 fresh S0 至少降低20%，或 uncached prompt tokens 显著减少；
  - 等权 M1 ACC 与 Coverage 均不比 S0下降超过1.0个百分点；
  - parse-valid rate至少99%。
- **Interpretation**：heldout-500 已被多次查看，本结果只用于 Prompt 工程与系统效率诊断，不能作为新的无偏泛化证据。
- **Priority**：MUST-RUN after B1 go。

### B3: Optional Prompt-Causality Diagnostic

仅当 S2 出现明显质量变化时增加一个中间版本：保持 System/User 拆分，但仍使用旧的 `question -> criterion -> A/B` 顺序。它用于区分质量变化来自 System role 还是 criterion-last，不属于首轮必须实验。

- **Priority**：NICE-TO-HAVE / conditional。

## 5. Measurement Controls

**Pilot amendment (2026-08-14)**: 首轮实现采用 `external_observation_no_reset`，不要求 vLLM development endpoints，不由 runner 清理或采集服务端缓存。runner 只记录 wall time、请求吞吐、latency、错误与质量；prefix-cache hit rate 由服务端后台日志另行提供。因此首轮时间对比标记为受运行顺序/热缓存影响的 exploratory 结果，不替代后续 cold-cache confirmatory timing。

- 三个 variant 必须在无其他实验占用8000/8001时运行。
- 每个 variant 开始前同时清理 prefix cache 与 multimodal cache，或重启并重新验证 endpoint identity；不得继承前一 variant 的 vLLM cache state。本地 vLLM 若未暴露 reset API，需以 `VLLM_SERVER_DEV_MODE=1` 启动，仅绑定可信本机地址。
- 记录服务端 prefix-cache counter 的 run-before/run-after 快照；全局日志中的瞬时百分比只作辅助，不能代替实验级计数。
- 为避免顺序和热状态偏置，discovery 的 variant 顺序冻结；若成本允许，再以反向顺序运行10样本 timing sanity。
- 报告逻辑请求数、API attempts 和成功输出数，禁止用更少请求或更低输出上限解释为缓存加速。
- retry 请求保持相同 endpoint；单独报告首轮请求和 retry 的耗时。

## 6. Artifacts

建议输出目录：

```text
output/evolving_structured_rubrics/
  rubric_evolution_phase5/
    pairwise_worker_cache_prompt_ablation_v1/
    frozen_manifest.json
    offline_audit.json
    smoke/
    discovery90/
      s0_control/
      s1_scheduler_only/
      s2_prompt_v2/
    heldout500/
      s0_control/
      s2_prompt_v2/
    final_report.json
    final_report.md
```

每个 variant 至少保存：

- prompt templates、hash 和 request spec；
- sample-to-endpoint schedule；
- seed/fan-out timing trace；
- prediction artifact；
- cache/usage/latency telemetry；
- M1、node、root/subtree metrics；
- corrected/harmed sample IDs；
- parse/retry failure analysis。

## 7. Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Main Risk |
|---|---|---|---|---|---|
| M0 | 离线协议验证 | unit tests + freeze/audit | request identity、content order、affinity 全部通过 | 分钟级，无模型请求 | System/User 消息未按预期序列化 |
| M1 | 10样本 smoke | S0/S1/S2，各220请求 | valid ≥99%，请求数相同，无调度违规 | 约660请求 | seed 与 fan-out 实际同时提交 |
| M2 | discovery消融 | S0/S1/S2，各1,980请求 | S2质量非退化且效率达到至少一项阈值 | 约5,940请求 | 小数据 timing 波动 |
| M3 | heldout确认 | fresh S0 + S2，各11,000请求 | S2速度提升且M1/coverage非退化 | 约22,000请求 | 多模态 prefix 不被 vLLM APC复用 |
| M4 | 条件诊断 | 仅在质量变化时运行B3 | 定位 System role 或顺序效应 | 最多额外11,000请求 | 实验成本膨胀 |

在此前同规模运行经验下，M3 使用两个独立端口预计约3–6小时总计；实际时间以 smoke 的每 endpoint throughput 重新估算，不预先承诺固定加速倍数。

## 8. Final Decision

- **Adopt Prompt v2 by default**：S2 在 discovery 与 heldout 均满足质量非退化，且 heldout fresh timing 达到效率成功标准。
- **Adopt scheduler only**：S1有效，但S2产生超过阈值的质量下降。
- **Keep current protocol**：S1/S2均无稳定效率收益，或多模态 prefix cache 不支持预期复用。
- 无论结果如何，旧 Prompt v1、旧 prediction artifacts 和既有实验 request identity 永久保留；不得用 v2 重新解释历史实验。

## 8.1 S3 Additive Extension

Discovery-90 showed that the affinity scheduler reduced effective concurrency
while Prompt v2 reduced per-request latency. S3 therefore isolates the
production-oriented combination:

```text
S3 = Prompt v2 + the original S0 available-slot dynamic scheduler
```

S3 is an additive extension: it freezes the completed S0/S2 artifact hashes,
generates only the missing 1,980 S3 predictions, and never rewrites the frozen
S0/S1/S2 outputs. Its report compares S3 against both S0 and S2 for wall time,
effective concurrency, M1, coverage, valid rate, and paired corrected/harmed
samples. The extension remains discovery-only and does not access heldout-500.

## 8.2 S3 Heldout-500 Semantic-drift Diagnostic

After S3 passes discovery, run only the two systems required to test deployment:
S0 (Prompt v1 + original dynamic scheduling) and S3 (Prompt v2 + original
dynamic scheduling). Each system evaluates the frozen 22-node Phase 10 rubric
on all 500 heldout samples, for 11,000 logical judgments per system. S2 is not
rerun because the deployment question no longer concerns affinity scheduling.

The primary safety gate is semantic non-inferiority: S3 M1 accuracy and coverage
may not fall more than 1 percentage point below S0, and final-valid rate must be
at least 99%. The report also includes corrected/harmed samples, exact McNemar,
all 22 node-level accuracy/coverage deltas, and wall-time/throughput diagnostics.
Because heldout-500 has been inspected previously, this is an exploratory
deployment diagnostic rather than a new unbiased generalization claim.

## 9. Final Checklist

- [ ] Prompt v2 不包含 RLHF-V 或 benchmark 名称
- [ ] 除最小任务泛化与字段重排外不新增判断规则
- [ ] criterion 位于 image/question/A/B 之后
- [ ] 同一样本所有 nodes 固定到同一 endpoint
- [ ] seed 完成后才 fan-out
- [ ] 三个系统请求数、Rubric、解码和聚合完全一致
- [ ] fresh cache 下测量速度
- [ ] discovery 与 heldout 都报告效率和质量
- [ ] heldout 明确标记 exploratory
- [ ] v2 使用独立 prompt version、request identity 和输出目录

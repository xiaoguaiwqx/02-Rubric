# Role-aware Refine Checkpoint Heldout Diagnostic

**问题**：Role-aware Refine 在 discovery-90 的 epoch 1 达到最高完整 M1，但 epoch 2–3 在继续接受局部 node-ACC 提升后回落。需要判断这是 discovery-only 的偶然峰值，还是 repeated Refine 的真实累计回归。

**方法命题**：在不重新选择或修改任何 criterion 的前提下，对冻结的 source、epoch 1、epoch 3 rubric 做同一 heldout-500 的配对 checkpoint 诊断。

**日期**：2026-08-10

## Claim Map

| Claim | 为什么重要 | 最小可信证据 | Block |
|---|---|---|---|
| C1 | epoch 1 的 role-aware Refine 收益不是仅限 discovery 的假象。 | epoch 1 heldout M1 高于 source，且 corrected > harmed。 | B1 |
| C2 | repeated Refine 可能损害已改善的全局组合。 | epoch 3 相对 epoch 1 的 heldout net corrected 为负，或至少未保留 epoch 1 的增益。 | B1 |

**反主张**：通过 heldout 选择最佳 checkpoint 或调节 Refine 参数获得收益。本实验明确禁止这两种行为。

## 冻结系统与比较

| ID | 系统 | Rubric 来源 | discovery-90 M1 | 角色 |
|---|---|---|---:|---|
| S0 | Source Split-only + Global Memory | `phase6_split_only_evolution_global_memory_v1/final/` | 57/90 = 63.33% | 冻结基线 |
| S1 | Role-aware Refine epoch 1 | `phase8_refine_role_aware_v2/epochs/epoch_01/` | 62/90 = 68.89% | discovery peak |
| S3 | Role-aware Refine epoch 3 | `phase8_refine_role_aware_v2/epochs/epoch_03/` | 60/90 = 66.67% | 当前 final checkpoint |

这不是 role-aware 与 uniform 的因果消融：`all_committed_nodes` 与 role-aware child trigger 同时存在。本实验只诊断同一 role-aware trajectory 中的 checkpoint 泛化。

## Block B1 — Frozen Checkpoint Paired Heldout Evaluation

- **Claim tested**：C1、C2。
- **数据**：同一 `heldout-500`；沿用既有 source heldout artifact 与 Pairwise request identity。
- **Worker**：Qwen3-VL-8B-Instruct，P05，单 replicate，仅经 `vllm-8001` 执行；不调用 397B Manager。
- **来源复用**：S0 的 17 个 heldout node outputs 必须逐项复用并验证 request identity。
- **新增预测**：相对 S0，S1 改写 4 个 description；S3 改写 9 个。S1 的 4 个 description 都是 S3 的子集，因此跨 checkpoint 只生成 **9 × 500 = 4,500** 个唯一 Pairwise 请求。
- **禁止事项**：不得生成 proposal、不得 Refine、不得改变 trigger/接受条件、不得根据 heldout 选择 checkpoint、不得重新跑 source。

### Freeze protocol

运行前写入 `heldout_checkpoint_diagnostic/frozen_manifest.json`，冻结：

1. S0/S1/S3 rubric SHA256、节点 description SHA256 与 preorder；
2. heldout dataset SHA256、500 个 sample IDs/fingerprints；
3. source `combined_pairwise` SHA256 与 request spec；
4. S1/S3 相对 source 的 changed-node map；
5. 9 个唯一待生成 request identities；
6. 当前 Worker endpoint identity 与 `P05` 解码配置；
7. `selection_after_heldout_forbidden=true`。

任何 Rubric、数据、request identity 或 changed-node map 漂移都硬失败。

### Execution

1. 验证并投影 S0 已有 heldout prediction；
2. 对 9 个 unique `(criterion name, description)` 生成缺失的 500-sample outputs；相同 description 在 S1/S3 之间只生成一次；
3. 为 S1、S3 分别把 source unchanged nodes 与新 outputs 合并为完整 Pairwise artifact；
4. 用相同 M1 offline executor 重放三个 checkpoint；
5. 仅生成报告，不写回 Role-aware Refine 的 discovery decision 或最终 rubric。

### Primary metrics

| 比较 | 主要指标 | 配对指标 |
|---|---|---|
| S1 vs S0 | heldout-500 M1 ACC | corrected、harmed、net corrected、exact McNemar |
| S3 vs S0 | heldout-500 M1 ACC | corrected、harmed、net corrected、exact McNemar |
| S3 vs S1 | heldout-500 M1 ACC | corrected、harmed、net corrected、exact McNemar |

每个系统额外报告 Coverage、covered ACC、Tie rate、正确数和 Wilson 95% CI。报告 epoch 1→3 的具体 corrected/harmed IDs，作为 repeated Refine 轨迹诊断。

### 判定与解释

| heldout 结果 | 解释 |
|---|---|
| S1 > S0，且 S3 < S1 | 支持“第一轮 role-aware 有效；重复 Refine 累计回归”。下一步做 `one-accept-per-node` 调度消融。 |
| S1 ≈ S0，S3 ≈ S0 | discovery 峰值未稳定泛化；role-aware 仅有 pilot evidence。 |
| S1 < S0 | epoch 1 的 discovery 增益主要是适应性收益；暂停扩大 Role-aware Refine。 |
| S3 ≥ S1 > S0 | discovery 的后期回落未泛化；说明 M1 discovery 方差较大，保留 repeated Refine 作为候选。 |

不以 McNemar 显著性作为唯一门槛；样本数 500 下未显著时仅报告 exploratory paired evidence。

## Run Order and Milestones

| Milestone | 目标 | Stage | 门槛 | 成本 | 风险 |
|---|---|---|---|---:|---|
| M0 | 冻结/离线审计 | `refine-role-checkpoints-freeze` | 9 个唯一 description、无 heldout API | 数秒 | checkpoint hash 漂移 |
| M1 | 生成与合并 | `refine-role-checkpoints-run` | 仅 9×500 缺失输出；可恢复 | 约 2.5–3.5 小时 | 8001 连接重试 |
| M2 | 配对报告 | `refine-role-checkpoints-report` | 三系统 hash 与 prediction coverage 一致 | 数秒 | 不得据此重选 rubric |

## Compute and Data Budget

- 新 Manager 请求：0。
- 新 Worker 请求：4,500。
- 基于此前单 criterion heldout-500 约 16.4 分钟，理论约 148 分钟；考虑连接重试，预算 2.5–3.5 小时。
- heldout 已在历史实验中使用；本实验必须标为 **exploratory paired checkpoint diagnostic**，而非新的 confirmatory test。

## 验收清单

- [ ] S0 outputs 完全复用，未重新请求。
- [ ] S1/S3 的共享 description 只生成一次。
- [ ] Worker identity、P05、数据 hash 与 source 一致。
- [ ] 运行阶段无法读取 discovery labels 以外的任何选择信号；不写回 discovery state。
- [ ] S0/S1/S3 三套完整 Pairwise artifact 均通过 M1 replay。
- [ ] 每个比较同时报告 ACC、Coverage、corrected/harmed、McNemar 和 CI。
- [ ] heldout 后没有新的 Refine、重新调度或 checkpoint 选择。

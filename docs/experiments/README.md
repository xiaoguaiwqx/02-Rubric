# 实验索引

方法、配置与统一命令入口见仓库 [README](../../README.md)，代码职责与输入边界见 [架构说明](../architecture.md)。主要框架图为 [assets/framework.png](../../assets/framework.png)。

## 当前方法与冻结数据

| 实验 | 主要内容 | 说明与证据 |
|---|---|---|
| 子树逐例反思与局部竞争 | R0、S0、局部候选验证、Global Arbiter 与 Final | [冻结协议](subtree-local-reflection/plan.md)、[历史结果表](subtree-local-reflection/results.md) |
| Hallucination100 生成 root | seed11 冻结训练100、幻觉留出648，Strict + Preserve5 | [原协议与历史结果](vlrb-hallucination100-generated-roots/plan.md)、[划分 ID](vlrb-hallucination100-generated-roots/seed11_split.json) |
| 英文 V3 Init Split | 偏好导向 signature/cluster，仅 children 参考本轮前序子准则 | [初始化与随机换位结果摘要](init-split-prompts/results.md) |
| Reasoning70 | 固定训练70，其余全部 Reasoning247 留出，十对预热、V3 初始化、五轮演化 | [协议、结果与完整命令](vlrb-reasoning70/README.md)、[划分 ID](vlrb-reasoning70/seed11_split.json) |

VLRB 训练子集放在本地 `data/VL_RewardBench/splits/`：

- Hallucination100：[discovery_100.jsonl](../../data/VL_RewardBench/splits/hallucination100/seed11/discovery_100.jsonl)。
- Reasoning70：[discovery_70.jsonl](../../data/VL_RewardBench/splits/reasoning70/seed11/discovery_70.jsonl)。

两者引用共享的 `data/VL_RewardBench/dataset_images/`，不复制图片。训练 JSONL、图像、实际服务配置及原始产物保持本地保存；Git 只保存示例配置、必要的结果摘要和冻结 split 元数据。

## 入口与计分

统一入口为 `experiments.evolving_structured_rubrics.run_subtree_experiment`：`prepare` 准备划分，`roots` 创建 F5/G5/GN，`init` 独立生成 S0，`evolve` 继续局部演化，`vlrb-r0`/`vlrb-s0`/`vlrb` 评测冻结版本，`report`/`report-init` 离线汇总已保存预测。G5/GN 默认五对预热，可通过 `roots --warmup-count` 设置；Reasoning70 的协议显式使用十对。

新外评使用 `independent-random-v1`：seed42 为每条样本的每次判断独立决定是否交换回答。先还原原回答身份，K=3 至少两票一致，否则弃权，Strict ACC 将弃权计错。旧 `alternating-v1` 结果保留为历史证据；不同换位协议不混入同一份配对报告。方法阶段 K=1 和正式外评 K=3 分别记录。

Hallucination100 的报告保留训练100、未训练1147、clean1146、幻觉留出648及三个官方类别；Reasoning70 使用训练70、全部非训练1177、Reasoning留出247及三个官方类别。全量1247与训练切片包含学习数据，不能作为独立未见样本证据。

## 主要观察

幻觉数据生成的 V3 在独立随机换位下，R0/S0/Final 的全量 Strict ACC 为 **74.26% / 77.63% / 76.82%**，幻觉留出648为 **82.56% / 86.88% / 86.73%**。这些是固定版本的外评，详细来源及历史交替换位对照见 [V3 摘要](init-split-prompts/results.md)。

Reasoning70 的 Reasoning247 Strict ACC 为 R0 **64.78%（160/247）**、S0 **63.16%（156/247）**、Final **61.94%（153/247）**，本次没有观察到主任务提升。Final 在 General 中有2次技术失败，完整报告汇总仍被阻断；[结果文档](vlrb-reasoning70/README.md#已观察结果g5-seed11)按保存预测重算并注明此状态。保留负结果，不能以局部 Worker 更新被接受代替整体系统改善。

## 历史分支与本地产物

`init-split-prompts` 分支保存完整提示词探索、计划、进度、各版本对照和 PPT 更新。主线保留英文 V3 核心实现与精简结果；历史 [框架 PPT](subtree-local-reflection/framework.pptx) 留作参考，其旧结果不代表当前随机换位协议。

旧 Gate/Cascade、joint、Phase17–22 和其他历史 runner 保存在 `codex/subtree-local-reflection`；原 CritiQ-V 保存在 `CritiQ-V`。整个 `output/` 只在本地保存，历史产物位置见 [目录归档清单](output-archive.md)。

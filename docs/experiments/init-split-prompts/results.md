# V3 Init Split 协议与结果摘要

当前初始化使用 `init-split-template-en-v3-sequential-children`。系统提示词和描述性用户模板集中在 `experiments/evolving_structured_rubrics/init_split.py`，运行中的提示词为英文，中文草稿保留为注释。完整版本探索、原计划、进度、v1/v2/v4 对照和 PPT 更新保存在 `init-split-prompts` 分支的 `52596ff`。

## 方法与输入边界

- 三个阶段均读取同一份 R0 的全部 root 职责。本次分析或生成只针对目标 root，根本身不修改。
- signature 读取图像、问题、两候选回答、目标 Worker 报告与整体人类偏好，提炼有证据支持的比较依据；不读取 Global Arbiter 报告。
- cluster 按共同适用情境、偏好因素与权衡依据归纳模式；至少两条不同 signature 支持一个 cluster，ID 不跨组重复。
- children 生成2–5条完整子准则。只有此阶段按 root 顺序读取本轮前面已生成的 children，保持各 Worker 可独立使用本组准则。
- 初始化先审查目标 Worker 与人类偏好不一致的训练案例；不足4条有效 signature 或2个 cluster 时补充其余训练案例，仍不足则停止。
- `init --source-run` 复用冻结 R0 与 K=1 训练报告；`--reuse-init-patterns` 另复用来源 signature/cluster，仅重新生成 children。默认完整初始化执行三个阶段。

模型为 Manager `Qwen/Qwen3.5-27B`、Worker/Arbiter `Qwen/Qwen3-VL-8B-Instruct`。Manager temperature=0.2、关闭 thinking；Worker temperature=0.5、max_tokens=2048。完整初始化、独立 S0 外评及来源复用命令见仓库 [README](../../../README.md#只验证-init-split)。

## 当前外评：独立随机换位 K=3

本地来源为 `output/init_split/seed11/template_en_v3_sequential_random_swap_k3/g5/`。它对已冻结的 R0、V3 S0 和五轮演化 Final 重新执行外评；表格来自该目录 `report.json` 中的 `generated_roots.systems`。本次整理仅核对已保存产物，没有重新调用模型。

协议为 `independent-random-v1`、seed42。每次判断独立随机交换回答，投票前还原原始回答身份，至少两票支持同一回答才得到结果；弃权计错。下表均为 **Strict ACC**，训练100和全量1247包含用于学习的样本。

| 评测切片 | R0：仅 root | V3 S0 | 五轮演化 Final |
|---|---:|---:|---:|
| 全量1247 | 926/1247；74.26% | 968/1247；77.63% | 958/1247；76.82% |
| 未训练1147 | 844/1147；73.58% | 880/1147；76.72% | 869/1147；75.76% |
| clean1146 | 843/1146；73.56% | 879/1146；76.70% | 868/1146；75.74% |
| 幻觉留出648 | 535/648；82.56% | 563/648；86.88% | 562/648；86.73% |
| 训练100 | 82/100；82.00% | 88/100；88.00% | 89/100；89.00% |
| General181 | 104/181；57.46% | 107/181；59.12% | 102/181；56.35% |
| Hallucination749 | 618/749；82.51% | 652/749；87.05% | 652/749；87.05% |
| Reasoning317 | 204/317；64.35% | 209/317；65.93% | 204/317；64.35% |

R0→S0 的幻觉留出净增加28个正确判断；S0→Final 减少1个。S0→Final 的全量净减少10个，General 和 Reasoning 各减少5个；局部演化没有继续提升整体外评，不能据训练100的上升认定泛化改善。

## 历史初始化对照：交替换位 K=3

2026-10-05 的 v3 初始化实验固定同一份五根 R0，复用偏好导向 v2 的 signature/cluster，仅重新生成 children。v2 与 v3 的本地目录分别为 `output/init_split/seed11/template_en_v2/g5/` 与 `output/init_split/seed11/template_en_v3_sequential/g5/`。这组历史外评采用 `alternating-v1`，与上面的独立随机换位结果分开比较。

| 历史切片 | v2 S0 | v3 S0 | v2→v3 净正确数变化 |
|---|---:|---:|---:|
| 全量1247 | 962/1247；77.15% | 963/1247；77.23% | +1 |
| 未训练1147 | 877/1147；76.46% | 873/1147；76.11% | −4 |
| 幻觉留出648 | 553/648；85.34% | 562/648；86.73% | +9 |
| General181 | 112/181；61.88% | 110/181；60.77% | −2 |
| Reasoning317 | 211/317；66.56% | 200/317；63.09% | −11 |

历史 v2→v3 的幻觉收益伴随其他类别损失；未训练1147和幻觉留出648的配对 bootstrap 95%区间分别为[−2.09, +1.31]、[−0.62, +3.40]个百分点，均包含零。顺序上下文不能据此认定已稳定解决跨根冗余。v3 仅新增5次 Manager 请求，初始化耗时5.90分钟不包含复用的 signature/cluster 生成成本。

## 产物与研究边界

- 当前随机换位汇总：[report.json](../../../output/init_split/seed11/template_en_v3_sequential_random_swap_k3/g5/report.json)。
- 当前逐样本预测：[vlrb/r0.json](../../../output/init_split/seed11/template_en_v3_sequential_random_swap_k3/g5/vlrb/r0.json)、[vlrb/initial.json](../../../output/init_split/seed11/template_en_v3_sequential_random_swap_k3/g5/vlrb/initial.json)、[vlrb/final.json](../../../output/init_split/seed11/template_en_v3_sequential_random_swap_k3/g5/vlrb/final.json)。
- 历史 v2→v3 初始化配对：[init_comparison.json](../../../output/init_split/seed11/template_en_v3_sequential/g5/init_comparison.json)。
- 在 Reasoning70 上重新挖掘的 V3 是另一套 R0/S0/Final，不是上述 Rubric；其主留出指标下降，见 [Reasoning70 结果](../vlrb-reasoning70/README.md)。

原始预测、缓存、图像与日志保持本地保存。历史与当前外评使用不同换位协议，不能合并成同一份配对报告；单次 seed 的观察也不代表初始化或演化必然改善未见人类偏好判断。

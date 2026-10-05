# Init Split 提示词实验结果

当前记录 v3 顺序 children 上下文实验，初始化与正式 VLRB K=3 外评于 2026-10-05 完成。固定 G5 seed11 的同一份五根 R0，复用偏好导向 v2 的 signature/cluster，只重新生成 children；v2 为主要对照，旧 S0 与英文模板 v1 保留作历史参考。本页比较初始化 S0，不包含后续局部演化结果。

## 1. 实验设置

| 项目 | 设置 |
|---|---|
| R0 与旧 S0 来源 | `output/subtree_reflection/seed11/g5/` |
| 英文模板 v1 | `output/init_split/seed11/template_en_v1/g5/`；侧重判断问题和执行困难 |
| 偏好模板 v2（对照） | `output/init_split/seed11/template_en_v2/g5/`；侧重人类选择的比较依据和权衡模式 |
| 顺序 children v3（当前） | `output/init_split/seed11/template_en_v3_sequential/g5/`；复用 v2 signature/cluster，仅重新生成 children |
| 初始化证据 | 同一份幻觉发现集 100 条、已保存的 R0 Worker K=1 报告；不重新预热或生成 root |
| Manager | `Qwen/Qwen3.5-27B`，temperature=0.2，关闭 thinking |
| Worker / Global Arbiter | `Qwen/Qwen3-VL-8B-Instruct`，temperature=0.5，max_tokens=2048 |
| 正式计分 | K=3，还原原始 A/B 映射，至少两票选择同一回答；否则弃权。Strict ACC 将弃权计错 |
| 共用上下文 | 每个阶段提供全部 root；signature 提供目标 Worker 报告，不提供 Global Arbiter 报告 |
| v2→v3 的改动 | 仅 children 阶段增加本轮前序 children：Root 1 无，Root 2 看 Root 1，依次递增至 Root 5 看 Root 1–4；signature/cluster 的材料与提示词不变 |

1147 为全量排除发现集；幻觉留出集 648 排除发现集和该近重复样本。全量 1247 与 Hallucination 749 均包含发现集，不作为独立未见样本结果。

## 2. 正式 K=3 结果

均为 **Strict ACC（弃权计错）**，括号内为正确数/总数；表中粗体表示当前 v3 结果。

| 评测范围 | R0，仅 root | 旧 S0 | 英文模板 v1 S0 | 偏好 v2 S0 | **顺序 children v3（当前）** |
|---|---:|---:|---:|---:|---:|
| VLRB 全量 1247 | 74.18%（925/1247） | 74.82%（933/1247） | 75.14%（937/1247） | 77.15%（962/1247） | **77.23%（963/1247）** |
| 未参与初始化 1147 | 73.32%（841/1147） | 74.19%（851/1147） | 74.37%（853/1147） | 76.46%（877/1147） | **76.11%（873/1147）** |
| 幻觉留出集 648 | 83.49%（541/648） | 83.95%（544/648） | 84.26%（546/648） | 85.34%（553/648） | **86.73%（562/648）** |
| 发现集 100，正式 K=3 | 84.00%（84/100） | 82.00%（82/100） | 84.00%（84/100） | 85.00%（85/100） | **90.00%（90/100）** |
| General 181 | 51.38%（93/181） | 54.14%（98/181） | 53.04%（96/181） | 61.88%（112/181） | **60.77%（110/181）** |
| Hallucination 749 | 83.58%（626/749） | 83.71%（627/749） | 84.25%（631/749） | 85.31%（639/749） | **87.18%（653/749）** |
| Reasoning 317 | 64.98%（206/317） | 65.62%（208/317） | 66.25%（210/317） | 66.56%（211/317） | **63.09%（200/317）** |

### v3 相对 v2 的配对变化

新增正确指 v2 错误而 v3 正确；失去正确反之。变化使用未四舍五入的 ACC 计算，单位为百分点。

| 评测范围 | 新增正确 | 失去正确 | 净变化 | ACC 变化 |
|---|---:|---:|---:|---:|
| 全量 1247 | 51 | 50 | +1 | +0.08 |
| 发现集 100 | 7 | 2 | +5 | +5.00 |
| 未参与初始化 1147 | 44 | 48 | -4 | -0.35 |
| 幻觉留出集 648 | 25 | 16 | +9 | +1.39 |
| General 181 | 10 | 12 | -2 | -1.10 |
| Reasoning 317 | 9 | 20 | -11 | -3.47 |

### 按数据来源拆分

以下切片包含发现集重叠部分。

| 数据来源 | 旧 S0 | 英文模板 v1 | 偏好 v2 | **顺序 children v3（当前）** |
|---|---:|---:|---:|---:|
| WildVision 171 | 52.63%（90/171） | 51.46%（88/171） | 59.65%（102/171） | **59.06%（101/171）** |
| VLFeedback 10 | 80.00%（8/10） | 80.00%（8/10） | 100.00%（10/10） | **90.00%（9/10）** |
| RLAIF-V 233 | 70.82%（165/233） | 70.39%（164/233） | 75.97%（177/233） | **78.11%（182/233）** |
| POVid 448 | 90.85%（407/448） | 91.74%（411/448） | 90.85%（407/448） | **93.08%（417/448）** |
| RLHF-V 68 | 80.88%（55/68） | 82.35%（56/68） | 80.88%（55/68） | **79.41%（54/68）** |
| Reasoning 317 | 65.62%（208/317） | 66.25%（210/317） | 66.56%（211/317） | **63.09%（200/317）** |

## 3. 各 root 与初始化诊断

各 root 单独的 Worker 判断采用正式 K=3，与整体人类偏好比较；它们不等同于该 root 职责是否执行正确，也不能直接解释为系统贡献。

| Root | 旧 S0：幻觉留出集 648 | v1：幻觉留出集 648 | v2：幻觉留出集 648 | **顺序 children v3（当前）** |
|---|---:|---:|---:|---:|
| factual_accuracy | 78.55%（509/648） | 71.45%（463/648） | 78.86%（511/648） | **80.25%（520/648）** |
| visual_grounding | 80.25%（520/648） | 83.02%（538/648） | 80.56%（522/648） | **79.01%（512/648）** |
| uncertainty_handling | 75.46%（489/648） | 80.71%（523/648） | 84.10%（545/648） | **82.25%（533/648）** |
| information_relevance | 74.23%（481/648） | 75.93%（492/648） | 74.85%（485/648） | **73.92%（479/648）** |
| hallucination_avoidance | 81.33%（527/648） | 80.71%（523/648） | 85.19%（552/648） | **85.19%（552/648）** |


## 4. 耗时与重要结论

| 项目 | 英文模板 v1 | 偏好 v2 | **顺序 children v3（当前）** |
|---|---:|---:|---:|
| init split | 28.00 分钟 | 40.94 分钟 | **5.90 分钟** |
| 正式 VLRB 外评 | 145.85 分钟 | 152.42 分钟 | **150.51 分钟** |
| 两阶段合计 | 173.85 分钟 | 193.36 分钟 | **156.41 分钟** |
| Manager 逻辑调用 / 尝试 | 148 / 149 | 148 / 180 | **5 / 5** |

v2 额外尝试包括 31 次服务端 503 和 1 次解析失败；v1/v2 初始化耗时差不能全部归因于提示词。v3 的 5 个 Manager 请求均一次成功。v3 初始化耗时包含 children 生成和 S0 发现集 K=1 评测，但不包含已复用的 signature/cluster 生成成本，不能将其与完整初始化耗时直接比较。正式预测均无未解决技术失败。

1. **全量基本持平，综合未见样本略降。** 相对 v2，v3 全量净增加 1 个正确判断（+0.08 个百分点），由发现集净增加 5 个、未参与初始化净减少 4 个组成。不能据全量小幅上升认定泛化改善。
2. **幻觉收益伴随其他类别损失。** 幻觉留出集净增加 9 个正确判断（+1.39 个百分点）；Hallucination 全类别净增加 14 个，主要来自 POVid（+10）和 RLAIF-V（+5），RLHF-V 减少 1 个。General 净减少 2 个，Reasoning 净减少 11 个（−3.47 个百分点）。
3. **尚无稳定改善证据。** v2→v3 的配对 bootstrap 95% 区间：未参与初始化 1147 为 [−2.09, +1.31] 个百分点，幻觉留出集 648 为 [−0.62, +3.40] 个百分点，均包含零。区间只反映固定预测的样本抽样，不涵盖重新生成 Rubric 和模型推理的运行波动。
4. **职责协调的作用仍需后续观察。** 顺序上下文使后续根能参考本轮已有指导，但 children 从 19 增至 20，描述长度也增加；当前不认定跨根冗余已被解决。v3 保留为当前 S0 和后续演化起点，观察其演化轨迹及最终结果。

历史 v1→v2：未参与初始化与 General 的改善较明显，主要来自 WildVision；幻觉留出集增量区间仍包含零。v2 已存在幻觉严重度跨根重叠、视觉扎根准则放宽外部知识限制等问题，继续保留为观察项。

## 5. 本地产物与身份

原始预测、缓存、图像和日志仅保存在本地 `output/`、`data/`，本目录只归档精简结果。

| 产物 | 本地路径 |
|---|---|
| v3 比较报告（old_s0 为 v2，new_s0 为 v3） | [init_comparison.json](../../../output/init_split/seed11/template_en_v3_sequential/g5/init_comparison.json) |
| v3 逐样本 K=3 预测 | [vlrb/initial.json](../../../output/init_split/seed11/template_en_v3_sequential/g5/vlrb/initial.json) |
| v3 正式计分摘要 | [vlrb/s0_report.json](../../../output/init_split/seed11/template_en_v3_sequential/g5/vlrb/s0_report.json) |
| v3 Rubric | [init/rubric.json](../../../output/init_split/seed11/template_en_v3_sequential/g5/init/rubric.json) |
| v3 冻结提示词与复用来源 | [init/protocol.json](../../../output/init_split/seed11/template_en_v3_sequential/g5/init/protocol.json) |
| v3 初始化摘要 | [init/summary.json](../../../output/init_split/seed11/template_en_v3_sequential/g5/init/summary.json) |
| v3 运行配置 | [run_config.json](../../../output/init_split/seed11/template_en_v3_sequential/g5/run_config.json) |
| v2 比较报告 | [init_comparison.json](../../../output/init_split/seed11/template_en_v2/g5/init_comparison.json) |
| v2 逐样本 K=3 预测 | [vlrb/initial.json](../../../output/init_split/seed11/template_en_v2/g5/vlrb/initial.json) |
| v2 Rubric | [init/rubric.json](../../../output/init_split/seed11/template_en_v2/g5/init/rubric.json) |
| v2 冻结提示词 | [init/protocol.json](../../../output/init_split/seed11/template_en_v2/g5/init/protocol.json) |
| v2 初始化摘要 | [init/summary.json](../../../output/init_split/seed11/template_en_v2/g5/init/summary.json) |
| v2 运行配置 | [run_config.json](../../../output/init_split/seed11/template_en_v2/g5/run_config.json) |
| v1 比较报告 | [init_comparison.json](../../../output/init_split/seed11/template_en_v1/g5/init_comparison.json) |

Rubric SHA256：

| 版本 | SHA256 |
|---|---|
| 共同 R0 | `6d17a9bc9e77c6b4102e35dfe14c1e404243f33a129fb4253452e30314a0f427` |
| 旧 S0 | `3827f27d936b18891d4028aeb3e3a9806e2aae364cd6204297e4a590335abf5d` |
| 英文模板 v1 S0 | `0d8c942bfbe1f6a691f86dfc776a83495b8e8ffda9dae314dde059dc50ef7d61` |
| 偏好 v2 S0 | `92cc8b18bfe1beac3ccb5416ae201b74d53c258c7e506ef7bc07f4dd392950d0` |
| 顺序 children v3 S0（当前） | `b84ff89f0931c38890e98138c5fc366bbed64b92d6f25263e508a1b8dea81fd8` |

v3 提示词版本为 `init-split-template-en-v3-sequential-children`
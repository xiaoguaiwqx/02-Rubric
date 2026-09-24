# VLRB Hallucination100 演化与同分布留出评估

状态：**seed11 的五轮演化、完整 VLRB K=3 推理和单种子报告已完成；seed29/47 尚未运行。** 以下预注册协议保持冻结，文末另列 seed11 已观察结果。工作分支 `codex/subtree-local-reflection`。运行命令见 [running.md](running.md)。每个 seed 的样本清单、配置、预测及日志保存在独立 `output/` 目录，不覆盖历史实验。

## 研究问题

此前在 Discovery100 上逐例子树反思的 Rubric，演化集从 62/100 提升到 68/100，但 Dev150 从 105/150 降至 103/150，VLRB 从 875/1247 降至 856/1247（官方 K=3 Strict ACC）。其中 Hallucination 为 584/749→572/749。现在只更换**演化样本来源**：从 VLRB Hallucination 组取 100 对，检验在未见过的同组样本上，Final 相对共同 Init 和 Discovery100 演化 Final 是否改善。组内改善仍不能单独区分「视觉任务分布更接近」与「人类偏好标注更一致」两种解释。

本实验的独立变量是演化集；不同时调整 Manager、Worker、prompt、接受规则或聚合方式。由于本课题此前已反复查看 VLRB 汇总和失败样例，留出集是**本实验未参与训练的评估集**，但不是研究过程中的全新盲测。

## 冻结数据与三个抽样种子

数据源为既有 VLRB Parquet，1247 对中官方 Hallucination 组 749 对：POVID 448、RLAIF-V 233、RLHF-V 68。先冻结种子 **11、29、47**，每个种子独立构成 100 对演化集，按来源固定配额 **60 / 31 / 9**；配额近似原组分布。只使用样本 ID、来源和内容去重信息抽样，不根据 Gold、模型预测或历史错误选择困难样本。演化输入的 A/B 展示顺序使用**冻结的完整 VLRB K=3 日程的第一轮**，并据此重标 Gold；不以 A/B 方向挑选样本。三个种子仅衡量**抽样变化**；Manager 与 Worker 的采样随机性没有因此被控制。

对每个种子按以下顺序生成清单，**在任何新模型调用前一次性保存并冻结**：

1. 使用与现有 VLRB 读取器相同的内部 `sample_id`，包括重复原始 ID 的 `__row_####` 后缀；记录源 Parquet 哈希、抽样算法与种子、训练 ID 的有序列表。**先**与旧 Discovery100（生成共同 Init 与历史 Final 所用的样本）做同图和近重复问答交叉筛查；重合条目从抽样池与主留出池一并剔除并列出 ID。文本完全一致只是筛查的一部分，不能替代图片相似性检查。
2. 在筛后的各来源按固定种子打乱；抽取 60/31/9 对。先用图片内容/感知哈希避免训练集内部重复图片；再用规范化问题及 A/B 文本检测近重复。若候选冲突，按打乱顺序取下一条，始终保持训练集恰好 100 个互异样本与来源配额。若配额无法满足，停止并记录原因，不悄悄改变样本数或配额。
3. 同 seed 的 Hallucination 留出集为筛后剩余样本，再剔除与训练集**同图**或近重复问题/回答的条目；保存每个被剔除 ID 和原因。因此留出集至多 649，实际分母以冻结清单为准。三次抽样的留出集可以相互重叠，不把三个 seed 的样本视为独立的 3 倍测试量。
4. General/Reasoning 样本不进入演化，仅用于次要迁移诊断。Dev150 也不用于训练、接受候选或选择轮次。记录它与所选 100 对是否有跨数据源的同图/近重复；如存在，次要诊断剔除并报告。

## 保持不变的演化协议

- **共同起点**：使用既有 `output/subtree_local_reflection/discovery100_27b_strict_preserve5/init/rubric.json` 作为三个 seed 的同一 Init；五根、初始孩子和描述逐字节不变。不能在 Hallucination100 上重新做初始化签名提取、聚类或 Split，否则同时改变起点。每个 seed 要在自己的 100 对上重新评估这个 Init，供局部竞争使用；历史 Discovery100 的 62/100 不能充当新基线。
- **模型与流程**：Manager `Qwen/Qwen3.5-27B`、关闭思考；Worker/Arbiter `Qwen/Qwen3-VL-8B-Instruct`。每根完整子树一次 Worker 判断、五根报告由 Global Arbiter 汇总；逐例独立反思、按根汇总 critiques、整组孩子 Split、五轮演化、每根随机提供 5 条当前正确案例，全部沿用现有协议。`preservation_seed=42` 固定，样本池随演化集改变；仅抽样 seed 为 11/29/47。
- **竞争指标**：继续在本 seed 的全量演化 100 对上以局部 `strict_acc` 严格提高才接受；保留现有 K=1 演化、无效输出处理、prompt、温度与并发设置。Manager 重试上限设为 10（续跑时复用已成功的调用），Worker 上限保持 4；三个 seed 使用同一设置。演化集的固定 A/B 展示顺序明确取官方 K=3 日程第一轮。不按外部留出结果挑选 epoch 或最佳 seed；正式比较使用五轮后的 Final。
- 每个 seed 保存完整运行配置、共同 Init 校验值、训练/留出 ID 哈希、每轮 Rubric 与局部/系统分数、Final Rubric。新运行目录按 seed 分开；不修改历史 `discovery100_27b_strict_preserve5` 产物。

## 对照与评估口径

**每个 seed 完成五轮演化并确定 Final 后，均在完整 VLRB1247 上做一次正式推理**，沿用现有官方 K=3、全量 A/B 顺序日程与聚合协议。首先报告全量 Strict ACC（正确数/1247）、覆盖率、Covered ACC，以及 General、Hallucination、Reasoning 和 Hallucination 三个来源的结果。全量统计包含该 seed 用于演化的 100 对，因此是完整基准表现，**不是独立泛化估计**。

再按该 seed 的冻结 ID 清单，从新 Final 的全量预测和两份历史预测中分别筛出**未参与演化的 Hallucination 留出集**，在同一批样本上比较三份 Rubric：共同 Init、历史 Discovery100 演化 Final、该 seed 新演化的 Hallucination100 Final。留出集 Strict ACC 为正确对数 / 留出对数，`None` 算错。另报覆盖率、Covered ACC、纠正/伤害对数及逐样本配对变化。全量和留出集每个分数都给出正确数与分母，不能只报百分比。

共同 Init 与历史 Discovery100 Final 已有全 1247 对的官方 K=3 逐样本预测。**本实验不为这两份旧 Rubric 发起任何新的 VLRB 推理调用**：核对源数据版本、内部 ID、Gold、A/B 顺序和聚合口径后，直接复用其全量结果，并按**各 seed 的冻结留出 ID**离线过滤重算。若旧产物不完整或口径不一致，先标记对应对照不可用并说明原因，不擅自重跑。只有三个 seed 各自的新 Final 需要完整推理 1247 对，使用同一全量 K=3 顺序日程。训练 100 对可以出现在全量统计中，但绝不计入该 seed 的主留出分数。

展示顺序为完整 VLRB1247 结果、Hallucination 留出集结果、各类别/来源诊断。用于回答「是否迁移到未见同组样本」的主比较仍为：①新 Final − 共同 Init；②新 Final − 历史 Discovery Final；③历史 Discovery Final − 共同 Init，均在每个 seed **相同留出 ID** 上做配对比较。分别给出三个 seed 的结果及增量均值、范围；配对 bootstrap 或精确 McNemar 作为不确定性辅助，不能将重叠留出集简单拼接成独立样本。Dev150 如需报告，单独运行并核对旧预测协议；它不是本次 VLRB 全量推理的组成部分。

| 方法 / rubric | 演化数据 | Seed | VLRB 全量正确/1247 | Hallucination 全组正确/749 | 该 seed 的 Hallucination 留出正确/总数 | 留出 Strict ACC | 留出纠正/伤害（对 Init） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 共同 Init | Discovery100 初始化 | 共用旧结果 | 旧预测复算，不重跑 | 旧预测复算 | 按各 seed 离线过滤 | 待评估 | 基线 |
| 历史 Discovery Final | Discovery100 | 共用旧结果 | 旧预测复算，不重跑 | 旧预测复算 | 按各 seed 离线过滤 | 待评估 | 待评估 |
| 新 Hallucination Final | VLRB Hallucination100 | 11 / 29 / 47 | 每 seed 完整推理 | 每 seed 统计 | 每 seed 过滤统计 | 待评估 | 待评估 |

全量与留出集的覆盖率、Covered ACC，以及 General/Reasoning、POVID/RLAIF-V/RLHF-V 明细另列诊断表；旧 Rubric 从历史逐样本预测统计，新 Final 从本次完整 VLRB 推理统计。

## 事先约定的判断与局限

主要假设是新 Final 在 Hallucination 留出集上相对**共同 Init 和历史 Discovery Final**均有正的配对 Strict ACC 增量。报告全部三个 seed，不挑最优；若至少 2/3 seed 为正且平均增量为正，可称「本实验支持同组迁移改善」，同时报告第三个 seed 和不确定区间；只有单个 seed 正向或仅演化 100 对提升，则不能得出该结论。若覆盖下降，应优先检查正确/错误/None 的逐样本转移，而非只看 Covered ACC。即使成立，也不能证明对 Dev、General/Reasoning 或新的标注人群泛化。

若同组仍无收益，需检查 Hallucination 三个来源的差异、Gold 偏好冲突、各根覆盖与 Arbiter 聚合，不直接归因于数据质量。若同组改善但 Dev/其他组下降，解释为较窄范围的适配，不能称系统整体改进。实验完成后才决定是否追加跨数据集人工偏好一致性审计；不得用本次留出表现回头筛选样本或 Rubric。

## 执行顺序与资源

先冻结三个清单与去重报告，验证两份旧评测产物可按内部 ID 离线复算；随后按 11→29→47 逐 seed 运行相同演化，**仅三个新 Final 各自推理完整 VLRB1247**，再统一生成全量、留出和分组表及 Final Rubric 变化摘要。历史 Preserve5 单条轨迹约 6 小时 20 分（包含实际等待，非保证时长）；三次演化按约 **19 小时以上**预算，另须计入 **3 次完整 VLRB K=3 评测**及服务端限流。调用前应从旧 usage 和当前单价估算费用；本计划不预设精确价格。只要一个 seed 的正式评测未完成，就标记为未完成，不用其他 seed 的高分填补。

离线抽样已得到三个 100 条演化集；seed 11/29/47 的留出集大小分别为 648/648/647，历史 Discovery100 交叉重合为 0，与选中训练样本重合而剔除的留出项分别为 1/1/2。以上只是数据准备结果，不是模型性能；正式运行状态以各 seed 的产物为准。

## 已观察结果：seed11（2026-09-24）

seed11 完成五轮演化，正式 VLRB 为 1247 对、K=3，最终产物技术失败数为 0。Final 相比共同 Init 只改动视觉依据和事实性两根子树；e02 接受这两根，e04 再次接受视觉依据，其他候选均未接受。演化集 K=1 系统正确数为 71→74→76/100；正式 VLRB K=3 在相同 100 条训练 ID 上为 70→76/100。

| Rubric | VLRB 全量 | Hallucination 全组 | seed11 训练 100 条 | 未见 Hallucination 648 条 |
| --- | ---: | ---: | ---: | ---: |
| 共同 Init | 875/1247 (70.17%) | 584/749 (77.97%) | 70/100 | 514/648 (79.32%) |
| 历史 Discovery Final | 856/1247 (68.64%) | 572/749 (76.37%) | 74/100 | 497/648 (76.70%) |
| seed11 Hallucination Final | 892/1247 (71.53%) | 590/749 (78.77%) | 76/100 | 514/648 (79.32%) |

主留出比较：新 Final 相对 Init 为 **23 纠正、23 伤害、净变化 0/648**（精确 McNemar 双侧 p=1.0）；相对历史 Discovery Final 为 **40 纠正、23 伤害、净增 17/648**（p≈0.043，未作多重比较校正）。留出覆盖率：Init 与新 Final 均为 645/648，历史 Discovery Final 为 647/648。

来源拆分显示抵消：未见 POVID 为 352→343/387（−9），未见 RLAIF-V 为 116→125/202（+9），未见 RLHF-V 为 46→46/59。全量 +17 正确相对 Init 分解为训练 Hallucination +6、未见 Hallucination 0、General +5、Reasoning +6；一条因与训练图像近重复而剔除的 Hallucination 样本没有变化。**seed11 不支持“相对共同 Init 改善未见同组样本”的主假设**；相对历史 Discovery Final 的改善主要是恢复了其损失。三个种子未完成，不据此作跨种子结论。

# VLRB General / Reasoning 来源分层十样例预热与类别路由实验

状态：**seed11 主实验及补充阶段对比均已完成**。结果以 `output/vlrb_general_reasoning_warmup10/seed11/report.json` 和 `stage_comparison.json` 为准。

## 1. 问题与比较对象

在原 VLRB seed11 类别划分和演化流程下，仅为 **General**、**Reasoning** 各按本类来源分布选择十条人类偏好样例预热 Manager，由 GN 提示词自选 2–7 个 root，随后初始 Split 和最多五轮逐例子树反思。检验两件事：

1. 新的本类 R0、S0、Final 在本类未见样本上的成绩是否逐阶段改善，以及新 Final 是否优于原随机五样例 Final。
2. 把这两套新 Final 与已完成的 **Hallucination150 6／3／1 十样例 Final** 组成三套 Rubric 后，留出集 3×3 迁移矩阵的对角线是否优于跨类 Rubric，已知类别路由是否优于任一单套 Rubric。

Hallucination 分支完全复用 `output/vlrb_hallucination_warmup631_10/seed11/gn/{final.json,vlrb/final.json}`，不重新预热、演化或推理。历史五根和原三类别实验只读用作参照。**三类别真实标签已知的路由**是本实验的上限式设置；本实验不训练类别分类器，也不声称未知类别输入能够自动路由。

本次同时改变预热的样例数和来源选择，因此“新 Final 优于旧随机五样例 Final”不能独立证明是哪一项造成收益。GN 生成的 root 个数和内容也允许随预热变化。

## 2. 冻结数据、配额与预热

直接复用 `output/vlrb_category_specialized_rubrics/seed11/split.json`、两个分支的 `discovery.jsonl`、配置及完整 VLRB 顺序；不重划分、不移动样本、不使用留出标签选样。类别样本数保持：

| 类别 | 演化集 | 留出集 | 演化集内来源 | 十条预热配额 |
| --- | ---: | ---: | --- | --- |
| General | 36 | 145 | WildVision 34；VLFeedback 2 | WildVision **9**；VLFeedback **1** |
| Reasoning | 63 | 254 | MathVerse 40；MMMU 23 | MathVerse **6**；MMMU **4** |
| Hallucination，复用 | 150 | 599 | POVID 90；RLAIF-V 48；RLHF-V 12 | 已完成的 **6／3／1**，不重跑 |

General 的旧随机五条预热全部来自 WildVision；Reasoning 的旧五条为 MathVerse 2、MMMU 3。新的十条在各来源内按 `sample_id` 排序，使用包含 `seed11`、类别名和来源名的独立固定随机流无放回抽样，再用独立固定随机流打乱输入顺序；沿用现有 `random_reverse` 的 A/B 翻转。记录样本 ID、来源、顺序及翻转后的实际输入，之后不按结果更换。**Reasoning 的训练行 `source` 字段统一写作 `reasoning_tasks`；MathVerse/MMMU 配额必须由 `sample_id` 的数据集前缀区分，不能按该字段抽样。**

General 的十条占 36 条训练集的 27.8%，Reasoning 占 63 条的 15.9%，Hallucination 占 150 条的 6.7%。这些比例不同，跨类别增益不能简单归因为相同强度的预热干预。General 的 VLFeedback 训练样本只有 2 条、留出样本只有 8 条，来源级结果只作描述。

## 3. 保持不变的算法路径

- Manager、Worker、Arbiter、K=1 演化评估、GN 的 2–7 root 数量指令、初始 Split、局部严格准确率接受规则和最多五轮上限保持原协议。预热仍调用现有 `generated_root_initialization.generate_r0_pair(..., warmup_examples=..., variants=("gn",))`；提示词只通过已有样例数变量渲染成 ten，其余内容不修改。
- 每类用完整的本类演化集进行 R0 推理、Split 和逐例反思；十条预热样本仍属于演化集，不从中移除。`signature`、`case_reflection` 阶段并发设为 **15**。这是 2026-09-26 用户在 General R0 已生成、Reasoning 首条预热未返回时作出的运行调度修订；此前未开始这两个 Manager 阶段。并发只改变执行调度，不更改研究方法。
- General 原有 36 条样本可能使某 root 的初始 Split 支持不足。若发生不可行，保留 R0 与失败产物，报告此结果；不临时放宽支持门槛或重抽十条。每个分支都记录 R0、S0、各轮与 Final 的训练预测和严格准确率，但**不以训练增益替代留出集结论**。
- 等两套 Final 全部冻结后再进行所有外部评估；不能依据外部成绩选择 root、轮次或重新演化。

## 4. 最少必要的新推理与复用

1. **General、Reasoning Final 各一次全量评估**：分别对完整 VLRB 1247 条按原 K=3、固定 A/B 日程和相同 Worker 配置推理一次。由每份预测只读切出全量和留出类别结果；不对 3×3 九格分别推理。
2. **阶段归因评估**：为了直接回答“演化是否变好”，在 Final 冻结后，再让 General 与 Reasoning 的 R0、S0 各自仅评估**本类留出集**（General 145、Reasoning 254），K=3，沿用从完整 1247 条生成的原 A/B 日程。Final 的本类结果从全量预测切片，不重复调用。逐样本比较 R0→S0→Final，区分预热根、Split、后续五轮的贡献。
3. **Hallucination 行只读复用**：从上述历史 `gn/vlrb/final.json` 读取逐样本预测，按与新两行相同的严格多数票规则重新切片。原随机五样例 General/Reasoning Final 和历史 Hallucination100 生成五根也只读复用，不补跑旧系统。

旧阶段的训练准确率是 K=1；新外部评估是 K=3，不能把两者直接当作泛化差距。若新 Final 有技术失败，按既有缓存和断点路径修复后续跑；技术失败不能当作正常 `None` 弃权。

## 5. 主要结果表和判据

先以 **留出 998** 形成主 3×3 矩阵，再报告全量 1247 的同结构矩阵。行是使用的 Final Rubric，列是测试样本类别；三个行总分分别表示不路由、只使用一套 Rubric。每格列出正确数/总数、Strict ACC（严格多数票、弃权计错）及覆盖率；路由等于三格对角线正确数相加。另给出三类别等权宏平均，避免 Hallucination 的 599 条主导全部解读。

本轮运行前，旧 General / 新 Hallucination / 旧 Reasoning 的留出矩阵为：

| Rubric | General 145 | Hallucination 599 | Reasoning 254 | 合计 998 |
| --- | ---: | ---: | ---: | ---: |
| 原 General | 76 | 451 | 171 | 698 |
| **已完成的新 Hallucination** | **79** | **492** | 163 | **734** |
| 原 Reasoning | 67 | 405 | 163 | 635 |

由此得出的临时混合路由是 **76＋492＋163＝731/998**，尚低于新 Hallucination 单套 **734/998**。设新 General 在 General 留出集上答对 `G` 条，新 Reasoning 在 Reasoning 留出集上答对 `R` 条，新路由的总正确数就是 **G＋492＋R**。它要严格超过已知的新 Hallucination 单套 734，须 `G＋R ≥ 243`；历史 Hallucination100 生成五根单套为 747/998，严格超过该参考值须 `G＋R ≥ 256`。这两个门槛只是预先说明的参考值；正式结论仍须和**运行后实际最佳的新单套 Rubric**逐样本比较。

预先报告三类判据，不以其中最有利的一项代替其余项：

1. **本类改进**：新 General 对原 General 在 145 条上的 76 个正确样本；新 Reasoning 对原 Reasoning 在 254 条上的 163 个正确样本，均做配对比较。同时报告各自 R0、S0、Final 的同类留出变化。
2. **专属性**：每个留出类别中，本类 Final 与其他两类 Final 逐样本配对比较。当前参照下，General 要超过新 Hallucination 的 79/145，Reasoning 要注意原 General 已有 171/254；正式判断以三套新 Final 的完整矩阵为准。
3. **路由效用**：已知类别路由逐样本对比三套新 Final 各自用于全部留出 998 条的结果，以及临时混合路由 731/998；同时列历史五根单套 747/998 供参考。主要读数为正确样本净变化，附配对纠正/伤害数、固定 seed 的配对 bootstrap 95% 区间与 McNemar 精确双侧 p 值。多项比较及多次查看过的 VLRB 留出集意味着显著性只作探索性证据。

如果 G/R 本类提高，但路由仍不胜最佳单套，只能说明专属 Rubric 有局部收益，不能声称类别标签路由有效。如果训练集提高而本类留出不提高，应保留负结果并分析样本量与偏好覆盖。不要用本次测试集反复挑选预热样本。

## 6. 最小实现、产物与执行顺序

新增一个薄入口 `experiments.evolving_structured_rubrics.vlrb_general_reasoning_warmup10`，以及独立目录 `output/vlrb_general_reasoning_warmup10/seed11/`。入口只做确定性选样、保存配置、顺序调用现有 root 生成和 `subtree_local_reflection.run`、调用现有 `framework_v6.evaluate`、读取已有预测生成报告；不修改旧三个类别的目录或全局常量，不复制 Worker、Manager、Split 和计分算法。

| 阶段 | 产物与完成标准 |
| --- | --- |
| M0 冻结输入 | 复用原 split，保存两组十条预热 ID、来源、顺序、配置和独立协议名 |
| M1 生成 R0 | General、Reasoning 各有预热 transcript、生成响应、GN R0 rubric |
| M2 Split＋演化 | 两组均有 R0、S0、最多五轮产物、`state.json` 与冻结 Final；失败则保留证据 |
| M3 外部推理 | 新 G/R Final 各完成全量 1247；新 G/R 的 R0、S0 各完成本类留出集；H 只读复用 |
| M4 报告 | 保存全量与留出 3×3、三行单套成绩、路由、阶段配对结果和来源细分；更新本文件结果节与实验索引 |

Final 两次全量推理的逻辑调用数为 `3×1247×[(N_G+1)+(N_R+1)]`，实际 root 数由 Manager 决定；另加 R0/S0 在 145、254 条上的诊断调用。优先完成两套 Final 与主矩阵，再完成阶段归因诊断；不因主矩阵结果改变冻结阶段评估范围。计划阶段不运行任何实验，不产生新结果。

已实现入口：`experiments/evolving_structured_rubrics/vlrb_general_reasoning_warmup10.py`。运行时在 `critiq` 环境中执行 `python -m experiments.evolving_structured_rubrics.vlrb_general_reasoning_warmup10 all --seed 11`；也可依次使用 `prepare`、`generate`、`evolve`、`vlrb`、`report` 阶段从已有产物恢复。输出独立保存在 `output/vlrb_general_reasoning_warmup10/seed11/`，其中 General/Reasoning 的 `selection.json` 固定十条样本 ID 与来源，`warmup/transcript.json` 记录实际输入，`report.json` 在完成后汇总全量与留出矩阵、路由、R0/S0/Final 阶段结果和配对比较。

2026-09-26 本地时间 19:43 断点恢复：确认本地部署的 Worker 端点提供所需的 `Qwen/Qwen3-VL-8B-Instruct`；仅把两类 `signature`／`case_reflection` 并发改为 15。General 的已完成十条预热和 R0 原样复用，Reasoning 从未返回的首条预热继续。实时追加日志为 `output/vlrb_general_reasoning_warmup10/seed11/experiment.log` 和 `experiment.stderr.log`；尚无本次外部评估结果。

## 7. 主实验完成后的补充阶段对比

根据演化集逐轮结果，补充评估 **General 第三轮 e03**（36 条上达到 22/36，而 Final 为 19/36）与 **Reasoning R0**（63 条上为 47/63，而 Final 为 46/63）。这两套 Rubric 在主实验的两套 Final 全量评估及报告完成后，分别使用相同的 Worker、1247 条 VLRB 样本顺序和 K=3 A/B 日程进行一次全量推理。General e03 与 Reasoning R0 均来自本次 seed11 的已冻结产物，不重新生成或演化。

补充结果单独写入 `stage_comparison.json`，报告全量 1247、留出 998、各自类别全量及本类留出集的正确数、Strict ACC、覆盖率，并与对应 Final 逐样本配对；另列完整和留出三类别切片。它是依据已观察到的演化集轨迹提出的**探索性阶段对比**，不改变原协议的 Final，也不以这次外部成绩回头选择轮次。入口为 `python -m experiments.evolving_structured_rubrics.vlrb_general_reasoning_stage_comparison --seed 11`；可重复执行以复用已完成预测缓存。

## 8. seed11 结果（2026-09-27）

主结论以未参与演化的 VLRB 留出 998 条为准。General、Reasoning 的 Final 分别在本类留出集答对 **74/145（51.03%）**、**157/254（61.81%）**；各自历史五样例 Final 为 76/145、163/254。两类新 Final 都未超过复用的 Hallucination Final 在相应类别的 79/145、163/254。

| Rubric / 留出类别 | General 145 | Hallucination 599 | Reasoning 254 | 全部留出 998 |
| --- | ---: | ---: | ---: | ---: |
| 新 General Final | 74 | 441 | 162 | 677（67.84%） |
| 复用 Hallucination Final | 79 | 492 | 163 | **734（73.55%）** |
| 新 Reasoning Final | 65 | 404 | 157 | 626（62.73%） |

按已知类别路由得到 **723/998（72.44%）**，低于单用 Hallucination Final 的 734/998，也低于历史 General/Reasoning 配合同一 Hallucination Final 的 731/998；历史生成五根单套为 747/998。新路由相对单用 Hallucination 的配对变化是纠正 24 条、损害 35 条（McNemar 精确双侧 p=0.193），因此这是本次 seed11 的负结果，尚不能断言类别路由普遍无效。

本类留出阶段结果：General R0/S0/Final 为 **70/72/74**（分母 145）；Reasoning 为 **165/168/157**（分母 254）。补充全量评估中，General e03 与 Final 在本类留出集为 **68/145 对 74/145**；Reasoning R0 与 Final 为 **159/254 对 157/254**。后者的 R0 与主实验较早的 R0 具有相同 Rubric 哈希、K=3 和 A/B 日程，却有 22/254 条保存预测不同；两次分数相差 6 条。小幅阶段差异须结合这种运行间波动解读，不能用补充结果回选轮次。

来源细分显示 Reasoning Final 相对 S0 在 MathVerse 留出集由 82/126 降到 76/126，在 MMMU 由 86/128 降到 81/128；下降不局限于单一来源。General Final 相对 R0 在 WildVision 留出集从 63/137 到 68/137；VLFeedback 仅 8 条，不作来源级推断。General e04 接受的单根改动在 36 条上使该根由 18 条正确增至 19 条，但整套系统由 22 条降至 19 条，提示逐根接受指标与全局仲裁结果可能不一致。以上结果仅一组 seed；十样例数量、来源配额与生成根内容共同变化，不能单独归因于其中一项。

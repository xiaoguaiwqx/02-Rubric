# VLRB 三类别专属 Rubric 演化与交叉测试计划

状态：**seed11 已完成（M0–M4）**。本文件前七节为冻结实验协议；第八节记录运行结果。完整数值以 `output/vlrb_category_specialized_rubrics/seed11/report.json` 为准。

## 1. 研究问题与解释边界

在 VLRB 的 General、Hallucination、Reasoning 三类中，分别由本类人类偏好样例归纳数量自选的 root，并沿用现有初始 Split 和最多五轮逐例子树反思。问题是：**对应类别的 Final Rubric 在该类未参与演化的样本上，是否比另外两类的 Final Rubric 更有效？**

首轮不训练共享 Rubric，不做“共享 root、分别演化”的附加对照。三套 Final Rubric 各推理一次完整 VLRB 1247 条，再从同一份逐样本预测分别统计**全量 1247**与**留出 80% 的 998 条**，各形成一张 3×3 矩阵。矩阵的对角线是使用 VLRB 已知类别标签选择 Rubric 的理想路由结果；三行各自的总体成绩是“不路由、只用这一套 Rubric”的对照。全量包含 249 条演化样本，仅留出 998 条用于判断未见样本表现。

本实验直接检验类别专属准则和已知类别路由的效果。类别之间还存在题目与来源差异，因此成绩差不能单独证明人类偏好差异是因果原因。VLRB 已在先前实验中反复观察，本次结果仍属同一基准上的后续探索。

## 2. 一次冻结的 20/80 样本划分

| 类别 | VLRB 总数 | 演化集 | 测试集 | 选取方式 |
| --- | ---: | ---: | ---: | --- |
| General | 181 | 36 | 145 | 从本类排序后的样本 ID 用固定种子随机抽 36 条 |
| Hallucination | 749 | 150 | 599 | 固定纳入旧 Hallucination100，再从其余 649 条随机抽 50 条 |
| Reasoning | 317 | 63 | 254 | 从本类排序后的样本 ID 用固定种子随机抽 63 条 |
| 合计 | 1247 | **249** | **998** | 约 20% / 80% |

- 旧 100 条名单读取 `output/vlrb_hallucination100_transfer/seed11/split.json` 的 `train_ids`；不把它们放回本次测试集。新抽样种子固定为 11，并将类别名纳入随机种子，避免输入顺序影响样本选择。一次写出三类训练 ID、测试 ID、VLRB parquet 哈希与抽样规则；之后不得按结果重抽。
- **按样本条目直接划分**；按照本次要求，不按图像或问答近重复聚类，也不据此删样本。报告中明确这是条目级划分，不把测试集称为近重复隔离集。
- 三类测试集互斥，其并集恰为 998 条。所有 Manager 的预热、错误签名、聚类、接受和停止决策只读取对应类别的演化集，不读取测试样本或其标签。
- 从完整 VLRB 顺序生成既有 A/B 顺序日程，再按 ID 截取三类训练与测试；不得仅对 998 条重算日程。训练继续使用现有 K=1，测试继续使用 K=3 和相同 Worker、Manager、Arbiter 配置。

## 3. 三个独立演化分支

每个类别执行同一流程，不预设三类具有相同 root 数或职责：

```text
本类演化样本 → 固定种子抽 5 条人类偏好样例预热 Manager
             → 现有 GN 提示词自选 2–7 个 root，形成 R0
             → 本类全部演化样本上的 root 推理、错误签名、聚类、初始 Split，形成 S0
             → 沿用原接受规则与最多五轮逐例子树反思，冻结 Final
```

`2–7` 是统一结构范围，**不指定实际 root 数**。本次 GN 数量指令固定为 `Choose the number of roots from two through seven according to the distinct recurring responsibilities in these examples.`，不附加旧实验中的 `five is allowed`；旧 Hallucination100 的已冻结提示词和产物保持历史原样。不得为某一类别根据测试成绩单独改范围或重新生成。保留五样本预热文本、随机 A/B 翻转、Manager 模型、Worker 模型、Split 的支持门槛、`strict_acc` 接受、五轮上限、Arbiter 提示词中随实际 N 渲染的数量词。General 虽只有 36 条，但不预先放宽初始 Split；若某个 root 无法形成两组有支持的簇，保存 R0 与失败证据并报告该分支不可行。

各分支独立保存 `warmup/`、`r0/`、`init/`、`e01`–`e05`、`state.json` 和缓存。训练阶段已有 R0、S0、Final 的逐样本预测及每 root 指标，应直接读取这些产物报告演化过程，不额外重跑训练推理。三套 Final 全部冻结后才执行外部测试；不根据外部指标选择 root、children 或轮次。

## 4. 全量一次推理，两套 3×3 统计

对每套 **Final** Rubric 在相同的完整 VLRB **1247 条**上做一次 K=3 系统评测，保存三份完整逐样本预测。对每份预测分别按全量类别和留出类别切片，即可得到两套九格矩阵，**不需要逐格独立评测**：

| 使用的 Final Rubric | General：全量 181 / 留出 145 | Hallucination：全量 749 / 留出 599 | Reasoning：全量 317 / 留出 254 |
| --- | --- | --- | --- |
| General | G→G | G→H | G→R |
| Hallucination | H→G | H→H | H→R |
| Reasoning | R→G | R→H | R→R |

- 每个格子并排报告**全量正确数/全量分母**与**留出正确数/留出分母**的 Strict ACC，弃权计错；每套 Rubric 再报告全量 `正确数/1247` 和留出 `正确数/998`。同时列出覆盖率、Covered ACC、技术失败数、root 数与调用成本。技术失败按原重试/断点恢复路径解决，不当作正常弃权。
- **已知类别路由**的全量 Strict ACC = 三个全量对角格正确数之和 `/1247`；留出 Strict ACC = 三个留出对角格正确数之和 `/998`。两者都报告三类等权宏平均，避免 Hallucination 的数量掩盖 General/Reasoning。全量数值包含演化样本，只作描述和与既有 VLRB 表格对齐；**留出 998 是主泛化读数**。
- 在留出 998 上预先比较每一列中本类 Rubric 与另外两套 Rubric 的逐样本纠正/伤害数；再把路由预测与“三套 Rubric 中任意一套用于全部留出样本”的各行预测分别配对比较。区间用按样本 ID 成对重采样；不看完结果再选择一个有利的行作为唯一对照。全量 1247 也列差值，但不据此作未见样本推断。
- 可选的后续诊断是在各自类别的留出子集评估 S0，以区分初始 Split 与五轮演化；**首轮主实验不因此增加外部调用**。训练阶段的 R0/S0/Final 已能先显示演化变化。

三份 Final 的理论系统调用数为 `1247 × 3 × [(N_G+1)+(N_H+1)+(N_R+1)]`（每个 root 加一次 Arbiter，K=3）。若三类各生成 5 个 root，约为 **67,338 次逻辑调用**；实际请求与耗时取决于缓存、并发和重试。每套 Final 只跑一次全量，留出指标从结果切片，不再为留出 998 或九个格子重复调用模型。

## 5. 历史产物复用：不重复运行旧对照

本次三套 Rubric 使用 **36/150/63 条新的类别演化集**，先前 Hallucination100 上训练的 F5/G5/GN Rubric 不能冒充新的 Hallucination150 分支；新三分支的生成、Split、演化与 Final 测试确实需要运行。

历史结果只作为**额外参考行**，不参与三类 Rubric 的生成或主结论：

| 历史系统 | 已有产物 | 本次处理 |
| --- | --- | --- |
| 预设五根 Hallucination100 的 S0/Final | `output/vlrb_hallucination100_fresh_init/seed11/vlrb/{initial,final}.json`；已有 F5 等价复用审计 | 直接读取保存的 1247 条预测，并按新测试 ID 重算留出 998 与两套三类别指标；不调用模型 |
| 偏好归纳五根 Hallucination100 的 S0/Final | `output/vlrb_hallucination100_generated_roots/seed11/g5/vlrb/{initial,final}.json` | 同样只读统计全量与留出；标明其演化集为旧 100 条而非本次 150 条 |
| 自选数量 Hallucination100 | GN 的 `initial.json` 已存在；`final.json` 在制定本计划时尚未完成 | Final 若之后自然完成且元数据一致，则只读切片；否则从历史参考表省略，不为补这行重跑 |

复用前核对样本 ID/顺序、原始 VLRB 数据文件、K=3、A/B 日程、预测解析与技术失败状态；历史与本次推理协议的任何差异必须在表旁注明。旧的早期裸 root 全量结果来自不同 Worker 端点，不作为同配置主对照。旧预测只读引用，不复制缓存、不重评全量 1247，也不为了补历史行启动额外旧实验。

## 6. 最小代码改动与执行顺序

1. **薄的新实验入口**：在 `experiments/evolving_structured_rubrics/` 新增一个类别实验 runner，只负责冻结 249/998 的 ID、写三份类别训练 JSONL/配置、依次调用现有生成与演化函数、对三份 Final 各调用一次现有 `framework_v6.evaluate` 评测完整 1247，最后从三份预测汇总全量与留出的 3×3 和历史只读切片。不要复制 Worker、Manager、聚类、接受或计分实现。
2. **小范围参数化 root 生成**：现有 `generated_root_initialization._warmup_rows` 强制输入恰好 100 条，`generate_r0_pair` 固定同时生成 G5/GN。仅把训练集长度和请求的变体列表变成可选参数，旧 Hallucination100 入口显式传入其冻结数量指令；新 runner 传本类长度、仅 `gn`，并使用上文去掉额外提示的新数量指令。其余预热/生成文本及解析规则不变。
3. **直接复用已有演化函数**：`subtree_local_reflection.run(..., rows=..., r0=..., manager=...)` 已支持显式传入训练行和生成的 R0；`local.make_manager(..., n_roots=...)` 已支持数量词渲染。外部测试复用 `framework_v6.evaluate(..., rows=..., orders=...)`；每套 Final 只评一次完整 1247。
4. **结果脚本只读汇总**：将三份全量 Final 预测分别切成全量三类别及留出三类别；另对通过核查的历史完整预测做相同切片。只写摘要指标、配对计数、配置/产物路径与必要哈希；原始图片、逐样本预测和缓存留在 `output/`。

新产物写入独立的 `output/vlrb_category_specialized_rubrics/seed11/`，不改动 Hallucination100 的旧目录。执行阶段：M0 冻结 split 与配置 → M1 三类五样本预热及 R0 → M2 三类各自 S0 与最多五轮 Final → M3 冻结后三份 1247 条外部推理 → M4 全量与留出两套 3×3、路由指标和历史只读参考。每阶段可从现有成功产物继续；不因某个 API 请求尚未返回而重启。首轮只做 seed11；其他种子属于后续稳定性实验，不以本轮测试分数选种子。

若三套 Final 均可行，最终报告应同时展示：三类生成的 root 数及职责、训练 R0→S0→Final、Final 的全量/留出 3×3、全量/留出对角线路由与各单套 Rubric 对照、历史参考行，以及 General 小样本造成的波动。若有分支不可行，保留失败产物并明确主矩阵无法完整形成，不通过改门槛补成一个看似完整的正结果。

## 7. 实现入口与运行命令

入口为 `experiments/evolving_structured_rubrics/vlrb_category_specialized_rubrics.py`。在 `critiq` 环境、仓库根目录执行以下阶段；每阶段重启时读取同一 `seed11` 输出目录的已保存产物。`prepare` 只冻结样本、训练 JSONL 和配置，不请求模型。`report` 只读新旧评测产物，不请求模型。

```powershell
conda activate critiq
python -m experiments.evolving_structured_rubrics.vlrb_category_specialized_rubrics prepare --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_category_specialized_rubrics generate --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_category_specialized_rubrics evolve --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_category_specialized_rubrics vlrb --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_category_specialized_rubrics report --seed 11
```

新产物的总划分保存在 `output/vlrb_category_specialized_rubrics/seed11/split.json`，三类分支在其 `general/gn`、`hallucination/gn`、`reasoning/gn`，汇总为 `report.json`。单次全流程也可用 `all --seed 11`；长时运行建议分阶段保留日志。历史 F5/G5/GN 仅由 `report` 读取保存的 `vlrb/{initial,final}.json`，不进入新实验的推理队列。

## 8. seed11 运行结果（2026-09-26）

三类均完成五轮演化，生成 root 数分别为 General **4**、Hallucination **3**、Reasoning **5**。训练集系统 Strict ACC 的 R0 → S0 → Final 分别为 General **19/36 → 21/36 → 21/36**、Hallucination **106/150 → 108/150 → 108/150**、Reasoning **42/63 → 48/63 → 47/63**。三份 VLRB Final 均完成 1247 条 K=3 推理，技术失败数均为零。Hallucination 曾有一条样本的子树报告在 10 次生成后仍解析失败；保留缓存，将尝试上限提高至 20 后从外部评测阶段恢复成功。

报告采用三次 A/B 顺序推理的**多数票**计算逐样本 Strict ACC。运行日志中的 `vlrb/final: Strict ACC` 是逐次推理的指标，不能代替下表的多数票结果。

| Final Rubric / 路由 | 全量 1247 | 留出 998 |
| --- | ---: | ---: |
| General Rubric 单独使用 | 869/1247（69.69%） | 698/998（69.94%） |
| Hallucination Rubric 单独使用 | 891/1247（71.45%） | 712/998（71.34%） |
| Reasoning Rubric 单独使用 | 791/1247（63.43%） | 635/998（63.63%） |
| 已知类别路由至本类 Rubric | **892/1247（71.53%）** | **714/998（71.54%）** |

留出集的 3×3 迁移矩阵如下；单元格为正确数/本列样本数（Strict ACC）。

| 使用的 Final Rubric | General 145 | Hallucination 599 | Reasoning 254 |
| --- | ---: | ---: | ---: |
| General | **76/145（52.41%）** | 451/599（75.29%） | **171/254（67.32%）** |
| Hallucination | 72/145（49.66%） | **475/599（79.30%）** | 165/254（64.96%） |
| Reasoning | 67/145（46.21%） | 405/599（67.61%） | 163/254（64.17%） |

对角线路由相对留出集最强的单套 Hallucination Rubric 仅多 **2/998** 个正确样本（配对纠正 32、伤害 30；McNemar 双侧 `p=0.899`）。General 与 Hallucination 两列由本类 Rubric 领先；Reasoning 列则由 General Rubric 领先本类 Rubric **8/254**。因此本轮不支持“三类分别演化并路由会明显提高 VLRB 泛化准确率”的强结论。

历史只读参考：旧 Hallucination100 的预设五根 Final 在本次留出 998 条上为 **735/998（73.65%）**，偏好归纳五根 Final 为 **747/998（74.85%）**。它们的演化集与本次 36/150/63 分支不同，仅作背景比较。全量 3×3、Covered ACC、配对区间、成本和历史产物核查结果保存在 `report.json`。

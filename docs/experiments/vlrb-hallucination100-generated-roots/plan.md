# Hallucination100：五样本预热生成 root 的对照实验

日期：2026-09-25。状态：**协议已冻结、代码已实现；seed11 实验运行中**。2026-09-25 02:58（Asia/Shanghai）启动；数据冻结检查通过，目前首条 Manager 预热请求仍在等待响应。运行日志、进程 ID 和断点产物记录在 `output/vlrb_hallucination100_generated_roots/seed11/`。

本实验接续 [子树逐例反思框架](../subtree-local-reflection/framework.pptx) 第 6–8 页的 Hallucination100 实验。上一实验使用五个固定 root，在目标分布的 100 条样本上生成初始 children，完成最多五轮逐例子树反思，再测试 VLRB。本实验只研究**初始 root 从何而来，以及是否必须固定为五个**；初始 Split、演化、Worker 和测试日程沿用上一实验。

当前主线的运行入口是仓库 README 中的 `run_subtree_experiment`；下文保留原实验冻结时的命令与目录，所涉及的完整历史 runner 可在 `codex/subtree-local-reflection` 分支查看。seed11 精确 split ID 和来源哈希已另存为本目录 `seed11_split.json`。

固定的 seed11 训练集现另存于本地 [data/VL_RewardBench/splits/hallucination100/seed11/discovery_100.jsonl](../../../data/VL_RewardBench/splits/hallucination100/seed11/discovery_100.jsonl)，与 Reasoning70 的训练文件统一放在 `data/VL_RewardBench/splits/`。100 条样本的 ID、顺序、问题、A/B 回答和人类标签与已保存实验一致；图片路径统一引用共享的 `data/VL_RewardBench/dataset_images/`。原输出文件、冻结划分和现有运行配置保持不变。

## 1. 研究问题与可支持的结论

| 编号 | 问题 | 所需证据 |
| --- | --- | --- |
| C1（主问题） | 根据 Hallucination100 的人类偏好生成 root，能否比原固定五 root 改善未见幻觉样本上的最终判断？ | 只将已有提示词里的 root 数量按实际 N 渲染，其他文字、数据、模型、初始 Split、五轮演化和 K=3 测试不变；比较 GN Final 与 F5 Final 的未见幻觉 648 条 Strict ACC 与配对变化。 |
| C2（辅助问题） | 在数据生成 root 的前提下，允许 Manager 自选数量是否比明确要求五个更有用？ | G5 与 GN 使用完全相同的五轮预热对话，仅最终生成指令不同；比较 R0、S0、Final、实际 root 数、覆盖和成本。 |

**解释边界**：G5 与 GN 的最终 root 内容也可能不同，因此 C2 检验的是“数量选择策略”的整体效果，不能把性能差严格归因为数量本身。训练 100 条上的提升不算泛化证据；VLRB 已在前序研究中多次用于观察结果，本次仍属于同一基准上的后续探索，强泛化主张需要独立确认集。

## 2. 冻结数据与实验单位

- 首轮使用已有 `output/vlrb_hallucination100_transfer/seed11/split.json`，继承同一份 VLRB 数据文件、样本 ID、A/B 顺序及去重规则；不重新挑选有利的训练集。100 条训练样本来自 POVid 60、RLAIF-V 31、RLHF-V 9。Manager 仅可在这 100 条内预热、生成和演化。
- **主测试集**：未见且通过现有近重复过滤的 Hallucination 648 条。它不参与 root 生成、初始 Split、五轮反思、接受或 checkpoint 选择。
- **干净的非训练集**：1146 条＝上述 648＋General 181＋Reasoning 317。旧表中的“1247 去训练 100＝1147”还含一条因与训练图像近重复而从 648 中排除的幻觉样本；1147 可作为算术切片报告，但不能称为完全去近重复的测试集。
- 完整 VLRB 为 1247 条，Hallucination 全组为 749 条，二者均包含训练 100 条。报告里始终把它们与 648、1146 分开标注。
- 先完成 seed11 的可比主实验。若要判断随机初始化是否稳定，再按预先指定的 seed29、seed47 分别运行完整三组；五样本抽样种子取对应的 split seed（11、29、47），翻转种子保持 100745534。不同 seed 的测试集大量重叠，不把三份样本当成独立的 3× 样本量。

## 3. 三组可比系统

| 组别 | R0 的来源 | root 数量 | 五样本预热 | 后续处理 |
| --- | --- | --- | --- | --- |
| F5：固定对照 | 现有 MultiCrit Open-Ended 五 root | 5 | 无；固定 root 不需要生成 | 若 N=5 渲染后的所有实际提示词和请求设置与旧 seed11 逐字一致，复用已保存的 S0/Final 预测；不一致才在新协议下重跑。 |
| G5：生成五 root | CritiQ 式五样本预热后归纳 | 最终指令要求恰好 5 | 与 GN 完全相同 | 用生成的 R0 独立运行初始 Split、最多五轮演化与外部评测。 |
| GN：自选数量 | CritiQ 式五样本预热后归纳 | Manager 自选 2–7 | 与 G5 完全相同 | 用生成的 R0 独立运行同一后续流程；本实验主方法。 |

G5 与 GN 在第五次预热回答后从**同一份对话历史**分叉，只更换第六轮的数量要求。不得分别抽样、重新预热或挑选看起来更好的生成结果。若 GN 恰好生成 5 个，仍照计划报告三个组；不能把一次 G5/GN 差值解释为纯数量效应。

现有 Manager/Arbiter 提示词的数量词是英文 `five`。实现时只在这几个位置放入数量参数，按 `rubric.root_ids` 渲染为 `two` 至 `seven`，**N=5 必须逐字得到原来的 `five` 和完整原提示词**；其他判别规则、段落与输出格式不改。因此旧实验 F5 Final `920/1247`、未见幻觉 `534/648` 有条件成为正式主对照：先核对保存请求的 system/user 提示词、rubric、模型与参数、图像、样本 ID、A/B 日程、K=3 聚合方式及技术失败状态。核对一致时直接引用保存的逐样本预测，不复制或伪装旧缓存；任一关键项不一致才重新运行 F5。现有 `framework_v6.evaluate` 对已存在结果只检查 rubric 哈希、样本顺序及技术失败数，**不能以其缓存命中代替这项请求等价核查**。F5 的 R0 外部预测也按同样条件检查，若找不到完全匹配的产物，则只补跑 R0 的 648 条。

## 4. CritiQ 式 root 生成协议

原 CritiQ-V 分支中 `Workflow.get_init_criteria` 的关键做法是：给同一个有状态 Manager 逐条展示带人类偏好的成对样本，在保留这些对话历史的情况下生成准则。原函数的 `n_criteria` 必须预设，输出是扁平 criterion；这里迁移预热机制，再将输出作为当前框架的 root，不直接调用原函数。此实验不接入知识库检索，以免又改变准则来源。

1. 按冻结的训练 ID 顺序，用 `random.Random(11).sample(rows, 5)` 等概率、无放回抽 5 条；不按最终结果改选。随后沿用 CritiQ 的 `random_reverse(seed=100745534)` 翻转并洗牌这五条 pair，保存最终预热顺序、来源、图像哈希、翻转后的回答和标签。五条的来源组成在报告中公开；纯随机五条不保证覆盖三个来源。
2. 使用与当前演化相同的 Qwen3.5-27B Manager、`temperature=0.2` 和非思考设置。五次**顺序**调用同一个有状态对话，每次提供该条的原图、问题、回答 A/B、人类偏好，并要求说明这一偏好的可观察依据。不得让默认纯文本 prompt 漏掉图像或问题；不得在预热指令中暗示原来的五个 root 名称。生成阶段若复用 `Agent`，显式设置 `api_retry_attempts=0`、外层最多 10 次尝试，并记录其当前 900 秒单请求超时；不要继承 `Agent` 默认的 50 次内部重试而误报调用成本。
3. 第六轮要求从前五例归纳可复用、互不重复的**上层评判职责**，不是直接生成 children。描述要能指导 root 对整对回答给出 A/B/None，且有空间让后续错误签名形成子准则。G5 明确请求 5 个；GN 不给目标数量，只给预先冻结的边界 **2–7 个**和“用足够少且足以覆盖主要偏好的职责”的原则。2 是多 root 仲裁下限，7 限制 100 条样本上的初始证据和测试成本；这两个边界是本实验新增规则，不是原 CritiQ 自动推断。
4. 输出结构为 `{"count_reason": "...", "roots": [{"name": "...", "description": "..."}]}`。仅做 JSON、数量范围、非空文本、规范化后名称唯一性和可构建 `StructuredRubric` 的结构校验。格式失败可在同一对话上按冻结重试次数修复；不能根据 Worker 分数、heldout 结果或主观喜好重新生成。语义重叠、职责遗漏和数量理由作为审计结果记录，不人工润色后伪装成原始生成。
5. 将生成结果转换成只有 root、没有 children 的 R0；root ID 按生成顺序稳定编号，lineage 记录生成协议与五条样本 ID，不写随机时间戳。保存第六轮原始输出、解析版本、提示词版本、模型参数、调用 usage。对话缓存可保存文本与图像哈希，并从冻结数据重建图片消息，避免把 base64 图像反复写入 JSON。

生成 root 的五条样本也属于随后用于 Split/演化的 100 条；这是有意复用训练数据。**不能**把这五条或训练其余 95 条上的收益称为未见集收益。

### 提示词草案：只新增 R0 生成阶段

五条预热样本逐条发送，`{image}` 是同条消息的图像内容，不是图片路径文本；每条回复留在同一对话历史中。下面的文字不点名原五维度：

```text
[Image: {image}]
Question: {question}
Response A: {A}
Response B: {B}
Human preference: {answer}
Why might a human prefer that response for this image and question?
Explain the evidence you can verify; say when the preference is uncertain.
```

第六轮的共同部分：

```text
From the five comparisons above, propose high-level root responsibilities for
judging new image/question/response pairs. A root must be reusable, distinct
from the others, and broad enough to support more specific child criteria later.
Describe what it checks and how it informs a relative A/B/None judgment.
Do not copy sample-specific facts, human labels, or a fixed preference for A/B.
Return JSON only: {"count_reason": "...", "roots":
[{"name": "short_unique_name", "description": "..."}]}.
```

G5 在共同部分后追加 `Return exactly five roots.`；GN 追加 `Choose the number of roots from two through seven according to the distinct recurring responsibilities in these examples; five is allowed.` 两组除这一句外完全相同。正式运行前固定英文原文与哈希，不按生成结果修改措辞。

## 5. R0 之后保持的算法

对三组分别在独立输出目录执行相同流程：

```text
R0 裸 root
  → 每个 root 在训练 100 条上独立推理
  → 各 root 的待核验错误案例 → 错误签名 → 语义聚类
  → 每个 root 生成 2–5 个初始 children，形成 S0
  → 最多五轮：逐例反思 → 每 root 一次整组 Split → 局部 Strict ACC 竞争
  → 冻结 Final → VLRB K=3 外部测试
```

- 保留原始初始 Split 门槛：每 root 至少四个适用签名、至少两个各有两条不同签名支持的簇；初次错误案例不足时按现有逻辑补充训练样本。若某生成 root 仍无法形成 S0，记录为**该次初始化不可行**，不降低阈值、不凭测试结果删 root 或反复生成直至成功。
- 演化仍为最多五轮；若现有 runner 因无可操作反馈提前停止，记录实际轮数。保留每 root 五条随机正确案例辅助 Split、`strict_acc` 接受规则、Worker Qwen3-VL-8B 和现有并发/采样设置。root 名称、描述、数量在整次运行中不变，只更新 children。
- 演化 Manager 的共用提示词只把两处 `five` 渲染为实际 root 数对应的英文数量词；Arbiter 也只替换 `five subtree assessments` 中的数量词。其他字句，包括 Arbiter 原有的视觉／事实优先规则，**逐字保留**。本实验的“新提示词”仅指五样本预热和 R0 生成阶段；后续提示词是旧模板的数量参数化。为保留旧缓存身份，不修改历史常量或模块全局字典；N=5 时还应保持原协议／提示词版本标记，N≠5 时用可区分的版本标记。原 Arbiter 的维度优先规则可能限制新 root 的表达，但三组共用同一规则；若要研究自适应仲裁，应另设实验，不在此处改变。
- 技术失败须在计分前按现有重试路径解决；解析失败不得算作正常 None。不同组、不同 root 内容不能复用旧的子树报告；同一组的 S0→Final 对未改变 root 可按现有哈希规则复用。

## 6. 终点、配对比较与诊断

**预先指定的主终点**：seed11 上 GN Final 对 F5 Final 的未见幻觉 648 条系统 Strict ACC 差值，报告正确数、百分比、逐样本纠正／伤害。差值的 95% 区间采用按样本 ID 成对重采样的 10,000 次 percentile bootstrap，固定分析随机种子 20260925；另报告双侧 exact McNemar。由于该基准此前已被反复观察，区间和 p 值均作探索性描述。

| 阶段或切片 | 必报内容 | 用途 |
| --- | --- | --- |
| R0、S0、Final 的训练 100 | 系统 Strict ACC；每 root 的正确数／有效 A/B 数／覆盖率／Covered ACC | 区分 root 生成、初始 Split 与五轮演化；训练指标不作泛化证据。 |
| R0 的未见幻觉 648（K=3） | F5/G5/GN 系统 Strict ACC 与 root 覆盖诊断 | 沿用完整 VLRB 中这些样本的原定 A/B 日程，直接观察裸 root 的差异；不在全量 1247 上重复 R0，以控制成本。 |
| S0、Final 的完整 VLRB 1247（K=3） | 官方系统 Strict ACC、覆盖率及所有切片 | 与上一实验同一测试日程；可从保存预测计算训练100、648、clean1146、General181、Hallucination749、Reasoning317。 |
| 组间 Final、组内 S0→Final | 同样本纠正／伤害、净变化、配对区间、McNemar | 识别生成 root 的收益是否被后续演化放大或抵消。 |
| 每个 root 的 R0/S0/Final | `correct/valid/total`、Covered ACC、coverage、Strict ACC、children 数量 | 分母因弃权而变，不能只报告 Covered ACC 百分比；不同组 root 不按序号强行配对。 |
| 运行成本 | 实际 root 数、总节点数、Worker/Manager/Arbiter 调用、tokens、耗时 | 判断收益是否来自更多子树调用或更长 Rubric。 |

主要辅助对照为 G5 Final vs F5 Final、GN Final vs G5 Final、以及三组各自 Final vs S0。若 GN 在 648 改善但 clean1146 或 General/Reasoning 退化，结论须限定于目标幻觉分布。若 S0 有益而 Final 退化，应优先诊断后续反思/接受机制；若生成 root 无法完成初始 Split，首先报告可行性问题。不得按 VLRB 指标选择“最佳 root 数”“最佳预热样本”或最佳轮次。

## 7. 实现边界与产物

| 代码位置 | 预期最小改动 |
| --- | --- |
| 新的 Hallucination100 root 生成／运行模块 | 五条样本选择与翻转、有状态多模态预热、G5/GN 同历史分叉、root JSON 转 R0、独立目录与报告。使用现有 `Agent` 的对话历史机制；其生成阶段超时／重试与现有 Manager 差异须显式记录。 |
| `framework_v6.initialize` | 可选注入 R0；未注入时仍使用旧固定五 root，保持历史调用行为。 |
| `subtree_local_reflection.run` | 由新 runner 先生成 R0 并初始化，再调用原五轮流程；仅在确有必要时增加可选参数。 |
| Manager/Arbiter 调用 | 给原模板的两处 Manager `five` 和一处 Arbiter `five` 加数量参数，N=5 渲染结果与原请求逐字相同；避免修改模块全局设置来切换实验。 |
| `framework_v6.external` 与新报告 | 复用现有 S0/Final K=3、官方投票、切片和配对计算；新 runner 增加 R0 的 648 测试，不套用旧四版本固定命名报告。 |

G5、GN 分别使用 `output/vlrb_hallucination100_generated_roots/seed11/{g5,gn}/`，保存配置、split 引用/哈希、预热 transcript、R0、S0、各轮 rubric 与局部比较、最终报告。F5 若通过请求等价核查，新目录只保存指向旧 seed11 产物的只读比较 manifest；否则建立独立的 `f5/` 运行目录。任何已成功的模型调用按请求内容缓存；请求、模型或提示词变更必须使用新目录，不覆盖历史结果。`output/` 中的图片、逐样本报告、缓存和密钥不提交；待结果核对后只把精简表与必要复现元数据写回本实验文档。

### 实现入口

在 `critiq` 环境和仓库根目录运行以下命令。各阶段可单独恢复；`all` 按同样顺序依次执行，且完整 VLRB 评测只在 G5/GN 的 Final 均冻结后开始。五条共享预热对话保存在 seed 目录的 `warmup/transcript.json`；各自的第六轮输出保存在 `g5/r0/generation.json`、`gn/r0/generation.json`。

```powershell
conda activate critiq
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_generated_roots prepare --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_generated_roots generate --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_generated_roots run --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_generated_roots vlrb --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_generated_roots report --seed 11
```

`run` 对 F5 seed11 逐条审计旧 S0/Final 请求；不等价时独立重跑。`vlrb` 为三组补充 R0 的 648 条 K=3 结果；`report` 写出 `seed11/report.json`，包括系统/各 root 的 Strict ACC、Covered ACC、覆盖、配对变化和调用成本。现有结果文件的顶层读取只核对部分身份字段；修改提示词或模型后必须换一个输出目录，不能依靠底层 Worker 缓存自动区分旧的完整评测文件。

## 8. 执行顺序、计算量与停走条件

### G5 Root create 全量补评（2026-10-02）

为补齐汇报表中“偏好归纳五根、仅 Root”的结果，直接使用 seed11 的 `g5/r0/rubric.json`，不重新生成 Root、不生成子准则、不演化。在同一 VLRB 1247 条与固定 A/B 日程上评估 K=3，保持现有 Worker 与 Global Arbiter 提示词、模型参数及并发50。独立结果为 `g5/vlrb/r0_full_k3.json`，切片报告为 `g5/vlrb/r0_full_k3_report.json`；日志原生追加至 `g5_r0_full.log` 和 `g5_r0_full.stderr.log`。保留历史演化集 K=1 与648条 K=3记录，汇报时使用本次全量 K=3的同口径切片。启动器为 seed11 下 `launch_g5_r0_full.cmd`，使用 `conda run --no-capture-output -n critiq`。

**观察结果：补评完成，1247/1247，耗时约101.9分钟，技术失败0。** 下表使用与既有PPT一致的原始索引 K=3汇总口径；逐样本配对分析保存在 `g5/vlrb/r0_full_k3_analysis.json`。

| G5 阶段 | 全量1247 | 未参与演化1147 | 演化集100 | 幻觉留出648 | General181 | Hallucination749 | Reasoning317 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 仅 Root | 899，72.09% | 820，71.49% | 79，79.00% | 523，80.71% | 85，46.96% | 603，80.51% | 211，66.56% |
| 初始子准则 | 929，74.50% | 850，74.11% | 79，79.00% | 538，83.02% | 99，54.70% | 618，82.51% | 212，66.88% |
| 五轮演化后 | 934，74.90% | 851，74.19% | 83，83.00% | 540，83.33% | 99，54.70% | 624，83.31% | 211，66.56% |

仅 Root→初始子准则在全量纠正85条、损害55条，净增30条（+2.41pp）；仅 Root→Final纠正80条、损害45条，净增35条（+2.81pp）。初始子准则→Final净增5条，而未参与演化1147仅净增1条。此seed观察到的收益主要发生于初始Split，后续五轮的泛化增益较小；Reasoning从仅Root到Final净增0条，但纠正与损害各17条。单seed、不同时间调用的配对结果不等于已排除解码随机性或复现波动。

| 阶段 | 工作 | 继续条件 |
| --- | --- | --- |
| M0 协议冻结 | 锁定 seed11 的训练/heldout 名单、五样本抽样与翻转规则、数量参数化位置及输出目录 | N=5 的请求与旧 F5 完全一致，或明确列出不一致并重跑 F5；648 和 clean1146 切片互斥关系正确。 |
| M1 接线检查 | 仅用训练样本检查六轮对话可恢复、图像和问题确实送达、G5/GN 共享前五轮历史、可变 root 可执行 | 不读取任何测试准确率作调参依据；格式问题修复后冻结代码与提示词。 |
| M2 seed11 主实验 | 核查并引用旧 F5，完成 G5/GN 的 R0→S0→最多五轮与训练指标 | 三组可比产物齐全；初始 Split 失败如实记录，不暗改规则。 |
| M3 冻结后评测 | G5/GN 的 R0 跑 648 K=3、S0/Final 跑完整 1247 K=3；F5 仅补齐未获等价产物的阶段，生成配对表 | 三组使用同一评测程序、样本顺序与 A/B 日程；无未解决技术失败。 |
| M4 稳定性与解释 | 先审 seed11 的 root 数/语义、覆盖和成本；如需稳健结论，按预先指定 seed29/47 重复，另寻独立确认数据 | 不从多次测试中挑最优生成结果；按全部完成 run 报告方向与变异。 |

一次全量 VLRB K=3 的理论系统调用上界为 `1247 × 3 × (N+1)`：五 root 为 22,446 次；GN 的 N=2–7 对应 11,223–29,928 次。R0 只测 648 K=3，另需 `648 × 3 × (N+1)` 次。S0→Final 未变的 root 可复用，真实调用量较两次独立全量评测少；训练演化、失败重试及生成阶段另计。上一实验的一次全量评测耗时不能直接按 root 数线性外推，M1 后再依据实际并发、缓存命中与重试估时。

**成功解释**：GN 对 F5 在 648 上有正的配对净变化，且变化不主要依赖弃权或技术失败；clean1146 与各类别不出现未披露的重大损害。单个 seed 的小幅正差仅称“观察到改善”。若 G5 与 GN 接近，只能说数据生成的内容可能比数量自由度更重要；若 GN 明显更好，后续仍需更严的数量消融和独立数据才能形成因果与泛化结论。

## 9. 已观察结果：G5 seed11 完整流程重跑（2026-10-03）

本节是运行后的结果归档，不修改前面的冻结协议或历史结果。第 8 节的 G5 表保留旧运行及其 2026-10-02 R0 全量补评；[框架 PPT](../subtree-local-reflection/framework.pptx)第 11 页最后三行已替换为下面的本次结果。这里只归档 G5，不能据此比较新的 F5/GN 或推断 root 数量策略的效果。

### 9.1 来源、完成状态与比较边界

- 本次本地产物：`output/subtree_reflection/seed11/g5/`，以 `report.json` 的 `generated_roots` 和 `vlrb/*_report.json` / `vlrb/report.json` 的 `official` 为正式指标来源。`completed=true`、`external_complete=true`，五轮完成，结束原因为 `max_epochs`。
- 历史本地产物：历史 worktree `CritiQ-framework-v6` 下的 `output/vlrb_hallucination100_generated_roots/seed11/g5/`；R0 全量来自 `vlrb/r0_full_k3_report.json`，三阶段切片来自 `vlrb/r0_full_k3_analysis.json`，S0/Final 正式结果来自 `vlrb/report.json`。原始产物继续留在本地，不纳入本次提交。
- 归档时代码基线为 `3ead5a4`，全套离线测试 46 项通过。它固定了当前共享图像、进度日志和正式 K=3 日志修复；不把事后日志修复当作产生本次实验分数的算法变化。
- 两次 100 条演化样例的 ID、顺序、图像 SHA、问题、候选回答和标签一致；五条 warmup 的样例、顺序、A/B 翻转和 prompt 一致。共享 warmup prompt SHA256 为 `fe6701ff20119ce98f6f32d1e700ddceb531bd9e912dfd0a282474fb23366951`，G5 生成 prompt SHA256 为 `6aa05c742982806889aeee073dea31e654420a2949a756ca3835fadd1610f9b1`。
- 保存配置中的 Worker 同为 Qwen3-VL-8B-Instruct、temperature=0.5、max_tokens=2048、并发50；Manager 同为 Qwen3.5-27B、temperature=0.2、关闭 thinking。接受指标为 Strict ACC，Preserve5、preservation_seed=42、五轮上限相同。
- 五条 warmup 的模型回答全部不同，R0 与后续 children 也不同。seed11 固定数据划分和抽样，不固定全部模型输出；这是完整流程的重复，不是固定 Rubric 的推理重复。保存的配置相同不证明远端服务状态完全相同，也不能将分数差直接归因于代码重构。

| 初始化证据 | 历史 | 本次 |
| --- | --- | --- |
| warmup history SHA256 | `beec222eacbb4221f7a56383a6295c9ab8e51b00a7a2519cdd7bbd9b7fad741a` | `39e171042f2beb4eb1f238232e0356bb7fab36f1f589abfaf7da05dd630ca651` |
| R0 Rubric SHA256 | `ea142b0c0dff347e441308bcbe91f25b724a8095c1f46fe276ff57652fc7f649` | `6d17a9bc9e77c6b4102e35dfe14c1e404243f33a129fb4253452e30314a0f427` |
| S0 Rubric SHA256 | `c9ae892fdbc4916762e8f91a5744d63f567d943b360d5cc91391a802704c4d6b` | `3827f27d936b18891d4028aeb3e3a9806e2aae364cd6204297e4a590335abf5d` |
| Final Rubric SHA256 | `f0e4c0306f0a0076db989c88de91dc4a4f2b03b48dc15b8d245481515f833274` | `e528e847e1d2534b8987161923ea318c90c045bc8cc8f2214e902329ee239c80` |
| 五个 root 名称 | factual_accuracy、assumption_handling、relevance_and_focus、completeness_within_bounds、interpretive_restraint | factual_accuracy、visual_grounding、uncertainty_handling、information_relevance、hallucination_avoidance |
| R0 / S0 / Final 节点数 | 5 / 26 / 26 | 5 / 24 / 24 |

### 9.2 本次正式 K=3 结果及历史比较

下表统一将每次输出还原到原始候选索引，A 或 B 至少获得两票才作决定，弃权计错。两次六份完整 VLRB 产物的 K、1247 条样例顺序、固定 A/B 日程均已核对；从保存预测重算的正式指标与报告一致，未解决技术失败均为 0。

| 本次 G5 阶段 | 全量1247 | 未参与演化1147 | 演化集100 | 幻觉留出648 | General181 | Hallucination749 | Reasoning317 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 仅 Root（R0） | 925，74.18% | 841，73.32% | 84，84.00% | 541，83.49% | 93，51.38% | 626，83.58% | 206，64.98% |
| 初始子准则（S0） | 933，74.82% | 851，74.19% | 82，82.00% | 544，83.95% | 98，54.14% | 627，83.71% | 208，65.62% |
| 五轮演化后（Final） | 945，75.78% | 861，75.07% | 84，84.00% | 545，84.10% | 102，56.35% | 630，84.11% | 213，67.19% |

1147 只排除演化集，clean1146 还排除 1 条近重复样例；PPT 使用 1147，独立样例结论另外检查 clean1146。

| clean1146 阶段 | 历史 | 本次 |
| --- | --- | --- |
| R0 | 819/1146，71.47% | 840/1146，73.30% |
| S0 | 849/1146，74.08% | 850/1146，74.17% |
| Final | 850/1146，74.17% | 860/1146，75.04% |

本次 Final 比历史全量净增11条（+0.88个百分点），clean净增10条（+0.87个百分点），幻觉留出净增5条（+0.77个百分点）。但本次 R0 已比历史全量多答对26条：历史 R0→S0→Final 为899→929→934，本次为925→933→945。初始 Split 的全量收益从+30条变为+8条，后续演化从+5条变为+12条；整个流程相对裸 root 的收益从+35条变为+20条，不能将更高的 Final 分数解释为更大的整体方法增益。

本次 R0→Final 的 clean 净增20条由 General +9、Reasoning +7、幻觉留出 +4组成；S0→Final 的 clean 净增10条中，幻觉留出仅+1条。VL-RewardBench 的官方 OverallAcc 排除弃权，历史/本次 Final 为75.63%/76.70%，官方 MacroAcc 为69.07%/70.27%；它们与表中的 Strict ACC 是不同指标。Final 全量覆盖为历史1235/1247、本次1232/1247，即弃权从12条增加到15条。

旧终端日志曾显示本次 Final 76.02%（948/1247）：运行时相对多数允许 `[A, None, None]` 判 A，而正式 K=3 将其计作弃权。4条样例受此影响，其中3条运行时判对。`3ead5a4` 后 VLRB R0/S0/Final 日志统一打印正式口径并标记 `official K=3`；保存产物的 runtime 字段仍保留为诊断，不与正式结果混用。

### 9.3 配对变化与解释限制

精确双侧 McNemar 使用正确/不正确二元结果，弃权属于不正确。下面 p 值未做多重比较校正，属于样例级探索性分析，不表示跨运行方差。

| 比较 | 切片 | 修正 / 损害 | 净增 | McNemar p |
| --- | --- | --- | --- | --- |
| 本次 R0→S0 | 全量1247 | 53 / 45 | +8 | 0.4797 |
| 本次 R0→Final | 全量1247 | 61 / 41 | +20 | 0.0594 |
| 本次 R0→Final | clean1146 | 56 / 36 | +20 | 0.0470 |
| 本次 R0→Final | 幻觉留出648 | 23 / 19 | +4 | 0.6440 |
| 本次 S0→Final | 全量1247 | 48 / 36 | +12 | 0.2299 |
| 本次 S0→Final | clean1146 | 44 / 34 | +10 | 0.3082 |
| 本次 S0→Final | 幻觉留出648 | 17 / 16 | +1 | 1.0000 |
| 历史 R0→Final | 幻觉留出648 | 34 / 17 | +17 | 0.0241 |
| 历史 Final→本次 Final | 全量1247 | 64 / 53 | +11 | 0.3553 |
| 历史 Final→本次 Final | clean1146 | 60 / 50 | +10 | 0.3909 |
| 历史 Final→本次 Final | 幻觉留出648 | 26 / 21 | +5 | 0.5601 |

样例级配对 bootstrap 使用10000次抽样、seed=20260925。本次 R0→Final 幻觉留出增益95%区间为[-1.39,+2.47]个百分点；本次 Final 相对历史 Final 的幻觉留出差区间为[-1.23,+2.93]个百分点，均跨0。clean 的 p=0.047 接近0.05且来自多切片分析，不能单独作为稳健的整体显著提升证据。当前只能说两次均观察到 R0→Final 正向变化，尚未证明稳定的幻觉留出增益。

### 9.4 演化轨迹、成本与负结果

以下演化训练指标为 K=1，不能与第 9.2 节演化集100的正式 K=3 混用。每行接受数是该轮被接受的整组 children 候选数量。

| 状态 | 历史训练 Strict ACC | 本次训练 Strict ACC | 历史接受数 | 本次接受数 |
| --- | --- | --- | --- | --- |
| R0 | 77% | 78% | — | — |
| S0 | 80% | 78% | — | — |
| e01 | 79% | 78% | 2 | 2 |
| e02 | 84% | 81% | 3 | 3 |
| e03 | 84% | 81% | 1 | 0 |
| e04 | 84% | 81% | 0 | 0 |
| e05 | 84% | 81% | 0 | 1 |

两次均接受6/25次候选。本次第二轮后系统训练正确数停在81；第五轮的局部更新没有继续增加系统正确数。核心生成、初始 Split、逐例反思、局部验证/接受、Global Arbiter 与冻结外评均实际执行。

| 保存产物中的耗时 | 历史（分钟） | 本次（分钟） |
| --- | --- | --- |
| 五轮演化累计 | 119.49 | 80.42 |
| R0 全量 VLRB K=3 | 101.88 | 98.94 |
| S0 全量 VLRB K=3 | 145.59 | 142.77 |
| Final 全量 VLRB K=3 | 133.82 | 139.06 |

这些是各产物保存的阶段耗时，不是从首个命令到末个命令的连续总耗时。节点数量、准则长度和调用状态不同，不能把耗时差全部归因于代码整理。

主要负结果保留如下：

- `factual_accuracy` 的训练 K=1 正确数69→79，但 S0→Final 全量 K=3 为872→857、幻觉留出为509→500；`hallucination_avoidance` 的训练为77→79，全量885→877、幻觉留出527→520。局部训练收益没有一致转化为外部收益。单 root 分数不等于其系统贡献，不能据此直接做因果分摊。
- 本次 Dev150（K=1）从 S0 的99/150（66.00%）降到 Final 的97/150（64.67%），修正12条、损害14条，Final覆盖148/150、技术失败0。历史 G5 未保存对应 Dev 结果，无法比较两次 Dev。
- 历史 Final→本次 Final 的 Hallucination749 总分净增6条，但 povid 为417→413（-4）、rlaif-v 为153→162（+9）、rlhf-v 为54→55（+1），不是所有幻觉来源都改善。

两次 G5 均保留为流程重复，不按测试分数选出一个“最佳运行”。若继续研究，优先固定同一份 R0 重复后续流程，再独立重复 root 生成，以区分初始化与演化的波动；本次归档没有启动新实验或更改算法。

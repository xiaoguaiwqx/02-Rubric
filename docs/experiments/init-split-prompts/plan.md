# Init Split 提示词重构与独立验证计划

日期：2026-10-03
分支：init-split-prompts
状态：代码重构与独立入口已实现，69项离线测试通过；正式实验待启动。

## 1. 目标与本轮范围

把初始化拆分明确为“案例问题提炼 signature → 共性归纳 cluster → 子准则生成 children”，使一个文件就能同时看到三个阶段的系统提示词、用户模板与执行流程。

采用用户提供的六个中文常量作为第一版。每个系统提示词独立、固定，不接收参数；用户模板通过 {xxx} 插入描述清楚的上下文。输入使用带说明的文本模板，输出仍使用现有 JSON schema。

实验问题：固定同一份五根 R0 和发现集后，新初始化方案生成的 S0，是否改善未见样本上的偏好判断？本轮评估的是“中文阶段提示词 + 描述性用户模板 + 全部根上下文 + 移除签名输入中的 Global Arbiter 报告”的组合效果，不分摊各项改动的因果贡献。

本轮只生成与评估 S0。五轮局部演化不参与对比；旧 Final 只作历史参考。生成 root、Worker、Global Arbiter、A/B 映射和正式投票口径维持原协议。

## 2. 已核实的来源与基线

来源目录：output/subtree_reflection/seed11/g5/。

| 产物 | 用途 | 处理方式 |
|---|---|---|
| r0/rubric.json | 已生成的五根 R0 | 原样复用；记录 SHA256，不重做 warmup 或 root 生成 |
| r0/system.json | 100 条发现集上的 R0 Worker 报告，K=1 | 核对 root、样本及 A/B 顺序后复用，用于选例与 signature |
| init/rubric.json | 旧初始化 S0 | 只读对照 |
| init/r01–r05/ | 旧 signature、cluster、children 与请求 | 用于流程诊断，不作为新方案的成功缓存 |
| vlrb/r0.json | R0 全量正式 K=3 预测 | 复用，不重复付费评测 |
| vlrb/initial.json | 旧 S0 全量正式 K=3 预测 | 同样本配对对照 |
| report.json | 已保存正式切片与成本摘要 | 读取 generated_roots 摘要，并用原始预测复核 |
| ../discovery_100.jsonl、../config.json | 同一份发现集与配置 | 复制必要元数据，保持样本内容和顺序一致 |

R0 Rubric SHA256：6d17a9bc9e77c6b4102e35dfe14c1e404243f33a129fb4253452e30314a0f427。
旧 S0 Rubric SHA256：3827f27d936b18891d4028aeb3e3a9806e2aae364cd6204297e4a590335abf5d。

正式 K=3，至少两次判断一致；弃权在 Strict ACC 中计错：

| VLRB 切片 | R0 | 旧 S0 |
|---|---:|---:|
| 全量 1247 | 925/1247，74.18% | 933/1247，74.82% |
| 未参与初始化 1147 | 841/1147，73.32% | 851/1147，74.19% |
| 去近重复 1146 | 840/1146，73.30% | 850/1146，74.17% |
| 幻觉留出 648 | 541/648，83.49% | 544/648，83.95% |
| 发现集 100，正式 K=3 | 84/100，84.00% | 82/100，82.00% |
| General 181 | 93/181，51.38% | 98/181，54.14% |
| Hallucination 749 | 626/749，83.58% | 627/749，83.71% |
| Reasoning 317 | 206/317，64.98% | 208/317，65.62% |

发现集选例使用 R0 的 K=1 报告，不能拿它的 78/100 与上表的正式 K=3 84/100 混用。

旧初始化共 148 个逻辑 Manager 请求：138 个 signature、5 个 cluster、5 个 children。五根均在 primary 阶段完成，无扩展选例。138 条 signature 中 137 条 applicable=true；该比例作为诊断基线，不作为新提示词的优化目标。

| 目标根 | 初次选例数 | 有效 signature | cluster 数 | children 数 |
|---|---:|---:|---:|---:|
| generated_root_01 | 33 | 33 | 5 | 4 |
| generated_root_02 | 21 | 21 | 4 | 4 |
| generated_root_03 | 25 | 24 | 4 | 3 |
| generated_root_04 | 35 | 35 | 4 | 4 |
| generated_root_05 | 24 | 24 | 4 | 4 |

## 3. 代码职责与依赖

只新增一个主要阶段模块 init_split.py，不再另外创建 init_split_prompts.py。

| 文件 | 重构后的职责 |
|---|---|
| experiments/evolving_structured_rubrics/init_split.py（新增） | 六个提示词常量、上下文渲染、阶段输入与校验、选例、signature → cluster → children、S0 初始化 |
| manager_runtime.py | 模型调用、凭证读取、重试、JSON 解析、缓存、并发和调用统计；接受调用方提供的提示词及阶段校验 |
| rubric_pipeline.py | 数据读取、Rubric 投影与组装、共享合法性检查、评测、配置冻结和成本工具 |
| subtree_local_reflection_manager.py | 后续演化专用提示词与校验；保持实际发送的 case_reflection/subtree_split 提示词一致 |
| run_subtree_experiment.py | 调度生成 R0、独立初始化、局部演化和外评 |
| generated_roots_report.py、vlrb_official.py | 复用正式计分、切片和配对函数，支持初始化对比报告 |

依赖方向：入口调用 init_split；init_split 调用 manager_runtime 与 rubric_pipeline。manager_runtime 不反向导入阶段模块，rubric_pipeline 不反向导入 init_split。局部演化需要初始化时，由上层明确调用 init_split；演化 Manager 不再携带初始化提示词。

先做结构搬迁并验证旧请求等价，再接入新模板。两步分别记录，便于区分代码接线问题和提示词实验变化，不长期维护两套默认初始化实现。

特别核对：
- 当前初始化提示词和演化提示词共享 COMMON。移出初始化提示词时，固定原有演化上下文，不连带改写后续演化提示词。
- generated_root_initialization.py 通过 Manager 读取凭证。改造默认提示词设置时保留此用法。
- aligned_system_runtime.py 仍引用根数量措辞辅助函数。只调整必要依赖，不扩大到 Worker 重构。
- JSON 结构保留为解析与请求元数据；模型的初始化用户输入改为实际渲染后的文本。缓存必须包含真正发送的系统提示词和用户文本。未改动的演化调用保持现有请求与缓存身份。

## 4. 三个阶段的上下文

全部根上下文在三个阶段中均来自同一份 R0，只含 root ID、名称、职责。不会随着前面某根完成子准则生成而发生变化。

| 阶段 | 输入材料 | 图像 | 输出 |
|---|---|---|---|
| signature | 全部根、目标根、原问题、A/B 原文、目标 Worker 的四个报告字段、整体人类偏好 | 当前案例图像 | applicable、signature、basis |
| cluster | 全部根、目标根、有效 signature 的 ID、问题模式与案例依据 | 不附图像 | clusters、unassigned_ids |
| children | 全部根、目标根、有效原始 signature 与依据、cluster 及成员 ID | 不附图像 | 完整 children 组、change_summary |

signature 不输入 Global Arbiter 的报告。其他根只提供职责上下文，不提供它们的逐例判断报告。

继续保留以下初始化算法：
- 按目标 Worker 与 gold 不一致选取 primary 案例，None 也进入候选审查；审查后仍可 applicable=false。
- 每根至少四条有效 signature 才尝试形成至少两个 cluster。
- 每个 cluster 至少两条不同 signature，同一个 ID 不跨 cluster 重复。
- 有效 signature 或 cluster 不足时，只向同一份发现集的剩余案例扩展。
- 支持材料仍不足时，如实报告初始化失败，不降低阈值、不强行补规则。
- 每根生成 2–5 个 children，五个 root 的 ID、名称、职责及顺序不变。
- signature ID 保持发现集中的稳定映射；A/B 与 Worker 报告逐字段对应。
- Strict 与 Preserve5 是后续局部演化的设置，不在此次初始化中新增接受门槛或保留案例机制。

先使用中文模板及模型直接生成的子准则，不再另加翻译步骤。本次因此也包含提示词/生成准则语言变化；若之后改为英文，应另记一个版本。

## 5. 独立入口与产物

现有 run_subtree_experiment 已增加三个 stage：

| stage | 行为 |
|---|---|
| init | 读取已有 R0 及 R0 发现集报告，运行三个初始化阶段，保存 S0 和发现集 K=1 判断，完成后停止 |
| vlrb-s0 | 读取已冻结 S0，独立运行全量 VLRB K=3 |
| report-init | 离线比较同一 R0、旧 S0、新 S0，输出系统/根切片、配对变化与初始化成本 |

通过 --source-run 读取既有 R0/基线变体目录，将来源路径及身份记录到新运行目录。复用时核对 Rubric、模型配置、样本和 A/B 顺序；匹配的 R0/system 无需调用模型。

当前 external 要求 state.completed=true，不能直接用于初始化独立外评。应在共享评测层抽出“评估指定已冻结 Rubric”的路径，原 S0/Final 外评继续调用它。新入口以明确的 S0 阶段状态表示初始化完成，不伪造 Final 或把演化标成已完成。

推荐新目录：

output/init_split/seed11/template_zh_v1/g5/

保存：
- 来源、代码版本、R0 SHA、配置、split 和六个模板的版本信息；
- r0/rubric.json 与复用的 r0/system.json；
- init/ 下的逐根 signature、cluster、children、Rubric 与发现集系统报告；
- vlrb/initial.json、正式 S0 报告与配对对比；
- 请求缓存、运行日志、调用量、tokens、实际阶段耗时。

旧基线目录只读；新 Manager 缓存与 Worker 评测缓存使用新目录。图像统一读取 data/VL_RewardBench/dataset_images，继续使用现有共享图像机制。原始输出继续仅本地保存，提交精简结果摘要。

## 6. 离线验证与小规模接线检查

正式模型调用前完成：

1. 运行当前全套离线测试，记录清理前基线。
2. 重构后验证选例、扩展选例、聚类约束、组装与初始化失败行为保持一致。
3. 核对六个常量与用户认可的中文文本一致；测试每个用户模板实际渲染内容，包含全部五根、目标根和所需证据。
4. 用包含引号、换行、公式花括号的案例验证原文能完整插入模板；系统提示词不做参数替换。
5. signature 请求只携带目标 Worker 与当前图像；cluster/children 不附图像，不携带 Global Arbiter 报告。
6. 新模板请求可恢复缓存；修改系统或用户模板不能误用旧成功缓存。
7. 固定未改动的演化提示词及请求样例，测试 R0 生成入口、完整 evolve 和正式计分兼容性。
8. 验证 init 停在 S0、vlrb-s0 不依赖 Final，报告使用正式 K=3 而不是 runtime 相对多数。

接线通过后，可用同一发现集中的少量案例检查 Manager 能读懂输入并返回正确 schema。这只是请求接线检查，不能代替完整 S0；接线时若需修改模板，固定新版本后再开始正式初始化。

## 7. 实验顺序与判定

### M0：基线冻结与离线回归

复核已保存的 R0/旧 S0 预测与正式摘要；固定配置、发现集、parquet、split、样本顺序、A/B 日程和 root 文本。全套离线测试通过后再执行模型调用。

### M1：新 S0 初始化

使用同一五根 R0、同一 100 条发现集、同一 R0 K=1 报告。Manager 保持 Qwen3.5-27B、temperature=0.2、关闭 thinking及现有并发；Worker 保持 Qwen3-VL-8B-Instruct、temperature=0.5、max_tokens=2048、并发50。调用重试规则沿用现状。

逐根记录 signature 适用率、扩展案例数、cluster 成员、未分配 ID、子准则数及长度，并检查证据是否支撑模式、模式是否转化为可执行检查。不能以提高 applicable 比例或增加规则条数作为成功标准。

五根均完成后固定新 S0；发现集 K=1 仅作诊断，不以其分数挑选或重生成一个“最佳 S0”。

### M2：新 S0 正式全量评测

对 1247 条 VLRB 运行同一冻结 A/B 日程的 K=3，复用现有 Worker/Global Arbiter 与正式计分。外评开始前已固定 S0，测试样本、测试 gold 和测试报告不进入初始化生成过程。

读取一次全量预测后离线计算所有切片。旧 R0、旧 S0 预测直接复用，本轮不重跑 warmup、R0、旧 S0、五轮演化或 Dev150。

### M3：配对分析与精简归档

主指标沿用 Hallucination100 实验的幻觉留出 648 Strict ACC；同时报告 clean1146、nontrain1147 和全量1247，防止仅看发现集重合部分的收益。

固定三组比较：
- 新 S0 vs 旧 S0：本轮初始化组合变化的直接比较。
- 新 S0 vs 同一 R0：新初始化的增量收益。
- 旧 S0 vs 同一 R0：旧初始化收益参照。

输出正确数/总数、Strict ACC、Covered ACC、coverage、弃权、A/B 复判不一致；同时输出 General/Hallucination/Reasoning 和各 root 的诊断。每个根的分数用于定位变化，不能直接视为它对系统收益的因果贡献。

对主要留出切片输出 corrected、harmed、净变化、McNemar 和配对 bootstrap 95% 区间，复用现有统计口径；保存配对样本 ID，人工查看有代表性的改进与伤害案例。

| 观察 | 解释与下一步 |
|---|---|
| 初始化可完成、648 提升且 clean1146 不退化 | 值得进一步复验；结合配对不确定性决定证据强度 |
| 指标接近、区间含零 | 结果尚不明确；保留更清晰的代码结构，另决定是否追加重复 |
| 仅发现集或含发现集的全量改善 | 不能认定未见样本效果改善 |
| 部分切片提升但其他切片退化 | 报告权衡与伤害案例，不只展示有利切片 |
| signature/cluster 不足导致失败 | 报告初始化可行性结果，不临时改阈值挽救 |
| 新 S0 退化 | 先定位阶段证据/子准则变化；下一版作为独立实验，不覆盖本次结果 |

这是一次固定 R0 的探索性比较。固定来源可减少 root 生成波动，但 Manager 与 Worker 的生成仍有随机性；一次运行不能证明各改动单独有效或收益稳定。若结果值得推进，再在同一 R0 上追加独立初始化重复；语言、全部根上下文、Arbiter 输入的单项消融属于后续可选实验，不阻塞本轮。

## 8. 调用预算、耗时与交付

- 旧初始化：148 个逻辑 Manager 请求。
- 新初始化：第一批仍为138个 signature 请求；若多根需要补充，最坏每根审查100条，最多500个 signature、10个 cluster、5个 children 逻辑请求；失败重试另计。
- R0 发现集报告与正式 VLRB 已保存，复用时无新增评测调用。
- 新 S0 发现集 K=1：理论约100 × (5+1) = 600个 Worker/Arbiter 逻辑请求。
- 新 S0 全量 K=3：理论约1247 × 3 × (5+1) = 22,446个 Worker/Arbiter 逻辑请求；实际缓存命中与失败重试单独记录。
- 旧 S0 全量评测保存耗时为8566.34秒，约142.77分钟。可先以约2小时23分钟作为同环境外评的预算参照；新准则长度、服务负载、重试会改变耗时，初始化耗时另计。
- 不需要本地 GPU 训练；主要成本是模型服务调用，tokens 与金额只能依据实际用量和已有服务计费记录统计。

交付物：
1. 职责清楚的 init_split.py、必要共享接口调整及离线测试。
2. 独立 init / vlrb-s0 / report-init 命令与准确的配置说明。
3. 新 S0、可恢复的请求缓存、日志和正式配对报告。
4. 一份“同一 R0 / 旧 S0 / 新 S0”的精简结果文档，含负结果、成本和适用范围。
5. 分别提交代码与测试、实验摘要；未经实验验证，不将新提示词替换进 main。


## 9. 实施记录

- 重构前60项测试通过；纯搬迁阶段对旧 S0 离线回放148个 Manager 请求，输入与产物完全一致。仓库重命名导致图像绝对路径别名不同，回放核对图像字节一致后使用旧别名；原产物未改写。
- 中文模板接入后新增9项离线测试，全套69项通过；演化、Worker/Arbiter固定提示词、接受规则、正式计分测试继续通过。
- 已实现独立 init、vlrb-s0、report-init 与 --source-run；共享图像及缓存恢复沿用既有机制。
- 已从原始 VLRB 预测独立复算 R0/旧 S0 的1247、1147、1146和648切片，分别为925/933、841/851、840/850和541/544，均与保存摘要一致。
- Worker 与 Manager 模型服务的只读可用性核对通过。初始化曾启动，随后按用户“不要commit代码和运行实验”的要求停止；只保留部分初始化缓存与日志，没有新S0或正式VLRB结果。
- 已撤回本次本地提交，保留全部未提交修改。以上实验与提交步骤属于原计划，当前不执行；本轮交付限定为代码重构、离线验证及文档。


```powershell
Set-Location "D:\3-Work\02-DD-LLM\02-Rubric"

$runRoot = "output/init_split/seed11/template_en_v2"
$config = "experiments/evolving_structured_rubrics/configs/generated_roots.example.json"
$sourceRun = "output/subtree_reflection/seed11/g5"

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

# 1. 复用五根 R0，使用英文提示词生成新 S0
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment init `
    --config $config `
    --output-root $runRoot `
    --variant g5 `
    --source-run $sourceRun `
    --attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/init.log"

if ($LASTEXITCODE -ne 0) { throw "Init Split 失败，请检查 init.log" }

# 2. 新 S0 的全量 VLRB 正式 K=3 评测
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment vlrb-s0 `
    --config $config `
    --output-root $runRoot `
    --variant g5 `
    --attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/vlrb-s0.log"

if ($LASTEXITCODE -ne 0) { throw "VLRB 评测失败，请检查 vlrb-s0.log" }

# 3. 对比同一 R0、旧 S0 与新 S0
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment report-init `
    --config $config `
    --output-root $runRoot `
    --variant g5 `
    --source-run $sourceRun `
    2>&1 | Tee-Object -FilePath "$runRoot/report-init.log"

if ($LASTEXITCODE -ne 0) { throw "结果对比失败，请检查 report-init.log" }
```
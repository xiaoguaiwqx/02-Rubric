# Framework v6：先 Split 初始化，再分层反思演化

日期：2026-09-17。状态：**代码实现与离线验证完成；尚未调用真实模型运行 smoke 或正式实验。**

本计划对应 [framework-editable-v6.pptx](framework-editable-v6.pptx) 的主框架，以及用户最新确认的“先五根 Split、最多五轮演化”设置。工作区为 `CritiQ-framework-v6`，分支为 `codex/framework-v6-clean`，实现起点为 `cf4a799`。旧版 27B Split-only 计划可在该提交的同路径查看；旧分支及其实验产物保留，不能作为本协议已完成的证据。

## 1. 要验证什么

**核心假设：固定 Worker 和 Global Arbiter，通过系统反思与子树反思，将有限能力 Worker 的实际失败转化为更易执行的子准则，可以逐步提高最终偏好判断能力。**

这里不假设 Worker 足够强。漏执行、作用域混淆、证据误读、理由与答案不一致，都可以成为改进 Rubric 的线索。不能仅因为问题被称为 `worker_execution` 就排除修改；也不能保证改写准则一定能修复视觉能力不足。

| 判断 | 本次需要的证据 | 解释边界 |
| --- | --- | --- |
| 演化后的 Rubric 比 Split 后的初始 Rubric 更有用 | S0 与 Final 的完整系统配对结果；每轮纠正和改错样本 | Discovery 上升是搜索结果，不能单独证明泛化 |
| 修改通过局部报告影响最终判断 | 反馈案例 → 子准则变化 → Worker 报告变化 → Arbiter 最终选择的可追溯实例 | 联合接受不能证明每个参与 root 独立有效；没有无反思对照时，不声称反思的独立因果增益 |

本次先完整跑通图中的一个主实验。多随机种子、无反思消融、位置交换诊断等留作后续，不插入主流程。

## 2. 固定设置与变化范围

| 项目 | 本次设计 |
| --- | --- |
| 五个 root | Completeness、Visual Grounding、Factuality、Creativity、Clarity；ID 与职责描述固定 |
| 初始 Rubric | 五个 root 全部生成孩子后得到 S0；裸 root 版本记为 R0 |
| 结构 | 仅一层孩子，每个 root 为 2–5 个孩子；后续替换该 root 的整组孩子，不递归扩深 |
| Manager | 硅基流动 `Qwen/Qwen3.5-122B-A10B`，开启思考；所有管理阶段使用同一模型 |
| Worker / Arbiter | `Qwen3-VL-8B-Instruct`，用户指定的 10.102.137.255 服务；实际地址写入本地配置 |
| Worker 并发 | 所有 Worker 与 Arbiter 请求共用全局上限 50，不能每个 root 各开 50 |
| Worker 参数 | 沿用当前基线 temperature=0.5、max_tokens=2048；演化 K=1，样本与 A/B 顺序固定 |
| Manager 调用 | 计划全局并发 4、单次 timeout=300s、每项最多 4 次总技术尝试；避免 SDK 与外层重试相乘 |
| Manager 长度参数 | 不把未经验证的 thinking_budget 当作硬上限；实现时记录实际请求参数和返回 usage。本次不发送 max_tokens / max_completion_tokens / thinking_budget，沿用服务端默认输出限制；不宣称输出或思考无限长，整次运行保持不变 |
| 演化次数 | S0 之后最多 5 轮；初始化不计入这 5 轮 |
| 接受规则 | 候选在全量 Discovery 上 Strict ACC 严格高于当前已接受版本，整批接受；相等或下降均拒绝 |
| 变化范围 | 仅 Rubric 孩子及必要的管理反馈；Worker、Arbiter 推理提示词与评价语义固定 |

开启思考指 Manager API 的运行设置；传给系统反思的 Arbiter reason 指模型实际输出的 `analysis_a`、`analysis_b`、`thought`，不要求供应商提供未暴露的内部推理轨迹。

## 3. 数据与比较对象

### 3.1 三类 Rubric

- **R0：五个裸 root。** 用于生成原始 root 错误材料；可报告其系统表现作背景。
- **S0：五个 root 全部 Split 后的 Rubric。** 正式演化基线，即本次的 init。
- **Current / Final：当前已接受版本 / 演化结束版本。** 每轮候选与 Current 比较，最后主要比较 S0 与 Final。

S0 不需要先胜过 R0 才建立，否则变成“挑一个有利的初始化”。必须记录 R0→S0 的变化；即使 S0 更差也如实报告，不重新生成到胜出为止。五根初始化缺失不能冒充完整 S0。

### 3.2 数据用途

| 数据 | 使用方式 |
| --- | --- |
| Discovery100 | 沿用已有冻结的 100 个样本 ID；初始化材料、演化反思与整批接受均来自这里 |
| 每轮反思案例 | 从这 100 条中确定性选择最多 40 条，按第 5 节规则覆盖错误和正确边界 |
| Dev150 | 演化完成、Final 冻结后比较 S0 / Final；不选轮次、不进 Manager |
| VLRB1247 | S0 / Final 最终配对评测；沿用现有 K=3 和确定性平衡 A/B 顺序协议 |
| 历史 heldout-500 | 本次不列为主流程必跑项；若后续追加，明确其已知来源重叠 |

选择 Discovery100 是为了先隔离框架变化、复用既有数据划分，并非认定 100 条足以代表全部任务。**接受使用全部 100 条，40 条反思子集不能代替接受集。** 不复用历史 71% 等基线，也不复用旧 init 的 VLRB 结果，因为本次 S0 不同。

数据集清单、样本顺序、模型参数与生成的 S0 在正式演化前保存。Dev / VLRB 不用于生成、反馈和接受；VLRB 已在历史研究中反复观察，应称为外部回归评测，不宣称全新盲测集。

## 4. 主流程：与 PPT 各模块逐一对应

```text
R0：五个裸 root
  → 各 root 原始错误材料 → 文本签名 → 语义聚类 → 五组孩子
  → S0：五根均已 Split → 全量系统评估
  → Current = S0

每轮，最多五轮：
  Current 的报告、Arbiter 理由与人工偏好
  → 系统级反思：按 root 输出有针对性的反馈
  → 子树级反思：各 root 决定 Preserve / Revise
  → Revise root：签名 → 聚类 → 完整新孩子组
  → 合并为一个候选 Rubric → 全量系统评估
  → 严格提高则 Current = Candidate，否则保持 Current
  → 带着本轮结果进入下一轮

冻结 Final = Current → S0 / Final 的 Dev、VLRB 评测 → 汇总分析
```

### 4.1 初始化：五根都要 Split

1. 在 Discovery100 上运行 R0 的完整系统：每个样本五份单节点子树报告，再由 Arbiter 判断。保留这些真实报告作为原始 root 材料，避免为同一用途额外执行一套重复诊断。
2. 按 root 收集局部可能失败案例。局部 A/B 与 gold 不同只能作为选例线索；局部 None、不同 root 偏好不同，不直接认定为错误。
3. Manager 在该 root 职责下提炼错误签名：具体漏掉什么、如何误读或错误比较、需要怎样的判断指引。证据不支持本 root 缺陷的案例标为不适用，不强行制造错误。
4. 对有效签名进行语义聚类，生成 2–5 个可复用的孩子。沿用最小簇大小 2；不要求每个签名必须归入某个簇，允许列为未聚类材料，防止强行合并无关案例。
5. 五组孩子一次合并得到 S0，运行完整系统并保存 init ACC、报告和 Arbiter 理由。

初始化不沿用旧的 ACC/coverage 阈值来决定“哪些 root 才允许 Split”；五根均进入生成。若某根有效案例不足以形成两簇，先从它在同一 Discovery100 的其余报告补充职责内疑难案例并提炼签名。仍不足则明确记录初始化无法满足五根 Split，停止检查材料，不能悄悄跳过该根或放宽簇规则。

原始 root 案例库及初始签名保存下来，供后续 root-local generation 使用；不把当时的 gold 分歧全部宣称为已核验局部错误。

### 4.2 每个模块的输入与输出

| 模块 | 输入 | 处理与输出 |
| --- | --- | --- |
| Unified-Subtree Worker | 原图、问题、A/B 原文、该 root 及其孩子 | 一次独立调用执行完整子树；输出局部分析、理由与 A/B/None |
| Global Arbiter | 原图、问题、A/B 原文、五份完整子树报告 | 输出 `analysis_a`、`analysis_b`、`thought`、`answer`；按原协议判断 |
| System assessment | 全量样本最终预测、人工偏好；候选评估时还有 Current 的预测 | Strict ACC、corrected/harmed/net、分组结果；为反思提供案例选择信号 |
| 系统级反思 | 简洁的当前 Rubric、代表性案例的图文原文/gold/五份报告/完整 Arbiter 理由、上轮结果摘要 | 识别证据或比较链条问题，按职责输出各 root 的反馈；无法归因的问题保持系统级记录 |
| 子树级反思 | 该 root 与当前孩子、对应反馈、相关案例与保留边界、该根上一轮修改结果 | Preserve 或 Revise；理由、案例 ID；Revise 时说明修改方向及需要保留的有效指导 |
| Root-local generation | 原始 root 签名库、该 root 最新反馈及相关当前案例、当前孩子与保留方向 | 按反馈更新相关签名、聚类，再生成完整新孩子组；输出可读的前后变化摘要 |
| 联合验收 | 合并后的候选、Current、同一完整 Discovery100 | 严格提高则整批提交，否则整批拒绝；输出纠正/改错案例，供下一轮使用 |

系统反思先完成，之后五根反思可并发；只有 Revise 的 root 进入生成。生成链保留 PPT 的“签名→聚类→孩子”，不是绕过这三步直接做任意单节点补丁。

### 4.3 修改应当具体改变什么

例如 Visual Grounding 的报告已发现 B 有图像依据，却因 B 未满足其他维度的文字要求而选 None：

- 系统反馈指出该 root 把全局文字要求混入局部判断，关联到实际案例和报告字段。
- 子树反思判断需要补充作用域和比较步骤，并保留已有的图像细节核对规则。
- 生成阶段将相关案例归纳为“局部视觉比较被全局条件覆盖”，在新孩子组中明确：先核对候选与图像对应的细节，再按本 root 的视觉依据比较；其他维度的失败交由对应报告和 Arbiter 处理。
- Worker 再次执行后，检查报告是否确实遵守该范围，以及最终系统的纠正是否多于改错。

这是一种待检验的修复假设，不能仅凭准则文字更清楚就宣布有效。

## 5. 代表性反馈：全量评估，分批系统反思，按 root 分配

### 5.1 每轮选择最多 40 条真实案例

选择过程由代码完成，不新增逐样本 Manager 诊断。使用固定 seed=42 和固定样本 ID 排序，保存所选 ID 与选择原因。

1. **最多 24 条当前系统错误案例**：在已有数据来源/任务类别内轮转取样，兼顾不同 root 报告的分歧模式，避免某一来源占满。
2. **最多 8 条上一轮候选引发翻转的案例**：corrected / harmed 各最多 4 条；无上一轮时用其他未选当前错误补充。
3. **至少预留 8 个名额给当前系统正确案例**：兼顾五根有分歧和一致的情况，为保留有效行为提供边界。

三组按样本 ID 去重；不足名额由未选错误优先、其他正确案例随后补齐，总量不超过 40 或数据集大小。若错误不足，应纳入全部错误，不为了比例制造错误案例。

分组优先用数据本身已有的标签，不另请模型给全部样本分类。若只有来源标签，就报告来源覆盖，不能称为已经覆盖所有任务类型。

### 5.2 系统反思按 10 条一批

40 条最多形成 4 次系统反思调用。每批看到相同的简洁 Rubric，以及本批每条例子的原图、问题、完整 A/B、gold、五份完整报告和 Arbiter 理由。逐例内容在该请求里只放一次。

每批输出一份按五个 root 分组的反馈列表。每条仅需：`sample_ids`、问题描述、证据依据、建议方向；另有未能分配给 root 的系统观察。代码合并分组并按案例去重，不增加一次“审计反馈的审计”调用。

这里的分批只是控制单次上下文，仍属于图中的系统级反思。每批提示词应允许无问题、证据不明确和某个 root 无反馈；不要求每个案例、每个 root 都必须生成缺陷。

### 5.3 各 root 获得自己的反馈包

- 每根仅接收分配给自己的反馈；同一案例可以关联多个 root，但每根必须说明各自职责上的问题。
- 根级反思每次最多看 20 条去重案例，优先最多 12 条相关问题，再最多 8 条职责相关的正确/保留案例；案例不足时不强行凑数。
- 超出上限时按问题类型轮转选例，保留反馈摘要和其余案例 ID。保留案例从已选系统案例中选取，避免额外扩张原图输入。
- 没有明确问题的 root 可以 Preserve；不能为了五根都有修改，把同一泛化建议复制五份。
- 每轮报告：所选总量、来源覆盖、错误/正确/翻转数量、各 root 的反馈数、独立案例数及实际输入数。某根只有两三个案例时必须如实标出。

40/20/10 是本次的可执行预算选择，不是统计充分性的保证。先用这些案例形成修改假设，再让完整 100 条上的联合比较判断效果；外部评测检查是否迁移。

## 6. 生成与接受：允许继续改进，也保留失败信息

### 6.1 当前反馈真正进入生成

原始 root 签名库保留作历史底座，但不能每轮不看反馈、照搬全部旧错误重新聚类。

- 按本轮 Revise 方向选择相关旧签名，并加入本轮反馈支持的新局部案例；旧签名文本无需重算，输入条件改变的签名单独生成。
- 不相关旧问题留在库里，当前孩子中有效指导作为保留要求；避免一次局部修复把整棵子树改成另一个任务。
- 聚类围绕此次要修复的模式进行，再将新模式与保留要求合成为完整孩子组。可以保留原孩子文本，不能简单追加至无限增长。
- 新准则使用角色无关表述，不写固定“Prefer A / Penalize B”，不写案例答案、样本 ID 或只适用某一张图的结论。本次不额外运行 A/B 交换实验。
- 每根每轮只有一个语义候选；传输/格式重试不等于允许性能拒绝后在本轮反复抽候选。

### 6.2 整批比较与状态更新

设当前版本为 S，候选为 C，固定接受集为 D：

`accept ⇔ correct_count(C, D) > correct_count(S, D)`

分母始终为完整 Discovery100，因此等价于 Strict ACC 严格提高。最终合法 None 按沿用的 Strict 规则计入分母、不计正确；局部 None 是合法局部报告。技术失败未解决时不进行接受比较。

例如：S0=65%，第一轮候选=68%则接受；第二轮候选=67%，虽然高于 S0，仍拒绝；第三轮候选=69%才再次接受。

- 未变 root 可复用完全相同输入的报告；变化 root 在全部 Discovery 上重算，随后对全部样本重新调用 Arbiter。
- 多根同时修改时只有一次联合接受，没有逐根性能预筛、最佳子集回退或隐含局部 ACC 门槛。
- 接受后候选预测直接成为当前基线，已接受 root 下轮仍可 Revise；不再使用旧版“Split 一次即完成”的 pending 逻辑。
- 拒绝后保留 Current，同时记录候选文本、报告变化和 harmed/corrected，下一轮系统反思可使用这些失败信息。联合结果不标成某 root 的因果贡献。
- 到五轮结束，或全部 root 明确 Preserve 时结束。全 Preserve 记为本轮无修改，不宣称已经达到最优；技术失败/非法提案不能伪装成 Preserve 或自然收敛。
- 不因连续两轮拒绝自动停止；无有效候选源于生成失败时，记录失败并暂停恢复，不默默漏掉某根后继续验收。

## 7. 必须修复的实现问题与最小边界

| 当前风险/历史问题 | 本次最小处理 |
| --- | --- |
| Arbiter 解析只留下 answer | 保留并传递 `analysis_a`、`analysis_b`、`thought`、`answer`，同时保留 raw；检查系统反思的实际输入，而非仅检查缓存存在 |
| 下游短摘要再次丢失关键理由 | 所选案例的原始分析字段完整传入，不沿用固定截取 400/600 字符作为唯一证据 |
| rubric examples 递归展开导致输入膨胀 | 为 Manager 显式构造 ID/名称/描述/孩子关系的投影，递归排除 examples、历史大对象；当前选中案例通过独立字段传一次 |
| 用最终 gold 判定每个 root 对错 | gold 用于系统评价及反思线索；Manager 必须按 root 范围核对，允许合理局部分歧和 None |
| 系统反馈集中于一个案例 | 固定选例预算、来源轮转、错误/正确/翻转覆盖；按 root 输出并记录覆盖情况 |
| 认为 Worker 出错就不能改 Rubric | 将其视为可尝试通过规则顺序、范围、判断映射改善的执行问题，最终用结果检验 |
| 输出格式复杂造成大量重试 | 系统反馈使用简短列表，root 仅 Preserve/Revise+reason；Revise 才需要目标；不做逐字引用匹配或强制所有字段非空 |
| 看不出哪个请求在等待 | 日志明确 stage、epoch、root/batch、attempt、开始/结束时间；保存 provider usage、finish_reason、超时和重试原因 |
| 缓存混用、目录膨胀 | 一次正式实验一个短输出目录；同参数恢复复用已完成请求，科学配置改变另起实验，保留历史产物 |

代码复用 `joint_split_evolution.py` 的整批系统评估/比较与数据读取、`split_evolution.py` 和现有 Manager 的签名/聚类/孩子生成思路、`aligned_system_runtime` 的 Worker/Arbiter 执行。

旧 `_prepare` 的阈值触发、剩余孩子容量及“一次 Split”语义不能直接当新调度器使用。新入口显式组织 S0 初始化、两级反思、整组孩子替换和五轮状态；共享函数仅添加必要的显式参数，历史协议默认不变。

不移植旧 evidence-routed 大流水线，不引入逐例 100 次诊断、严格引用账本、手动逐级审批、重复机制门槛或通用迁移框架。最小检查仅覆盖真实结构/引用错误，例如样本 ID 存在、孩子数量合法、reason 确实传递；它们不承担事实正确性的自动证明。

## 8. 实现与运行顺序

| 阶段 | 工作与完成条件 | 是否模型调用 |
| --- | --- | --- |
| P1 数据与理由通路 | 保存完整 Arbiter 理由；Manager Rubric 投影排除 examples；样本/标签边界正确 | 离线假后端验证 |
| P2 框架调度 | 五根 S0 初始化、分批系统反思、五根反思、签名/聚类/孩子替换、五轮联合接受与恢复 | 离线验证关键分支 |
| P3 一次小型联通检查 | 固定 10 条样本检查真实图文、完整 reason、Manager thinking 请求、各阶段解析及端点；复用离线夹具覆盖小样本不触发的生成分支 | 少量真实调用；不作为收益结论 |
| P4 正式主实验 | 固定完整 Discovery100，生成 S0 后最多五轮；一条入口顺序运行，不按阶段反复手动批准 | 正式调用 |
| P5 最终评测与报告 | 冻结 Final，S0/Final 在 Dev150、VLRB1247 配对评测，汇总收益、退化、成本 | 正式调用 |

离线验证重点：严格提高才接受；高于 S0 但低于 Current 时拒绝；接受后仍能再次修改；所有 root 必须完成初始化；合法 None 与技术失败分开；多根整批提交/拒绝；恢复不重做已完成调用；gold 不进入 Worker/Arbiter；examples 不进入 Manager。

小型联通检查只验证服务及信息传递，不要求在 10 条上证明提升，也不为少量案例强行降低聚类规则。真实主实验若初始化失败，应解决具体错误再恢复，不随运行结果调整指标或数据。

建议正式产物统一放 `output/fw6`，配置放 `.local/framework_v6/config.json`。目录内部区分 `r0/`、`init/`、`e01/`…`e05/`、`dev/`、`vlrb/`；smoke 放同目录下独立子目录，不污染正式状态。实现后提供准确的一次性运行与断点恢复命令；本计划不列尚不存在的 CLI。

## 9. 预算与结果报告

### 9.1 逻辑调用预算

在 Discovery100、K=1 下：

- R0 完整系统：100 × (5+1) = **600** 次 Worker/Arbiter 调用。
- S0 完整系统：同为 **600** 次。
- 每轮 m 根变化：100 × (m+1) 次；五轮每次五根变化最多 **3,000** 次。
- 因此演化相关最多 **4,200** 次，不含联通检查、Manager 与技术重试。
- 每轮系统反思最多 4 次，子树反思最多 5 次；五轮最多 **45 次反思调用**。
- 签名、聚类与孩子生成另计：若有效初始案例总数为 E，逐例签名约 E 次；每次 root 生成另有一次聚类及一次完整孩子组生成调用。实际数量由错误分布与复用量决定，不能把 Manager 成本只算成 45 次。
- Dev150，S0/Final 各 K=1：最多 **1,800** 次系统调用。
- VLRB1247，S0/Final 各 K=3：最多 **44,892** 次系统调用。外部评测很可能是主要 Worker 成本。

S0 与 Final 相同时复用同一份评测；未变 root 的可复用调用会降低上述上限。真实耗时根据联通检查及首轮的服务吞吐估算，不能简单用调用次数除以并发 50。Manager 费用按返回 token usage 和实际账单统计，不预设 thinking_budget 已限制思考长度。

### 9.2 必须产出的结果

1. **主表**：R0/S0/Final 的 Discovery 指标；S0/Final 的 Dev、VLRB Strict ACC、配对 corrected/harmed/net，并报告现有分类指标。
2. **演化表**：每轮 Current ACC、Candidate ACC、接受/拒绝、修改 roots、孩子数、耗时、Manager tokens 与技术重试。
3. **机制案例**：至少展示修复成功与修改造成退化两类实际例子，串起反馈、准则、Worker 报告及 Arbiter 判断。
4. **反馈覆盖表**：每轮各 root 获得多少独立案例、是什么问题，是否只围绕同一案例反复修改。

可以报告配对置信区间或既有统计检验帮助解释小幅差异，但不把统计显著性临时改成接受门槛。K=1、同一 Discovery 生成与选择，以及随机采样都会带来选择偏差；一次主实验也不能证明跨 seed 稳定性。

### 9.3 如何解释成败

- Discovery 与外部评测均改善，且出现可追溯的报告修复：支持该框架的可行性与迁移潜力。
- Discovery 改善但外部持平/下降：搜索可能过拟合，不能把严格单调的接受曲线当作泛化证据。
- Rubric 改变但 Worker 报告未改善：重点检查指引是否可执行、是否被其他孩子干扰，以及视觉能力边界。
- Worker 报告改善但 Arbiter 不受益：定位信息整合问题；本次保持 Arbiter 固定，如实记录，不中途改提示词救结果。
- S0 建立但后续均 Preserve/拒绝：这是完整负结果；不能强制修改或挑最后候选替换 Final。

**本次交付边界：首先完成并运行 PPT 所描述的框架，然后基于保存的证据决定下一项改进。**


## 10. 实现交付与运行命令（2026-09-17）

入口为 `experiments/evolving_structured_rubrics/framework_v6.py`，Manager 提示词及缓存调用在 `framework_v6_manager.py`。系统推理复用 aligned runtime，联合指标复用原有实现。原 Specialize Manager 将签名限制为明确 wrong，且要求签名全部归簇，不能表达本协议的职责内核验和不适用案例；本次用一个小型 Manager 适配器承载五种任务，不修改历史 Manager 的默认约束。聚类后一次生成完整孩子组，使保留与修改指令共同约束最终子树。

共享修改只有：Arbiter 增加可选的完整理由解析路径，aligned runtime 显式启用该路径，VLRB 数据加载增加显式 parquet 路径参数。旧协议默认不变。

本机配置已经生成在 `.local/framework_v6/config.json`。它只读引用原工作区的数据和 `.env`，不复制密钥、不复用旧预测；新产物在当前工作区 `output/fw6`。Manager 请求为 `temperature=0.2`、`enable_thinking=true`，timeout=300 秒、SDK 自动重试关闭，由外层累计尝试上限统一管理。未设置输出/思考 token 预算。

### 一次性执行所有阶段

在 PowerShell 执行以下完整命令。使用新工作区的代码；`.env` 由程序自动加载，无需再手动设置 `GUIJI_API_KEY`。

```powershell
Set-Location 'D:\3-Work\02-DD-LLM\CritiQ-framework-v6'
conda activate critiq
$Py = 'C:\Users\wenqx\miniconda3\envs\critiq\python.exe'
$Module = 'experiments.evolving_structured_rubrics.framework_v6'
$Config = '.local/framework_v6/config.json'
$Out = 'output/fw6'

& $Py -u -m $Module configure --config $Config --data-root 'D:\3-Work\02-DD-LLM\CritiQ' --env-file 'D:\3-Work\02-DD-LLM\CritiQ\.env' --worker-url 'http://10.102.137.255:8000/v1'
if ($LASTEXITCODE -ne 0) { throw '配置生成失败' }

& $Py -u -m $Module all --config $Config --output-dir $Out --attempt-limit 4
if ($LASTEXITCODE -ne 0) { throw '实验中断；检查最后一条错误及对应调用文件，然后断点续跑' }
```

`all` 顺序执行：本地检查 → 10 条 smoke（系统推理及两级反思）→ 五根 Split 初始化 S0 → 最多五轮演化 → Dev150 → VLRB1247 → 汇总。不需要中间手动审批；真实技术或材料不足错误会停止，不将失败伪装成正常完成。

### 恢复与检查

- 普通中断：重新执行同一条 `all` 命令，已成功请求读缓存；不删除结果目录。
- 某调用已用满 4 次：先检查原因；确认可以续试后，以 `--attempt-limit 8` 重新运行。8 是累计上限，不是再加 8 次；成功请求不重跑。
- 只检查本地配置/数据：将 `all` 改为 `check`，不会调用 API。
- 最终结果：`report.json`；过程：`eXX/summary.json`、`eXX/coverage.json`、`eXX/system_feedback.json`；Rubric：`init/rubric.json`、`final.json`；机制实例：`mechanism_examples.json`。
- 每项 Manager 调用文件保存实际输入、输出、供应商 reasoning_content（若返回）、usage、finish_reason、失败原因和耗时；API 密钥不写入文件。

离线测试覆盖五轮状态转换、全 Preserve、五根初始化缺失、整组替换、反馈分配、完整理由、examples 排除、合法 None、无 gold 泄漏、请求缓存/累计重试，以及外部 S0/Final 与 K=3 协议。真实模型返回质量和服务吞吐尚未验证，运行中的 smoke 将检查实际端点及图文反思链。

验证记录：75 项相关离线测试通过（其中 19 项为 Framework v6 专项测试）；本地 Discovery100/Dev150 图像检查通过，VLRB parquet 元数据为 1247 条。独立审查子智能体因模型额度限制未完成，已进行本地复核，不能视为独立审查通过。


### 运行修复：TPM 限流后等待（2026-09-17）

首轮 root 反思遇到 HTTP 429 / TPM limit reached；旧调用循环立即重试，导致四次机会在几秒内耗尽。现改为 Manager 共享冷却：连续 429 分别至少等待 60、120、240 秒，尊重更长的 Retry-After 秒数。所有尚未发出的 Manager 请求遵守同一冷却时间，日志打印等待；不把限流/传输异常作为 JSON 格式纠错信息塞给模型。累计尝试上限、模型、提示词、并发、timeout 和接受规则不变。

检查时 r01/r02/r04/r05 已在第5次成功；r03累计8次，其中前4次为429、后4次为约300秒超时。超时本身不能证明仍是TPM限流。没有删除或修改这些运行产物；已成功请求继续复用。修复后21项Framework专项离线测试通过，包括共享冷却与Retry-After等待。未自动启动续跑。

r03已有8次记录，续跑需提高累计尝试上限，例如 `--attempt-limit 12`；这最多再给该调用4次机会。如果仍持续300秒超时，应检查服务响应与该请求规模，不能宣称本次429修复已解决超时。

### Manager timeout 调整为 500 秒（2026-09-17）

用户已将本地配置的 Manager timeout 从300改为500秒。续跑检查现仅允许这个操作参数变化，模型/采样等科学设置仍必须匹配；保留原始run_config快照，每次新Manager调用记录timeout_seconds。check已打印实际配置500秒，22项专项离线测试通过。该检查未发出API请求，不代表500秒的真实调用已成功。修改配置后直接重启all命令，不要重新执行configure覆盖用户设置。正在运行的进程不会自动加载改后的配置。

### 27B Manager 独立重跑（2026-09-17）

因122B完整反思请求连续超时，另建27B对照运行。配置`.local/framework_v6/config_27b.json`从当前122B配置复制，仅将Manager模型改为`Qwen/Qwen3.5-27B`；保留思考开启、temperature=0.2、并发4、timeout=500秒，以及Worker并发50和最多五轮的算法设置。122B配置与`output/fw6`产物原样保留，待服务稳定再续跑或另行重跑，不覆盖历史。

27B输出独立放`output/fw6_27b`，重新执行R0、五根Split建立S0、最多五轮演化和最终Dev/VLRB；不复用122B初始Rubric或任何预测。已完成本地check，尚未启动付费推理。直接运行如下命令，不执行会恢复122B模板的configure：

```powershell
Set-Location 'D:\3-Work\02-DD-LLM\CritiQ-framework-v6'
conda activate critiq
python -u -m experiments.evolving_structured_rubrics.framework_v6 all --config .local/framework_v6/config_27b.json --output-dir output/fw6_27b --attempt-limit 4
```

若122B实验仍在运行，先在其终端按Ctrl+C退出，避免同时占用同一API账户的配额。27B主实验从独立S0开始，因此最终差异包含初始化与演化轨迹的差异，不能当作固定输入下的单次反思能力对照。


## 冻结 Rubric 的显式递归对照（2026-09-18）

本组不重新演化，也不调用 Manager。固定 `output/fw6_27b` 的 init 和最终接受的 e02 Rubric，分别用历史 Prompt-v2 单准则 Worker + 显式递归聚合评测 VLRB，与同一 Rubric 已完成的 Unified-Subtree + Global Arbiter 结果配对比较。

- init 为 22 节点，Final 为 21 节点；VLRB 1247 条，复用相同 K=3 和 A/B 呈现顺序。
- Worker 为 Qwen3-VL-8B-Instruct，沿用当前配置服务、全局并发 50、temperature=0.5、max_tokens=2048。
- 复用历史 `vl_rewardbench_prompt_v2` 的节点推理、缓存、递归聚合和官方评测函数，不修改历史模块全局变量。技术失败单独补跑，累计最多 8 次；语义 None 不重试。
- 输出统一放入 `output/fw6_27b/recursive/{initial,final}`。日志、节点预测、修复记录、配对报告均保留。以落盘报告为完成依据。
- 主比较：同一 Rubric 下两种推理方案；次比较：递归下 init 与 Final。统一报告官方 OverallAcc、MacroAcc、Strict ACC、Coverage 和配对纠正/损害。
- 本对照同时改变推理粒度、Prompt 和聚合方式，不是纯聚合器消融，也不是等计算预算对照。两组逻辑节点请求分别为 82302、78561。

启动与断点续跑（使用 critiq 环境）：

```powershell
python -u -m experiments.evolving_structured_rubrics.framework_v6_recursive --config .local/framework_v6/config_27b.json --source-dir output/fw6_27b --attempt-limit 8
```

加 `--check` 可离线验证 Rubric、样本、呈现顺序和请求配置，不发送推理请求。

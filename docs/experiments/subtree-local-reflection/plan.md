# 子树逐例反思与局部竞争：实现计划

日期：2026-09-19。状态：**最小实现和离线验证已完成；真实 smoke 与正式实验尚未运行。**

框架：[framework.pptx](framework.pptx)。分支：`codex/subtree-local-reflection`。本计划只覆盖当前新方案；旧分层反馈实验的结果不是本方案的结果。

## 1. 目标与边界

检验：固定 Worker 与 Arbiter，通过单案例独立反思，将有限能力 Worker 的局部失败转化为更容易执行的孩子准则，是否能改善子树判断，并进一步改善最终系统偏好判断。

初始化保留“错误签名 → 语义聚类 → 孩子组”。演化改为“局部案例 → 独立 reflection → 按 root 收集 critiques → 一次 split → 局部竞争”。删除系统级反思、反馈路由、后续签名重提取与聚类；不引入逐节点 refine。

两项待验证判断：

| 判断 | 必须观察的证据 | 不能据此声称 |
| --- | --- | --- |
| 局部修改有效 | 每根 Current/Candidate 的正确数、覆盖数、Covered ACC 与变化案例；Dev 上的对应结果 | Discovery 上升不等于泛化；覆盖下降不等于判断能力提高 |
| 演化对整个系统有用 | 同一 S0 与 Final 在 Dev/VLRB 的配对系统结果 | 局部竞争获胜不保证 Arbiter 的最终 ACC 上升 |

本轮先做可诊断的完整主实验。没有同初始化、同模型的对照，不能将变化单独归因于逐例 reflection 或局部竞争。

## 2. 固定协议

- 五个 root 的 ID、名称、完整职责描述固定；仅一层孩子，每根 2–5 条。允许保留、改写、合并、新增，输出完整孩子组。
- Worker 对 **root + 完整子树**一次调用，输出 `analysis_a`、`analysis_b`、`thought`、`answer`。不是逐孩子调用，也不是递归多数投票。
- 五根报告交给原 Global Arbiter；保留其完整公开输出分析字段，用于系统结果诊断，不进入本次局部反思输入。
- Manager 按最新选择使用硅基流动 `Qwen/Qwen3.5-27B`，关闭思考（`enable_thinking=false`）；其余参数与实验流程保持不变。不得把历史 27B/35B 产物混为本实验调用。
- Worker/Arbiter 沿用 `10.102.137.255` 的既有 API 配置和 8B 模型；全局并发 50，五根不能各开 50。temperature、max_tokens、提示词保持原基线设置（当前计划基准 0.5、2048）。
- Manager 错误签名提取和逐案例反思并发各为 6（两阶段不同时运行），其他阶段默认并发 4；timeout 300s、总尝试上限 4。线程池与请求信号量同时按阶段设置，沿用共享 429 冷却与 SDK `max_retries=0`。Worker 并发仍为 50。
- 不依赖未经验证的 `thinking_budget` 硬限制；保留原管理端长度参数方案，保存实际请求和 usage。不能把服务端默认限制称为无限输出。
- 初始化不计轮次；S0 后最多五轮演化。
- 同一轮冻结完整 Current Rubric；五根独立提出并评估候选，轮末只组装局部获胜的孩子组。下一轮才使用组装后的完整 Rubric 背景。

## 3. 数据与初始化

沿用既有冻结 Discovery100、Dev150 和 VLRB1247 的 ID 清单与顺序，先隔离算法变化，不新增数据选择器。

| 数据 | 用途 |
| --- | --- |
| Discovery100 | 初始化、全部局部案例反思、全量局部竞争；K=1，沿用原始 A/B 顺序 |
| Dev150 | Final 冻结后比较 S0/Final；不反馈、不选轮次 |
| VLRB1247 | 外部回归评测，沿用既有 K=3 和 A/B 排序、聚合协议；不是全新盲测 |

初始化步骤：

1. 建立五个裸 root 的 R0，得到各根在 Discovery 的报告。
2. 每根从与全局 Gold 不一致的报告中提取局部失败签名；None 也进入核验。Manager 可以判定签名不适用，不把分歧直接认定为局部错误。
3. 复用原初始化的语义聚类和孩子生成规则：至少四个适用签名、至少两个有支持的簇；生成 2–5 条孩子。材料不足时沿用原补充案例逻辑；仍不足则报告初始化未完成，不伪造 S0。
4. 五根全部生成后形成 S0。S0 不需要先战胜 R0；不重复抽样挑选有利初始化。
5. 保存 S0 及其 Discovery 五根完整报告。默认重新建立本实验基线；若后续明确复用已生成 S0，只能作为注明来源的固定初始化输入，不能复用不匹配的评测结果。

## 4. 每轮实际数据流

### 4.1 选择该根的全部待反思案例

对每个根 Rᵢ，检查 Discovery 全部样本：有效 A/B 判断与 Gold 不一致，或有效输出 None，就纳入待反思集合。**不设置条数上限**，按冻结样本顺序去重。

全局 Gold 是总体偏好，因此集合应命名为 `review_cases`，不是已经证明的局部错误。技术失败先重试修复；不能当作 None，也不能进反思。

### 4.2 每个 `(root_id, sample_id)` 独立反思

每次调用包含：

- 当前完整 Rubric：五根及全部孩子的 ID、名称和完整描述、父子关系；排除 criterion examples、lineage、旧候选和历史长对话。
- 当前目标 root ID 及其完整子树，明确“全局准则是背景，修改对象仅为这根的孩子”。复用同一投影，不另造简略描述。
- 这一条案例的原图、问题、A/B 原文、Gold。
- 该根 Worker 的完整 `analysis_a`、`analysis_b`、`thought`、`answer`。

不输入其他四根报告、Arbiter 输出或其他案例。每例使用新的请求上下文；共享背景不等于复用跨案例对话，也不假设 API 自动提供免费缓存。

简单输出：

```json
{"analysis": "该案例中的局部判断与依据，以及是否存在可改进之处", "critique": "少量可泛化的孩子准则修改建议；无合理修改建议时为空字符串"}
```

要求检查局部职责、图像/候选事实和执行困难；允许指出“当前局部判断或 None 合理”。Worker 漏执行、误解规则也能产生准则改进建议，不因归类为执行错误就排除。禁止把固定 Prefer A/Penalize B、样本 Gold 或具体答案写入准则。不要要求逐字引用匹配，不增加证据账本和复杂 action 分支。

### 4.3 按根收集，进行一次 Split

程序汇集该根所有非空 critiques，携带 sample_id 用于追溯；没有额外模型汇总调用。

Split 输入：同一轮完整 Current Rubric、目标子树、全部 critiques。默认不再附原图、A/B、Worker 长报告或逐例 analysis，避免把短反馈重新膨胀为案例包。analysis 保存在产物中供审查。

Manager 在一次调用内处理建议的重复、冲突和适用边界，保留有效能力，输出：

```json
{"children": [{"name": "criterion_name", "description": "完整准则描述"}], "change_summary": "改了什么，以及保留了什么"}
```

不要求“一条 critique 对应一个孩子”；不做语义聚类。无待反思案例或全部 critique 为空时直接保留该根。整组描述没有变化时跳过重复 Worker 调用。若全部 critiques 超过模型上下文，不静默截断或自动分层总结：明确报告长度问题，保留输入供后续调整方案。

### 4.4 独立局部竞争

每个有变化的候选只重跑目标 root 的完整子树，在 **全量 Discovery100** 上比较；不是仅在反思案例上评分，也不运行 Arbiter 决定是否接受。

设 N 为总样本数，C 为有效 A/B 判断且与 Gold 一致的数量，V 为有效 A/B 判断数量：

- `covered_acc = C / V`；V=0 时为 null，不能获胜。
- `coverage = V / N`；`strict_acc = C / N` 同时记录。
- Current 与 Candidate 均有覆盖时，仅当 `C_candidate * V_current > C_current * V_candidate` 才接受，避免浮点误差；相等或更低拒绝。
- Current 覆盖为零时，计划约定 Candidate 必须 V>0 且 C>0 才能建立首个有效版本；单独记录该零覆盖分支，不能宣称常规 ACC 差值。
- 比较前必须没有技术失败；解析失败不是弃权，不能靠丢失困难案例获益。

按已讨论方案，**不新增最低覆盖门槛、Strict ACC 门槛或系统级否决**。必须记录正确→错误、错误→正确、A/B→None、None→A/B，以及共同覆盖样本的配对变化，揭示只靠弃权提高 Covered ACC 的情况。

### 4.5 组装与下一轮

轮末把接受根的孩子替换到 Current，其余根原样保留；复用该版本已得到的逐根报告，不无意义地重跑不变根。完整系统评测使用这五根报告重算 Arbiter，保存公开分析输出，仅作监控。

之后用新的完整 Current Rubric 进入下一轮。全部无可用建议时可以提前结束；有候选但全部拒绝不自动等同于算法收敛，默认继续至最多五轮。失败中断可以恢复，不跳过失败 root 伪装成完成。

## 5. 最小代码落点

| 位置 | 实现任务 |
| --- | --- |
| 新 `experiments/evolving_structured_rubrics/subtree_local_reflection.py` | 小型 runner：初始化接入、全量逐根选例、逐例调用、一次 Split、局部竞争、组装、外部评测与报告 |
| 新 `subtree_local_reflection_manager.py` | 两个新阶段 `case_reflection`、`subtree_split` 的提示词/简洁校验；初始化沿用旧 signature/cluster/children |
| `framework_v6.py` 的纯函数 | 复用 `project_rubric`、`replace_groups`、`load_rows`、初始化规则；不要直接调用旧 `reflect/run` |
| `manager_runtime.py` | 复用请求、日志、usage、缓存、429 冷却；必要时仅增加可选 prompts/validator 注入，当前阶段行为不变，禁止修改模块全局字典切换协议 |
| `aligned_system_runtime.py` | 先查现有子树调用和缓存能力；如无入口，仅抽出最小 `evaluate_subtrees` 能力供新 runner 使用，现有系统评测继续沿用原语义 |
| 新配置示例与 tests | 数据/模型设置、最多五轮；离线测试指标、输入隔离、局部接受、恢复路径 |

特别避免两个旧行为：`case_payload(root=...)` 仍附带 Arbiter 内容，必须新增局部 payload；`joint_decision` 按系统 Strict ACC 整批接受，必须替换成独立 root 决策。

先确认局部 runtime 返回与完整系统一致的报告格式及 A/B 映射，不能另写第二套 Worker 提示词。参数替换不改变算法；不复制整个旧 runner，不增加通用调度平台。

## 6. 输出、恢复和可观测性

一个正式 run 只用一个目录，例如 `output/subtree_local_reflection/<run_name>/`：

```text
config.json, state.json
r0/, init/                         # 初始化签名、聚类、孩子、报告与S0
e01/...e05/
  r01/...r05/
    selected_cases.json
    reflections/s0000.json       # 按数据行编号；内容保留 sample_id、请求、响应、usage、尝试
    critiques.json, split.json
    candidate_reports.json, comparison.json
  rubric.json, summary.json, system.json
dev/, vlrb/, report.json
```

沿用缓存成功项跳过、失败项恢复和原子写入；缓存核对实际 Rubric/输入/模型参数，不复用过期批次。epoch/root/sample/stage/attempt、开始/完成/耗时/冷却均写日志。轮末完成后再更新 state，拒绝候选仍保留供诊断。

请求次数、输入/输出/reasoning tokens、重试和失败缺失 usage 分别统计；累计请求延迟不能当作实际 wall time。API key 仅从 `.env`/环境读取，不进入日志或文档。

## 7. 实现顺序与完成条件

| 步骤 | 工作 | 完成条件 |
| --- | --- | --- |
| M1 局部执行与指标 | 子树报告读取/执行、覆盖指标、局部比较 | 无 Arbiter 的子树执行可用；None/零覆盖/技术失败/同分测试通过 |
| M2 反思与 Split | 全局 Rubric 投影、独立案例 payload、两个管理阶段 | 不含 examples/其他根报告/Arbiter；无反馈保留；完整孩子组有效且根不变 |
| M3 演化与恢复 | 轮初冻结、五根独立竞争、混合组装、最多五轮 | 假后端验证部分根接受/部分拒绝、重启不重复成功调用、最终版本正确 |
| M4 小规模 smoke | 独立目录，少量样本走通初始化或测试fixture后的完整一轮 | 真实请求格式与视觉输入正确；所有入选局部案例均反思；不将 smoke ACC 当结论 |
| M5 正式主实验 | 新建正式 S0，最多五轮，冻结 Final 后 Dev/VLRB | 全部阶段完成且无未解决技术失败；报告完整 |

测试必须在 `conda activate critiq` 环境，使用 fake backend 做确定性验证；真实 smoke 是单独模型调用。重点测试：A/B 原文对齐、局部 None 的合法性、C/V 比较、候选覆盖为零、全局与局部偏好区分、root 不变、轮初快照一致性、混合接受与缓存恢复。改共享 runtime 时跑相关回归测试，不能只检查文件能导入。

已实现 CLI：`configure / check / smoke / run / dev / vlrb / report / all`。`all` 顺序执行 smoke、run、dev、vlrb、report；失败即停止，恢复时复用成功缓存。smoke 使用前十条 Discovery 和固定测试孩子走一轮，仅检查接线，不替代正式签名聚类初始化。

## 8. 成本、结果表和后续决策

Discovery100 下，每轮每根最多 100 条待核验案例，因此 Manager 最多 **500 次逐例 reflection + 5 次 Split**（不含技术重试）；五轮最多 2525 次，初始化另计。全量反思无上限指不对实际错误集合截断，不代表 API/数据规模无限。

候选 Worker 每轮最多 500 次子树调用；组装系统监控最多另加 100 次 Arbiter。S0/Final 的外部完整系统每样本每次重复需要 5 根 + 1 Arbiter；VLRB K=3 的两个版本理论上为 `1247×3×6×2=44892` 次请求，缓存命中可减少实际调用。拒绝版本不跑 VLRB。

不凭历史不同模型的平均耗时承诺完成时间。smoke 后按实测输入/输出 token、p50/p95 延迟、429 重试和并发估算；全局 Rubric 每例重复发送是主要成本之一。汇集 critiques 长度也要监控，不能只看逐例请求短。

主表：S0/Final 的 Discovery、Dev、VLRB 系统 Strict ACC、Overall ACC、Macro ACC、Coverage，口径复用现有评测。局部表：每轮每根 C/V/N、Covered ACC、Coverage、Strict ACC、接受与否、critique 数、孩子数。附至少几条“局部报告 → critique → 准则变化 → 新报告”的真实证据链。

如果 Discovery Covered ACC 上升但 Coverage 明显下降、Dev 或系统表现下降，如实判为“当前局部目标未转化为系统增益”，不能宣称方案成功。若结果值得继续，再做同一 S0/模型下的“无逐例反思直接 Split”或“旧系统反馈”对照，以及多次独立重复；这些不阻塞首轮最小实现，也不同时改变多个机制来解释因果。

当前状态：M1–M3 最小实现完成，M4–M5 等待真实运行。新增与 framework-v6 相关离线测试共 29 项通过。全仓测试曾运行 541 项，其中 6 项未通过：四项缺少历史数据/产物，两项 phase17 历史协议配置不一致。新增流程的数据检查已通过，数据来自原 CritiQ 目录。

## 9. 完整运行命令

在 PowerShell 中执行以下整段。读取原项目 `.env` 中的 `GUIJI_API_KEY`，不会将密钥写入配置。

```powershell
Set-Location 'D:\3-Work\02-DD-LLM\CritiQ-framework-v6'
conda activate critiq
$Py = 'C:\Users\wenqx\miniconda3\envs\critiq\python.exe'
$Module = 'experiments.evolving_structured_rubrics.subtree_local_reflection'
$Config = '.local\subtree_local_reflection\config.json'
$Out = 'output\subtree_local_reflection\main'

& $Py -u -m $Module configure --config $Config --data-root 'D:\3-Work\02-DD-LLM\CritiQ' --env-file 'D:\3-Work\02-DD-LLM\CritiQ\.env' --worker-url 'http://10.102.137.255:8000/v1'
if ($LASTEXITCODE -ne 0) { throw '配置生成失败' }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
& $Py -u -m $Module all --config $Config --output-dir $Out --attempt-limit 4 2>&1 | Tee-Object -FilePath "$Out\run.log" -Append
if ($LASTEXITCODE -ne 0) { throw '实验中断，请检查 run.log；成功调用已缓存' }
```

中断后保留同一输出目录，重新执行 `all`。若某项已耗尽四次总尝试，将 `--attempt-limit` 增至 8 以追加尝试；不要删除成功缓存。若修改模型或协议，另建运行目录，避免混合比较。

主要结果：`report.json`（系统与外部评测、各轮局部指标及成本）、`final.json`（最终 Rubric）、`mechanism_examples.json`（修复/损害样例及修改路径）。`eXX/rYY/comparison.json` 保留覆盖变化与逐例前后判断，反思原文位于同目录 `reflections/`。


## Controlled experiment: five random correct cases (2026-09-22)

Pre-run protocol: compare against Discovery100 / Qwen3.5-27B non-thinking / local Strict ACC acceptance. Reuse the same Init Rubric and initial Discovery, Dev and VLRB reports. Start evolution at epoch 1, not from the previous final Rubric.

For each root and epoch, sample up to five cases without replacement from current local predictions matching global gold. Use seed 42 combined with epoch and root ID for reproducible random sampling. If fewer than five qualify, use all. Save the selected cases to `eXX/rXX/preservation_cases.json`.

Only the subtree Split request gains these cases: original image, question, A/B, gold and complete local report. Keep them in a separate `preservation_cases` field from all error critiques, with instructions to preserve evidence-supported, in-scope capabilities. Matching global gold does not prove the local reasoning is correct. Correct cases do not receive extra reflection calls. Roots without actionable critiques remain unchanged.

Keep five evolution rounds, existing error reflection, Worker settings and Strict ACC acceptance unchanged. Add no separate preservation gate. Config: `.local/subtree_local_reflection/config_strict_preserve5.json`; output: `output/subtree_local_reflection/discovery100_27b_strict_preserve5`. Compare final Discovery, Dev, VLRB metrics, root-level regressions and Split input usage against the Strict baseline. Reused initialization costs are not new run costs.

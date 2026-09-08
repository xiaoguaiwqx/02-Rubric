# 27B Manager：局部错误生成、联合系统接受的 Split-only 实验

日期：2026-09-08。状态：**最小实现与本地检查完成；67项相关离线测试通过，尚未真实smoke或正式运行**。沿用现有生成与评估组件，仅新增必要的实验调度；不改历史 Phase17/Phase19 默认行为。

## 1. 问题与验证目标

此前局部递归竞争拒绝了多数 Split，但最后一轮完整候选 Rubric 在外部评测上有收益。本次检验：保留 root 自身错误驱动的候选生成，取消逐 root 性能筛选，改为整批 Unified-Subtree + Global-Arbiter Strict ACC 竞争，能否接受有用的结构。

**首次实验仅做 Split，不调度 Refine，不做准则精简。** 不引入强孩子锁定、逐候选系统评估、最佳单候选回退、组合搜索或逐样本 Reflection 新流水线。模型生成、JSON解析、提示词主体、子树推理和Arbiter算法继续复用。

| 待验证判断 | 最小证据 | 不能据此声称 |
|---|---|---|
| 联合接受能保留局部竞争可能丢失的结构 | 联合候选、全样本纠正/改错、提交记录；初始/最终系统比较 | 每个被接受root都独立有益 |
| 接受的结构具有迁移价值 | 冻结最终Rubric后，独立评测Initial与Final并做样本配对 | Discovery上升即泛化上升，或纯粹由接受公式造成收益 |

这是推理方式与提交粒度一起改变的首轮可行性实验，不是单一指标消融。历史“含Refine、8B Worker、局部接受”只作背景，不作严格因果对照。若试运行值得继续，再用同模型的Split-only局部接受组隔离机制；首轮不强制增加该实验。

## 2. 设置与待确认项

| 项目 | 首轮计划 |
|---|---|
| 初始Rubric | 原始五个roots、无孩子；不从Epoch 5完整候选开始 |
| Manager | Qwen3.5-27B，thinking关闭；ErrorSignature并发30，聚类/孩子生成沿用串行 |
| Manager输出与技术设置 | 沿用当前本地配置：16384输出上限、900秒超时、解析失败最多重试3次；不恢复无效的thinking_token_budget |
| 推理模型（用户已确认） | Pairwise Worker、Unified-Subtree Worker、Global Arbiter全部沿用Qwen3-VL-8B-Instruct；只有Manager用27B，先验证8B效果 |
| 系统执行 | 复用8B Clean S5-v2（A/B-preferred、原生None）；Pairwise、Unified-Subtree与Arbiter仅使用同一个8B服务端点，单端点上限及全局总并发均为100 |
| Worker生成设置 | temperature=0.5、max_tokens=2048；复用已有解析/技术重试，不自动提高至4096 |
| 生成与接受数据 | 沿用Discovery100；Dev150只观察，不进Manager、不参与接受 |
| 外部评测 | 冻结最终结构后评测heldout-500及VL-RewardBench 1247；保持现有重叠标注 |
| 演化重复次数 | 建议沿用Phase19的K=1，固定同一A/B顺序和已保存基线；最终VLRB用既有K=3平衡顺序 |
| 轮数 | 最多5个epoch；每个待处理root每轮最多一个新候选，不在同一轮反复采样至接受 |
| Split触发 | 沿用Phase17有效运行设置：covered ACC < 0.75、coverage > 0.80、support ≥ 15、wrong ≥ 15 |
| 候选约束 | 沿用当前本地实验：cluster最小2、最多5个孩子、原有至少两个cluster等合法性检查；使用现有短ID映射 |

模型组合已确认：27B只负责Manager，三类推理角色均为8B，演化与最终评测保持一致。K=1为沿用Phase19的首轮计划值，其他值也应在实现时与实际配置核对。Manager服务与8B推理服务分开配置，不能把此前27B Worker端点直接当成8B；真实地址仅放本地配置。首次实现交付准确运行命令，本计划不列不存在的CLI命令。

用户已确认当前部署：一个服务运行27B Manager，另一个服务运行8B推理模型；具体地址写入本地配置。8B池只登记一个真实端点，总并发100，不将Manager服务加入推理池，也不将同一8B地址登记两次模拟双服务。Manager并发保持原设置；签名阶段与系统评估阶段顺序执行，不为本实验新增跨角色调度器。运行前smoke核对实际模型，本文记录的是用户提供的部署信息。

## 3. 单轮算法：同一基线、一次联合判断

### 初始化

在Discovery上对五个父root做Pairwise预测，按现有定义收集明确判断错误的样本；合法None不是明确错误，技术失败单独处理。生成初始系统基线：每棵单节点子树一份Unified报告，再交给Arbiter。

Split期间父root描述不变，且没有Refine，因此父节点预测和错误签名在输入、模型与请求设置不变时复用。错误集合**不与系统错误或Unified报告错误取交集**。

### 每轮候选与评估

1. 读取上轮已提交Rubric及对应系统报告，冻结本轮共同基线。
2. 所有未完成且满足原触发条件的初始roots，独立执行ErrorSignature → Cluster → Child。只做提案合法性检查，不调用局部递归性能竞争。
3. 本轮合法候选一起加入候选Rubric；未触发或提案无效的root保留旧版本。提案无效不等于性能拒绝；传输中断或不完整系统预测不得伪装为无效候选跳过。
4. 在**全部Discovery100**上重算变化root的Unified报告，复用未变root的原报告，调用Arbiter得到候选最终判断。gold只用于离线评价和Manager反馈，不进入Worker/Arbiter推理输入。
5. 只比较这一份联合候选与共同基线，不预筛各root收益。

令$F_0$为本轮当前系统、$F_1$为联合候选系统，$y_i$为人工偏好：

$$
G=\sum_{i=1}^{|D|}\left(\mathbf{1}[F_1(x_i)=y_i]-\mathbf{1}[F_0(x_i)=y_i]\right).
$$

$$
G>0\Rightarrow\text{整批提交};\qquad G\leq0\Rightarrow\text{整批拒绝}.
$$

分母固定，与Strict ACC严格上升等价；平局/合法弃权计不正确。先解决技术失败再比较，不将不完整调用当成模型错误。

### 提交、重试与终止

- 接受：只提交本轮合法且实际参与评估的Split，记录共同的batch结果；不能将batch收益写成各root独立收益。候选系统报告直接成为新基线，不重新采样。
- 拒绝：保持已提交Rubric和基线不变；候选保留在本轮产物中，下一轮生成新候选。
- 已接受root完成Split后不再拆分或替换；其他root可继续尝试。只调度原始roots，不递归拆孩子。
- 因此若第一轮五个root均合法并整体接受，**Split-only可以第一轮结束**，不是必须再演化到第五轮。再次替换已接受子树是另一种结构搜索，首轮不加入。
- 同一attempt恢复时复用缓存，不重新聚类；被拒后进入新的epoch/attempt才允许重新聚类。没有强孩子锁定，下一轮可重新分组，但复用输入未变的错误签名。
- 五轮用尽或全部root完成/无可调度root时结束。零接受也是有效负结果；绝不自动改用“最后完整候选”作为正式最终Rubric。
- 若候选有跨root命名冲突，沿用现有结构合法性检查，先标明冲突候选；不静默改名或改变评估后结构。

## 4. 联合拒绝反馈：先做最小版本

联合失败不证明每棵子树有害。第一版复用现有`prior_failures` / `retry_feedback`输入通道，构造简短反馈，不新增Reflection模型、长历史拼接或反事实归因搜索。

每个参与root获得：本轮系统基线/候选Strict、纠正/改错/净收益、本root候选摘要，以及固定上限的代表样本。建议最多8条，优先“系统对→错且候选子树错”，其次“系统错→错且子树错”，并保留少量成功边界。按稳定样本顺序取例，不依据未来测试集。

每条例子明确区分系统前后答案与该root前后答案，只提供本root相关报告及必要的短Arbiter解释；反馈写明“联合结果，非该root因果贡献”。若本root正确而系统错误，不要求它迎合错误判断。新一轮生成仍使用该root**完整原始错误签名集合**，这些例子只补充建议，不变成新的allowlist。

反馈只描述已观察结果，由原聚类/孩子Manager据此调整。若现有反馈schema绑定局部竞争字段，只在新实验入口做显式小型适配或添加可选反馈参数；不能把系统指标冒充局部ACC，不修改历史默认解释。候选生成提示词主体保持不变，只增加本实验需要的反馈说明。

## 5. 最小实现范围（代码已检查）

仅靠改配置不能完成本实验：现有Phase19调度同时执行逐候选评估、局部锁定诊断、联合回退和Refine。建议**一个小型Split-only实验入口**，不复制Phase19整文件，也不新增实验框架。

| 现有组件 | 复用方式 | 不沿用的路径 |
|---|---|---|
| `split_evolution._prepare`、现有Specialize Manager | root错误触发、签名、聚类、孩子与候选；allowlist不传 | `split._evaluate`局部性能接受与强孩子锁定 |
| `aligned_system_runtime.evaluate` / `paired` | Clean S5报告、变化root复用、全样本配对；传入显式设置 | 不重写Worker、Arbiter、JSON解析、K次聚合 |
| `unified_subtree_arbiter_evolution` | 参考epoch快照及报告组织；仅复用确实适用的小函数 | `_evaluate_candidates`、singleton回退、Refine初始化/调度 |
| 现有数据/指标/外部评测函数 | 读取相同划分，Initial/Final同协议评测 | 不为独立输出目录伪造Phase10 Rubric或依赖其文件布局 |

新入口负责候选集合、合并、一次联合评估、提交/拒绝、简短反馈和日志。配置与输出独立，建议本地配置`.local/joint_split_27b/config.json`、短输出目录`output/js27b`，避免Windows长路径复发。

优先通过现有显式参数传协议；确有阻碍时只补默认值不变的局部参数，并加回归测试。**不修改其他模块全局变量选择协议，不改变Phase17/19默认算法。** 现有aligned runtime含双端点限定，实现时需用显式配置支持本实验的单端点，保留历史默认行为；只调整端点选择/调度并测试，不改推理或聚合逻辑，不伪造第二端点。

初版不需要新的通用缓存、身份绑定、迁移层、部署脚本、Refine模块或统计框架。历史模块函数若要求局部评估产物，不为“复用”伪造这些数据；宁可在小入口直接记录真实batch结果。

## 6. 验证顺序与最小证据

| 阶段 | 必做工作 | 通过标准/输出 |
|---|---|---|
| M0 离线测试 | 假Manager/假Worker；联合正、零、负收益；部分无效；缓存恢复 | 只有正收益整批提交；绝不调用Refine/局部性能竞争；失败反馈不冒充归因 |
| M1 小规模smoke | 少量固定Discovery样本，验证生成→合并→Unified→Arbiter→反馈 | 不将smoke结果用于正式接受；核对真实模型、thinking、token上限 |
| M2 正式演化 | 保存Initial；最多5轮Split-only | 每轮batch、root状态、纠正/改错、候选/提交结构和请求统计完整 |
| M3 冻结后评测 | Initial与Final同模型同协议，在heldout/VLRB配对；Dev只作观察 | Strict主指标，Overall/Macro/覆盖/类别与McNemar辅助；不挑最佳epoch |
| 可选后续 | 同模型Split-only局部接受对照、多seed、独立接受集 | 首轮之外，不以此阻塞最小实现 |

若Initial与Final完全一致，直接复用同一评测结果，不靠重新采样制造差异。历史缓存只有数据、Rubric、提示词、模型、顺序和参数确实匹配时才复用；否则重算，不因模型名相同就复用。

日志至少打印：epoch、参与/无效roots、初始/候选Strict、corrected/harmed/net、`batch accepted/rejected`、每个root的`committed / rejected_with_batch / proposal_invalid / not_eligible`。下一轮打印是否复用签名、是否新聚类。另记录技术重试、输入/输出tokens、API次数和墙钟耗时，不新增独立状态平台。

## 7. 预算与解释边界

K=1时Initial系统基线约$100\times(5+1)=600$次逻辑调用，初始root Pairwise约$100\times5=500$次。每轮有$m$棵变化子树，系统候选约需$100\times(m+1)$次；五轮均变化五棵树时，共约4,100次Pairwise/系统调用，**不含Manager生成和技术重试**。root错误不变时无需每轮重推500次。

VLRB每份Rubric使用$1247\times3\times6=22446$次逻辑调用，Initial与Final均新算时44,892次，可能比演化更昂贵。heldout重复次数沿用既有评测协议并在运行配置中记录。GPU小时与周转时间由smoke实测估算，不套用节点递归吞吐。

Discovery用于生成也用于接受，K=1且净增加一条即可提交，仍可能选择过拟合或采样偶然收益；缓存固定基线只保证比较可追溯，不消除噪声。禁止性能拒绝后仅重抽Arbiter直到通过。外部评测不反馈给Manager。

成功分两层：能够生成并联合接受是**流程可行性**；冻结Final在外部数据上相对Initial提升才是**迁移证据**。若Discovery提升、外部下降，报告负结果，不增加局部回退、Refine或临时阈值修饰首轮实验。

## 8. 实现交付清单

- [x] 用户确认Pairwise/Unified/Arbiter全部使用Qwen3-VL-8B-Instruct，Manager使用27B。
- [x] 用户确认单个8B服务承担全部推理，总并发100；27B服务仅用于Manager。
- [x] 本地配置为单8B端点100并发、27B Manager，演化K=1；实际服务模型由smoke检查。
- [x] 一个Split-only入口及确定性测试；历史默认行为不变。
- [x] 本地配置生成方式及完整运行命令见下节。
- [x] 输出路径与恢复说明；技术失败与性能拒绝分开。
- [ ] 结果完成后更新本计划对应状态及主文档，不新建PLAN/TRACKER副本。

相关背景：[主文档18.2及19.8节](../../Evolving%20Structured%20Rubrics%20Implementation%20Plan.md)、[此前27B实验与运行协议](../phase17-manager-qwen35-27b/plan.md)。

## 9. 实际入口与运行命令

实现位于`experiments/evolving_structured_rubrics/joint_split_evolution.py`。共享代码仅两处可选参数：runtime的`endpoint_ids`支持单端点，Split `_prepare`的`prior_failures_override`接收联合反馈；默认均不变。

父节点证据固定在Initial上：候选生成仍使用Initial Rubric对应的真实Pairwise反馈，Manager全局记忆则来自当前已提交Rubric。对尚未Split且文本未变的root，将合法的加法patch合并到当前结构；不伪造孩子预测或完整树反馈。最新一次联合反馈最多8个例子，下一轮重新聚类；同轮恢复保留已完成缓存和提案无效记录。

`check`已执行，只读取本地数据并保存运行配置，不访问模型。真实`smoke`检查服务模型并对5条Discovery样本运行父节点Pairwise和完整系统推理；候选生成/提交路径由离线假后端测试覆盖，smoke不人为放宽Split触发阈值。正式运行前需用户确认8B服务使用既定非thinking模型配置。

### 本机直接运行

本地配置`.local/joint_split_27b/config.json`已生成，Manager和Worker地址按用户部署分别写入，不需要重新生成。以下命令由用户运行；任一阶段失败立即停止：

```powershell
conda activate critiq
$Config = ".local/joint_split_27b/config.json"
$Output = "output/js27b_names"
$ErrorActionPreference = "Stop"

function Run-JointSplitStage([string]$Stage, [int]$AttemptLimit = 4) {
    python -m experiments.evolving_structured_rubrics.joint_split_evolution $Stage --config $Config --output-dir $Output --attempt-limit $AttemptLimit
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Stage" }
}

Run-JointSplitStage "check"
Run-JointSplitStage "smoke"
Run-JointSplitStage "run"
Run-JointSplitStage "report"
Run-JointSplitStage "dev"
Run-JointSplitStage "heldout"
Run-JointSplitStage "vlrb"
```

Dev及heldout沿用aligned评测K=1，VLRB沿用K=3及既有确定性平衡顺序。Dev只在演化结束后作Initial/Final观察，本版不新增每轮Dev模型调用。外部结果直接比较本次Initial和Final，不读取历史Phase10 Rubric。heldout保留“存在已知来源重叠”的探索性标注，不声称其为干净独立测试。

### 技术失败后继续

不要删除输出。传输中断时重跑失败阶段；如果系统调用四次总尝试后仍有解析失败，可将**同一阶段**的总技术尝试上限提升到11，例如：

```powershell
Run-JointSplitStage "run" 11
# 若失败发生在外部评测，改成对应阶段：
# Run-JointSplitStage "vlrb" 11
```

11表示累计上限，不是每次再加11次。完整预测直接复用，仅不完整调用继续；达到上限仍失败则停止检查，不自动4096。此参数只控制Unified/Arbiter，Manager仍使用原配置，父节点Pairwise若残留技术失败会停止，需检查其错误缓存，不会拿错误集合继续演化。科学配置改变须另选输出目录，不能覆盖已存在实验。

主要产物：`state.json`为已提交状态；`eXX/summary.json`记录联合决定；`eXX/committed.json`保留结构；`final.json`为最终Rubric；`report.json`为Discovery汇总；`dev/report.json`、`heldout/report.json`、`vlrb/report.json`为配对结果。系统成本按本目录唯一缓存调用汇总；Pairwise与Manager成本保留在其原有遥测/签名产物中，不把不同来源重复相加。

### 在其他机器重新生成配置（仅首次）

复用此前已验证的Phase17本地配置，不手写另一套Manager参数。目标已存在时生成命令拒绝覆盖：

```powershell
python -m experiments.evolving_structured_rubrics.joint_split_evolution configure `
  --source-config ".local/phase17_manager_qwen35_27b/config.json" `
  --config ".local/joint_split_27b/config.json" `
  --manager-url "http://<manager-host>:8000/v1" `
  --worker-url "http://<worker-host>:8000/v1"
```

保留源配置的checkpoint路径及请求超时；更换机器时核对这些本地字段。继承配置内的历史Refine设置不会被本入口读取或执行。

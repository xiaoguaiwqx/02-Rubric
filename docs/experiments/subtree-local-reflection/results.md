# Subtree local reflection：首轮完整实验结果

分析日期：2026-09-20。权威产物：`output/subtree_local_reflection/main/`。Manager 为 Qwen/Qwen3.5-27B，非思考；五轮，局部 Covered ACC 严格提高才接受；未加入失败 memory。

## 总体结果

| 数据集 | Init 正确数 | Final 正确数 | Strict ACC 变化 | 纠正/损害 |
|---|---:|---:|---|---|
| Discovery100 | 62 | 67 | 62% → 67% | 11 / 6 |
| Dev150 | 105 | 106 | 70% → 70.67% | 15 / 14 |
| VLRB1247，报告 official 区块 | 875 | 861 | 70.17% → 69.05% | 45 / 59 |

VLRB Overall ACC：70.74% → 69.38%；Macro ACC：64.08% → 63.17%；Coverage：99.20% → 99.52%。配对精确 McNemar p=0.202，方向为负但不能据此宣称显著下降。Discovery 是选择集，不能用其 p 值支持泛化主张。Dev 的配对 p=1。

VLRB 日志 69.21% 使用 runtime 的相对多数：A 比 B 多即 A；official 区块要求三次至少两票一致。因此 [A,None,None] 两者不同。历史对比应统一使用 report.official 口径，不混用终端数字。

## 演化路径

| 轮次 | 反思 root–案例数 | 接受 root | 系统 Strict ACC |
|---|---:|---|---:|
| e01 | 200 | 视觉依据、事实性 | 63% |
| e02 | 196 | 完整性、事实性 | 61% |
| e03 | 199 | 清晰性 | 63% |
| e04 | 195 | 创造性 | 67% |
| e05 | 192 | 无 | 67% |

25 次 Split 接受 6 次，拒绝 19 次。Final 等于 e04 接受版本；Init/Final 均 25 节点（5 根+20 孩子）。

## 局部泛化

| Root | Discovery Covered ACC Init→Final | Dev Covered ACC Init→Final | Dev 正确数 Init→Final |
|---|---|---|---|
| 完整性 | 68.13%→69.88% | 76.26%→72.99% | 106→100 |
| 视觉依据 | 65.56%→65.93% | 75.94%→70.80% | 101→97 |
| 事实性 | 59.78%→68.60% | 70.31%→69.05% | 90→87 |
| 创造性 | 67.02%→69.47% | 66.91%→67.57% | 91→100 |
| 清晰性 | 64.21%→67.71% | 73.79%→73.61% | 107→106 |

局部数字是与全局 Gold 的一致率，不是人工标注的各维度真值。

完整性 e02 从 62/91 变为 58/83：正确数下降但 Covered ACC 上升，按设定规则接受。说明该指标有分母变化问题。当前数据不证明所有局部提升都是这种情况：创造性、清晰性正确数确实增长。

VLRB 类别正确数：General 86→87，Hallucination 584→570，Reasoning 205→204。主要净损失来自 Hallucination，尚不能仅凭组别把责任归到事实性 root，需逐例检查局部报告和 Arbiter。

## 准则与反馈检查

Final 完整性将“有尝试回答”作为覆盖条件，两个孩子仍有重叠。创造性 narrative_coherence_and_flow 与清晰性有职责重叠。Final 事实性 e02 已明确排除格式/字数等约束违反，不能沿用 e01 存在该问题的结论。失败 memory 只讨论未实现，被拒候选的结果未进入下轮生成。

982 次逐例反思只有 9 条空 critique，973 条提出修改。此比例提示可能存在过度提出修改建议的倾向，但不是伪缺陷率；仍需核验案例。

## 成本和技术状态

最终评估 technical_failure_count 均为 0。正式 Manager 1202 次逻辑调用、1354 次尝试；输入 7,060,961 token、输出 437,580 token，reasoning=0，150 次缺 usage。错误：83 次 RateLimitError、57 次 InternalServerError、10 次 APITimeoutError、2 次 ValueError。

按用户提供单价（输入0.0006元/K、输出0.0048元/K），已知正式 Manager usage 约6.34元；不是完整账单，缺失 usage 和其他费用不能由此推出。

按 stage 重算：signature 185 次；cluster 5 次；children 5 次；case_reflection 982 次；subtree_split 25 次。总调用1202正确。报告 smoke_manager_cost.calls=135 混入 Worker 空 attempts 缓存，真实 Manager smoke 为15次（10反思+5Split）；其15次 attempts和token不受该计数问题影响。累计 latency 是并发请求耗时之和，不是实际运行时长。

## 结论与下一步

目前未证明该方案带来外部泛化收益；Discovery 改善，Dev 基本持平，VLRB 方向下降。优先离线检查59条VLRB损害与45条纠正的判断链，核验职责漂移、局部正确是否被Arbiter覆盖、边界型投票。失败 memory 与 Covered ACC 覆盖约束应分开实验，使用相同 S0；避免同时修改多个机制。下一轮完整实验前先做单根小范围验证，并保留独立验证集，VLRB继续仅作最终测试。


## 与第17章聚合机制的核对（2026-09-20）

本次为代码、提示词和保存结果的离线审计，没有新增模型调用。第17章参照为 Phase17 E4 的27节点实验，不是后来28节点的27B Manager实验。

### 实现一致性

当前 aligned_system_runtime 调用 internal_global_arbiter_k1 的 Unified-Subtree 提示词，以及 global_arbiter_ab_only 的 Clean S5 提示词。两个 system prompt 均逐字出现在第17章中。每个 root+完整子树独立生成 analysis_a、analysis_b、thought、answer；五份报告全文随图像、问题、A/B进入 Arbiter，没有先投票或只传标签。每个 replicate 内报告和 Arbiter 使用相同 A/B 顺序，最后还原原始标签。输入准则只包含名字和描述，不包含 examples。

Init/Final 各核验18,705份子树报告和3,741份Arbiter结果，全部解析成功且四字段非空。当前 additionally 保留并校验 Arbiter reason；历史解析仅要求 answer，属于技术校验差别，不是标签级聚合。

存在指标实现差异：aligned_system_runtime.metrics 使用 A/B 相对多数，A/None/None 会输出 A；第17章 K=3 要求至少两次一致，否则 None。framework_v6.external 的 official 块已按第17章规则重算。当前 Init runtime 为878正确、official875；Final runtime863、official861。本文比较采用 official，避免混用。

### 同一报告上的聚合对照

| Rubric | 子树等权多数投票 Strict | 完整报告 Arbiter Strict | 净增正确数 | Arbiter corrected / harmed |
|---|---:|---:|---:|---:|
| 第17章 Phase17 E4，27节点 | 832/1247，66.72% | 887/1247，71.13% | +55 | 74 / 19 |
| 当前 Init，25节点 | 828/1247，66.40% | 875/1247，70.17% | +47 | 73 / 26 |
| 当前 Final，25节点 | 797/1247，63.91% | 861/1247，69.05% | +64 | 91 / 27 |

当前多数投票由已保存的同一批报告离线重算：每次先对五根非None答案作等权多数（平票None），再K=3至少两次一致。它是第17章S3类型对照，不是逐节点显式递归S0。第17章S0为873/1247（70.01%），S5虽多14条但配对p=0.215，未证明稳定优于递归。

第17章值来自计划文档，当前值直接重算保存产物。不同Rubric之间不是纯聚合消融，不能将差值都归因于聚合器。

### 解释与证据边界

1. Arbiter没有失效：Final相对同报告投票多判对64条（+5.13pp），甚至比Init的净收益更大。演化后投票少31条，Arbiter少14条，说明聚合仍有补偿作用。
2. 优化目标不匹配：compare只要求Discovery100上局部Covered ACC提升，没有系统接受门槛。完整性62/91→58/83，正确数减少4但分母减少8，仍被接受。它在VLRB上的正确数712→534，覆盖1182→1084；这是最突出局部退化，但不能把系统全部损失直接归因于它。
3. 全局Gold并非每根的维度真值。筛选用local answer != gold，包含全部None；反思提示词虽明确禁止强行追随Gold，接受指标仍用全局Gold衡量局部答案。保留维度分工与追求每根预测全局偏好之间存在张力。
4. 100条Discovery同时用于提建议和反复竞争，K=1；外部VLRB使用K=3换序。Discovery系统62→67，Dev105→106，VLRB875→861，未证明泛化收益。VLRB corrected45/harmed59，单次配对p≈0.202，不宜宣称统计确定的下降机制。
5. 逐例上下文缩短不等于反馈无噪声；982次反思产生973条非空critique。全部建议在一次Split整合，旧正确案例和拒绝候选经验未显式随建议输入。完整性孩子重叠、创造性与清晰性重叠等现象值得逐例核验；这些目前是候选解释，不是因果证据。
6. 第17章full-report neutral也达到887正确，与factuality-first相同；label-only neutral仅820。不能优先把此次差距归因于缺少事实优先提示词（该段实际存在）或reason丢失（已排除）。

后续最小定位实验：保持推理提示词和Arbiter不变，用Final其他四根报告固定、仅恢复Init完整性报告，再重跑Arbiter做配对归因。VLRB上的此类检查只作为事后诊断；新接受规则应在独立开发数据上选择，不能据此反复挑VLRB最优版本。


## Discovery90 数据替换对照（2026-09-20 完成）

产物：output/subtree_local_reflection/discovery90。13:15:48启动，20:19:05完成，约7小时3分钟。初始化和五轮演化均使用原RLHF-V Discovery90；Dev150和VLRB不变。算法、模型、提示词和接受规则未更改。注意main的冻结配置记录最初Manager并发30，运行中曾改为10再6；两组最终runtime均为6。因此是算法配置保持一致的数据替换pilot，但不是完全相同服务调度历史或多seed对照。两组Init也由各自数据重新生成，不是固定Init后仅替换反馈数据。

### 系统结果

VLRB采用official K3至少两票一致。终端runtime Final68.81%与official68.24%不同，原因见上节指标口径审计。

| 数据 | Init正确数 / Strict | Final正确数 / Strict | 变化 |
|---|---|---|---|
| Discovery90 | 66/90，73.33% | 65/90，72.22% | -1 |
| Dev150 | 107/150，71.33% | 109/150，72.67% | +2 |
| VLRB1247 | 858/1247，68.81% | 851/1247，68.24% | -7 |

Dev corrected10/harmed8；VLRB corrected40/harmed47，exact McNemar p=0.5203。VLRB Overall69.42%→69.36%，Macro63.39%→63.11%，Coverage99.12%→98.40%（1236→1227）。外部收益未得到支持，但单次下降不显著。

Discovery100对照：Dev105→106；VLRB875→861。Discovery90 Final比Discovery100 Final少10条VLRB正确，但其Init已经少17条；不能把最终差距全部归因于后续演化。两种数据下均未观察到正向VLRB演化增益。

VLRB类别正确数：General85→83；Hallucination564→560；Reasoning209→208。RLHF-V来源子集50/68→48/68，未观察到该子集收益；这不等于对独立RLHF-V heldout500的评估，本轮没有运行heldout500。

### 五轮与接受机制

系统正确数66→63→66→67→65→65。每轮答案变动7、5、5、8、6条，并非整个系统停止变化。

25次Split接受8次：e01完整性/事实性/创造性；e02创造性；e03事实性；e04完整性；e05完整性/事实性。视觉依据、清晰性没有接受修改，其Final报告复用Init。Init/Final均25节点。

8次接受中的4次正确数下降：e01事实性60/88→59/86；e01创造性56/81→53/76；e03事实性59/86→54/77；e04完整性62/89→60/86。比值为正确数/覆盖数，均使Covered ACC提高。e04完整性仅提高约0.10pp；e05事实性54/77→59/84仅提高约0.11pp，接受没有最小收益或统计稳定性门槛。

VLRB局部正确数：完整性758→742，视觉依据830→830，事实性858→784，创造性754→721，清晰性767→767。局部指标使用全局Gold，不是各维度人工真值。完整性覆盖1203→1155，Covered ACC63.01%→64.24%；事实性覆盖1211→1121，Covered ACC70.85%→69.94%；创造性覆盖1107→1138，Covered ACC68.11%→63.36%。

### 聚合与准则诊断

同报告离线聚合：VLRB Init多数投票808/1247（64.80%），Arbiter858（68.81%）；Final多数投票803（64.39%），Arbiter851（68.24%）。Arbiter净收益+50/+48，依然有效。Dev多数投票两版本均113/150（75.33%），高于Arbiter107/109，说明聚合收益有数据依赖，不能宣称普遍优于投票。

Final完整性包含detect_invented_elements_and_structures、validate_premises_and_hypotheticals；创造性包含verify_grounding_prerequisite、handle_shared_grounding_failures，显示明显视觉事实性职责重叠。事实性和创造性都有双方严重错误时输出None的规则；事实性将缺少motion blur等视觉证据的动态描述视为幻觉，可能把合理静态图像动作推断过度排除。此为文本层面的风险，尚未逐例证明造成系统损害。完整性、事实性和创造性职责趋同可能减少互补性，需要案例或替换单根报告实验验证。

准则description字符总数9119→14047（+54.0%），节点数不变；长度增长不等于质量提高。更集中数据没有自动解决Covered ACC选择偏差、全局Gold监督局部维度的冲突，以及有限样本K1反复选择问题。

### 成本

正式Manager890次逻辑调用、972次尝试，输入4,204,022、输出321,029 token，reasoning0，82次usage缺失。137次signature、5次cluster、5次children、718次reflection、25次split；718次reflection均进入critique汇总。错误11次429、63次503类InternalServerError、8次timeout。最终系统评估技术失败为0。按历史用户单价估算已知usage约4.06元，不包含缺失usage、smoke和其他费用；比前组约6.34元少36%。

结论：数据源变化会影响初始化和演化方向，但本组不支持仅换回RLHF-V即可恢复VLRB收益。下一步优先隔离接受目标而非继续盲换数据；对局部正确数下降仍接受的候选做离线审计，再在独立开发数据上预先固定新的接受规则，保持VLRB为最终评估。


## Discovery100 + 122B非思考Manager（2026-09-21）

产物output/subtree_local_reflection/discovery100_122b，completed/external_complete均true。2026-09-20 22:37至09-21 09:33，约10小时56分。Manager122B、enable_thinking=false、并发signature/reflection6、默认4、timeout300，Worker保持8B/50。reasoning_tokens=0。Init/Final均28节点，description字符12472→17820（+42.9%）。

| 数据 | Init正确数 / Strict | Final正确数 / Strict | 差值 |
|---|---|---|---|
| Discovery100 | 61/100，61% | 65/100，65% | +4 |
| Dev150 | 109/150，72.67% | 98/150，65.33% | -11 |
| VLRB official | 868/1247，69.61% | 831/1247，66.64% | -37 |

VLRB official K3至少两票，与终端runtime67.76%不同。official corrected45/harmed82，exact McNemar p=0.001307。单轨迹内下降显著，但不是多个独立演化seed的统计结论。Overall70.23%→70.19%，Macro64.90%→64.62%，Coverage99.12%→94.95%（1236→1184）；None11→63。correct→None29、wrong→None24、None→correct1，净正确损失中28条来自弃权转换，其余净9条来自A/B翻转。Dev correct→None11，None→correct0，corrected11/harmed22；覆盖148→134。

对照27B Discovery100：Init875→Final861（-14）；122B868→831（-37）。122B Final比27B少30条，Init已少7条。各自重新初始化，不能只看Final差距判断纯reflection能力。初始122B Dev比27B高4条，VLRB低7条，初始化并非全面更好。

五轮系统61→60→63→67→64→65。接受9/25：e01 R2/R4/R5，e02 R2/R3，e03 R5，e04 R1/R3，e05 R1。四次接受正确数下降：e01R2 57/93→55/86；e03R5 63/94→58/81；e04R3 57/86→46/63；e05R1 63/93→62/90。尤其R3的Covered ACC66.28%→73.02%，正确数反而少11。

VLRB局部正确/覆盖/coveredACC：R1 748/1198/62.44%→756/1196/63.21%；R2 771/1165/66.18%→763/1183/64.50%；R3 853/1193/71.50%→688/962/71.52%；R4 828/1203/68.83%→830/1208/68.71%；R5 771/1191/64.74%→728/1107/65.76%。局部Gold仍是全局偏好，不是维度真值。

同报告多数投票→Arbiter：Init816→868（+52），Final798→831（+33）。Final仍优于多数投票，但Arbiter增益缩小；Init corrected77/harmed25，Final77/44。完整报告中的弃权信号可能影响Arbiter，但单根因果贡献尚未验证。

类别正确：General91→85，Hallucination567→542，Reasoning210→204。Final事实性新增/包含factual_equivalence_and_none_protocol，清晰性包含tie_resolution_and_none_preference；局部合理None本身不是错误，但当前全局Gold+CoveredACC选择可能强化拒答方向。事实性将格式/禁词违反定义为factual hallucination，清晰性也审查事实前提、格式约束，存在跨根职责重叠。不可将全局弃权全部直接归因于某条准则，需逐例报告或单根替换验证。

成本：1260逻辑Manager调用、1755attempts，输入8166231输出426961，reasoning0，492attempt缺usage；1036逐例reflection，汇总critique同为1036。日志记录RateLimitError468、timeout5、InternalServerError19（日志口径，不等价严格去重正式attempt分类）。已完成评估技术失败0。与27B相比已知输入增加约15.6%，输出减少约2.4%；不沿用27B单价估算122B费用。

结论：仅提高Manager规模没有改善本协议泛化；覆盖下降和局部CoveredACC接受的目标偏差证据更强。下一步优先固定Manager/数据/Init，仅比较接受规则或先恢复Init事实性/清晰性报告做事后归因；不要依据VLRB反复选择最终版本。所有后续测试尚未运行。


## 局部Strict ACC接受对照（已完成）

产物output/subtree_local_reflection/discovery100_27b_strict。Init rubric、Discovery system、Dev initial、VLRB initial四文件与main逐字节一致。接受指标为strict_acc，Manager仍27B非思考。运行期间调整过并发和API key轮换，非多seed确定性对照。

Discovery62→69；Dev105/150→112/150（70%→74.67%，corrected14/harmed7）；VLRB official875→864（70.17%→69.29%，corrected40/harmed51，p=0.29447）。终端runtime69.53%不是official。VLRB Coverage1237→1240（99.20%→99.44%），Overall70.74%→69.68%，Macro64.08%→63.30%。没有覆盖坍缩，下降来自判断准确性。相同Init的Covered对照Final为Discovery67、Dev106、VLRB861；Strict分别多2、6、3条。单轨迹VLRB差3条不能证明稳定优越。

五轮系统62→67→68→68→69→69。25次候选接受5次：e01事实性55→62、创造性63→67；e02创造性67→68、清晰性61→65；e04清晰性65→66。所有接受均增加局部正确数；完整性和视觉依据未修改。不能把这5次理解为原Covered轨迹过滤后的结果，因为候选生成和后续反馈轨迹重新采样。

VLRB局部正确/覆盖：完整性712/1182保持；视觉依据784/1189保持；事实性844/1199→842/1189；创造性839/1177→752/1213；清晰性759/1211→722/1215。创造性-87和清晰性-37是最突出泛化退化，非弃权造成。五根等权多数投票828→789；Arbiter875→864，聚合净收益47→75，仍提供补偿。类别正确General86→84，Hallucination584→570，Reasoning205→210。

Final25节点，description字符9988→12151（+21.7%）。清晰性排除视觉错误、格式约束，边界较明确，但清晰性将arithmetic execution errors归给Factuality，而事实性又排除正确视觉前提下的calculation mistakes，存在算术错误责任缺口。该文本问题尚未通过具体案例证明因果贡献。更清楚的分工不自动提高与全局Gold的一致率。

Manager965calls1010attempts，输入5978741输出371817、reasoning0、44attempt缺usage；初始化复用，因此本次成本不包含原初始化成本。已完成系统技术失败0。

结论：Strict修复了可变覆盖分母导致的接受问题，Dev观察到改善，但VLRB收益仍未实现。下一步优先独立接受集或正确案例保留约束的单变量验证，以及创造性/清晰性单根恢复的事后归因；不继续把全部失败归为Covered ACC或Manager规模。


## Strict ACC + five random correct cases (completed 2026-09-22)

Run: `output/subtree_local_reflection/discovery100_27b_strict_preserve5`. completed/external_complete=true. Frozen run configs differ from Strict baseline only in preservation_case_count=5 and preservation_seed=42. Four Init artifacts are byte-identical. All 25 Split requests contain five images and five preservation cases. Single stochastic trajectory; not a multi-seed causal estimate.

| System | Init | Strict baseline Final | Preserve5 Final |
|---|---:|---:|---:|
| Discovery100 correct | 62 | 69 | 68 |
| Dev150 correct | 105 | 112 | 103 |
| VLRB official correct | 875 | 864 | 856 |

VLRB uses official K3 majority, not runtime plurality. Preserve5 Strict=68.64%, Overall=68.92%, Macro=62.24%, Coverage=99.60%. Versus Init: corrected51/harmed70, net-19, exact McNemar p=0.10137. Dev corrected14/harmed16. VLRB group correct General86->84, Hallucination584->572, Reasoning205->200. No system coverage collapse (1237->1242 covered).

Evolution system correct: 62->69->72->71->69->68. Accepted 8/25: e01 R3/R4/R5; e02 R4; e03 R2; e04 R4/R5; e05 R1. Sum of five roots' correct counts: 300->313->314->315->321->323. Thus accepted local improvements do not guarantee system improvements, even on the same Discovery data. K1 stochastic judging is a remaining confound.

VLRB root correct Init->Final: R1 712->548, R2 784->791, R3 844->842, R4 839->856, R5 759->731. Dev: R1 106->99, R2 101->93, R3 90->95, R4 91->98, R5 107->104. Discovery: R1 62->64, R2 59->60, R3 55->63, R4 63->68, R5 61->68. Root metrics use global preference gold, not dimension-specific human labels. R1 VLRB coverage1182->1062 and covered accuracy60.24%->51.60%, so both coverage and covered accuracy deteriorated.

Preservation audit: 102/125 case-occurrences remain correct in candidate evaluation (81.6%); accepted candidates retain35/40 (87.5%). These are occurrences, not unique samples. Eight accepted candidates correct84 and harm61 old-correct case-occurrences. Five examples are prompt guidance, not a hard preservation gate; they neither protect every provided example nor all previous successes. e04 R4 retains3/5 while accepted65->68; e05 R1 retains4/5 while accepted62->64.

Same-report five-root majority VLRB: Init828, Strict789, Preserve5816; Arbiter875/864/856. Preserve5 majority improves27 over Strict but Arbiter loses8, showing individual-root/global-report aggregation differences; this is not explicit node-recursive evaluation. Dev majority110/114/109 versus Arbiter105/112/103.

Final Rubric25->24 nodes, description characters9988->12628 (+26.4%). R1 explicitly counts attempts as complete even when hallucinated; it excludes factual errors and changes visual_grounding_verification to visual_reference_attempt. This sharply narrows Completeness toward mention coverage, consistent with—but not a proven cause of—its external degradation. R2 map_visual_data_to_reasoning treats explicit mapping as satisfied even if observations are wrong, while sibling rules demand accurate visual claims. R3 excludes calculation/setup errors but also validates derived visual values and option mismatches; boundaries remain potentially conflicting. R4 separates style from accuracy and reaches856 VLRB correct, so preservation is not uniformly harmful. R5 now explicitly penalizes internal mathematical impossibilities, partially addressing the earlier arithmetic responsibility gap, but loses28 VLRB correct. Need case-level or one-root replacement tests for causal attribution.

Split measured input tokens: Strict min/mean/max9883/10651/11522; Preserve514515/20358/35625 (+91.1% mean). Mean output711->714. Reflection mean input6071->5981. Known formal Manager input6112990/output373809;962 logical calls1048 attempts,86 usage-missing attempts,reasoning0. No final system technical failures. Run artifacts span approximately00:27-06:47 (~6h20m, including elapsed gaps). Latency sums are concurrent request time, not wall time. Longer Split inputs are verified; attention dilution is a hypothesis, not established by this run.

Conclusion: this five-random-case prompt intervention did not improve external system generalization. Priority diagnostic: freeze reports and restore Init R1 only, rerun Arbiter as a post-hoc attribution experiment. Do not tune on VLRB. A later pre-registered experiment could separate critique generation data from local acceptance data, or test compact positive capability summaries, one variable at a time. Do not simply increase positive example count or Manager size based on this trajectory.


### Historical Discovery100 peak correction

Expanded audit searched 772 small report/summary/trajectory/metric JSON artifacts across both CritiQ and CritiQ-framework-v6 output trees (excluding inference caches), plus the historical implementation document. Preserve5 e02=72/100 is NOT a historical record. Phase19 system-acceptance epoch trajectory is69,69,69,71,71,72; its formal Final is72. Phase20 mixed local acceptance trajectory is69,69,73,66,66,68; e02 reached73, but Phase21 protocol explicitly retains Phase20 only as a diagnostic reference because of protocol mixing. Phase20 frozen Discovery hash equals current dataset SHA25606551a775bdec55818df5fc2db3aaebdd418c7327248d66811266faa2f98a1c3. Phase21 trajectory is62 throughout; Phase22 is68,65,69,69,69,69. Phase21 K1 rejected-candidate audit peaks at71; K3 exhaustive coalition audit at70. These diagnostics must not be conflated with accepted evolution Final checkpoints. Earlier Phase17 explicit-recursive peak66/Final65 and Qwen2.5 Phase18 peak63/Final62 use different inference setups. Thus this run matches Phase19's72 but does not exceed Phase20's diagnostic73; its own Final remains68.

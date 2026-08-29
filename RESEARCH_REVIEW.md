# 独立研究贡献 Review：Evolving Structured Rubrics from Multimodal Preferences

## 审查范围与状态

本审查只依据 `docs/Contribution.md` 的贡献叙述；没有读取或检索仓库中的实验、消融、结果或相关工作。评审标准为 NeurIPS / ICML / ICLR senior reviewer。二次 reviewer 使用 GPT-5.6-Sol（ultra），与主审查为同一模型家族，因此 `review_independence: same-family`、`acceptance_status: provisional`；这不是跨模型的最终裁决。

## 总体判断

Experience → Evolution → Execution 的递进非常清楚，问题也有现实意义：冻结多模态 judge、把外部偏好反馈保存为可审计的结构化记忆，并尝试让重复错误触发规则分化和修正。可是，仅从当前叙述看，工作仍更像一个有吸引力的系统愿景，而不是科学增量已经可辨识的顶会方法。

当前暂定评分为 **4/10（weak reject / borderline）**，信心 **3/5**。主要原因不是方向不重要，而是：三项贡献可能只是同一系统的表示、更新规则和推理编排；failure attribution、Split/Refine、执行语义和 transfer 的定义与证据尚不闭合；与 external memory/RAG、hard-example cache、reflection/prompt evolution、hierarchical/decomposed judge、process/reward modeling 等类别的 algorithm-level 区分没有被写出。

## 优点

1. 叙事围绕同一个对象（结构化评价知识）递进，不是简单罗列模块。
2. 明确区分 rubric 的“内容”和 rubric 的“执行语义”，这是比“把更长 prompt 塞给模型”更有潜力的研究假设。
3. 冻结底模、显式外部记忆、父节点 fallback、失败归因等设计具有可解释、可审计和跨 worker 复用的潜在价值。
4. 文本至少意识到结构化执行和跨 worker 迁移需要单独验证，而非只报告单一总体分数。

## 主要问题（按严重性排序）

### 1. 三项贡献的独立性与科学增量不够清楚

Contribution 1 是 framing/表示（外部演化记忆），Contribution 2 是记忆更新规则，Contribution 3 是推理编排；后两项都依赖第一项，因此把三层都写成平行 novelty 容易被认为是 contribution inflation。建议把 **failure-driven structural evolution** 作为唯一主方法贡献，把 Experience 降为问题设定，把 Execution 降为必要的系统设计与支持性证据；除非能给出独立形式化语义和严格因果对照，否则不要把执行结构单独包装成第二个科学贡献。

### 2. “failure”与错误归因没有可操作定义，闭环可能自证

文本没有说明 failure 是与外部人类偏好不一致、低置信度、自批评，还是 criterion-level oracle。整体 pairwise 标签 `y` 通常不能唯一识别“缺少哪个评价知识”；如果没有独立反馈，系统可能只是把自己的解释当成错误原因。必须写清反馈来源、时间线、噪声/标注分歧处理、credit assignment，以及何时允许更新。

### 3. 从错误模式到“专家知识”存在不可辨识跳跃

ErrorSignature、语义聚类、children 生成和 failure attribution 都只是名称，未说明输入、目标、距离、阈值和人工/模型审核。重复错误也可能来自样本偏斜、worker 能力不足或标签噪声，并不等于存在新的评价维度。没有独立验证集或盲审，不能把结果称为“长出专家知识”。

### 4. Split/Refine 目前不可复现且易过拟合

需要形式化 `R_t` 的状态、候选生成、适用边界、accept/reject、停止、合并、删除、回滚和版本化。 “specialized accuracy 不劣于 parent” 还需要定义指标、held-out 划分、统计检验、多重尝试修正和复杂度惩罚；在自适应 scope 上挑选 child 会产生选择偏差，parent fallback 也可能掩盖 child 失败。

### 5. 稳定性与成本完全没有进入主张

树是否会无界膨胀？重复/冲突 criteria 如何处理？旧规则何时废弃？深度增加带来的 token、调用、延迟和标注成本是多少？`R_t` 随时间变化时，跨版本分数是否可比；若用于下游 RL，reward 是否非平稳？这些是“持续演化 reward judge”能否成立的必要条件。

### 6. Structured execution 的因果证据可能被预算混杂

文中以 S6 负消融支持“完整 rubric 文本不能替代结构化执行”，但只给出内部编号和定性描述。“one-shot”与多 subtree/arbiter 可能在 token 数、调用次数、上下文长度、ensemble 效应、顺序和随机性上不匹配。因此不能从一个对照推出“必须”隔离证据或显式冲突解析。至少要同内容、同总 token/call、同 seed 的 matched-budget 对照，并加入 map-reduce、并行 criterion voting、debate 等强结构基线。

### 7. Transfer / generalization claim 过度且不可判定

“未见 preference distribution”“明显收益”“在一定程度迁移”都没有定义 shift、目标反馈是否为零、worker 与 arbiter 分别换了什么、指标和置信区间。Qwen3-VL→Qwen2.5-VL 的例子在当前文件中不足以证明模型无关性；收益也可能来自额外指令或预算。应将结论限制到明确的 shift 和零目标反馈设置，并比较 source-evolved、target-worker-directly-evolved、static/random rubric。

### 8. 与既有方法类别的差异没有钉在算子层

仅说 failure-driven、structured、evolving 不足以排除“hard-example cache + RAG”“reflection/rewrite”“固定层级 judge”“prompt/program evolution”“test-time adaptation”“process reward modeling”等解释。需要逐类别说明：哪些 baseline 具有相同信息与预算；你的方法唯一新增、且可证伪的算子是什么；它是否确实在未见失败实例上产生可迁移的局部准则。

### 9. 多模态特异性与输出目标不清

除 `(I,Q,A,B,y)` 和 Visual Grounding 例子外，机制几乎可原样用于文本 judge。需说明视觉证据如何抽取/隔离、模态冲突如何处理、worker 感知差异如何进入 failure taxonomy，以及输出究竟是 pairwise label、标量 reward、criterion scores 还是 policy feedback。

## 最可能导致拒稿的三个问题及最低修复

1. **新颖性不可辨识**：被评为“memory + reflection + hierarchy + map-reduce”的重命名。最低修复：一页形式化算法和逐类别 novelty/delta 表；至少一个 matched-budget 小实验或受控案例证明只有 failure-driven structural induction 能在未见实例上产生可迁移 child。
2. **错误归因/标签泄漏**：没有独立 oracle 时闭环不成立。最低修复：外部或 held-out 偏好反馈、严格 train→update→test 时间线、criterion-level attribution 盲审、噪声鲁棒性分析。
3. **执行与迁移结论被混杂变量支撑**：one-shot 对照及跨 worker 证据不足。最低修复：同 rubric 内容与总预算的执行对照；明确定义 shift、worker/arbiter 角色、零目标反馈，并报告绝对效应与 CI。

近致命问题是树膨胀、冲突和成本；需复杂度正则、merge/prune/rollback、终止条件与成本曲线。

## 不大幅增加计算量时的最小证据包

- 给出完整伪代码和时间线：failure event、ErrorSignature、聚类、候选、accept/reject、回滚/停止、执行与聚合。
- 在小型受控审计集上验证：重复真实模式应 split，随机/标签噪声不应 split；报告 cluster coherence、跨 seed 稳定性和盲审 attribution。
- 用相同失败信息与 token/call budget 对照 hard-example cache/RAG、单父节点 rewrite/reflection、固定 hierarchy、one-shot/map-reduce/debate；在未见失败实例上测 accuracy 与 calibration。
- 做 leave-one-cluster-out 或 counterfactual 测试，报告规则数、覆盖率、重复/回滚率、调用/延迟成本。
- 对 execution 保持 rubric 内容不变，仅改变上下文隔离与 aggregator；对 transfer 比较 source-evolved、target-directly-evolved、static/random，并确保目标分布无反馈。

## 建议的收敛版贡献表述

> 我们研究冻结多模态 judge 的 failure-driven external-memory adaptation。给定能够识别 judge 错误的偏好反馈，我们把重复错误签名聚合为候选局部评价准则，并仅在独立验证集和复杂度预算下接受 split/refine 更新。推理时通过隔离的 subtree 评审与显式聚合器执行这些规则。我们的经验性结论限定在明确的分布转移和零目标反馈的 worker-transfer 设置；我们不声称 pairwise 标签能唯一恢复人类 ontology，也不声称该评价记忆对所有模型普遍无关。

对应的英文核心句可压缩为：

> “We study failure-driven external-memory adaptation of a frozen multimodal judge. Recurring, externally validated error signatures are aggregated into candidate local criteria, and split/refine updates are accepted only on held-out data under a complexity budget; the rubric is executed by isolated subtree evaluations followed by an explicit aggregator. Claims are limited to specified shifts and zero-feedback worker transfer.”

## 最终建议

当前不建议以“三个平行 conceptual/method/system contributions”直接投稿顶会。应将主线收敛为一个可证伪的方法命题：**外部验证的重复 judge failures 是否能驱动结构化规则归纳，并在未见实例上带来超越记忆缓存/重写/固定层级的增益？** 先补齐定义、泄漏安全、matched controls、稳定性与成本，再决定是否把 execution 升格为独立贡献。达到 borderline accept（约 6/10）的门槛，是形式化算法 + 强基线/预算匹配 + 有限且量化的 transfer 证据；达到 7+ 还需要跨 seed/跨任务稳健增益或非平凡理论性质。

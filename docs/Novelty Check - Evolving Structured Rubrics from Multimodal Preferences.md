# Novelty Check: Evolving Structured Rubrics from Multimodal Preferences

> 更新日期：2026-08-26  
> 评估范围：当前最优的显式结构化 rubric 演化与执行方案。尚在实验中的 Implicit-All / Unified-Subtree 不纳入本报告。
> 检索范围：ACL Anthology、arXiv、OpenReview 及 2024–2026 年公开论文，重点检查截至 2026-08-26 的近六个月工作。本文是研究查新与投稿定位分析，不是穷尽性专利检索。

## 1. 研究方案概述

本研究关注如何用少量多模态人类成对偏好数据，使冻结的视觉语言模型（VLM）judge 更接近人类判断。输入数据由图像、问题、两个候选回答和人类偏好标签组成。系统首先以五个通用多模态评价维度为根节点，然后让 VLM 在每条准则下分别判断候选回答。系统收集与人类标签不一致的错误案例，由强 manager 模型生成错误签名、聚类反复出现的失败模式，并通过以下方式演化 rubric：

- **Split**：把过于宽泛的父准则拆分为更具体的子准则；
- **Refine**：根据失败案例改写准则的适用条件、排除条件和判断规则；
- **Competition and rollback**：新准则只有在冻结数据和预定义指标上优于旧准则时才接受，否则回退；
- **Lineage and memory**：保存准则来源、错误簇、接受/拒绝记录以及失败经验，避免重复探索。

最终得到的不是一个经过参数训练的 reward model，而是一组可检查、具有迁移潜力、显式执行的自然语言准则。当前 Phase-17 只用 100 条 Discovery 数据生成 rubric proposal，并决定 Split/Refine 的接受与回退，形成 5 个根节点和 22 个子节点；根节点来自 Multi-Crit 提供的通用评价维度，子节点主要从目标多模态偏好数据中的错误与失败经验归纳产生。

因此，这不是无先验的准则发现：搜索同时受到三类先验约束——Multi-Crit 提供的根准则、只能在既有父节点下进行的子准则搜索空间，以及强预训练 manager 已有的多模态知识。本文后续所称“发现”，均指在这些先验下发现目标偏好中反复出现的失配模式及其更具体的判断边界。

当前端到端执行语义如下：

1. Qwen3-VL-8B Pairwise Worker 在每个准则下分别输出 `A`、`B` 或 abstain；
2. 每个父节点及其直接子节点形成一个子树，子节点存在唯一 A/B 多数投票时使用子节点多数投票，否则回退父节点本地投票；
3. 每个根子树最多向全局贡献一票，五个根子树采用等权多数聚合，无唯一多数时输出 Tie；
4. Phase-17 的错误签名、语义聚类和子准则生成使用 Qwen3.5-397B-A17B manager；
5. 新准则在固定的 Discovery100 cohort 上与旧准则比较。Split 要求父节点加子节点后的局部子树准确率不低于原父节点，Refine 要求目标节点准确率严格提升；不满足规则的候选被拒绝并记录失败原因。

这里的“冻结竞争”表示样本集合、worker request、聚合语义和接受规则在一次候选比较中保持不变，**不表示使用了独立的 validation split**。同一 Discovery100 被反复用于准则选择，因此仍存在选择性过拟合风险。Dev150 仅作诊断，不参与接受、拒绝或 early stopping。

数据隔离记录显示：Discovery100 与 Dev150 来自 RLHF-V、MM-RLHF、ViLReward-73K、MMPR-v1.2、VisionArena-Battle 和 MMIF-23K 六个来源，两者在 sample ID、图像 SHA、问题 SHA、无序回答对 SHA 和 source/sample ID 上均为零重叠。构建两者时还用 VL-RewardBench 的 sample ID、图像、image+question 和无序回答对指纹做排除；最终审计中，这些强重叠均为 0。

一个真实的 root-to-child 演化例子是：

- 父准则 `completeness_and_coverage` 倾向于把更长、更详细的回答判断为更完整；
- 多个失败案例显示，一些回答只是通过编造图像中不存在的对象、文字或属性获得了表面上的“覆盖率”；
- 系统据此生成子准则 `grounded_coverage_over_hallucinated_volume`，只在候选回答的视觉覆盖差异主要来自幻觉内容时适用，并明确规定应偏好有视觉依据的覆盖，而不能把幻觉体量计为完整性；
- 候选子准则只有通过上述冻结竞争规则才写入最终 rubric，并保存其错误簇、示例和父节点 lineage。

### 1.1 术语

- **Native judge**：不提供学习所得 rubric，仅使用原生 pairwise 判断提示的 VLM worker。
- **Initial rubric**：只使用来自 Multi-Crit 的五个通用根准则。
- **Evolved rubric**：五个根准则加上从 Discovery100 失败案例中接受的 22 个子准则。
- **Error signature**：manager 对单个 human–judge disagreement 的结构化失败归因。
- **Failure cluster**：多个错误签名中语义一致、可由同一更具体判断规则解释的错误集合。
- **Applicability boundary**：子准则中的 `Applicable only when`、`Not applicable when` 和 `Decision rule`，用于限制准则的判断范围。
- **Transfer**：冻结演化后的 rubric 文本与结构，更换 Pairwise Worker 后重新执行；当前证据只有 Qwen3-VL-8B 到 Qwen2.5-VL-7B 的同家族迁移。

## 2. 当前主要结果

在 VL-RewardBench 的 1,247 个样本上，当前结果如下。表内全部准确率均为 **Strict ACC**：始终以 1,247 为分母，最终输出 Tie、abstain 或技术未解析均不计为正确；因此不同方法不会因选择性弃权而缩小分母。

| Worker | Native judge | Initial 5-root rubric | Evolved rubric | 相对 native 增益 | 相对 initial 增益 |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 54.13% | 57.10% | 70.01% | +15.88 pp | +12.91 pp |
| Qwen2.5-VL-7B | 41.30% | 41.38% | 56.94% | +15.64 pp | +15.56 pp |

这里应使用“提高约 16 个百分点”，而不是“相对提高 16%”。Qwen3-VL 上的结果表明，收益不只是来自加入五个通用根准则；从偏好错误中演化出的子准则额外带来了 12.91 个百分点。将 Qwen3-VL 上演化出的同一 rubric 直接交给 Qwen2.5-VL-7B 后仍有明显收益，说明自然语言 rubric 是一种有潜力的模型参数外人类偏好接口，并提供了初步的同家族迁移证据。当前报告尚未给出多随机种子置信区间、完整 paired significance，以及准则生成阶段的总调用次数/token 成本；这些属于投稿前必须补充的证据。

对应实验记录：

- [Phase-17 VL-RewardBench 探索性评估](../output/evolving_structured_rubrics/vl_rewardbench_phase17_discovery_v2_prompt_v2_v1/final_report.md)
- [Qwen2.5-VL worker transfer](../output/evolving_structured_rubrics/vl_rewardbench_qwen25_phase17_e4_transfer_v1/final_report.md)
- [Phase-17 演化协议与内部结果](../output/evolving_structured_rubrics/rubric_evolution_phase5/phase17_discovery_v2_prompt_v2_split_refine_v1/final_report.md)

## 3. 核心创新主张及初步判断

| 核心主张 | 新颖性判断 | 原因 |
|---|---|---|
| 从少量 pairwise preference 中自动显式化评价准则 | **低** | CritiQ、Auto-Rubric、PReMISE 已在文本任务中实现相似目标。 |
| 从多模态 judge 的重复错误和失败经验中发现偏好失配 | **中等** | 错误驱动的反思已有先例，但把视觉证据相关失败聚类为可执行准则边界仍有一定差异。 |
| 将粗准则 Split 为带 Applicable / Not applicable / Decision rule 的子准则 | **中等** | 这是当前最清晰的方法差异，但与 Auto-Rubric、RRD 的层次分解仍有概念重合。 |
| 使用冻结竞争、回退、lineage 和 memory 控制自然语言准则演化 | **中等** | 单项机制并非全新，但形成了较完整、可审计的准则优化协议。 |
| 演化出的多模态 rubric 能迁移到另一个 VLM worker | **中等（作为发现）** | 文本领域已有跨 judge rubric transfer；当前多模态结果较强，但只有 Qwen3-VL 到 Qwen2.5-VL 一条迁移路径。 |
| 用 100 条人类偏好观察到约 16 个百分点的 benchmark 收益 | **较高（作为探索性实证结果）** | 绝对增益较大且有实际价值，但尚无完整显著性检验；性能增益本身也不能单独构成方法新颖性。 |

综合创新性评估为 **5.7/10**：方法创新约 **4/10**，多模态应用创新约 **6/10**，当前实证发现约 **7/10**。研究值得继续，但需要将贡献准确定位为“少样本、多模态、失败驱动的准则特化与迁移”，而不是“从零发现人类潜在准则的新范式”。

## 4. 最接近的工作

最直接威胁方法新颖性的五篇工作可以先概括为：

| 工作 | 主要重合类型 | 与当前研究最危险的重合点 | 当前仍可能保留的差异 |
|---|---|---|---|
| CritiQ | 算法与数据效率 | 少量偏好、manager-worker、错误反思、准则演化 | 多模态错误结构、带边界子准则、冻结竞争与 lineage |
| Auto-Rubric | 算法与可迁移性 | 少量偏好、验证驱动迭代、层次 rubric、跨 judge | failure-cluster-to-child 和 image-conditioned reusable specialization |
| PReMISE | 目标与审计 | 从 pairwise preference 发现可复用 policy rubrics | 显式父子谱系和视觉适用边界 |
| RRD | 结构优化 | 递归分解、过滤失配与冗余准则 | 人类失配驱动的持久子准则及回退记录 |
| CriterAlign | 失败经验 | 从 human–judge rationale gaps 提炼指导并改进判断 | 把多模态失败簇转化为有父节点和适用域的子准则 |

### 4.1 直接接近核心方法的工作

#### 4.1.1 CritiQ: Mining Data Quality Criteria from Human Preferences

- **论文**：[CritiQ, ACL 2025](https://aclanthology.org/2025.acl-long.792/)
- **做了什么**：使用约 30 个文本领域的人类偏好对自动挖掘数据质量准则。Manager agent 根据判断结果演化准则，worker agents 执行成对判断，并通过反思错误案例改进准则。最终准则还被用于训练 scorer 和执行数据选择。
- **与我们相似的地方**：都从少量人类 pairwise preference 中挖掘自然语言准则；都使用 manager-worker 架构；都根据错误判断进行反思和准则更新；都强调准则的可解释性和可复用性。
- **主要差异**：CritiQ 面向文本数据质量选择，当前研究面向图像条件下的开放式回答判断；当前研究进一步把失败模式聚类为带适用边界的子准则，保存显式父子谱系，并用候选竞争与回退控制结构演化。
- **新颖性风险**：这是最危险的直接先例。若不能证明多模态异质性使准则表示、失败归因或演化机制发生实质变化，审稿人可能把当前研究概括为“CritiQ for multimodal judging”。

#### 4.1.2 Auto-Rubric: From Implicit Weights to Explicit Rubrics

- **论文**：[Auto-Rubric, 2025/2026](https://arxiv.org/abs/2510.17314)
- **做了什么**：提出 training-free rubric learning。它先对偏好对执行 verification-driven 的局部准则归纳和反复修订，再用信息论编码率从候选准则中选出紧凑、非冗余的核心集合，并组织成“高层主题—具体检查项”的层次结构。论文报告只用 70 个文本偏好对即可改善多个 judge，并分析跨模型兼容性。
- **与我们相似的地方**：都把隐式偏好转化为显式自然语言准则；都强调少样本、无参数更新、迭代验证、层次结构和跨 judge 可迁移性。
- **主要差异**：Auto-Rubric 主要通过局部归纳与全局压缩寻找通用文本 rubrics；当前研究从 VLM judge 的实际错误签名出发，对已有粗准则执行 failure-cluster-to-child 的结构化特化，并显式编码适用、不适用条件和决策边界。
- **新颖性风险**：少样本、层次 rubric 和跨模型迁移均不能单独作为当前研究的首次贡献。

#### 4.1.3 PReMISE: Policy Rubrics as Measurement Specifications for LLM Judges

- **论文**：[PReMISE, 2026](https://arxiv.org/abs/2605.30803)
- **做了什么**：从文本 pairwise human preferences 中发现可复用的 policy-level rubric set，并从结构充分性、可靠性、偏好拟合和对抗鲁棒性四个方向审计 rubric。它还使用 preference-rank selection 和 reliability-constrained refinement 修复准则，并在多个 judge 上评估。
- **与我们相似的地方**：都从人类偏好中得到跨样本复用的准则，而不是为每个问题单独生成 rubric；都关心 applicability、准则质量、偏好一致性和跨 judge 使用。
- **主要差异**：PReMISE 主要对文本 policy rubrics 进行聚合、选择、审计和修复；当前研究显式追踪父子结构，并利用多模态错误案例生成视觉相关的应用边界和决策规则。
- **新颖性风险**：“从偏好发现可复用 rubric”和“跨 judge 使用”已被直接覆盖。

#### 4.1.4 RRD: Rethinking Rubric Generation

- **论文**：[RRD, 2026](https://arxiv.org/abs/2602.05125)
- **做了什么**：通过 recursive decompose-filter 循环，把粗准则拆成更细、判别性更强的准则，同时移除失配和冗余准则；还通过相关性感知的 whitening 权重减少相似准则的重复计权。
- **与我们相似的地方**：都认为粗粒度准则无法覆盖异质失败；都递归拆分准则，并根据实际判断能力保留或淘汰候选准则。
- **主要差异**：RRD 的重点是准则集合的信息性、覆盖性、非冗余性及聚合权重；当前研究的重点是从具体多模态错误簇生成带 applicability boundary 的子准则，以及保留演化 lineage、失败归因和回退记录。
- **新颖性风险**：不能笼统声称“首次通过递归分解改善 rubric”或“首次解决准则冗余”。

#### 4.1.5 CriterAlign: Criterion-Centric Rationale Alignment

- **论文**：[CriterAlign, 2026](https://arxiv.org/abs/2605.19665)
- **做了什么**：面向代码偏好判断，执行直接的 criterion-level pairwise judgment、tie-driven criterion refinement、A/B swap consistency filtering 和最终 pairwise synthesis。其 Human-Preference-Aligned Guidance 从人类偏好与 monolithic judge 预测之间反复出现的 rationale gaps 中提炼指导信息，再注入准则生成器和 judge。
- **与我们相似的地方**：都不是简单读取正确/错误标签，而是从人类偏好与 judge 失配的重复模式中提取失败经验，再用这些经验调整准则和判断过程。
- **主要差异**：CriterAlign 面向代码文本任务，主要形成全局 guidance；当前研究在多模态场景中把失败簇变成有父节点、适用域和具体视觉证据边界的持久子准则。
- **新颖性风险**：“从失败经验中挖掘人类潜在偏好”已有很接近的表述和机制，必须突出多模态失败结构以及 failure-to-child 的具体差异。

### 4.2 直接接近多模态问题设定的工作

#### 4.2.1 MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria

- **论文**：[MLLM-Bench, NAACL 2025](https://aclanthology.org/2025.naacl-long.256/)
- **做了什么**：为开放、主观的多模态任务构建 per-sample evaluation criteria，并让强 MLLM 对两条回答执行成对判断。论文报告其评价结果与人类判断达到 88.02% 一致性。
- **与我们相似的地方**：输入同样包含图像、问题和两个候选回答；都使用显式 criteria 改善多模态 pairwise judge 的人类一致性。
- **主要差异**：MLLM-Bench 的准则是面向单个样本构建的 benchmark annotation；当前研究从少量全局偏好数据中学习可跨样本复用、可以继续演化的准则集合。

#### 4.2.2 Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following

- **论文**：[Multi-Crit, 2025/2026](https://arxiv.org/abs/2511.21662)
- **做了什么**：构建带多准则人类标注的多模态 judge benchmark，评估模型能否遵循不同细粒度标准、切换标准，以及识别准则之间的偏好冲突。论文发现即使强模型也难以稳定遵循多元评价准则。
- **与我们相似的地方**：都把多模态人类偏好视为多个可能冲突的评价维度，而非单一总体分数；都执行 criterion-level 判断。
- **主要差异**：Multi-Crit 主要提供 benchmark 和人工定义的 criterion-level labels，并不从总体偏好错误中自动演化可复用 rubric。当前研究使用了 Multi-Crit 的五个通用维度作为初始 roots，再从目标偏好数据中学习子准则。
- **新颖性风险**：由于初始 roots 来自 Multi-Crit，当前系统不能声称从目标偏好中“从零发现”顶层人类准则；更准确的说法是从偏好中发现粗准则遗漏的子类型和适用边界。

#### 4.2.3 Prometheus-Vision

- **论文**：[Prometheus-Vision, Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.672/)
- **做了什么**：训练一个能够理解用户给定评分标准、对视觉语言回答提供细粒度反馈和评分的开源 VLM evaluator。
- **与我们相似的地方**：都让 VLM 根据显式自然语言准则评估图像条件回答，并强调细粒度、可解释的判断。
- **主要差异**：Prometheus-Vision 关注训练一个更好的 rubric-following evaluator，准则由用户给出；当前研究冻结 judge 参数，自动从少量偏好和错误中改进准则本身。

#### 4.2.4 PerceptionRubrics

- **论文**：[PerceptionRubrics, ICML 2026](https://arxiv.org/abs/2606.28322)
- **做了什么**：为信息密集图像建立大量 instance-specific rubrics，将标准分为必须正确的视觉事实和容易出错的细节，并使用 gated scoring 强烈惩罚关键视觉事实错误。
- **与我们相似的地方**：都认为视觉评价不能只做整体语义匹配，需要把视觉事实、细节和关键失败拆成原子标准；都使用类似 gate 的优先级思想避免流畅回答掩盖视觉错误。
- **主要差异**：PerceptionRubrics 的标准来自 golden captions 和专门的数据构建流程，主要用于 benchmark；当前研究从总体人类偏好和 judge 错误中自动生成可复用准则，并以偏好判断准确率驱动接受或回退。

### 4.3 同时接近多模态设定与 rubric learning 的混合型近邻

#### 4.3.1 AutoRubric-T2I

- **论文**：[AutoRubric-T2I, 2026](https://arxiv.org/abs/2605.17602)
- **做了什么**：从文本到图像生成的偏好对中合成候选 rubrics，让 VLM 在每条准则下比较成对图像，再用 L1 正则 logistic regression 选择最有判别力的准则。
- **与我们相似的地方**：都将多模态人类偏好从隐式 pairwise label 转化为显式、可解释准则；都使用准则在偏好对上的判别能力进行选择。
- **主要差异**：AutoRubric-T2I 的候选输出本身是图像，目标是图像生成 reward；当前研究评估的是给定图像和问题后的文本回答，并重点处理视觉证据、语言表现、推理和指令遵循之间的冲突。
- **新颖性风险**：“首次从多模态偏好自动学习 rubric”这一宽泛主张已经不成立，必须限定到 image-conditioned response judging 和 failure-driven reusable specialization。

#### 4.3.2 Auto-Rubric as Reward (ARR)

- **论文**：[ARR, 2026](https://arxiv.org/abs/2605.08354)
- **做了什么**：面向文本到图像生成和图像编辑，将 VLM 内部的隐式偏好外化为 prompt-specific、可独立验证的多维准则，并进一步把 rubric-conditioned preference decision 用作策略优化奖励。
- **与我们相似的地方**：都强调标量或整体偏好**会压缩人类判断中的多维结构**，显式 rubrics 可以提高可解释性、数据效率和人类一致性。
- **主要差异**：ARR 主要从 VLM 已有知识按 prompt 生成 instance-specific criteria，并服务于生成模型训练；当前研究从真实人类偏好与 judge 失败中学习跨样本复用的 policy-level 准则，不更新被评价模型。

#### 4.3.3 Visual Preference Optimization with Rubric Rewards

- **论文**：[rDPO, 2026](https://arxiv.org/abs/2604.13029)
- **做了什么**：为每个 image-instruction 构建包含 essential 和 additional checks 的 instance-specific rubric，用 VLM judge 逐项评分并筛选 on-policy preference data，随后执行视觉偏好优化。
- **与我们相似的地方**：都利用准则分解粗粒度多模态偏好，并区分关键视觉事实和补充质量维度。
- **主要差异**：rDPO 的重点是构建训练奖励和优化 VLM policy；当前研究的目标是用少量真实人类偏好改造 judge 的显式评价规则，主要贡献位于 evaluation 与 rubric specialization，而不是 policy training。

## 5. 与最接近工作的能力对照

下表中的“是”表示论文明确包含该能力，“部分”表示目标相近但实现或实验范围不同。

| 工作 | 人类 pairwise preference | 少样本准则学习 | 错误/失败驱动 | 层次或递归分解 | 多模态 | 学习所得 rubric 跨 judge 迁移 |
|---|---|---|---|---|---|---|
| CritiQ | 是 | 是 | 是 | 部分 | 否 | 非重点 |
| Auto-Rubric | 是 | 是 | 是 | 是 | 否 | 是 |
| PReMISE | 是 | 部分 | 是 | 部分 | 否 | 是 |
| RRD | 部分 | 非重点 | 部分 | 是 | 否 | 未说明/非重点 |
| CriterAlign | 是 | 非重点 | 是 | 部分 | 否 | 非重点 |
| MLLM-Bench | 是 | 否 | 否 | 否 | 是 | 否 |
| Multi-Crit | 是 | 否 | 否 | 否 | 是 | 否 |
| AutoRubric-T2I | 是 | 是 | 部分 | 否 | 是 | 未说明/非重点 |
| ARR | 部分 | 非重点 | 否 | 部分 | 是 | 未说明/非重点 |
| **当前研究** | **是** | **是（100 对）** | **是** | **是** | **是** | **初步同家族证据** |

没有一篇工作与当前方案在所有列上完全相同，但多个近邻工作的组合能够覆盖大部分单项机制。因此，创新不能建立在任意一个单独属性上，而应建立在以下交叉点：

> **从少量 image-conditioned human preferences 暴露出的 judge failures 中，学习带适用边界和演化谱系的可复用多模态准则，并检验这些准则能否成为模型参数之外的显式 human-alignment interface。当前证据仅初步支持同家族 worker transfer。**

## 6. 最危险的审稿人反驳

最可能的负面评价是：

> CritiQ 已经从少量人类偏好和错误反思中演化自然语言准则；Auto-Rubric、RRD 和 PReMISE 已覆盖少样本归纳、递归分解、层次结构和跨 judge transfer。本文使用 Multi-Crit 提供的顶层准则，将相似流程应用到图像条件回答。多模态改变了准则内容，但没有从根本上改变学习算法，因此主要是领域适配而不是新方法。

以当前证据，这一反驳只能部分回应。最终 rubric 中出现了视觉验证、空间几何、视觉歧义、可靠推断和视觉事实优先级等多模态相关准则，但尚未报告各类准则的出现频率和独立贡献，也缺少直接实验来证明这些收益不能通过图像 caption、更多通用文本准则或同等计算量的 CritiQ-style refinement 获得。真正需要反驳的不是“系统有没有生成视觉准则”，而是“这些准则及其收益是否必须由当前的多模态失败驱动机制产生”。

## 7. 推荐的论文定位

建议使用以下核心定位：

> **Few-shot failure-driven specialization of multimodal rubrics with preliminary cross-worker reuse for human-aligned VLM judging.**

对应中文表述：

> 从少量多模态人类偏好中，将通用评价维度逐步特化为带视觉适用边界的准则，并初步检验其同家族跨 worker 复用，从而显式修正 VLM judge 与人类偏好之间反复出现的失配。

建议将贡献组织为：

1. 观察到粗粒度多模态准则在视觉事实、语言表现、推理有效性和指令遵循之间反复出现的混淆模式；
2. 提出从多模态错误签名到 failure cluster、再到 applicability-bounded child criterion 的准则特化机制；
3. 使用冻结竞争、回退、lineage 和失败记忆形成可审计的自然语言准则演化协议；
4. 使用 100 条人类偏好作为准则优化数据，在 VL-RewardBench 的探索性 post-hoc 最优 checkpoint 上，使 Qwen3-VL-8B 相对 native judge 提高 15.88 个百分点；
5. 演化 rubric 在 Qwen2.5-VL-7B 上仍提高 15.64 个百分点，提供自然语言准则同家族 worker transfer 的初步证据。

应避免以下表述：

- “首次从偏好中自动发现评价准则”；
- “首次提出层次化或递归演化的 rubric”；
- “首次实现跨模型 rubric transfer”；
- “从零发现多模态人类潜在准则”；
- “只需 100 个样本的低成本方法”，而不说明使用强预训练 manager 和较高推理成本。

更准确的说法是：系统使用少量**人类标签**，借助强预训练 manager，把通用多模态准则特化为更贴合目标人类偏好的显式规则。

## 8. 将创新性提高到 7/10 所需的证据

1. **证明图像是准则发现所必需的**：比较 image、image caption、移除图像和打乱图像条件下的错误归因、准则结构和最终收益。
2. **与同预算近邻方法比较**：实现 CritiQ-style revise-only、flat rubric evolution、Auto-Rubric/RRD-style decomposition，并固定 worker、偏好数量、准则数量和推理预算。
3. **排除 root seed 依赖**：比较无外部 roots、随机通用 roots、Prometheus-style roots 和 Multi-Crit roots，观察最终准则是否收敛到相似视觉维度。
4. **排除规模和调用次数效应**：将 27-node evolved rubric 与同规模人工扩展、随机扩展和未使用错误簇的 rubrics 比较。
5. **扩大跨模型迁移**：至少增加两个非 Qwen 家族的 VLM worker，形成 source-worker × target-worker transfer matrix。
6. **证明多模态异质性对应实际收益**：按 OCR、空间关系、图表、密集场景、视觉歧义、视觉常识和开放式描述分组报告增益。
7. **提高确认性证据强度**：在完全未参与开发选择的外部数据上进行多随机种子、预先锁定协议的确认性评估，并报告 paired significance 和置信区间。

上述实验不会自动保证创新性达到 7/10；它们对应的是条件性判01据。至少需要看到：image 条件显著优于 caption/no-image；当前结构化方法在等节点数、等调用预算下显著优于 revise-only 和 flat evolution；并且至少两个非 Qwen worker 保留统计显著收益，才能较有把握地把方法创新性评价提高到约 7/10。

## 9. 最终判断

当前研究具有真实但较窄的创新性。最强部分不是“偏好到 rubric”这一已有范式，而是：

1. 在 image-conditioned response judging 中，从少量人类偏好暴露出的失败模式学习视觉相关准则边界；
2. 将这些边界组织为可执行、可回退、带完整谱系的持久 rubric；
3. 在已参与开发期诊断的 VL-RewardBench 上观察到约 16 个百分点的探索性提升，并展示初步的同家族 VLM worker transfer。

当前最合理的总体评价为 **5.7/10，Proceed with caution**。如果补齐多模态必要性、同预算近邻基线、root-independent discovery 和跨家族迁移，创新性有机会提高到约 **7/10**。

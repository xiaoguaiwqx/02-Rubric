# 论文笔记：CritiQ: Mining Data Quality Criteria from Human Preferences

> 重点关注“少量人类偏好 -> 可解释质量标准 -> 大规模数据选择”这条技术链路。

---

## 基本信息

- **论文标题**: CritiQ: Mining Data Quality Criteria from Human Preferences
- **作者**: Honglin Guo, Kai Lv, Qipeng Guo, Tianyi Liang, Zhiheng Xi, Demin Song, Qiuyinzhe Zhang, Yu Sun, Kai Chen, Xipeng Qiu, Tao Gui
- **会议/期刊**: ACL 2025 Long Paper
- **年份**: 2025
- **引用数**: 0
- **综合评分**: 8.8/10
- **arXiv链接**: https://arxiv.org/abs/2502.19279
- **代码链接**: https://github.com/KYLN24/CritiQ
- **ACL链接**: https://aclanthology.org/2025.acl-long.792/
- **DOI**: 10.18653/v1/2025.acl-long.792
- **阅读日期**: 2026-03-24

---

## 一句话总结

> CritiQ 用大约 30 对人工偏好比较样本，借助“管理者-执行者”多智能体流程自动挖掘可解释的数据质量标准，再把这些标准蒸馏成轻量评分器，实现低标注成本、可解释、可扩展的数据筛选。

---

## 核心问题

**要解决什么问题？**
- 如何在尽量少的人工标注成本下，得到与人类偏好一致、跨领域可迁移、还能扩展到大规模语料打分的数据质量标准。

**为什么这个问题重要？**
- LLM 的持续预训练和指令训练都高度依赖高质量数据，但“什么是高质量”本身难以手写清楚。
- 手工启发式、PPL、分类器、prompt-based judge 都有明显偏置，要么依赖专家经验，要么依赖昂贵大模型，要么解释性弱。
- 对你当前的研究主题“数据选择 in MLLMs/LLMs”来说，这篇论文的价值不只是筛数据，而是**把“质量定义”本身当成可学习对象**。

**现有方法的局限性是什么？**
- **手工规则**: 规则脆弱，跨领域泛化差，隐藏设计者偏置。
- **困惑度/现成模型打分**: 本质依赖已有模型偏好，且难解释为什么某样本更优。
- **训练分类器**: 需要较大标注集，且标签空间通常是预先定义好的。
- **LLM prompt 直接判分**: 仍然需要手写 prompt 和标准，且成本高、不稳定。
- **固定单一准则**: 如仅看 educational value，容易遗漏质量的多维度本质。

---

## 主要贡献

1. 提出 **CritiQ Flow**，从极少量人类偏好对中自动演化出文本质量标准，而不是先由研究者写死标准。
2. 构建了一个来自既有数据集论文的 **criteria knowledge base**，为不同领域提供质量标准初始化。
3. 用多 criterion 的 pairwise judgment + majority voting 获取更稳的人类偏好近似，再训练 **CritiQ Scorer** 做大规模高效打分。
4. 在 code、math、logic 三个领域上证明：自动挖掘出的 verbal criteria 比手工单标准或直接 prompt 优势更明显。
5. 通过持续预训练验证筛出的数据确实更好，而不只是“更像人工偏好测试集”。

---

#### 算法流程
#### 符号与整体流程

先把论文里的对象统一一下，后面更容易看清每一步到底在优化什么：

- `D`：待筛选的原始语料库。
- `D_human`：约 30 对人工偏好样本，用来“学标准”，不是直接学 scorer。
- `C_knowledge`：从已有数据集论文中抽出的质量标准知识库。
- `C_domain`：针对当前领域（code/math/logic）从知识库检索出来的候选标准。
- `C_final`：经过多轮 evolution 后保留下来的最终标准集合。
- `D_agent`：由 worker agents 用 `C_final` 自动标注的更大规模 pairwise 数据，用来训练 scorer。

整个方法不是“人工标签 -> 直接训练一个质量分类器”，而是两阶段：

```text
阶段 1：从 D_human 中挖掘和优化 criterion
阶段 2：用最终 criterion 自动生成 D_agent，再训练轻量 scorer
```

它的技术主线可以概括为：

```text
1. 采样并人工标注少量偏好对，得到 D_human。
2. 用知识库 + manager 初始化一组质量标准。
3. 用单标准 worker 对 D_human 做 pairwise judgment。
4. 依据 accuracy 对标准做保留 / 删除 / 反思修改。
5. 多轮迭代后得到 C_final。
6. 用 C_final 自动标注 25k 对样本，形成 D_agent。
7. 用 Bradley-Terry 风格 scorer 学习单样本分数函数 s_theta(d)。
8. 对全量语料打分并按温度采样得到高质量子集。
```

#### 3.2 Knowledge Base：知识库如何初始化标准

这一部分的目标是避免 manager 从零开始“拍脑袋”定义高质量标准。

具体流程：

1. 从 Hugging Face Hub 的数据集出发，抓取其引用论文。
2. 只保留有 HTML 版本的 arXiv 论文，避免 PDF 解析噪声。
3. 用 `GPT-4o-mini` 根据标题和摘要筛出真正介绍 dataset 的论文。
4. 再用 `GPT-4o-mini` 从这些论文中系统抽取质量标准，形成候选 criterion。
5. 对 criterion 名称做归一化，用 `Jaccard similarity > 0.3` 去重。
6. 最终得到 `342` 条 distinct quality criteria，构成 `C_knowledge`。

当面对某个具体领域的数据集时，系统不是直接把 `C_knowledge` 整库喂进去，而是：

- 先根据领域描述检索出更相关的 `C_domain`
- 再在 `D_human` 上测试这些 criterion 的效果
- 只保留 `acc_i > 0.5` 的 criterion 作为初始化候选
- 如果数量不够，再让 manager 额外生成新标准

这个设计的本质是“先用已有研究给出先验，再用当前数据做局部校正”。知识库的价值不在于给出最终答案，而在于：

- 减少 cold start 难度
- 避免 manager 一开始生成过于空泛的 criterion
- 把既有 dataset 研究中反复出现的 quality signal 变成可复用模块

#### 3.3 Multi-Criteria Pairwise Judgment：多标准成对判断

给定一个样本对 `p = (textA, textB)` 和一组标准 `C = {c_1, c_2, ..., c_n}`，CritiQ 不让一个 judge 一次性综合所有因素，而是拆成多个 worker：

- 每个 worker 只负责一个 criterion `c_i`
- 每个 worker 的输出是 `A / B / null`

这里 `null` 很关键，表示：

- 当前 criterion 对这个 pair 不适用
- 或者 A 和 B 在该 criterion 下难分高下

所以每条 criterion 允许“高精度但不全覆盖”，而不是被迫在所有样本上都硬判。

最终对一个 pair 的整体判断是多数投票：

```text
judge(p, C) = majority over { worker_i(p, c_i) }
```

更细地说，每个 worker 实际做的是：

1. 阅读 criterion 的名字和描述。
2. 在该 criterion 下分析 A 和 B。
3. 如果可比较，则输出更优者。
4. 如果不适用或势均力敌，则输出 `null`。

这样设计的好处是：

- 降低单 prompt 过载，单个 worker 的任务更简单。
- 每条标准都变成可解释、可调试的局部视角。
- 后面可以精确知道“哪条标准好、哪条标准差”，为 evolution 提供反馈。

#### 3.4 Criteria Evolution：标准如何被自动优化

这一部分是 CritiQ Flow 的核心优化器。它优化的不是模型权重，而是 **criterion 的自然语言描述**。

先对每条 criterion `c_i` 在 `D_human` 上计算 accuracy：

- 只统计该 criterion 给出有效判断的 pair
- `null` 不记为对，也不记为错

也就是说：

```text
1. 从目标数据集 D 中抽取约 30 对样本，请人工标注每对中谁质量更高，得到 D_human。
2. 从知识库中按领域检索初始 criteria；不够时由 manager agent 生成新 criteria。
3. 对每个 criterion 分配一个 worker agent，让其在该 criterion 下对 D_human 做 pairwise judgment。
4. 统计每个 criterion 的 accuracy：
   - 高于高阈值：保留；
   - 低于低阈值：删除，并记录到 memo，避免重复生成；
   - 介于两者之间：manager 根据错例和 worker reasoning 做 reflection/refinement。
5. 重复迭代若干轮，保留全流程中 accuracy 最好的 criteria，形成 final criteria。
6. 用 final criteria 在更大的随机 pair 集 D_agent（25k 对）上自动打偏好标签。
7. 训练 CritiQ Scorer（Bradley-Terry 风格轻量打分模型）。
8. 用 scorer 对全量语料打分，再按 exp(score / tau) 进行温度采样，而非简单 top-k。
acc_i = 正确判断数 / 有效判断数
```

作者更关注的是“这条标准在适用时是否准确”，而不是“它是否覆盖全部样本”。

然后根据两个阈值分三类：

- `acc_i >= t_high`：直接保留
- `acc_i <= t_low`：删除，并写入 bad-criterion memo，避免 manager 反复生成同类差标准
- `t_low < acc_i < t_high`：交给 manager 做 reflection/refinement

其中 reflection 的输入包括：

- 当前 criterion
- 它判错的样本对
- 人类正确答案
- worker 在该 criterion 下的 reasoning

manager 要做的不是重判这对样本，而是分析：

- 为什么这个 criterion 会误导 worker
- 应该如何修改 criterion 描述，让它更精确、更符合当前领域

修改后的新版本记为 `c_i'`。为了防止文本优化越改越差，论文加了一个非常重要的约束：

```text
只有当 acc_i' >= acc_i 时，才接受这次修改
```

所以 CritiQ 的 evolution 更像一个受验证集约束的离散搜索过程，而不是无约束 prompt 改写。

最终输出的不是“最后一轮的标准”，而是整个演化轨迹中表现最好的那些 criterion，组成 `C_final`。

#### 3.5 Train the Scoring Model：从 pairwise preference 到单样本分数

得到 `C_final` 后，CritiQ 不再继续用昂贵的 manager，而是让 worker 依据这些标准去自动标注更大规模的 pair，形成 `D_agent`。

这里有两个重要细节：

1. pair 是从 `D` 中随机采样得到的。
2. 采样前会先按文本长度分组，减少 worker 的长度偏置。

之后训练 `CRITIQ Scorer`。它初始化自 `Qwen2.5-1.5B`，目标是学习一个单样本分数函数：

```text
s_theta(d) ∈ R
```

对一对偏好数据 `(d_high, d_low)`，它不直接输出 A/B 类别，而是分别给两条文本打分：

- `s_theta(d_high)`
- `s_theta(d_low)`

然后通过 Bradley-Terry 风格的 pairwise loss 训练。等价写法是：

```text
P(d_high ≻ d_low) = sigmoid(s_theta(d_high) - s_theta(d_low))
L = -log sigmoid(s_theta(d_high) - s_theta(d_low))
```

这个设计的直觉是：

- 高质量文本应该得到更高分
- 不需要人为规定“绝对质量分数是多少”
- 只要保证在大量 pair 上，高质量样本的分数 consistently 高于低质量样本

因此，Qwen2.5-1.5B 在 scorer 里输出的是 **一个具体实数分数**，不是直接输出概率。概率只是在训练时对“分数差”做 `sigmoid` 得到。

结合论文说他们使用 `trl` 训练 scorer，可以把这一阶段理解成标准 reward-model / sequence-level ranking 训练：

- backbone：Qwen2.5-1.5B
- 输出：单条文本的 scalar score
- 监督：`D_agent` 中的 chosen / rejected pair
- 验证指标：pairwise validation accuracy

#### 3.6 Selecting Data：如何从分数变成最终子集

训练完 scorer 后，系统对整个语料库 `D` 中每条文本 `d_i` 计算：

```text
s_theta(d_i)
```

然后把这些原始分数归一化得到最终质量分数 `s_i`。作者没有用硬性的 top-k 截断，而是采用温度采样：

```text
p_i ∝ exp(s_i / τ)
```

其中 `τ` 是温度。这样做的原因是：

- 如果直接 top-k，很容易把数据分布挤压到极少数模式
- 温度采样能保留“高质量偏置”，同时减少模式坍缩
- 这和 QuRating 的 reward-weighted regression 视角是一致的

为了高效实现“无放回采样”，作者使用了 **Gumbel top-k trick**。

所以最终的数据选择不是：

```text
找出分数最高的若干条
```

而是：

```text
按照质量分数诱导出的概率分布，偏向高分样本地进行采样
```

这一步很重要，因为它体现出作者并不把质量当成唯一目标，还隐含地保留了多样性。

#### 关键公式
- **多标准投票判定**:

- **多标准投票判定**：
  - 对每个 pair `(A, B)`，每个 worker 在单一 criterion 下输出 `A / B / null`。
  - 最终结果由所有 criteria 的多数投票得到。
- **Criterion accuracy**:
  - 只在该 criterion 给出有效判断时计算，不把 `null` 情况算入错误。
  - 这意味着作者更关注“criterion 在适用时是否准”，而不是“是否覆盖所有样本”。
- **Scorer 训练**:
  - 使用 Bradley-Terry 偏好建模，将 pairwise preference 转换为单样本分数。
  - 本质上学习 `s(d_high) > s(d_low)` 的排序约束。
- **最终采样**:
  - 不是 deterministic top-k，而是按 `p_i ∝ exp(s_i / τ)` 采样。
  - 这和 QuRating 一脉相承，优点是保留高质量偏置的同时避免过度模式坍缩。
- **Criterion accuracy**：
  - 只在该 criterion 给出有效判断时计算，不把 `null` 算入错误。
  - 本质上度量“criterion 在适用时是否准确”。
- **Bradley-Terry scorer**：
  - `P(d_high ≻ d_low) = sigmoid(s_theta(d_high) - s_theta(d_low))`
  - `L = -log sigmoid(s_theta(d_high) - s_theta(d_low))`
- **最终采样分布**：
  - `p_i ∝ exp(s_i / τ)`
  - 目的不是硬截断，而是在质量和分布覆盖之间做平衡。

#### 架构图
> 整体结构可以概括为三层：
> 1. **知识初始化层**：从 dataset paper 中抽 criteria，构成知识库；
> 2. **标准演化层**：manager 负责生成/反思/修改 criterion，worker 负责单 criterion 比较；
> 2. **标准演化层**：manager 负责生成、反思和修改 criterion，worker 负责单 criterion 比较；
> 3. **扩展打分层**：用演化后的 criteria 生成大规模 pair 标签，再训练轻量 scorer 做全量数据选择。

![alt text](CritiQ.png)
![CritiQ pipeline](assets/CritiQ.png)

---

## 实验设置

### 数据集
- **Code**: Stack v2 的 Python 子集
- **Math**: OpenWebMath 的非代码子集
- **Logic**: Zyda-2

### 标注与数据规模

| 领域 | #D_human | #D_agent | #D_test |
|---|---:|---:|---:|
| Code | 40 | 25,000 | 178 |
| Math | 30 | 25,000 | 70 |
| Logic | 30 | 25,000 | 134 |

- `D_human` 用于挖掘和演化 criteria。
- `D_agent` 用于训练 CritiQ Scorer。
- `D_test` 是三位标注者一致同意的高置信测试对。

### 基线方法
- **Vanilla**: 不给 criterion，直接 prompt worker 做质量比较。
- **TextGrad**: 文本优化方法，尝试自动优化 prompt。
- **QuRating 单一标准**:
  - Writing Style
  - Facts & Trivia
  - Educational Value
  - Require Expertise
- **DRPO**: 作为额外 text-based optimization baseline，仅补充在 code 域实验。

### 评估指标
- **人类偏好一致率（accuracy on D_test）**：衡量方法是否真正对齐人类质量判断。
- **持续预训练后的下游性能**：验证筛出来的数据是否真的能改善模型能力，而不只是 overfit judge。

### 实验环境
- **Manager agent**: GPT-4o（文中注明版本 `gpt-4o-2024-11-20`）
- **Worker agent**: Qwen2.5-72B-Instruct
- **CritiQ Scorer base**: Qwen2.5-1.5B
- **Flow 超参数**:
  - 各领域初始 `#Criteria = 20`
  - Code / Math / Logic 的迭代次数分别为 `3 / 5 / 3`
- **Continual pretraining**:
  - 基座模型：Llama-3.2-3B
  - 训练 4 epochs
  - 序列长度 8192
  - 全局 batch size 4M tokens
  - 使用 32 张 NVIDIA H800

---

## 实验结果

### 主要结果

**1. CritiQ Flow 显著优于 vanilla / TextGrad / 单一手工标准**

| 方法 | Code | Math | Logic | Avg. |
|---|---:|---:|---:|---:|
| Vanilla | 82.02 | 72.86 | 72.99 | 75.96 |
| TextGrad | 72.70 | 78.57 | 75.22 | 75.50 |
| QuRating 最优单标准 | 85.39 | 68.57 | 84.33 | 79.43 |
| **CritiQ Flow** | **89.33** | **84.57** | **88.06** | **87.32** |
| **CritiQ Scorer** | **89.89** | **90.00** | **90.22** | **90.04** |

关键观察：
- CritiQ Flow 平均比 vanilla 高 **11.36** 个点。
- CritiQ Scorer 比 Flow 还高，说明“criteria -> scorer”的蒸馏没有丢失能力，反而提升了泛化。
- QuRating 的单一标准在某个域偶尔有效，但整体不稳定，验证了“质量是多维而非单维”的判断。

**2. 消融证实 knowledge base 和 evolution 都有价值**
- `w/o evo.` 平均 83.46
- `w/o k.b.` 平均 83.80
- `w/o evo. & k.b.` 平均 75.89，几乎退回 vanilla

这说明：
- 知识库不是装饰，而是让初始化站在已有研究肩膀上；
- reflection/evolution 也不是噱头，而是真正把 criterion 从“泛化描述”收紧到“领域可执行标准”。

**3. 持续预训练结果证明筛到的是“更有训练价值的数据”**

- **Code**
  - Raw 平均：38.80 / 32.38
  - Stack uniform：44.16 / 36.87
  - QR-Edu：47.93 / 39.84
  - **CRITIQ：53.88 / 40.98**
- **Math**
  - Raw：22.70
  - OWM uniform：22.19
  - QR-Edu：22.02
  - **CRITIQ：26.04**
- **Logic**
  - Raw：32.66
  - Zyda-2 uniform：30.06
  - QR-Edu：31.24
  - **CRITIQ：34.36**

这部分非常关键，因为它回答了一个常见质疑：
- “你只是把 judge 偏好的数据挑出来了，真的更利于训练吗？”
- 作者的答案是：**是的，至少在 continual pretraining 场景下，的确转化成了下游能力收益。**

### 消融实验
- **Majority voting** 很重要：
  - Ours：87.32
  - w/o voting：83.51
- 这说明“把所有标准塞进一个 prompt”不如“单标准拆开判 + 投票融合”。

### 可视化分析
- criterion 的 accuracy 分布会随着迭代整体右移，说明 evolution 确实在把标准推向更高质量区域。
- refuse rate 大多较低，但部分高精度 criterion 在适用范围窄时也有保留价值。

---

## 优点

1. **问题定义很强**：不是学一个黑盒质量分，而是学习“质量标准”本身，解释性明显优于常见打分器。
2. **标注效率高**：只需要约 30 对人工偏好样本就能启动，这是这篇工作的最大卖点之一。
3. **工程链路完整**：从 criteria mining、agent annotation、scorer training 到最终 data selection，闭环很完整。
4. **实验设计合理**：既测 human alignment，也测 downstream pretraining performance，两条证据链比较扎实。
5. **和已有工作形成互补**：相比 QuRating 的手工 criteria、相比纯 reward/ppl/filtering 方法，CritiQ 更强调“标准自动发现”。

---

## 局限性

1. **领域仍然偏窄**：只在 code / math / logic 三类高结构化文本上验证，尚未证明在开放域网页文本上同样成立。
2. **人工偏好本身仍有主观偏差**：作者承认质量定义仍然带主观性，只是把偏差从“研究者手写规则”转移到“标注者偏好”。
3. **annotator 设置不够独立**：标注者是作者本人，虽然有专业性，但会引入潜在确认偏差。
4. **成本仍不算低**：Flow 运行时 code 域用了大量 worker token；只是相对“全量人工/全量 GPT-4 judge”更便宜。
5. **scorer 依赖 agent 标注扩展集**：如果 worker 在某些局部区域系统性偏差，scorer 会把偏差进一步蒸馏和放大。

---

## 启发与思考

### 对我的研究的启发
- 这篇论文把“数据质量标准”变成**可挖掘对象**，这对你的 survey 很有价值：它不是传统的 sample scoring，而是 **criteria induction for data selection**。
- 对 MLLM 数据选择来说，可以借鉴成：
  - 从少量人类偏好对中自动归纳“多模态高质量标准”；
  - 再用这些标准去构建便宜的 proxy scorer；
  - 最后大规模选图文对或指令样本。
- 它和你当前收集的 CoIDO、PreSel、Concept-skill transferability 形成互补：
  - 那些工作更偏“如何选样本”；
  - CritiQ 更偏“如何定义好样本”。

### 可能的改进方向
1. 把 criterion 从纯文本描述扩展为 **层级化 taxonomy**，区分 domain-level、task-level、sample-level 标准。
2. 不只用 pairwise preference，可以引入 **listwise preference** 或 **active selection**，减少人类标注浪费。
3. 在 scorer 训练时显式建模 **criterion disagreement**，而不是把所有 worker judgment 都压成单一偏好标签。
4. 对多模态场景引入图像相关 criterion，如视觉信息密度、图文一致性、问题可回答性、OCR/布局复杂度等。
5. 把静态一次性 data selection 推进为 **curriculum-aware dynamic selection**，让标准随训练阶段变化。

### 可以借鉴的技术
- **多 criterion 分解**：不要把“高质量”塞成一个 prompt。
- **manager-worker + reflection**：适合做标准挖掘，而不只是答案生成。
- **知识库初始化 + 演化**：先借鉴既有论文，再针对目标数据做局部适配。
- **pairwise -> scorer 蒸馏**：用高成本 judge 生成偏好，再蒸馏成低成本全量筛选器。
- **温度采样替代 top-k**：避免极端集中到少数模式。

### 值得进一步探索的问题
- “高质量标准”是否会随着模型规模改变？
- 不同训练阶段是否需要不同 criteria？
- criterion 的可迁移单位到底是 domain、task 还是 capability？
- 多模态数据里，文本标准和视觉标准如何联合演化？
- 能否从 model learning dynamics 中反向验证某条 criterion 是否真的带来增益？

---

## 相关论文

### 引用的重要论文
1. **QuRating: Selecting High-Quality Data for Training Language Models**
   - 提供了手工多标准质量评估视角，是 CritiQ 最直接的对照系。
2. **Wettig et al., 2024 / Korbak et al., 2023**
   - 支撑其 pairwise preference -> reward/scorer 的建模方式。
3. **TextGrad**
   - 作为文本优化 baseline，帮助说明“普通 prompt 优化”不等于“有效质量标准挖掘”。

### 被引用的后续工作
1. 待后续专门追踪
2. 待后续专门追踪
3. 待后续专门追踪

### 同期相关工作
1. **QuRating**
2. **Clustering and Ranking (CaR)**
3. **Superfiltering / LESS / DATAMAN / ADO / Nemotron-CLIMB**

---

## 代码与资源

- **官方代码**: https://github.com/KYLN24/CritiQ
- **论文 PDF**: `arxiv_papers/2502.19279v3.pdf`
- **解析 Markdown**: `arxiv_papers/2502.19279v3.md`
- **ACL PDF**: https://aclanthology.org/2025.acl-long.792.pdf
- **数据来源**:
  - Stack v2 (Python)
  - OpenWebMath (non-code)
  - Zyda-2
- **预训练模型**:
  - Qwen2.5-72B-Instruct
  - Qwen2.5-1.5B
  - Llama-3.2-3B

---

## 笔记标签

`#数据选择` `#LLM` `#偏好学习` `#质量标准挖掘` `#agent-based-filtering` `#pairwise-ranking`

---

## 阅读进度

- [x] 第一遍：快速浏览（摘要、引言、结论）
- [x] 第二遍：深入阅读（方法、实验）
- [x] 第三遍：批判性思考（局限性、改进）
- [ ] 代码复现
- [ ] 实验验证

---

## 更新日志

- **[2026-03-16]**: 建立初版笔记，记录论文定位与一句话总结。
- **[2026-03-24]**: 基于 `2502.19279v3` 和 ACL 2025 正式版本补全方法细节、实验结果、局限性与研究启发。
- **[待后续]**: 若继续做 survey，可把 CritiQ 单独归为“criteria mining / preference-driven data selection”一类。

# Evolving Structured Rubrics from Multimodal Preferences

> 版本：2026-07-24

---

## 1. 问题定义

给定多模态偏好数据集 $\mathcal{D} = \{(I_i, Q_i, a_i^+, a_i^-)\}$。

**目标**：设计一套演化机制，使量规 $\mathcal{R} = \{c_1, \ldots, c_K\}$ 从少量种子出发，在偏好判断准确率的驱动下逐步生长、分化和精简。标准之间的级联关系是演化过程的**自然产物**，而非直接优化的目标——正如实验中观察到的那样，系统从未被要求生成层级，但标准之间自发涌现了前置依赖。

---

## 2. 核心概念

### 2.1 标准 (Criterion)

一条标准 $c$ 由两部分构成：

- **描述 (Description)**：一段自然语言，声明该标准评判什么、如何判断、在什么条件下不适用。
- **示例 (Examples)**：至少 1 个偏好对及正确的判断结果，用于在推理时锚定 VLLM 的判断行为。

**示例**：

```
[描述]
评估回答中明确的空间关系主张（如"在左侧""重叠""在后方"）
是否与图像中至少两个已确认物体的实际布局一致。
仅在两个候选回答都通过视觉依据一致性检查后才适用。

[示例]
图像：一张办公桌照片，电脑在左侧，水杯在右侧
问题：描述桌上的物品摆放
回答a⁺：电脑在水杯的左侧
回答a⁻：水杯在电脑旁边
判断：偏好 a⁺（a⁺ 明确描述了左右关系且与图像一致）
```

### 2.2 量规 (Rubric)

量规 $\mathcal{R} = \{c_1, \ldots, c_K\}$ 是一组标准的集合。

推理时，对每个样本 $(I, Q, a^+, a^-)$，每条标准独立判断——偏好 $a^+$、偏好 $a^-$、或不适用。最终偏好由所有适用标准的加权多数投票决定，权重为各标准的历史准确率。

### 2.3 性能指标

- **适用样本集**：$\mathcal{S}(c) = \{s \in \mathcal{D} \mid c(s) \neq \text{NA}\}$
- **准确率**：$\text{Acc}(c) = \frac{|\{s \in \mathcal{S}(c) \mid c(s) = p_s\}|}{|\mathcal{S}(c)|}$，$p_s$ 为样本 $s$ 的人类偏好
- **覆盖率**：$\text{Cov}(c) = |\mathcal{S}(c)| / |\mathcal{D}|$
- **判断向量**：$\mathbf{v}_c \in \{-1, 0, +1\}^{|\mathcal{D}|}$（$+1$ = 偏好 $a^+$，$-1$ = 偏好 $a^-$，$0$ = 不适用）
- **长度**：$\text{len}(c)$ = 标准描述的字符数

---

## 3. 适应度

### 3.1 适应度公式

$$\text{Fitness}(c) = \text{Acc}(c) - \lambda \cdot (e^{\alpha \cdot L_{\text{norm}}(c)} - 1) \cdot \text{Cov}(c)$$

其中 $L_{\text{norm}}(c) = \text{len}(c) / \text{len}_{\max}$，$\alpha$ 控制陡峭度（典型值 2.0），$\lambda$ 控制整体强度（典型值 0.1）。

**效果**：覆盖越广的标准，同样的长度受到的惩罚越重——粗准则应简洁，专家准则可以更详细。


### 3.2 生存门槛与全局参数

以下两个参数同时服务于 Drop 和 Create，构成统一的"质量标准"：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| $\tau_{\text{acc}}$ | 一条标准在适用样本上的最低可接受准确率 | 0.55 |
| $N_{\min}$ | 判断准确率所需的最小样本数（低于此数则准确率估计不可靠） | 30 |

这两个参数在不同层面的应用：

- **准则层面（Drop）**：连续 $k$ 轮 $\text{Acc}(c) \le \tau_{\text{acc}}$ 或 $|\mathcal{S}(c)| < N_{\min}$，则 $c$ 被淘汰。
- **量规层面（Create）**：定义空缺域 $\mathcal{D}_{\text{gap}} = \{s \in \mathcal{D} \mid \forall c \in \mathcal{R},\ \text{Acc}_s(c) \le \tau_{\text{acc}}\}$——所有现有标准在该样本上均不达标。若 $|\mathcal{D}_{\text{gap}}| \ge N_{\min}$，说明存在一个未被覆盖的评价维度，触发 Create。（**定义有问题，使用一个准则c去判断样本s为什么会有准确率，不应该表示对不对吗？**）

### 3.3 冗余度

用于触发 Merge，不进入适应度。两条标准的判断向量均为 $\{-1, 0, +1\}$ 三元值，在共同适用样本集 $\mathcal{S}_{ij} = \mathcal{S}(c_i) \cap \mathcal{S}(c_j)$ 上计算汉明距离：

$$\text{Agreement}(c_i, c_j) = 1 - \frac{1}{|\mathcal{S}_{ij}|} \sum_{s \in \mathcal{S}_{ij}} \mathbb{I}[\mathbf{v}_{c_i}[s] \neq \mathbf{v}_{c_j}[s]]$$

若 $\text{Agreement}(c_i, c_j) > \tau_{\text{merge}}$（如 0.85），触发合并。

---

## 4. 演化算子

演化算子是指**直接改变量规结构的操作**。本节先枚举场景，再定义算子，最后说明算子依赖的公共机制。

### 4.1 场景枚举

| # | 场景 | 算子 |
|---|------|------|
| A | 缺少某个评价维度：存在样本子集，所有现有标准的准确率均低于 $\tau_{\text{acc}}$ | **Create** |
| B | 标准覆盖率大但准确率不高——覆盖太广导致无法精确，错误可聚类为 $\ge 2$ 个语义类别 | **Split** |
| C | 标准通过了生存门槛，准确率仍有提升空间，但覆盖率不足以触发 Split——问题出在措辞精度而非覆盖粒度过粗 | **Refine** |
| D | 两条标准在共同适用样本上投票高度一致 | **Merge** |
| E | 标准连续多轮准确率低于生存门槛 | **Drop** |

### 4.2 算子定义

#### Create

**触发**：$|\mathcal{D}_{\text{gap}}| \ge N_{\min}$（$\mathcal{D}_{\text{gap}}$ 定义见 §3.2）。

**操作**：LLM 从 $\mathcal{D}_{\text{gap}}$ 中归纳一条新标准（含描述和示例）。在 $\mathcal{D}_{\text{gap}}$ 上 $\text{Acc} > \tau_{\text{acc}}$ 则加入 $\mathcal{R}$，否则丢弃。

#### Split

**触发**：$\text{Acc}(c) < \tau_{\text{split}}$ 且 $\text{Cov}(c) > \tau_{\text{cov}}^{\text{high}}$——准确率不高且覆盖很广，说明标准试图覆盖过多异质场景。LLM 将 $c$ 的错误样本聚类为若干语义类别，为每个类别生成一条子标准（含描述和示例）。

**竞争**：子标准集在 $\mathcal{S}(c)$ 上的集体适应度超过 $c$，则替换 $c$；否则回退并记录 split 失败历史。

#### Refine

**触发**：$\tau_{\text{acc}} < \text{Acc}(c) < \tau_{\text{refine}}$，且 $\text{Cov}(c) \le \tau_{\text{cov}}^{\text{high}}$——准确率有提升空间，但覆盖不广，不适合 Split。问题出在措辞而非粒度过粗。

**操作**：LLM 根据错误样本改写描述。精炼版 $c'$ 与 $c$ 比较适应度——注意这里比的是适应度而不仅是准确率，因此措辞变长会触发惩罚。$c'$ 适应度更高则替换，否则回退。

#### Merge

LLM 将 $c_i$ 和 $c_j$ 合并为一条泛化标准 $c_{\text{merged}}$。在 $\mathcal{S}(c_i) \cup \mathcal{S}(c_j)$ 上适应度不低于原两条的加权平均，则替换；否则保留原标准对。

#### Drop

**触发**：连续 $k$ 轮不满足生存门槛（$\text{Acc} \le \tau_{\text{acc}}$ 或 $|\mathcal{S}| < N_{\min}$）。

**操作**：从 $\mathcal{R}$ 中移除。描述写入历史记录以防重复探索。

### 4.3 触发条件与竞争判定

各算子对"何时触发"和"如何判定成功"使用不同的度量，形成两层筛选：

| 算子 | 触发条件（决定是否执行） | 竞争判定（决定是否接受结果） |
|------|------------------------|---------------------------|
| **Create** | $\|\mathcal{D}_{\text{gap}}\| \ge N_{\min}$ | $\text{Acc}(c_{\text{new}}) > \tau_{\text{acc}}$ |
| **Split** | $\text{Acc}(c) < \tau_{\text{split}}$ 且 $\text{Cov}(c) > \tau_{\text{cov}}^{\text{high}}$ | $\text{Fitness}(\{c_1,\ldots\}) > \text{Fitness}(c)$ |
| **Refine** | $\tau_{\text{acc}} < \text{Acc}(c) < \tau_{\text{refine}}$ 且 $\text{Cov}(c) \le \tau_{\text{cov}}^{\text{high}}$ | $\text{Fitness}(c') > \text{Fitness}(c)$ |
| **Merge** | $\text{Agreement}(c_i, c_j) > \tau_{\text{merge}}$ | $\text{Fitness}(c_{\text{merged}}) \ge$ 原两条加权平均 |
| **Drop** | 连续 $k$ 轮 $\text{Acc} \le \tau_{\text{acc}}$ 或 $\|\mathcal{S}\| < N_{\min}$ | — |

**设计理由**：触发条件用准确率和结构条件（错误聚类、一致性、空缺域）——它们是"是否需要干预"的信号。竞争判定用适应度——它是"干预是否真的更好"的最终度量。适应度中的长度惩罚在竞争时生效：一条更准确但大幅变长的精炼版本可能因惩罚而竞争失败，从而被回退。

---

## 5. 演化流程

```
输入：偏好数据集 D，T_max = 10，收敛容忍 s = 3
输出：量规 R

1. 初始化
   - LLM 从 D 中随机采样 50-100 对偏好数据，归纳 3-5 条初始标准
   - 在 D 全集上评估，淘汰不满足生存门槛的

2. 循环至收敛：
   a. 评估
      - 每条标准 c ∈ R：计算 Acc(c), Cov(c), len(c), Fitness(c)
      - 检查生存门槛（Acc > τ_acc 且 |S| ≥ N_min），标记不达标的标准
      - 所有标准对：计算 Agreement
      - 检测 D_gap（所有标准 Acc 均 < τ_acc 的样本子集）

   b. 演化（每轮按以下顺序执行。触发用 Acc/Cov/Agreement，竞争用 Fitness）：
      - Drop     ← Acc 不满足生存门槛连续 k 轮 → 移除
      - Merge    ← Agreement > τ_merge → 合并（Fitness 判定）
      - Create   ← |D_gap| ≥ N_min → 创建新标准（Acc 判定）
      - Split / Refine（并行分支，每条标准走其中一条）：
          · Acc < τ_split 且 Cov > τ_cov^high → Split（Fitness 判定）
          · τ_acc < Acc < τ_refine 且 Cov ≤ τ_cov^high → Refine（Fitness 判定）

   c. 收敛检查
      若连续 s=3 轮无变化，或达到 T_max，停止

3. 输出 R
```

**算子顺序**：Drop 最先（清除无效标准），Merge 次之（消除冗余，避免 Split 重复分裂等价标准），Create 在 Split/Refine 之前（先填补空缺维度以提供完整竞争参照）。Split 和 Refine 是并行分支——覆盖率高的走 Split（覆盖太广需要细分），覆盖率低的走 Refine（措辞精度不够需要改写），每条标准每轮只走其中一条。

---

## 6. 级联图的形成

Split 是级联关系形成的唯一来源。

当 $c$ 分裂为子标准时，子标准的适用域天然是 $\mathcal{S}(c)$ 的子集——LLM 被要求基于 $c$ 的错误聚类来生成更精确的判断逻辑，这些聚类正分布在 $\mathcal{S}(c)$ 的特定子区域中。多轮 Split 后，不同分支的子标准适用域形成嵌套与偏序，构成一张**级联图 (Cascade Graph)**——形式上是 DAG，语义上编码了"先评估谁、再评估谁"的偏序关系。

级联图不是被显式推断或优化的。它是对 Split 历史记录的追溯：谁从谁分裂而来，适用域自然嵌套。推理时沿级联图自上而下评估——基础标准先于专家标准，父标准先于子标准。

---

## 7. 关键性质

- **粒度分化**：Split（细分）+ 竞争（子标准须超越父标准）使标准在不同粒度处停止演化，自然形成从粗到细的谱系。
- **质量单调**：Split 和 Refine 内嵌竞争，量规整体判断能力不会退化。
- **长度自约束**：长度惩罚与覆盖率耦合，粗标准自动简洁，专家标准保留细节空间。
- **Scenario-specific**：每条标准由多样本归纳而来，标准之间形成稳定的适用域关系。
- **无参数更新**：LLM 是"标准改写器"，演化的对象是自然语言文本。

---

## 8. 框架总结

| 算子 | 触发条件 | 竞争判定 | 粒度 |
|------|---------|---------|------|
| **Create** | $\|\mathcal{D}_{\text{gap}}\| \ge N_{\min}$ | $\text{Acc}(c_{\text{new}}) > \tau_{\text{acc}}$ | — |
| **Split** | Acc 偏低且 Cov 偏高 | Fitness(子标准集) > Fitness(父) | 变细 |
| **Refine** | Acc 偏低但 Cov 不高 | Fitness(精炼版) > Fitness(原版) | 不变 |
| **Merge** | Agreement > $\tau_{\text{merge}}$ | Fitness(合并版) ≥ 原加权平均 | 变粗 |
| **Drop** | 连续 k 轮 Acc ≤ τ_acc 或 |S| < N_min | — | — |

**竞争与回退**：Split、Refine、Merge 的结果通过适应度判定——适应度同时衡量准确率和长度，因此更准确但过度冗长的改写会被回退。Split 失败额外记录历史。

**初始化**：Seed 从偏好数据中归纳 3-5 条初始标准，是流程的启动步骤而非算子。

---

## 9. 待验证问题

1. **演化稳定性**：不同随机种子下，量规的规模和粒度分布是否收敛到相似结果？
2. **Split 数量**：不限制分裂数量是否导致碎片化？与固定 Split=2 相比如何？
3. **Seed 敏感性**：差的初始种子能否被后续 Create + Split + Refine 修复？
4. **长度惩罚的敏感度**：$\alpha$ 和 $\lambda$ 对最终标准的长度分布和准确率的影响？
5. **级联图质量**：演化产生的级联图，与标准描述中的前置依赖语言是否一致？

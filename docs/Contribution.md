把三点贡献组织成一个非常清晰的递进：
$$
\boxed{ \text{Learn what to judge} \rightarrow \text{learn how the judgment knowledge evolves} \rightarrow \text{learn how it is executed} }
$$
而不是再把贡献写成“我们提出 Split / Refine / hierarchy”这种机制罗列。

## Contribution 1 — 从偏好标签到可积累的评价经验：Evolving Reward Judge

这是**最大、最高层的 conceptual contribution**。

传统 preference-based reward modeling 通常把每条偏好看成一次监督：
$$
(I,Q,A,B,y) \rightarrow \text{learn } y
$$
我们的观点是，一条 preference 的价值不止在于告诉 Judge：

> A 比 B 好。

更重要的是，当当前 Judge 判断错误时，这个失败揭示了：

> **Judge 缺少什么评价知识？在哪方面偏好没有与人类对齐？在哪类场景下原有判断准则失效？**

因此我们提出 **Evolving Reward Judge**：

> 一个底层多模态模型参数保持固定，但能够从历史偏好判断的成功与失败中，持续积累、组织和修改显式评价知识的 reward judge。

它不是把经验全部吸收到模型参数中，而是保存为一个持续演化的 **Structured Rubric**：
$$
\mathcal R_t = \text{explicit evaluation memory at time }t
$$
并形成闭环：
$$
\boxed{ \text{Preference} \rightarrow \text{Judgment} \rightarrow \text{Failure Experience} \rightarrow \mathcal R_{t+1} \rightarrow \text{Future Judgment} }
$$
所以第一贡献真正想说的是：

> **我们把 preference 从一次性的监督标签，转化成可以被 Judge 持续复用的 evaluation experience。**

我认为这是 **Contribution #1，也是论文 headline contribution**。

------

## Contribution 2 — Failure-driven Rubric Evolution：让 Judge 从自己的系统性错误中长出专家知识

核心 **methodological contribution**。

重点不能简单写：

> We use Split and Refine to evolve rubrics.

这样太像 operator engineering。

真正有价值的是：

> **Rubric 的结构变化不是由 LLM 凭空产生或者是从单个偏好中发现，而是由 Judge 在真实多个偏好判断中的 preference experience 上反复出现的 failure pattern 驱动。**

流程是：
$$
\text{Judge failures} \rightarrow \text{ErrorSignatures} \rightarrow \text{semantic failure clusters} \rightarrow \text{specialized children}
$$
即，当一个宽泛 criterion 频繁出错时，我们不是简单重写它，而是问：

> 这些错误是不是实际上来自几个不同的评价子问题？

于是 **Split** 把不同 failure modes 转化为不同局部专家：

```text
Visual Grounding
├── object existence
├── spatial grounding
├── fine-grained attributes
└── ...
```

这意味着 hierarchy 是按照**错误模式**挖掘的，而不是预先人为设计 ontology。

然后 **Refine** 解决另一类问题：

> 这个专家方向是对的，但 applicability boundary / decision rule 还不够精确。

它根据新的错误经验继续修改 criterion 的：

- `Criterion focus`
- `Applicable only when`
- `Not applicable when`
- `Decision rule`

Split 产生**新的知识分支**，Refine 修正**已有知识边界**。

并且失败 proposal 也不会完全丢失，而会产生 failure attribution，进入下一轮 Manager memory，避免重复探索已经失败的方案。当前正式 Split 也已经采用固定 parent scope 上的 Specialized Accuracy 竞争，只有不劣于 parent 的完整 children 集合才被接纳，同时保留 parent fallback。

所以第二贡献可以概括成：

> **我们提出一种 failure-driven structural learning mechanism，使 Judge 能把重复错误抽象成新的局部评价专家，并利用后续失败继续修正它们的适用边界与决策能力。**

------

## Contribution 3 — Structured Reward Judging：不仅演化“判断什么”，还执行“如何判断”

如果只有前两点，会被 reviewer 理解成：

> “一个自动 rubric generation / refinement system。”

但 Rubric 不是生成完就作为一大段 prompt 塞进去直接 judge。

它真正被**执行**。

每个样本经过：
$$
x \rightarrow  \text{subtree evidence and reasoning} \rightarrow \text{cross-root conflict resolution} \rightarrow \text{final preference}
$$
因此：
$$
\boxed{ \text{Rubric knowledge} + \text{execution semantics} = \text{Reward Judge} }
$$
这是第三贡献的核心。

已经有一个非常关键的负消融：

> **把完整 Rubric 文本一次性交给模型，并不能替代结构化执行。**

S6 的结果说明 Rubric 的收益不仅来自 description 里“写了什么知识”，还来自：

- 局部 evidence 被隔离；
- 每棵 subtree 独立分析；
- 不同评价维度不会过早互相污染；
- 最终阶段显式解决跨 root 冲突。

文档因此明确保留了：

> ```
> 5 Unified-Subtree reports → Clean Global Arbiter
> ```

而不是把全部 criteria 压缩进一次判断。

这件事很重要，因为它让我们的方法从：
$$
\text{learned rubric}
$$
升级成：
$$
\boxed{ \text{learned + executable evaluation system} }
$$
此外，第三贡献还可以自然带上**泛化证据**：

Evolved Rubric 不只改善产生它的 Worker。Phase17 中，用 Qwen3-VL 的错误经验演化出的 Rubric，在保持 Rubric、数据、Prompt 和聚合方式不变、只将 Worker 换成 Qwen2.5-VL-7B 后，仍然获得明显收益。这个实验专门验证了这些评价规则并非只是针对原 Worker 的 prompt patches。

而执行结构也进一步做了 Qwen2.5 上的跨模型迁移验证。

因此第三点可以写成：

> **我们展示 structured rubric 必须作为一个可执行的 judgment system，而不仅是一组文本 criteria；这种显式评价知识及其执行机制能够迁移到未见 preference distribution，并在一定程度上迁移到不同 multimodal Workers。**

------



### **1. Evolving Reward Judge — 新的问题定义 / paradigm**

> Preferences are not merely labels; they are **evaluation experience**.

### **2. Failure-driven Structured Rubric Evolution — 核心学习机制**

> Systematic judgment failures become **new reusable evaluation expertise**.

ErrorSignature → Split → specialist children → Refine → failure memory，使评价体系根据真实错误逐渐专业化。

### **3. Structured Reward Judging — 执行与泛化**

> The evolved knowledge is not merely stored; it is **executed compositionally**.

局部 criterion → subtree evidence → global conflict resolution，并验证其外部分布与跨 Worker 迁移。



---

$$
\boxed{ \begin{aligned} \textbf{Experience} &: \text{从偏好失败中学习什么}\\ \downarrow\\ \textbf{Evolution} &: \text{评价知识如何生长和修正}\\ \downarrow\\ \textbf{Execution} &: \text{这些知识如何组合成最终判断} \end{aligned}}
$$

> **我们提出 Evolving Reward Judge：它不通过更新模型参数学习，而是将偏好判断中的失败经验积累成一个显式的结构化评价记忆，使评价准则能够持续分化、修正，并通过结构化执行产生后续奖励判断。**


## exp4: epoch=10

| 阶段                | train / valid             | heldout500          |
| ------------------- | ------------------------- | ------------------- |
| warmup              | -                         | **0.646 = 323/500** |
| iter 0              | train 0.556, val100 0.630 | -                   |
| iter 1              | train 0.633, val100 0.650 | -                   |
| iter 2              | train 0.600, val100 0.650 | -                   |
| iter 3              | train 0.578, val100 0.620 | -                   |
| iter 4              | train 0.656, val100 0.720 | -                   |
| iter 5              | train 0.611, val100 0.670 | -                   |
| iter 6              | train 0.611, val100 0.670 | -                   |
| iter 7              | train 0.578, val100 0.590 | -                   |
| iter 8              | train 0.589, val100 0.580 | -                   |
| iter 9              | train 0.567, val100 0.570 | -                   |
| final current       | train 0.511               | -                   |
| final best criteria | -                         | 0.696 = 348/500     |

**final 相对于 warmup 提升了5%。相对于Vanilla（0.598） 提高9.8%**

**Final Criterion Stats**

| criterion                            | train score | heldout acc | correct/app | coverage |
| ------------------------------------ | ----------- | ----------- | ----------- | -------- |
| visual_coherence                     | 0.810       | 0.821       | 142/173     | 0.346    |
| temporal_coherence                   | 0.857       | 0.807       | 176/218     | 0.436    |
| visual_ambiguity_resolution          | 0.694       | 0.750       | 180/240     | 0.480    |
| temporal_consistency                 | 0.600       | 0.749       | 173/231     | 0.462    |
| logical_consistency                  | 0.875       | 0.744       | 32/43       | 0.086    |
| multimodal_consistency               | 0.721       | 0.738       | 265/359     | 0.718    |
| factual_consistency                  | 0.600       | 0.712       | 321/451     | 0.902    |
| completeness                         | 0.634       | 0.688       | 319/464     | 0.928    |
| visual_grounding                     | 0.615       | 0.687       | 311/453     | 0.906    |
| contextual_fidelity                  | 0.735       | 0.686       | 321/468     | 0.936    |
| semantic_specificity                 | 0.792       | 0.742       | 118/159     | 0.318    |
| action_or_state_accuracy             | 0.773       | 0.704       | 107/152     | 0.304    |
| biological_or_realistic_plausibility | 0.867       | 0.694       | 75/108      | 0.216    |
| avoidance_of_hallucination           | 0.610       | 0.682       | 309/453     | 0.906    |
| multimodal_alignment                 | 0.683       | 0.679       | 315/464     | 0.928    |
| specificity                          | 0.792       | 0.670       | 301/449     | 0.898    |
| calibrated_uncertainty               | 0.648       | 0.654       | 270/413     | 0.826    |

- **挖掘出很多局部专家。**

- criterion 冗余。比如说：temporal_coherence和 temporal_consistency  

- 看日志：mid criterion 的改写越来越长，输入token数量激增，导致推理成本增加，推理速度变慢。[训练log.txt](.\论文复现\log.txt)

- 最后的 final criterion 位于：[final_criteria_table_full_translation_zh.md](final_criteria_table_full_translation_zh.md)

- 最后演化出基石/基础准则和高精度专家准则，前者通用覆盖率高，后者覆盖率极低，但它们在适用场景下的验证精度极高。

- Agent 自动演化出了严格的“评估优先级“。几乎所有的高级专家准则中，都包含类似这样的话语：

  - 在 `visual_coherence` 中：“该准则只在以下条件下适用：（1）两个候选回答都通过 multimodal_alignment...”
  - 在 `temporal_coherence` 中：“该准则严格处于次要位置，绝不推翻 visual_grounding 或 multimodal_alignment...”
  - 在 `completeness` 中：“完整性只在已经确认视觉依据一致性和事实一致性之后进行评估...”

  传统的 LLM-as-a-Judge 往往是把所有标准平行地抛给模型去打分（例如：请从准确性、流畅度、完整性给出 1-5 分）。而框架迭代最后发现（我并没有让他去做这种层次结构，仅仅给他错误样例reflection），**多模态评估是树状的（Tree-structured）或级联的（Cascaded）**。必须先满足物理存在（Visual Grounding），才能谈相对空间关系（Visual Coherence），最后才能作为决胜条件（Tie-breaker）去比较具体性（Specificity）。这一点似乎契合人类的视觉信息处理机制。

- 在纯文本大模型中，“幻觉”通常指代事实错误。而在多模态场景中幻觉问题更复杂，包括图像和文本事实性幻觉，在前面几个实验中发现仅仅使用“幻觉”这个宽泛的准则并不好，但是在这次实验中，把迭代次数设置成了10，似乎将粗粒的”幻觉“准则演化出了很多细粒度的幻觉准则。最后将多模态幻觉进行了极其细腻的**正交分解（Orthogonal Decomposition）**：

  - **空间关系幻觉：** `visual_coherence` 专门处理“在左侧”“重叠”等空间介词的误用。
  - **时序/动态幻觉：** `temporal_coherence` 和 `temporal_consistency` 专门捕捉静态图像中错误推断的动作（例如把“悬空”强行解释为“正在降落”）。
  - **生物学幻觉：** `biological_or_realistic_plausibility` 专门处理违反物理现实和生物解剖学的错误（例如“狗在飞”）。
  - **内部逻辑幻觉：** `multimodal_consistency` 和 `logical_consistency` 捕捉文本自身与视觉锚点产生的自相矛盾。

  自动生成了一套**更细粒度的多模态偏好分类法**

- 目前大多数 MLLM 评估方法都假设“图片有一个绝对真实的描述（Ground Truth）”。但准则演化出了 `visual_ambiguity_resolution` 和 `calibrated_uncertainty` 这两条高级的标准。

  - 准则明确指出：当遇到遮挡、模糊、阴影等客观存在的视觉歧义时，模型不应该去“编造（Fabricate）”，而应该使用“可能”“看起来像”等谨慎措辞（Cautious phrasing）。



**小结**：人类对多模态内容的偏好判断是一个高度异质且层次化的过程。然而，当前的大语言模型偏好对齐（如 RLHF-V）通常依赖于单一、宽泛的评分标准或简单的多数投票，这导致了严重的维度灾难和对细粒度视觉错误的忽视。

揭示了多模态偏好评估的两个核心性质：

1. **长尾与异质性：** 存在大量低覆盖但高精度的“专家准则”（如时间连贯性、生物合理性）。
2. **认知级联（Cognitive Cascade）：** 准则之间存在严格的先决依赖关系（如必须先满足 Visual Grounding，才能评估 Visual Coherence）。


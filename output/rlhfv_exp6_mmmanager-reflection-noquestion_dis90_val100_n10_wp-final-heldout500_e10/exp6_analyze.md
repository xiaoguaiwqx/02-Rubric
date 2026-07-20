## Exp6：多模态 Manager + Reflection 不显式输入 Question（epoch=10）

Exp6 保留 Exp5 的多模态 Manager，但只在 Warm-up 阶段输入 Image + Question；逐错误样例 Reflection 阶段输入 Image + A/B + Human Preference + Worker Thought，不再显式输入 Question。该实验用于观察 Question 是否会让 Manager 的反思过度依赖具体任务语义。

| 阶段                | train90 / valid100         | heldout500          |
| ------------------- | -------------------------- | ------------------- |
| warmup              | -                          | **0.632 = 316/500** |
| iter 0              | train 0.600, val100 0.680  | -                   |
| iter 1              | train 0.633, val100 0.640  | -                   |
| iter 2              | train 0.567, val100 0.720  | -                   |
| iter 3              | train 0.633, val100 0.660  | -                   |
| iter 4              | train 0.622, val100 0.680  | -                   |
| iter 5              | train 0.622, val100 0.670  | -                   |
| iter 6              | train 0.633, val100 0.700  | -                   |
| iter 7              | train 0.700, val100 0.700  | -                   |
| iter 8              | train 0.644, val100 0.690  | -                   |
| iter 9              | train 0.678, val100 0.690  | -                   |
| final current       | train 0.656                | -                   |
| final best criteria | -                          | **0.678 = 339/500** |

> 表中的逐轮结果由 `workflow_agent.log` 中的原始投票重建。`iter i` 的 train 评估进入本轮前的准则，而 val100 评估本轮 Reflection/Revise 后、保存于 `epoch_i.json` 的准则，因此同一行的 train 和 valid 并非完全相同的 criterion 版本。

**Final 相对 Warm-up 提升 4.6 个百分点；相对 Vanilla（0.598）提高 8.0 个百分点。**

Exp6 最终比 Exp5（0.666）高 1.2 个百分点，但同一批样本上的配对比较为：Exp6 独有正确 30 条，Exp5 独有正确 24 条，McNemar 检验 `p=0.497`。因此目前不能断言“不输入 Question”显著优于“输入 Question”。Exp6 仍比 Exp4（0.696）低 1.8 个百分点。

### Final Criterion Stats

| criterion                                      | train score | heldout acc | correct/app | coverage |
| ---------------------------------------------- | ----------- | ----------- | ----------- | -------- |
| temporal_coherence_of_visual_sequence          | 0.897       | 0.790       | 113/143     | 0.286    |
| visual_stability_and_motion_consistency        | 0.692       | 0.766       | 85/111      | 0.222    |
| biological_or_contextual_plausibility          | 0.769       | 0.752       | 173/230     | 0.460    |
| visual_grounding                               | 0.688       | 0.736       | 332/451     | 0.902    |
| multimodal_coherence                           | 0.709       | 0.734       | 318/433     | 0.866    |
| calibrated_uncertainty_in_ambiguous_scenes     | 0.746       | 0.724       | 280/387     | 0.774    |
| specificity                                    | 0.769       | 0.717       | 327/456     | 0.912    |
| temporal_or_action_coherence                   | 0.659       | 0.714       | 150/210     | 0.420    |
| factual_consistency_with_the_image             | 0.756       | 0.696       | 313/450     | 0.900    |
| logical_consistency                            | 0.625       | 0.683       | 297/435     | 0.870    |
| completeness                                   | 0.720       | 0.671       | 310/462     | 0.924    |
| calibrated_uncertainty                         | 0.667       | 0.667       | 312/468     | 0.936    |
| clarity_of_implied_relationships               | 0.632       | 0.658       | 306/465     | 0.930    |

完整的逐样本统计见 [final_heldout_diagnostics.json](final_heldout_diagnostics.json)，原始投票矩阵见 [final_heldout_raw_prediction.json](final_heldout_raw_prediction.json)。

### 主要观察

- **Question 消融在实现上基本成功。** Exp6 共进行了 1,879 次逐错误样例 Reflection，全部附带图片，全部没有显式 Question 字段；只有 3 次（0.16%）因为 Worker Thought 复述了原问题而间接泄漏 Question。因此该实验确实近似实现了“Manager 看图，但 Reflection 不看问题”。

- **Exp6 从更差的初始准则出发，获得了更大的演化增益。** Warm-up 只有 0.632，最终达到 0.678，提升 4.6 个百分点；Exp5 则从 0.666 开始，最终仍为 0.666。这说明 Exp6 的优化过程比 Exp5 更有效，但两次实验的初始准则不同，Manager temperature 也为 1.0，因此不能把差异完全归因于 Question。

- **不输入 Question 后，准则变得更加宽泛。** Exp6 final criteria 的平均 coverage 为 0.723，明显高于 Exp5 的 0.604；每个样本平均有效 A/B 投票数从 Exp5 的 7.25 增加到 9.40。平票数仍为 24，但非平票样本准确率从 0.700 提高到 0.712。

- **覆盖率上升并不等于每条准则都更准确。** 与 Exp5 的同名准则相比，`visual_grounding` 的 heldout 准确率从 0.679 提升到 0.736；`specificity` 从 0.702 提升到 0.717，覆盖率从 0.516 提升到 0.912。另一方面，`completeness` 从 0.684 降到 0.671，`factual_consistency_with_the_image` 从 0.712 降到 0.696。没有 Question 时，Manager 更容易总结通用视觉错误，但更难判断“哪些信息与当前问题真正相关”。

- **准则演化不是单调过程。** valid100 在 iter 2 后达到最高 0.720，train90 在 iter 7 进入时达到最高 0.700；之后继续迭代没有超过这些峰值。最终 current criteria 的 train90 为 0.656，说明 10 轮已经出现优化过头，适合考虑 early stopping。

- **仍然演化出了基础准则与局部专家。** `visual_grounding`、`specificity` 和 `completeness` 的覆盖率都超过 0.90；`temporal_coherence_of_visual_sequence` 与 `visual_stability_and_motion_consistency` 的覆盖率只有 0.286 和 0.222，但条件准确率达到 0.790 和 0.766。该结果再次支持多模态偏好评价具有“高覆盖基石 + 低覆盖专家”的长尾结构。

- **第 8 轮发生了准则提前收缩。** 当时一条 Low Criterion 被淘汰，Manager 被要求生成一条替代准则，但返回了约 65,670 字符、至少包含 700 多个枚举条款的未闭合 JSON。Low replacement 路径没有解析重试，也没有在失败后保留旧准则，因此 current criteria 从 10 条降到 9 条。之后没有恢复到目标数量。

- **Mid revise 重试机制有效，但没有解决全部结构化输出问题。** `completeness` 和 `visual_grounding` 有 3 次 revise 在 11 次尝试后仍失败；这些场景正确保留了旧准则，因此 Mid Criterion 没有因解析失败而消失。然而 Reflection 本身没有重试：1,879 次 Reflection 中约 367 次无法解析，失败率为 19.5%，无效 suggestion 会以 `None` 形式进入后续汇总。

- **准则描述膨胀比 Exp5 更严重。** 最终参与评估的准则描述平均长度约为 6,236 字符；最后 current criteria 平均达到 10,693 字符，最大为 20,264 字符。Exp5 对应值分别约为 3,877 和 5,963。后期 Manager 会逐条复制 suggestion，而不是对错误模式进行聚类、去重和抽象，最终导致上下文膨胀、输出截断和日志体积增加。[workflow_agent.log](workflow_agent.log) 达到约 743 MB。

- **Final 仍然是历史准则组合。** 最终 checkpoint 中有 9 条 current criteria、23 条历史 archive criteria 和 14 个 banned names；最终评估从 archive 中选出 13 条训练分数不低于 0.6 的准则。13 条中只有 3 条描述与最后 current 版本完全一致。因此，第 8 轮收缩没有直接把最终投票数降到 9，但它阻止了一条新准则进入后续演化。详细状态见 [epoch_final.json](epoch_final.json)。

- **当前 final criterion 选择方式会引入弱准则。** 使用全部 13 条准则时准确率为 0.678；按照 train score 排序，仅使用前 9 条进行离线投票可达到 0.708，并将平票从 24 个减少到 20 个。Top-9 独有正确 22 条，全部 13 条独有正确 7 条，配对检验 `p=0.008`。但这是查看 heldout 后的诊断结果，不能作为 Exp6 的正式成绩；它说明下一次实验应预先固定 Top-k，或者把 coverage、相关性和冗余纳入选择标准。

### 小结

Exp6 表明：在 Reflection 阶段移除 Question 后，Manager 仍能利用图片和错误案例形成有效的准则改进，并使一个较弱的 Warm-up 准则集合从 0.632 提升到 0.678。演化出的准则覆盖率更高，`visual_grounding` 等基础视觉准则也优于 Exp5。

但当前证据不足以证明“不输入 Question”本身优于“输入 Question”：两次实验起点不同，最终差异只有 1.2 个百分点且不显著。与此同时，Exp6 暴露出三个更直接的系统问题：Reflection 解析失败率接近 20%、criterion 描述持续膨胀，以及 Low replacement 解析失败会导致准则数量收缩。

下一次严格消融应从完全相同的 `epoch_init` 开始，仅切换 `manager_reflection_include_question`，固定数据顺序和采样随机性，并运行多个 seed。同时应为 Low replacement 增加“解析失败重试、数量校验、耗尽后保留旧准则”，限制每条 criterion 汇总的错误案例数量和最终描述长度，并预先定义基于 train score、coverage 与冗余的 final criterion 选择策略。

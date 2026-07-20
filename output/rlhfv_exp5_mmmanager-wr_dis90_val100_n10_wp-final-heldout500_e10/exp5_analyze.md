## Exp5：多模态 Manager + Reflection 输入 Question（epoch=10）

Exp5 在 Warm-up 和逐错误样例 Reflection 阶段都让 Manager 查看图片；Warm-up 和 Reflection 阶段会显式输入 Question。Worker 始终使用 Image + Question + A/B + Criterion 做判断。

| 阶段                | train90 / valid100         | heldout500          |
| ------------------- | -------------------------- | ------------------- |
| warmup              | -                          | **0.666 = 333/500** |
| iter 0              | train 0.578, val100 0.660  | -                   |
| iter 1              | train 0.589, val100 0.660  | -                   |
| iter 2              | train 0.556, val100 0.660  | -                   |
| iter 3              | train 0.522, val100 0.640  | -                   |
| iter 4              | train 0.567, val100 0.630  | -                   |
| iter 5              | train 0.611, val100 0.660  | -                   |
| iter 6              | train 0.644, val100 0.620  | -                   |
| iter 7              | train 0.578, val100 0.680  | -                   |
| iter 8              | train 0.644, val100 0.680  | -                   |
| iter 9              | train 0.656, val100 0.660  | -                   |
| final current       | train 0.611                | -                   |
| final best criteria | -                          | **0.666 = 333/500** |

> 表中的逐轮结果由 `workflow_agent.log` 中的原始投票重建。`iter i` 的 train 评估进入本轮前的准则，而 val100 评估本轮 Reflection/Revise 后、保存于 `epoch_i.json` 的准则，因此同一行的 train 和 valid 并非完全相同的 criterion 版本。

**Final 相对 Warm-up 没有提升；相对 Vanilla（0.598）提高 6.8 个百分点，但比 Exp4（0.696）低 3.0 个百分点。**

Exp5 与 Exp4 在同一批 500 条样本上的配对比较为：Exp5 独有正确 21 条，Exp4 独有正确 36 条，McNemar 检验 `p=0.063`。差异接近但尚未达到常用的 0.05 显著性阈值。

### Final Criterion Stats

| criterion                                      | train score | heldout acc | correct/app | coverage |
| ---------------------------------------------- | ----------- | ----------- | ----------- | -------- |
| visual_stability_under_transformation          | 0.824       | 0.841       | 58/69       | 0.138    |
| temporal_consistency_of_visual_elements        | 0.750       | 0.821       | 64/78       | 0.156    |
| logical_coherence_of_visual_sequence           | 0.710       | 0.750       | 135/180     | 0.360    |
| robustness_to_visual_noise                     | 0.719       | 0.738       | 121/164     | 0.328    |
| factual_consistency_with_the_image             | 0.692       | 0.712       | 306/430     | 0.860    |
| specificity                                    | 0.756       | 0.702       | 181/258     | 0.516    |
| accuracy_of_visual_interpretation              | 0.691       | 0.701       | 314/448     | 0.896    |
| consistency_with_common_sense_or_domain_knowledge | 0.675    | 0.694       | 311/448     | 0.896    |
| avoidance_of_ambiguous_visual_references       | 0.755       | 0.688       | 198/288     | 0.576    |
| completeness                                   | 0.778       | 0.684       | 234/342     | 0.684    |
| visual_grounding                               | 0.605       | 0.679       | 311/458     | 0.916    |
| multi-element_consistency                      | 0.671       | 0.657       | 302/460     | 0.920    |

完整的逐样本统计见 [final_heldout_diagnostics.json](final_heldout_diagnostics.json)，原始投票矩阵见 [final_heldout_raw_prediction.json](final_heldout_raw_prediction.json)。

### 主要观察

- **给 Manager 增加多模态能力和 Question 后，最终 heldout500 没有超过 Warm-up。** 初始 10 条准则已经达到 0.666；经过 10 轮演化后，最终历史准则组合仍为 0.666。优化过程改变了准则组成，但没有形成可测量的整体增益。

- **准则演化过程波动明显，后期没有稳定收敛。** valid100 在 iter 7 和 iter 8 后达到最高 0.680，随后回落到 0.660；最终 current criteria 的 train90 也从 iter 9 进入时的 0.656 降到 0.611。继续迭代并不保证 ensemble 单调改善。

- **演化出了低覆盖、高精度的局部专家。** `visual_stability_under_transformation` 和 `temporal_consistency_of_visual_elements` 的 heldout 条件准确率分别达到 0.841 和 0.821，但覆盖率只有 0.138 和 0.156。这与 Exp4 观察到的“长尾专家”现象一致。

- **通用准则覆盖率高，但准确率相对有限。** `visual_grounding` 和 `multi-element_consistency` 的覆盖率超过 0.91，但 heldout 准确率只有 0.679 和 0.657。由于当前聚合方式对所有适用准则等权投票，高覆盖的弱准则可能抵消低覆盖专家的正确判断。

- **最终组合存在明显的投票稀疏性。** 12 条 final criteria 的平均覆盖率为 0.604，每个样本平均只有 7.25 条准则实际投出 A/B，其余返回 None；最终出现 24 个平票样本。对 gold=A 的准确率为 0.590，而对 gold=B 为 0.738，表现出较明显的 B 侧偏好。

- **Final 并不等于最后一轮 current criteria。** 最终 checkpoint 中有 10 条 current criteria、26 条历史 archive criteria 和 18 个 banned names；最终评估从历史 archive 中选出了 12 条训练分数不低于 0.6 的准则。12 条中只有 3 条描述与最后 current 版本完全一致，因此 0.666 衡量的是“历史最佳版本组合”，不是最后一轮准则本身。详细状态见 [epoch_final.json](epoch_final.json)。

- **Reflection 和 Revise 的结构化输出仍不稳定。** Exp5 共进行了 1,433 次逐错误样例 Reflection，其中约 295 次无法解析，失败率为 20.6%。Mid Criterion 的 revise 重试避免了解析失败后准则直接消失，但大量失败请求增加了推理成本。

- **准则描述持续膨胀。** 最终参与评估的准则描述平均长度约为 3,877 字符；最后 current criteria 平均约为 5,963 字符，最大达到 8,653 字符。模型倾向于把每条 suggestion 继续追加到旧描述中，而不是进行抽象、去重和压缩。[workflow_agent.log](workflow_agent.log) 因此达到约 548 MB，即使图片 base64 已经被替换为占位符，重复的长文本 history 仍会造成很大的日志开销。

### 小结

Exp5 证明了 Manager 可以在 Warm-up 和 Reflection 中稳定接收图片与 Question，也再次产生了“基础准则 + 长尾专家”的多模态评价结构。但本次实验没有证明这种多模态 Manager 配置能够提升整体偏好判断：最终准确率与 Warm-up 相同，并低于 Exp4。

主要问题不是缺少局部高精度准则，而是：准则覆盖率差异很大、等权投票没有利用这种差异、Reflection 输出有约 20% 无法解析，以及 criterion 描述在多轮迭代中持续膨胀。后续实验应优先控制初始准则和随机性，并对 suggestion 数量、description 长度和最终 criterion 选择策略进行约束。

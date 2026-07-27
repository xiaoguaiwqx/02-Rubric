## Exp7：多模态 Manager + Reflection 不显式输入 Question（N=20，epoch=5）

Exp7 延续 Exp6 的多模态 Manager 设置：Warm-up 阶段输入 Image + Question，逐错误样例 Reflection 阶段输入 Image + A/B + Human Preference + Worker Thought，但不显式输入原始 Question。本次实验将准则数量从 10 增加到 20，并将 epoch 从 10 减少到 5，目标是获得更细粒度的局部专家，同时缓解多轮 Reflection 导致的过度改写和描述膨胀。

| 阶段                | train90 / valid100         | heldout500          |
| ------------------- | -------------------------- | ------------------- |
| warmup              | -                          | **0.654 = 327/500** |
| iter 0              | train 0.622, val100 0.660  | -                   |
| iter 1              | train 0.656, val100 0.670  | -                   |
| iter 2              | train 0.622, val100 0.660  | -                   |
| iter 3              | train 0.611, val100 0.640  | -                   |
| iter 4              | train 0.622, val100 0.680  | -                   |
| final current       | train 0.656                | -                   |
| final best criteria | -                          | **0.686 = 343/500** |

> Train90 结果按照 `workflow.optimize()` 内部真实执行的 `random_reverse()` 重新计算：训练集会先固定随机打乱，并随机交换部分样本的 A/B，同时翻转 gold label。表中的 `iter i` 与 Exp4、Exp6 的记录口径一致：train 是进入本轮改写前的准则，val100 是本轮 Reflection/Revise 后的准则，因此同一行的 train 和 valid 并非完全相同的 criterion 版本。

**Final 相对 Warm-up 提升 3.2 个百分点；相对 Vanilla（0.598）提高 8.8 个百分点，相对提升约 14.7%。**

Warm-up 与 Final 在同一批 heldout500 样本上的配对比较为：Final 独有正确 38 条，Warm-up 独有正确 22 条，McNemar 检验 `p=0.052`。该结果接近显著性阈值，但目前仍不足以断言演化带来了稳定提升。

Exp7 最终比 Exp6（0.678）高 0.8 个百分点，但 Exp7 独有正确 27 条、Exp6 独有正确 23 条，McNemar 检验 `p=0.672`；Exp7 仍比 Exp4（0.696）低 1.0 个百分点。因此现有单次实验不能证明 N=20、epoch=5 的联合配置优于 Exp4 或 Exp6。

### Final Criterion Stats

| criterion | train score | heldout acc | correct/app | coverage |
| --- | ---: | ---: | ---: | ---: |
| `plausibility_of_inferred_states` | 0.733 | 0.756 | 102/135 | 0.270 |
| `temporal_consistency_in_sequence` | 0.639 | 0.749 | 161/215 | 0.430 |
| `visual_anomaly_detection` | 0.705 | 0.747 | 233/312 | 0.624 |
| `accuracy_of_positioning` | 0.806 | 0.734 | 213/290 | 0.580 |
| `object_boundary_accuracy` | 0.741 | 0.734 | 113/154 | 0.308 |
| `factual_consistency` | 0.719 | 0.728 | 289/397 | 0.794 |
| `specificity_of_visual_description` | 0.750 | 0.716 | 257/359 | 0.718 |
| `semantic_coherence_with_image_context` | 0.716 | 0.712 | 280/393 | 0.786 |
| `scale_and_proportion_accuracy` | 0.692 | 0.712 | 210/295 | 0.590 |
| `visual_grounding` | 0.679 | 0.709 | 305/430 | 0.860 |
| `answer_confidence_indication` | 0.716 | 0.709 | 224/316 | 0.632 |
| `avoidance_of_vague_quantifiers` | 0.783 | 0.706 | 101/143 | 0.286 |
| `multi_object_relationship_accuracy` | 0.712 | 0.706 | 168/238 | 0.476 |
| `biological_or_contextual_plausibility` | 0.700 | 0.704 | 245/348 | 0.696 |
| `accuracy_of_attributes` | 0.675 | 0.703 | 308/438 | 0.876 |
| `completeness` | 0.724 | 0.703 | 296/421 | 0.842 |
| `visual_detail_preservation` | 0.667 | 0.700 | 313/447 | 0.894 |
| `object_part_identification_accuracy` | 0.712 | 0.697 | 265/380 | 0.760 |
| `attribute_ambiguity_handling` | 0.649 | 0.675 | 262/388 | 0.776 |
| `visual_cue_alignment` | 0.643 | 0.667 | 98/147 | 0.294 |
| `calibrated_uncertainty` | 0.655 | 0.656 | 217/331 | 0.662 |
| `consistency_within_answer` | 0.649 | 0.654 | 183/280 | 0.560 |
| `cultural_or_situational_appropriateness` | 0.627 | 0.651 | 298/458 | 0.916 |
| `temporal_consistency` | 0.688 | 0.650 | 223/343 | 0.686 |
| `visual_resolution_adaptation` | 0.647 | 0.639 | 295/462 | 0.924 |
| `visual_analogy_or_comparison_accuracy` | 0.714 | 0.545 | 12/22 | 0.044 |

完整逐样本统计见 [final_heldout_diagnostics.json](final_heldout_diagnostics.json)，原始投票矩阵见 [final_heldout_raw_prediction.json](final_heldout_raw_prediction.json)。

### 主要观察

- **Exp7 没有发生 Train90 性能崩溃。** Train ensemble 在 0.611～0.656 之间波动，final current 为 0.656，与演化过程中的最高值持平。它没有单调提升，但这与当前优化目标一致：Workflow 根据每条 criterion 的条件准确率进行分桶，并不直接优化多数投票准确率。

- **准则变得更有选择性，支持“更细粒度局部专家”的假设。** 从初始到 final current，单准则平均 Train 条件准确率从约 0.595 上升到 0.671，而平均 Coverage 从 0.889 下降到约 0.638。这说明准则主要通过缩小适用范围、在不相关样本上回答 U，逐渐成为局部专家，而不是在所有样本上统一变强。

- **Exp7 挖掘出了新的细粒度视觉维度。** `plausibility_of_inferred_states`、`temporal_consistency_in_sequence`、`object_boundary_accuracy` 等准则覆盖率较低，但 heldout 条件准确率达到 0.734～0.756；同时还演化出物体部件、尺度与比例、视觉异常、属性歧义和低分辨率适配等更具体的视觉错误类型。

- **更多、更细的专家尚未转化为更高的明确判断质量。** Exp7 最终 26 条准则的平均 heldout 条件准确率为 0.695，低于 Exp6 的 0.716；非平票准确率为 0.703，也低于 Exp6 的 0.712。Exp7 相比 Exp6 的主要变化是平票从 24 个降到 12 个、每个样本平均有效票从 9.40 增加到 16.28，因此 0.8 个百分点的表面提升主要来自 Coverage 和投票数增加，而不是非平票判断更准确。

- **epoch=5 明显缓解了描述膨胀，但没有根治。** Final current criterion 的平均描述长度约为 4,095 字符，低于 Exp6 的 10,693 字符，下降约 61.7%；但仍比初始化的 227 字符膨胀约 18 倍，最大描述达到 9,324 字符。[workflow_agent.log](workflow_agent.log) 仍达到约 585 MB。

- **Reflection 解析失败随描述增长而增加。** Exp7 共进行了 1,466 次逐错误样例 Reflection，其中 51 次解析失败，整体失败率约 3.5%；失败率从 epoch 0 的 0% 上升到 epoch 4 的约 10.8%。Reflection 本身没有重试，失败 suggestion 会以 `None` 进入后续汇总。Mid revise 重试机制则有效：64 个改写任务经历 77 次生成尝试和 13 次无效输出后全部成功，没有因解析失败丢失准则。

- **停止在 5 个 epoch 时，current criteria 尚未收敛。** Final current20 中没有 score 不低于 0.8 的 Good criterion，19 条仍属于 Mid，`non_visual_context_relevance` 以 0.481 属于 Low。如果继续下一轮，19 条仍会被 Reflection/Revise，1 条仍会被替换。因此 epoch=5 是成本控制和防止继续膨胀的人工停止点，而不是准则已经稳定的停止点。

- **Final 仍然评估历史准则组合，而不是最终 20 条 current criteria。** [epoch_final.json](epoch_final.json) 中有 20 条 current criteria、38 条 archive criteria，最终通过 `score >= 0.6` 选出 26 条历史准则参与 heldout500。26 条中只有 19 条与 current 同名，只有 12 条描述与 current 版本完全一致。因此 0.686 不能直接解释为“最终 20 条细粒度准则”的性能，也不能与 Exp6 的 13 条历史准则做严格的等规模比较。

- **Train score 对低覆盖准则存在明显的高方差。** `visual_analogy_or_comparison_accuracy` 在 Train90 上只对 7 条样本适用，判断正确 5 条，因此获得 0.714；但 heldout500 上只有 12/22 正确，条件准确率为 0.545。所有 Final criteria 的 Train score 与 heldout 条件准确率相关系数仅约 0.374，说明固定阈值 0.6 没有考虑适用样本数、置信度和选择偏差，容易把“幸运的低覆盖版本”收入历史 ensemble。

- **等权多数投票仍受冗余准则影响。** 157 个 Final 错误样本中，有 134 个至少存在一条判断正确的 criterion，但正确局部专家经常被相关准则簇的多数错误票覆盖。离线 leave-one-out 中，去掉 `scale_and_proportion_accuracy` 后可从 0.686 上升到 0.694；这是查看 heldout 后的诊断，不能作为正式成绩，但说明准则越多并不一定越好，后续应考虑可靠性、Coverage 和准则相关性加权。

- **Low replacement 会重新生成已经淘汰的准则。** Epoch 1 新增 4 条准则，其中 3 条已在 banned 中；epoch 2 为 7/8；epoch 3 为 7/7。`multi-object_relationship_accuracy` 与 `multi_object_relationship_accuracy` 还会因为连接符不同被当成两个名字。Prompt 中要求避免重复并不足够，需要在代码层面进行名称标准化和 archive/banned 强制去重。

- **Final 存在明显的 B 方向偏置。** Gold A 的准确率为 152/244 = 0.623，Gold B 为 191/256 = 0.746，相差 12.3 个百分点；最终预测 A/B/平票分别为 212/276/12。下一步需要通过交换 A/B 后再次推理，检查 Worker 的位置一致性。

- **Train/Valid 缺少结构化原始结果，增加了分析出错风险。** Final heldout 保存了完整原始投票，但 Train 和 Valid 只打印到终端并记录 Agent 调用日志；训练数据还会经过固定 shuffle 和 A/B 翻转。后续应逐轮保存 `train_eval_epoch_i.json` 和 `valid_eval_epoch_i.json`，记录 sample ID、实际 A/B 顺序、翻转后的 gold、criterion version、A/B/U 投票及聚合结果。

### 小结

Exp7 表明，将准则数量增加到 20、epoch 减少到 5 后，系统可以稳定维持目标准则数量，并演化出覆盖范围更窄、视觉维度更细的局部专家。Train ensemble 最终达到 0.656，没有出现提前收缩或性能崩溃；较少的 epoch 也显著降低了 criterion 描述长度和 Reflection 解析失败率。

但当前证据不足以证明 Exp7 优于 Exp6 或 Exp4：最终相对 Exp6 只提高 0.8 个百分点且不显著，非平票准确率和平均单准则 heldout 准确率反而更低。更重要的是，Final 实际使用了 26 条历史准则而不是 final current20，准则数量、epoch、初始生成随机性和代码修复同时发生变化，使得本次实验不能单独归因于“更多准则”或“更少 epoch”。

下一步应优先完成三项低成本验证：单独评估 final current20 的 heldout500、跨实验使用固定 Top-k 历史准则、执行 A/B swap consistency 测试。随后应为 criterion 选择加入最小适用样本数、置信下界和冗余惩罚，为 Low replacement 增加强制去重，并补跑 `{N=10,20} × {epoch=5,10}` 的多 seed 消融实验。

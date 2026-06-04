## exp2 500-heldout set

其他设置和exp1 一模一样，只是数据集变了。

**数据集**：

- train_set: 90 条，用于优化 criteria；

- valid_set: 从500-heldout set随机选择100条，只用于观察泛化，不参与当前轮 criteria 改写。（如果用500-heldout set太耗时了，先使用子集来观察，在这里valid_set 没有参与训练步骤所以不算数据泄露）
- 500-heldout set：在**criterion warm up** 和 **final criterion** 做真实测试。看迭代之后的criterion 是否有被优化得更好。

总共调用模型`11659`次，**实验总耗时约：5 小时 38 分 22 秒**。

Heldout final evaluation 的吞吐约为 **1.04 criterion-level judgments/s**，即每次“样本 × criterion”的 worker 判断约 0.96s。由于 final eval 使用 8 条 criteria，500 个 heldout 样本共触发约 4000 次判断，因此样本级平均耗时约 7.7s/sample。相比之下，reflection 阶段单次调用更慢，通常数秒到十几秒，因为 manager 需要**基于错误案例分析并改写 criteria**；worker 判断本身也不只是输出 A/B，而是**会分析 answer A、answer B、偏好依据，并返回结构化判断**。

| 阶段   | train / valid             | heldout500          |
| ------ | ------------------------- | ------------------- |
| warmup | -                         | **0.644** = 322/500 |
| iter 0 | train 0.578, val100 0.710 | -                   |
| iter 1 | train 0.567, val100 0.720 | -                   |
| iter 2 | train 0.567, val100 0.630 | -                   |
| iter 3 | train 0.556, val100 0.590 | -                   |
| iter 4 | train 0.578, val100 0.620 | -                   |
| final  | train 0.611               | **0.672** = 336/500 |

Warmup 到 final：322/500 -> 336/500，净提升 +14 条，+2.8pt。

**Final Eval Criteria**

这次 final eval 实际用了 8 条 criterion：

| criterion                          | train score | heldout500 per-criterion |
| ---------------------------------- | ----------- | ------------------------ |
| temporal_consistency               | 0.786       | 0.782                    |
| completeness_and_specificity       | 0.763       | 0.715                    |
| visual_grounding                   | 0.663       | 0.714                    |
| factual_consistency_with_the_image | 0.653       | 0.705                    |
| visual_contextual_accuracy         | 0.652       | 0.687                    |
| question_relevance                 | 0.713       | 0.676                    |
| calibrated_uncertainty             | 0.649       | 0.667                    |
| response_ambiguity_resolution      | 0.605       | 0.639                    |

最稳的是 visual_grounding、factual_consistency_with_the_image、completeness_and_specificity：它们 train/heldout 都不错，而且覆盖语义基础。**temporal_consistency 数值最高，但要谨慎，因为 val100 里它拒答率很高**，iter4 达到 91/100 refuse；当前 final 日志没有打印 heldout500 refusal，所以它的 0.782 可能是“低覆盖高精度”。

**Warmup Vs Final Criterion Signal**

| same-name criterion                      | warmup heldout | final heldout | change |
| ---------------------------------------- | -------------- | ------------- | ------ |
| visual_grounding                         | 0.700          | 0.714         | +1.4%  |
| question_relevance                       | 0.579          | 0.676         | +9.8%  |
| factual_consistency_with_the_image       | 0.722          | 0.705         | -1.7%  |
| completeness_and_specificity             | 0.685          | 0.715         | +3.1%  |
| avoidance_of_hallucinated_visual_details | 0.642          | not selected  | -      |

最明显的改善是 question_relevance，从 warmup 的弱项变成 final 可用项。avoidance_of_hallucinated_visual_details 继续没进 final，幻觉这个criterion 可能太粗了，需要拆细。

**Key Findings**

- **final 比 warmup 好**，但幅度不大。
  0.644 -> 0.672 是正向信号，不是强结果。适合作为下一轮优化方向依据，不适合作为最终结论。
- **val100 仍然不能直接当模型选择指标。**
  val100 最好是 iter1 的 0.720，后面掉到 0.590/0.620，但 final heldout 仍有 0.672。过程曲线波动大，说明 val100 只适合观察，不适合强 early stopping。
- **最终用了被 banned 过的历史 criteria。**
  get_best_criteria(0.6) 从 all_criteria 取历史高分，不排除 banned_criteria。所以像 visual_grounding、question_relevance 这些虽然在 banned 列表里，仍然进入 final eval。因为这些指标一开始是mid criterion, 但是在经过reflection 和rewrite 的时候变成low criterion了，因此被加入了banned list。
- **记录 refusal coverage。**
  现在 final per-criterion acc 很有用，但缺少 heldout500 上每条 criterion 的 refuse 数。尤其 temporal_consistency 这种高分项，如果覆盖率很低，它更像 tie-breaker，不该和 broad criterion 等权投票。**给 final eval 额外打印每条 criterion 的 Refuse to Respond**
- **final per-criterion acc > final acc**。像visual_grounding这些criterion 可以达到0.714，但是最后acc只有0.672。可以细挖其中的原因，是因为集体投票机制导致的吗？如果是有没有更好集成的方法。
-   其他问题还是和exp1一样


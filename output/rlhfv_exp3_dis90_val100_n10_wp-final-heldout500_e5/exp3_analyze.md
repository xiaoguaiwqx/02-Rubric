## exp3 n_criterion=10 与 heldout 诊断

除了n_criterion配置之外，其他设置都和exp2一样。这次实验**将n_criterion设置成10**，看会不会生成更多细粒度的criterion。除此之外，加了更多**诊断信息**，final eval 后额外打印：

- 每条 criterion 的 refuse
- coverage = applicable / total
- correct / applicable
- 每个样本的 A / B / None vote counts
- ensemble_wrong_with_correct_criteria：final ensemble 判错、但至少有单条 criterion 判对的样本列表

| 阶段   | train / valid             | heldout500      |
| ------ | ------------------------- | --------------- |
| warmup | -                         | 0.636 = 318/500 |
| iter 0 | train 0.556, val100 0.640 | -               |
| iter 1 | train 0.600, val100 0.690 | -               |
| iter 2 | train 0.567, val100 0.660 | -               |
| iter 3 | train 0.567, val100 0.660 | -               |
| iter 4 | train 0.633, val100 0.690 | -               |
| final  | train 0.622               | 0.676 = 338/500 |

warmup 到 final：318/500 -> 338/500，净提升 +20 条，**+4.0%**。

虽然设置了 N_CRITERIA=10，final eval 实际用了 **11 条**，因为 get_best_criteria(0.6) 是从 all_criteria 里筛 score >= 0.6，不是 top-10。

| criterion                                     | train score | heldout acc | correct/app | refuse | coverage |
| --------------------------------------------- | ----------- | ----------- | ----------- | ------ | -------- |
| temporal_consistency_in_motion                | 0.674       | 0.731       | 158/216     | 284    | 0.432    |
| specificity                                   | 0.727       | 0.706       | 314/445     | 55     | 0.890    |
| calibrated_uncertainty                        | 0.689       | 0.700       | 240/343     | 157    | 0.686    |
| visual_coherence_of_composition               | 0.718       | 0.693       | 147/212     | 288    | 0.424    |
| visual_grounding                              | 0.688       | 0.682       | 317/465     | 35     | 0.930    |
| contextual_sensitivity                        | 0.650       | 0.675       | 307/455     | 45     | 0.910    |
| factual_consistency                           | 0.709       | 0.672       | 310/461     | 39     | 0.922    |
| completeness                                  | 0.679       | 0.665       | 314/472     | 28     | 0.944    |
| interpretive_accuracy_under_ambiguous_context | 0.630       | 0.652       | 307/471     | 29     | 0.942    |
| adaptation_to_visual_quality                  | 0.633       | 0.650       | 321/494     | 6      | 0.988    |
| sensitivity_to_implied_meaning                | 0.646       | 0.568       | 151/266     | 234    | 0.532    |

**Key Findings**

1. **N=10 有帮助，但不是巨大提升。**
   final heldout 0.676，比 warmup 0.636 高 +4.0pt。这是目前较好的正向信号，但还不是“解决问题”的幅度。

2. **高分低覆盖 criterion 适合做 tie-breaker，不适合等权投票。**
   temporal_consistency_in_motion、visual_coherence_of_composition 的 heldout acc 高，但 coverage 只有 0.432/0.424。它们有用，但应该在适用时加权，而不是和 broad criteria 等权。

3. **specificity 是这次最强的实用 criterion。**
   它 train 0.727、heldout 0.706、coverage 0.890，同时准确率和覆盖率都不错。下一轮可以保留。

4. **sensitivity_to_implied_meaning 明显拖后腿。**
   heldout acc 只有 0.568，coverage 0.532，低覆盖且低精度。它虽然 train 过了 0.6 阈值，但泛化不好，建议从 final eval 中排除或降权。

5. **投票机制仍然是主要问题。**
   final 错了 162 条，其中 140 条至少有一个 criterion 判对。更关键的是，102/162 个错误是 margin >= 3 的强错误多数，不只是平票问题。这说明有些 criteria 会集体偏向错误答案，单纯增加 criterion 数量不一定继续提升。

   **能否使用一个 manager 充当 soft router**，让他来选择哪些criterion对于这个样本是有用的，然后在投票的时候设计一个**投票机制**，比如说**按照训练集acc来加权**。

6. **代码层面还有一个 selection 口径问题**。final eval 用了 11 条历史最佳，而 epoch_final.current_criteria 只有 6 条？如果 manager 返回数量不足、名字重复、parse 后合并覆盖，criterion就减少了，后面没有机制补回来

 **Next Experiments**

- **做 final ablation**：best11、top5_by_train、top7_by_train、no_sensitivity_to_implied_meaning、broad_only。
- 明确限制 final eval 数量：比如 **top-k**，而不是阈值筛出任意数量。
- **单独分析 strong wrong majority 的样本，找出哪些 criteria 经常一起把答案带偏。**
- **Key Findings 5** 的问题解决方案


# Phase19 最终准则与 Split/Refine 分析

## 数据范围

分析对象为 `output/evolving_structured_rubrics/rubric_evolution_phase5/phase19_unified_subtree_arbiter_aligned_evolution_v1`，并与上一版 `phase18_qwen25_discovery_v2_prompt_v2_split_refine_v1` 对照。Phase19 使用 Qwen3-VL-8B worker；Phase18 使用 Qwen2.5-VL-7B，数据规模和协议不完全相同，绝对准确率不能直接视为同一 benchmark 上的严格比较。

## Phase19 最终准则

初始 5 个 root 保留，最终增加 5 个 children，共 10 个节点。有效提交来自两个 Split，Refine 没有一次被接纳。

### Completeness root

- `verifiable_visual_coverage`：要求“完整性”只能由图像中可验证的视觉内容贡献，防止虚构细节被当作覆盖度。
- `task_intent_completion_priority`：要求按任务意图、约束和逻辑完成任务，而不是把冗长、步骤多或外围细节误当作完整。

这两个 child 把原本宽泛的 completeness 分解成“可验证视觉覆盖”和“任务意图完成”两个维度。它们语义上有价值，但与 factuality、visual grounding、instruction following 存在明显重叠。

### Creativity root

- `visual_precision_over_expressive_elaboration`：视觉/事实任务中，精确和可观察证据优先于叙事修辞。
- `utility_appropriate_elaboration`：根据任务需要在简洁和解释性之间选择合适的展开程度。
- `reasoning_validity_over_presentation`：客观推理任务中，逻辑和计算正确性优先于自信语气、格式和步骤数量。

这三个 child 实际上把 creativity 从“创造性”改造成了条件化的 style-vs-substance 冲突解决规则；对开放式创作任务的适用边界仍需谨慎。三个 child 的边界并非完全正交，最终记录的 sibling conflict rate 为 14.7%–31.8%。

最终 rubric 的 5 条 parent→child edge 全部为 `condition: always`。因此这些 child 并非严格按路由只在适用样本上执行，而是由全局 arbiter 处理大量重叠意见；这可能正是 sibling conflict 较高的原因之一。

## Split/Refine 成功率

| 版本 | Split 尝试 | Split 接纳 | 接纳率 | Refine 尝试 | Refine 接纳 | 接纳率 | 节点数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase19 | 23 | 2 | 8.7% | 4 | 0 | 0% | 5→10 |
| Phase18 | 9 | 5 | 55.6% | 61 | 15 | 24.6% | 5→25 |

作为历史背景，Phase17 的 Qwen3-VL exploratory benchmark report 使用 1247 条 VL-RewardBench 样本，node_count=27，initial strict accuracy 0.57097、final 0.69687（+12.59pp）。它的评估规模、演化协议和报告口径与 Phase19 不同，不能与 Phase19 的 100 条 discovery 或 500 条 held-out 直接比较；这里只能说明 Phase19 不是简单延续“更大树必然更好”的结果。

Phase19 的最终报告另有 `operator_transition_counts = accepted 2 / rejected 25 / tie_rejected 1`，合计 28；但显式 `split_attempts + refine_attempts` 数组只有 27 条。该计数差异应在论文前统一。

### Phase19 两次接纳的细节

| 接纳 root | child 集合 | child 局部 specialized accuracy | 系统级变化 |
|---|---|---|---|
| Completeness（epoch 3） | verifiable_visual_coverage；task_intent_completion_priority | 相对 parent 的 single-child specialized delta 分别约 −8.2pp、−5.2pp | corrected 5、harmed 3，net +2，system strict accuracy +2pp |
| Creativity（epoch 5） | visual_precision_over_expressive_elaboration；utility_appropriate_elaboration；reasoning_validity_over_presentation | 约 −12.9pp、−5.4pp、−1.1pp | corrected 8、harmed 7，net +1，system strict accuracy +1pp |

这是本次演化最重要的诊断：两次被接纳的 Split，其 child 局部准确率都低于 parent，且 `strong_child=false`。它们是因为完整 subtree + global arbiter 的端到端结果略有正增益而被接纳，并不是因为 child 本身成为了更强的局部专家。

Refine 的 4 次尝试全部被拒绝：两个 completeness child 各尝试两次；原因包括 `subtree_evidence_regression` 和 `no_system_effect`。因此当前最终 rubric 的所有新增节点 lineage 都是 `operator: split`，没有一个最终节点来自成功 Refine。

## 端到端效果

| 评估 | 初始 | Phase19 最终 | 变化 |
|---|---:|---:|---:|
| Discovery（100 条）strict accuracy | 0.690 | 0.720 | +3.0pp |
| Discovery paired | — | corrected 11 / harmed 8 | net +3 |
| Diagnostic dev（150 条，禁止用于选择） | 0.7067（epoch 0） | 0.7133（epoch 5） | +0.67pp；中间 epoch 2 达到 0.740，随后回落 |
| Held-out（500 条）strict accuracy | 0.754 | 0.748 | −0.6pp |
| Held-out paired | — | corrected 21 / harmed 24 | net −3 |

因此 Phase19 当前更准确的结论是：选择集上有小幅净改进，但没有 held-out 泛化收益；演化过程也不是单调的。Discovery 的 +3pp 只有 11 个纠正对 8 个伤害，不能单独作为强证据。

### 最新 VL-RewardBench 外部基准

最新的 1247 条 VL-RewardBench 对齐评测（同样本顺序、同 prompt/decoding；benchmark 结果未参与选择）给出更有说服力的端到端证据。下表采用 `final_report.json` 中的 `macro_strict_accuracy`；仓库内 `final_report.md` 展示的是另一口径的 macro 数值（例如 aligned final 为 0.6494），两者必须在论文中明确区分并统一命名：

| 系统 | Strict accuracy | Covered/overall accuracy | Macro strict accuracy | Coverage |
|---|---:|---:|---:|---:|
| Initial five roots | 0.6808 | 0.6819 | 0.6153 | 0.9984 |
| Phase17 E4 control | 0.7113 | 0.7159 | 0.6565 | 0.9936 |
| Phase19 aligned final | **0.7057** | **0.7097** | 0.6439 | 0.9944 |

配对比较进一步限定了结论：

- 相对 raw initial：corrected=65、harmed=34、net=+31；strict delta 的 paired-bootstrap 估计为 **+2.486pp**，95% CI [0.962, 4.090]pp，McNemar exact p=0.00239。也就是说，最终准则对最初五根准则有统计显著的外部收益。
- 相对 Phase17 E4 control：corrected=57、harmed=64、net=−7；delta 估计 **−0.561pp**，95% CI [−2.245, 1.203]pp，McNemar p=0.586。该差异不显著，不能声称 Phase19 超过已有的 Phase17 演化系统。

因此，Phase19 的“独特增益”应定义为：在不使用 benchmark 选择的前提下，相比 raw five-root baseline 泛化约 2.5pp；相对更强的 Phase17 E4 演化 checkpoint，则尚未显示增益。注意 macro strict accuracy 低于 Phase17 E4（0.6439 vs 0.6565），说明平均组间平衡并未改善；总体收益更可能集中在部分数据族，而不是所有子任务均匀提升。Coverage 仅小幅高于 Phase17 E4、但低于 initial，故 strict 提升不是由更高覆盖率造成的。

### 为什么最新的 “Initial five roots” 比早期 initial 高约 11pp

早期 Phase17 的 `initial_five_root_prompt_v2` strict accuracy 为 0.5710，而本次对齐评测中的 `initial_five_root` 为 0.6808，差值为 **+10.99pp**（相对提升约 19.2%）。这两个名称相同，但并不是同一个部署协议：本次 `initial_five_root` 已经运行在 `unified-subtree → global-arbiter` runtime 下；“initial”只表示 rubric 内容仍是五个 root，并不表示退回到旧的单阶段/平面聚合流程。因此这 11pp 不能归因于 rubric 演化本身。

仓库中较早的同类组件实验给出方向性分解：unified-subtree-only（S3）strict=0.6672，global-arbiter（S4）strict=0.6744，AB-only global-arbiter（S5）strict=0.7145；但它们的 prompt、None 语义和冻结报告与本次 aligned runtime 仍不完全一致。因此可以说“子树报告 + 全局 arbiter 很可能是主要增益来源”，却不能从现有结果识别出 arbiter 单独贡献了多少个百分点。需要在同一 aligned runtime 中固定五个 root，做 direct root aggregation、subtree-only、subtree+arbiter 三路消融，才能做因果归因。

按配对样本的净变化，Phase19 相对 initial 的收益主要来自 RLAIF-V（+9）、hallucination（+9）、wildvision（+8）和 mathverse（+5）；相对 Phase17 E4，hallucination（−10）和 wildvision（−9）出现净回退，而 RLAIF-V（+8）和 mmmu_pro（+4）有所改善。这里的分组数字是配对计数，不能在缺少各组分母时直接解释成组级准确率。

运行审计显示最终 unresolved technical failures=0，且 benchmark selection_after_benchmark_forbidden=true；不过初次运行曾有 93/168 条技术失败并通过 retry 清零，论文中应报告这一操作事实以说明评测稳定性和重试策略。

## 与 Phase18 之前方法的差异

| 维度 | Phase18 | Phase19 |
|---|---|---|
| Worker | Qwen2.5-VL-7B | Qwen3-VL-8B |
| 演化风格 | 更激进、更密集地扩展所有 5 个 root | 保守，只接受两个 root 的 Split |
| 最终结构 | 25 nodes，20 children | 10 nodes，5 children |
| Split/Refine 接纳 | 5/9；15/61 | 2/23；0/4 |
| Discovery M1 | 0.58→0.62（+4pp） | 0.69→0.72（+3pp） |
| Held-out M1 | qwen25 specific final 0.608→0.664（+5.6pp） | 0.754→0.748（−0.6pp） |
| 结构语义 | 大量细粒度子域：视觉引用、空间关系、结构化数据、约束等 | 少量更高层的“可验证覆盖/任务意图/精度-表达”规则 |

Phase18 的高接纳率和更大树可能意味着更强的适配能力，但也增加规则膨胀和过拟合风险；Phase19 的低接纳率体现了更严格的端到端保护，却同时暴露出 under-evolution：实际没有成功 Refine，且最终 held-out 下降。由于 worker、数据与协议不同，上述百分比应解释为演化行为差异，而不是方法优劣的无条件排名。

## 结论

1. Phase19 最终准则的主要新知识集中在两个错误族：把幻觉/冗长误当 completeness，以及把表达风格误当 substantive quality。
2. 它更像一个“保守的结构化 arbiter 修正”而不是成功的持续 Refine 系统；当前没有证据表明 Refine 已经带来独立收益。
3. 两次接纳都依赖 system-level interaction，而非 child-level specialization；因此论文不应声称这些 child 本身是已验证的局部专家。
4. 选择集 +3pp、held-out −0.6pp 的组合不支持“泛化改善”这一强主张。
5. 外部 1247 样本 benchmark 支持相对 raw initial 的 +2.5pp 泛化收益，但不支持相对 Phase17 E4 的优越性；Phase19 的贡献更接近“更保守、可审计的演化路径”，而不是新的性能 SOTA。

## 建议的下一步

- 把 Split acceptance 和 child local quality 分开报告，避免将端到端 arbiter 增益误称为 specialist gain。
- 对 5 个 child 做 matched no-child / parent-only / child-only / all-children 对照，并报告 sibling conflict 与 arbiter override 率。
- 为 Refine 构造独立 held-out criterion audit；否则应把 Refine 从当前已实现贡献中降级为未验证组件。
- 在同一 worker、同一 500 held-out 集上复跑 Phase18/Phase19 rubric，才能比较“激进扩展 vs 保守接纳”。
- 修正 27 条显式尝试与 28 条 aggregate transition 的日志计数不一致问题。

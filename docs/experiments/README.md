# 实验索引

Init Split 提示词重构与独立验证见[计划](init-split-prompts/plan.md)、[执行进度](init-split-prompts/tracker.md)和[实验结果](init-split-prompts/results.md)。在 init-split-prompts 分支复用已保存的 G5 seed11 五根 R0，只生成新 S0 并做正式 VLRB K=3 配对对比；英文模板 v1 和偏好导向 v2 均已完成。本次 v2 全量 Strict ACC 为 77.15%，未参与初始化 1147 条为 76.46%，幻觉留出集 648 条为 85.34%；结果文档保留各版本、切片与各根指标及主要负结果。

主要框架图为 [assets/framework.png](../../assets/framework.png)。方法概览、三个配置示例的区别与运行命令统一维护在仓库 [README](../../README.md)。当前 Hallucination100 参考配置使用 Strict + Preserve5；原 Discovery100 的 Covered/Strict 示例均使用 0 条保留案例。G5/GN 共用预热历史，默认五样例，可通过 `roots --warmup-count` 设置；F5 为人工固定五根对照。

当前主线是[子树逐例反思与局部竞争](subtree-local-reflection/plan.md)。[最终实验结果](subtree-local-reflection/results.md)集中展示八行主对比表、当前配置和主要结论；[最新框架 PPT](subtree-local-reflection/framework.pptx)说明方法与实验。Hallucination100 的详细实验记录见[冻结方案及已观察结果](vlrb-hallucination100-generated-roots/plan.md)，可复现的 seed11 样本 ID 和来源哈希见 [split manifest](vlrb-hallucination100-generated-roots/seed11_split.json)。

G5 seed11 的 2026-10-03 重跑与历史运行对比已归档于生成 root 实验文档第 9 节，包含正式 R0/S0/Final、clean1146、配对统计、演化轨迹、耗时及负结果。PPT 第 11 页最后三行使用本次结果；文档第 8 节保留旧结果。两次初始化内容不同，不能把分数差直接解释为代码重构收益。

当前代码入口是 `experiments.evolving_structured_rubrics.run_subtree_experiment`：`prepare` 还原发现集，`roots` 创建 F5/G5/GN，`vlrb-r0` 测裸 root，`evolve` 运行 S0 与局部演化，`dev`/`vlrb` 做冻结后的独立评测，`report` 汇总结果。具体命令见仓库 [README](../../README.md)。VLRB 正式指标和阶段日志使用 K=3 至少两票一致；保存产物中的运行时相对多数值仅用于诊断。

生成 root 实验的 `report` 对同一批完整 VLRB 预测离线计算训练 100、非训练 1147、去近重复 1146、未见幻觉 648 和官方类别切片，并保留 R0/S0/Final、各 root、配对变化及成本。三组变体均有完整报告时，seed 目录再输出 F5/G5/GN 配对汇总；全量 1247 指标包含训练样本。

旧 Gate/Cascade、joint、Phase17–22、类别专属 Rubric、十样例预热和迁移审计的代码与实验文档保存在 `codex/subtree-local-reflection` 分支；原 CritiQ-V 保存在 `CritiQ-V`。这些探索中的负结果仍属于研究历史，不能当作当前方法的运行入口。历史本地原始产物归档于 `output/archive/`，两个核心实验目录保留原位。整个 `output/` 仅本地保存，不再由 Git 跟踪；研究证据只提交必要的结果摘要、split 元数据和可编辑 PPT。

本地输出目录已按[目录归档清单](output-archive.md)整理：最新完整运行和核心对照保留原路径，历史探索按系列整目录迁移；原来跟踪的 268 个历史文件已取消 Git 跟踪，本地文件内容不改写。

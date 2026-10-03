# 实验索引

主要框架图为 [assets/framework.png](../../assets/framework.png)。方法概览、三个配置示例的区别与运行命令统一维护在仓库 [README](../../README.md)。当前 Hallucination100 参考配置使用 Strict + Preserve5；原 Discovery100 的 Covered/Strict 示例均使用 0 条保留案例。G5/GN 共用五样例预热历史，F5 为人工固定五根对照。

当前主线是[子树逐例反思与局部竞争](subtree-local-reflection/plan.md)。[最终实验结果](subtree-local-reflection/results.md)集中展示八行主对比表、当前配置和主要结论；[最新框架 PPT](subtree-local-reflection/framework.pptx)说明方法与实验。Hallucination100 的详细实验记录见[冻结方案及已观察结果](vlrb-hallucination100-generated-roots/plan.md)，可复现的 seed11 样本 ID 和来源哈希见 [split manifest](vlrb-hallucination100-generated-roots/seed11_split.json)。

G5 seed11 的 2026-10-03 重跑与历史运行对比已归档于生成 root 实验文档第 9 节，包含正式 R0/S0/Final、clean1146、配对统计、演化轨迹、耗时及负结果。PPT 第 11 页最后三行使用本次结果；文档第 8 节保留旧结果。两次初始化内容不同，不能把分数差直接解释为代码重构收益。

当前代码入口是 `experiments.evolving_structured_rubrics.run_subtree_experiment`：`prepare` 还原发现集，`roots` 创建 F5/G5/GN，`vlrb-r0` 测裸 root，`evolve` 运行 S0 与局部演化，`dev`/`vlrb` 做冻结后的独立评测，`report` 汇总结果。具体命令见仓库 [README](../../README.md)。VLRB 正式指标和阶段日志使用 K=3 至少两票一致；保存产物中的运行时相对多数值仅用于诊断。

生成 root 实验的 `report` 对同一批完整 VLRB 预测离线计算训练 100、非训练 1147、去近重复 1146、未见幻觉 648 和官方类别切片，并保留 R0/S0/Final、各 root、配对变化及成本。三组变体均有完整报告时，seed 目录再输出 F5/G5/GN 配对汇总；全量 1247 指标包含训练样本。

旧 Gate/Cascade、joint、Phase17–22、类别专属 Rubric、十样例预热和迁移审计的代码与实验文档保存在 `codex/subtree-local-reflection` 分支；原 CritiQ-V 保存在 `CritiQ-V`。这些探索中的负结果仍属于研究历史，不能当作当前方法的运行入口。历史本地原始产物继续保存在 `output/`，当前主线只提交必要的结果摘要、split 元数据和可编辑 PPT。

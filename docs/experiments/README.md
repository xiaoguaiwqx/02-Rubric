# 实验索引

VLRB 训练子集统一保存在本地 `data/VL_RewardBench/splits/`：Hallucination100 为 [hallucination100/seed11/discovery_100.jsonl](../../data/VL_RewardBench/splits/hallucination100/seed11/discovery_100.jsonl)，Reasoning70 为 [reasoning70/seed11/discovery_70.jsonl](../../data/VL_RewardBench/splits/reasoning70/seed11/discovery_70.jsonl)。两者引用已有共享图像，不复制图片；冻结 ID 和划分说明保留在各自实验文档中。

Reasoning70 的 seed11 训练集已按原 Hallucination100 方法抽取并固定：Reasoning 317 条中，现有 70 条作为训练集，其余全部 247 条作为留出集，不进行留出集近重复排除。训练实际包含 MathVerse 35、MMMU-Pro 35 条；未额外设置子任务配额。方法与数据位置见 [划分说明](vlrb-reasoning70/README.md)，冻结 ID 和来源哈希见 [split manifest](vlrb-reasoning70/seed11_split.json)。训练 JSONL 与共享图片仅保存在本地 `data/VL_RewardBench/`，尚未运行该划分的模型实验。

RLAIF-V 挖掘数据已于 2026-10-05 完整下载至 `data/RLAIF-V/`，来源为 [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)，版本 `cdfc8c13778434e38afd538b0641ea942df4af78`。14 个 Parquet 分片共 83,132 条、12,708,910,785 字节，全部通过官方 SHA-256 核对；图片字节包含在分片中。来源和问题类型分布见本地 `selection_summary.json`，版本、行数和哈希见 `download_manifest.json`，筛选示例见 `LOCAL_USAGE.md`。本次只准备数据，尚未选定挖掘子集、检查与 VLRB 的重叠或启动新实验；后续固定挖掘划分时应记录与全量 VLRB 1,247 条的去重规则。

Init Split 提示词重构与独立验证见[计划](init-split-prompts/plan.md)、[执行进度](init-split-prompts/tracker.md)和[实验结果](init-split-prompts/results.md)。在 init-split-prompts 分支复用已保存的 G5 seed11 五根 R0，比较初始化 S0 的正式 VLRB K=3 结果；英文模板 v1、偏好导向 v2 与顺序 children v3 均已完成。当前 v3 全量 Strict ACC 为 77.23%（963/1247），未参与初始化为 76.11%（873/1147），幻觉留出集为 86.73%（562/648）。相对 v2，全量净增加 1 个正确判断、未参与初始化净减少 4 个、幻觉留出集净增加 9 个；结果文档保留各版本、配对变化与主要负结果。

顺序 children v3 通过 `init --reuse-init-patterns --source-run` 复用 v2 的 signature/cluster，仅重新生成 children；各根在 children 阶段参考本轮前面已生成的子准则，signature/cluster 不读取前序 children。共生成 20 条子准则，仅新增 5 个 Manager 请求；本次归档范围止于 S0。范围与完整命令见上述计划文末。

主要框架图为 [assets/framework.png](../../assets/framework.png)。方法概览、三个配置示例的区别与运行命令统一维护在仓库 [README](../../README.md)。当前 Hallucination100 参考配置使用 Strict + Preserve5；原 Discovery100 的 Covered/Strict 示例均使用 0 条保留案例。G5/GN 共用预热历史，默认五样例，可通过 `roots --warmup-count` 设置；F5 为人工固定五根对照。

当前主线是[子树逐例反思与局部竞争](subtree-local-reflection/plan.md)。[最终实验结果](subtree-local-reflection/results.md)集中展示八行主对比表、当前配置和主要结论；[最新框架 PPT](subtree-local-reflection/framework.pptx)说明方法与实验。Hallucination100 的详细实验记录见[冻结方案及已观察结果](vlrb-hallucination100-generated-roots/plan.md)，可复现的 seed11 样本 ID 和来源哈希见 [split manifest](vlrb-hallucination100-generated-roots/seed11_split.json)。

G5 seed11 的 2026-10-03 重跑与历史运行对比已归档于生成 root 实验文档第 9 节，包含正式 R0/S0/Final、clean1146、配对统计、演化轨迹、耗时及负结果。PPT 第 11 页最后三行使用本次结果；文档第 8 节保留旧结果。两次初始化内容不同，不能把分数差直接解释为代码重构收益。

当前代码入口是 `experiments.evolving_structured_rubrics.run_subtree_experiment`：`prepare` 还原发现集，`roots` 创建 F5/G5/GN，`vlrb-r0` 测裸 root，`evolve` 运行 S0 与局部演化，`dev`/`vlrb` 做冻结后的独立评测，`report` 汇总结果。具体命令见仓库 [README](../../README.md)。VLRB 正式指标和阶段日志使用 K=3 至少两票一致；保存产物中的运行时相对多数值仅用于诊断。

生成 root 实验的 `report` 对同一批完整 VLRB 预测离线计算训练 100、非训练 1147、去近重复 1146、未见幻觉 648 和官方类别切片，并保留 R0/S0/Final、各 root、配对变化及成本。三组变体均有完整报告时，seed 目录再输出 F5/G5/GN 配对汇总；全量 1247 指标包含训练样本。

旧 Gate/Cascade、joint、Phase17–22、类别专属 Rubric、十样例预热和迁移审计的代码与实验文档保存在 `codex/subtree-local-reflection` 分支；原 CritiQ-V 保存在 `CritiQ-V`。这些探索中的负结果仍属于研究历史，不能当作当前方法的运行入口。历史本地原始产物归档于 `output/archive/`，两个核心实验目录保留原位。整个 `output/` 仅本地保存，不再由 Git 跟踪；研究证据只提交必要的结果摘要、split 元数据和可编辑 PPT。

本地输出目录已按[目录归档清单](output-archive.md)整理：最新完整运行和核心对照保留原路径，历史探索按系列整目录迁移；原来跟踪的 268 个历史文件已取消 Git 跟踪，本地文件内容不改写。

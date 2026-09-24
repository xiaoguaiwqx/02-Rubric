# 实验索引与维护约定

本页是研究实验的导航入口，不复制完整结果。主计划是历史设计与结果汇总；各实验的冻结协议描述当时约束，不代表当前进度。

| 实验 | 已确认状态 | 设计与结果 |
| --- | --- | --- |
| Hallucination100 本地 Split 初始化 | seed11 已准备独立配置与输出目录 | [设计与运行命令](vlrb-hallucination100-fresh-init/plan.md)；五个固定 root 在目标100条上初始化，再五轮局部演化；评估新 Init 与 Final，复用旧对照 |
| 子树逐例反思与局部竞争 | 已完成多组演化与外部评测 | [实现计划](subtree-local-reflection/plan.md) · [结果](subtree-local-reflection/results.md) · [可编辑框架图](subtree-local-reflection/framework.pptx)；含局部 Strict ACC 与 5 条随机正确案例对照 |
| VLRB Hallucination100 同组演化迁移 | seed11 完成：未见 Hallucination 留出集相对共同 Init 为 514→514/648；seed29/47 待运行 | [实验计划与 seed11 结果](vlrb-hallucination100-transfer/plan.md) · [运行指南](vlrb-hallucination100-transfer/running.md)；仅三个新 Final 各做一次完整 VLRB K=3 推理，旧两份对照复用历史预测 |
| Framework v6：五根 Split 初始化与分层反思演化 | 2026-09-17 实现及离线验证完成，未真实运行；122B thinking Manager，Worker 全局并发50 | [计划与完整运行命令](joint-split-27b/plan.md)、[框架PPT](joint-split-27b/framework-editable-v6.pptx)；S0之后最多五轮，严格优于当前版本才整批接受 |
| Phase17：27B Manager 替换 | 最小参数化实现完成；未正式运行 | [实验设置与运行命令](phase17-manager-qwen35-27b/plan.md)；只更换Manager、Worker端点与并发 |
| Phase19 系统级接受 | 已完成；尚无稳定泛化优势 | [主计划18.2节](../Evolving%20Structured%20Rubrics%20Implementation%20Plan.md#phase19) |
| Phase20 混合局部接受 | 历史探索；不作为新实验默认入口 | [冻结协议](phase20/protocol.md)；保留其已被后续协议引用的基线产物 |
| Phase21 子树原子竞争 | 已完成；25个候选全部拒绝 | [冻结协议](phase21/protocol.md)、[主计划18.3节](../Evolving%20Structured%20Rubrics%20Implementation%20Plan.md#phase21) |
| Phase21 反事实审计 | 结果已记录；不反向选择正式候选 | [主计划18.5节](../Evolving%20Structured%20Rubrics%20Implementation%20Plan.md#counterfactual-audit) |
| Phase22 全样本竞争 | 演化、heldout、VLRB均已完成；2/25操作接受 | [冻结协议](phase22/protocol.md)、[运行指南](phase22/running.md)、[主计划18.7节](../Evolving%20Structured%20Rubrics%20Implementation%20Plan.md#phase22) |

Phase22 的 Worker 是 Qwen3-VL-8B-Instruct，不是 Qwen3.5-27B。其 VLRB Strict ACC 为67.92%，相对 Initial净−2条、相对Phase17 E4净−40条；实际没有触发重新聚类。完整统计只维护在主计划18.7节。

## 哪些信息以哪里为准

Framework v6 在 `codex/framework-v6-clean` 上从 `cf4a799` 重新起步。该提交中 `docs/experiments/joint-split-27b/plan.md` 保存旧版27B协议；当前同路径明确描述新协议，旧实验状态和运行命令不适用于新框架。旧分支及产物保持原样。

- **设计约束**：对应实验的冻结协议及运行 manifest；结果不好也不回改历史假设。
- **当前运行状态**：该次运行的 `stage_status.json`、完整性检查与最终报告。旧 tracker 不覆盖实际产物。
- **研究结论**：主计划相应章节；以后优先维护现有章节，不再复制时间戳版总结。
- **运行命令**：该实验的 running 文档。更换模型、数据或科学协议必须使用独立身份与产物，不能覆盖旧实验。
- **代码验证**：测试与 Git 历史；历史 review 中的通过数只代表当时验证，不代表当前分支。

## 产物与草稿

`output/`、本地数据仓库、端点配置和缓存留在本地，不随清理删除。新运行产物默认忽略，已有 Git 跟踪的历史报告保留；新增精简结果经审核后放入 `docs/experiment-results/`。仅凭 `final_report` 文件名不能判定可提交，原报告可能含逐样本预测与本地路径。

`refine-logs/` 中有独立历史价值的协议与审计仍保留。重复的时间戳快照、自动 review 轮次只作可恢复本地归档，不作为当前状态入口。旧固定入口保留简短跳转，防止既有引用断裂。

以后每个实验只维护一个协议和一个运行指南；临时草稿写入 `.local/`。确需修改已冻结协议时，创建明确的新协议版本，不覆盖原版本，也不自动生成一组新的 PLAN/TRACKER/REVIEW 副本。

# Prompt v2 Aligned Evolution 实验 Tracker

## 冻结目标

- Treatment：从五个 initial roots 开始，使用 Prompt v2 完成完整 Locked-Split + Role-aware Refine 演化。
- Primary Control：Phase 10 Rubric + Prompt v2。
- discovery 接受数据：discovery-90。
- 外部评估：heldout-500 与 VL-RewardBench 均必跑，均不得反向选择 Rubric。

## 实现状态

| 项目 | 状态 | 产物 |
|---|---|---|
| Prompt v2 通用 Pairwise 入口 | DONE | `run_rubric_evolution.py` 中显式 `v1/v2_cache` 模式；默认 v1 不变 |
| Prompt-v2-aligned evolution | DONE | `prompt_v2_aligned_evolution.py` |
| 新 Rubric heldout-500 | DONE | evolution 的 `heldout/final-report` stages |
| 新 Rubric VL-RewardBench | DONE | `vl_rewardbench_prompt_v2_evolved.py` |
| 配置与 CLI stages | DONE | example config 与统一 runner |
| Offline regression | DONE | 302/302 unittest、compileall、diff-check |
| 在线 smoke/full run | PENDING | 由用户在双服务器在线时执行 |

## 里程碑

| ID | 阶段 | 状态 | 关键产物/验收 |
|---|---|---|---|
| M0 | 实现前协议冻结 | pending | config、Prompt v2 hash、五-root 起点、Phase 10 Control hash |
| M1 | Offline audit | pending | cache/request identity、可复用 initial-root predictions、heldout 隔离 |
| M2 | 单-root smoke | pending | P2 feedback→signature→cluster→children→competition 全链路 |
| M3 | Full discovery evolution | pending | 3–5 epochs、同步提交、final rubric hash |
| M4 | Discovery report | pending | 轨迹、接受/拒绝、节点增长、成本 |
| M5 | heldout-500 | pending | Initial P2 / Phase10 P2 / New P2 paired comparison |
| M6 | VL-RB freeze/audit | pending | 1,247 pairs、K=3 schedule、Control artifacts |
| M7 | VL-RB smoke | pending | 双 endpoint、swap 映射、parse valid |
| M8 | VL-RB full run/retry | pending | 全请求完成、失败最多重试10次 |
| M9 | VL-RB final report | pending | OverallAcc、MacroAcc、三类别、paired stats、成本 |
| M10 | 回归与文档 | pending | full unittest、compileall、Implementation Plan 结果记录 |

## 核心结果表（运行后填写）

### heldout-500

| 系统 | ACC | Coverage | Correct / 500 | Corrected | Harmed | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Initial five-root + P2 | TBD | TBD | TBD | - | - | - |
| Phase 10 \(R_{v1}\) + P2 | 76.6% | TBD | TBD | - | - | - |
| New \(R_{v2}\) + P2 | TBD | TBD | TBD | TBD | TBD | TBD |

### VL-RewardBench

| 系统 | OverallAcc | MacroAcc | General | Hallucination | Reasoning | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Initial five-root + P2 | 58.12% | 54.60% | TBD | TBD | TBD | TBD |
| Phase 10 \(R_{v1}\) + P2 | 69.53% | 63.37% | TBD | TBD | TBD | TBD |
| New \(R_{v2}\) + P2 | TBD | TBD | TBD | TBD | TBD | TBD |

## 实验审计

| 检查项 | 状态 | 备注 |
|---|---|---|
| 全部 discovery Worker 请求为 Prompt v2 | pending | |
| Prompt-v1 cache 不会误命中 | pending | |
| Prompt 改变后 signatures 未错误复用 | pending | |
| Split/Refine/aggregation 与 Phase 10 一致 | pending | |
| discovery 未读取 heldout/VL-RB | pending | |
| final rubric 在外部评估前冻结 | pending | |
| VL-RB 无按 heldout 结果选择是否运行 | pending | |
| K=3 独立请求与 A/B 映射正确 | pending | |
| 所有解析/技术失败可追溯 | pending | |
| 报告包含请求数、token、耗时与节点数 | pending | |

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-17 | 主要 Control 使用 \((R_{v1},P_{v2})\)，不是 \((R_{v1},P_{v1})\) | 隔离演化期 Prompt 对齐收益 |
| 2026-08-17 | 新 Rubric 必须运行 VL-RewardBench | 检验跨数据分布迁移并避免选择性外测 |
| 2026-08-17 | 不加入 Gate/Root Pre-Refine/后验权重 | 除明确记录的 Phase16-v2 Split 调度阈值外，不再引入其他结构变量 |
| 2026-08-17 | Prompt 改变时重新生成错误链路 | predictions 与 decisive-wrong IDs 可能改变 |
| 2026-08-17 | Phase16-v2 将专用 Split trigger 调整为 `ACC < 0.75, Coverage > 0.80` | Prompt v2 使三个 root 越过旧 0.70 阈值；本轮希望五个 initial roots 均先接受 Split 候选检验。全局 Split v1、竞争条件和 Refine 阈值不变 |

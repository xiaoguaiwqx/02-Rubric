# Split-retry v2 Experiment Tracker

| Run ID | 阶段 | 目的 | 数据 | 优先级 | 状态 | 验收/结果 |
|---|---|---|---|---|---|---|
| SRV2-M0-01 | 协议冻结 | 冻结 parent scope、ErrorSignatures、cluster、Worker/Manager identity | Visual discovery-90 | MUST | DONE | 已冻结 source candidate/combined/signatures；heldout 未访问 |
| SRV2-M0-02 | 离线 lock audit | 验证强 child 定义与兼容锁定选择 | archived Visual attempt | MUST | DONE | 锁定 peripheral child；locked-only 复算为 68/89 = 76.40% |
| SRV2-M0-03 | 回归测试 | 验证旧 Split 不漂移、hash 复用、partial gate | offline tests | MUST | DONE | focused tests 33/33、full unittest 216/216 与 compileall 通过 |
| SRV2-B1-01 | 固定 cluster retry 1 | sample-driven rewrite/replace 弱 children | Visual discovery-90 | MUST | TODO | locked hash 不变；完整 Specialized ACC 上升 |
| SRV2-B1-02 | 固定 cluster retry 2 | 在完整失败归因后做最后一次修复 | Visual discovery-90 | CONDITIONAL | TODO | 仅当 B1-01 未 full accept 时运行 |
| SRV2-B1-03 | 最终决策 | full accept 或 partial locked accept | Visual discovery-90 | MUST | TODO | 明确区分 full/partial/reject |
| SRV2-B2-01 | 稳定性 | 3 Manager seeds | Visual discovery-90 | CONDITIONAL | TODO | 仅在 single-seed 机制为正时运行 |
| SRV2-B2-02 | heldout diagnostic | 冻结 discovery candidate 后一次评估 | heldout-500 | CONDITIONAL | TODO | 不允许反向选择 candidate |

## 当前已知基线

| 项目 | 数值 |
|---|---:|
| Parent ACC | 62/89 = 69.66% |
| 当前 retry 最好完整 Specialized ACC | 61/89 = 68.54% |
| Archived strong child + parent | 68/89 = 76.40% |
| Archived strong child net corrected | +6 |

## 决策日志

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-10 | v2 不重新聚类 | 先隔离“保留强 child + 修复弱 child”是否有效，并降低 Manager 成本 |
| 2026-08-10 | 强 child 使用相对 parent 的净增益定义 | 不同 root 的基础能力不同，绝对 child ACC 阈值不可比 |
| 2026-08-10 | 最多两轮修复后允许 partial locked accept | 避免弱 siblings 永久阻止已验证局部专家进入 rubric，同时保留集体非退化约束 |

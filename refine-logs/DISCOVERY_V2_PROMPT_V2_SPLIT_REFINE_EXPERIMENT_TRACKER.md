# Discovery-v2 Prompt-v2 Split+Refine 实验 Tracker

## 状态

- 当前阶段：代码实现与离线验证完成，等待在线 smoke/full run。
- 目标分支：`discovery-data`。
- 实验目录：`phase17_discovery_v2_prompt_v2_split_refine_v1`。
- 外部评测目录：`vl_rewardbench_phase17_discovery_v2_prompt_v2_v1`。

## P0：数据隔离

- [x] 记录 Discovery100 中 2 个 heldout 重叠实例。
- [x] 记录 Dev150 中 6 个 heldout 重叠实例。
- [x] 用户授权保留重叠；heldout 降级为 exploratory regression check。
- [x] 验证 Discovery/Dev 强去重键交集为零。
- [x] 验证 Discovery+Dev 与 VL-RB 的 sample/image/pair 强重叠为零。
- [x] 保留 source-label exploratory 标记。

## P1：独立 runner 与协议冻结

- [x] 新增 Phase17 独立 config block。
- [x] 新增 `discovery_v2_prompt_v2_evolution.py` 独立 runner。
- [x] 从五个初始 roots 生成 Discovery100 Prompt-v2 baseline。
- [x] 禁止复用 Discovery90 prediction 和 ErrorSignature。
- [x] 冻结 Phase16 Split/locked retry/Refine/Global Memory 协议。
- [x] 双 endpoint available-slot + sample-major 调度。
- [x] freeze/audit stage 与 resume identity 校验。

## P2：Dev150 逐轮诊断

- [x] 生成 epoch-0 Dev baseline 的实现。
- [x] 每个 committed epoch 生成唯一 Dev report。
- [x] 共享 cache 只刷新新增/改写 criterion description。
- [x] 输出 Overall/domain/source/root/node 指标。
- [x] 输出 corrected/harmed/net corrected 与轨迹。
- [x] Dev loader 与 Manager/trigger/acceptance/early-stop 单向隔离。
- [ ] 测试改变 Dev 不改变最终 rubric hash。

## P3：Smoke 与完整演化

- [x] `discovery-v2-evolution-freeze` 已在临时输出完成离线验证。
- [x] `discovery-v2-evolution-audit` 已在临时输出完成离线验证。
- [ ] `discovery-v2-evolution-smoke`
- [ ] 验证 smoke 不访问 heldout/VL-RB。
- [ ] `discovery-v2-evolution-run`
- [ ] `discovery-v2-evolution-report`
- [ ] 冻结最后 committed rubric，不按 Dev 选 checkpoint。

## P4：heldout-500

- [x] final rubric 前置 hash 校验实现。
- [x] 只生成新/变更 criteria 的 Prompt-v2 prediction。
- [x] Initial/Phase10/Phase16/Phase17 同协议对照。
- [x] strict ACC、Coverage、paired transitions、McNemar、CI。
- [x] 明确 exploratory 与 selection forbidden。

## P5：VL-RewardBench

- [x] freeze/audit 数据与 baseline artifact 实现。
- [x] 20-sample smoke 覆盖双 endpoint、K=3 和 orientation mapping。
- [x] sample-major available-slot 完整运行。
- [x] 不跨 replicate 复用随机输出。
- [x] 最多 10 次技术/解析重试。
- [x] unresolved technical failures 单独报告。
- [x] OverallAcc、MacroAcc、三类 ACC、Coverage、paired comparison。
- [x] Initial、Phase10、Phase16 与 Phase17 四系统同协议结果表。
- [x] 不启用 Gate/Router/后验权重。

## P6：测试与交付

- [x] focused unittest：21/21 passed。
- [x] Split/Refine/Prompt-v2 regression。
- [x] 全量 unittest：341/341 passed。
- [x] compileall。
- [x] 实验相关文件 `git diff --check` 通过。
- [x] 生成紧凑 final report JSON/Markdown 的实现。
- [ ] 将实验设计与结果补入主实施文档。

## 已知风险

| 风险 | 当前处理 |
|---|---|
| source label 未经全量人工复核 | 全程标记 exploratory，不作 confirmatory claim |
| 100 条 discovery 方差较大 | 固定 seed 与协议；后续再做 multi-seed/human-reviewed replay |
| Dev 被误用作 checkpoint selector | 代码级单向隔离与 counterfactual test |
| VL-RB 请求量大 | 两端口、sample-major、description-hash 缓存、断点恢复 |
| 旧 Prompt/预测污染 | request identity 硬校验；Prompt-v1 和 Discovery90 artifacts 禁止复用 |

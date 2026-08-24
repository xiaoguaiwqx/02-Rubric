# Qwen2.5-VL Full Evolution Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics / Artifact | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q25E-M0-01 | M0 | 冻结数据、协议、端点和 Initial rubric | Qwen2.5 Initial | Discovery100 | frozen manifest、request specs、hash | MUST | TODO | 禁止复用 Qwen3 evolution artifacts |
| Q25E-M0-02 | M0 | 离线身份与 heldout 隔离审计 | Qwen2.5 Initial | Offline | audit checks | MUST | TODO | Qwen3 reuse count必须为0 |
| Q25E-M1-01 | M1 | 验证 fresh signature→candidate→competition | Qwen2.5 smoke | Discovery100 subset | validity、decision、endpoint calls | MUST | TODO | 不修改正式轨迹 |
| Q25E-M2-01 | M2 | 完整 Locked-Split + Role-aware Refine | Qwen2.5-specific | Discovery100 | 3–5 epoch evolution history | MUST | TODO | formal Final由协议自然停止产生 |
| Q25E-M2-02 | M2 | 每 epoch 独立诊断 | Qwen2.5-specific | Dev150 | Overall/Macro/domain ACC、Coverage | MUST | TODO | 不选择checkpoint |
| Q25E-M2-03 | M2 | 冻结 discovery Final | Qwen2.5-specific Final | Discovery100 | final rubric/prediction/report hash | MUST | TODO | heldout_accessed=false |
| Q25E-M3-01 | M3 | 同域 exploratory regression | Initial / Transferred E4 / Specific Final | heldout-500 | Strict/Covered ACC、paired、roots | MUST | TODO | 不改变Final选择 |
| Q25E-M4-01 | M4 | 冻结外部评测 | 三个Qwen2.5 systems | VL-RewardBench | data/rubric/schedule hashes | MUST | TODO | 复用既有K=3 schedule |
| Q25E-M4-02 | M4 | 外部 smoke | Specific Final | VL-RewardBench 20 | parse、双端点、Initial离线投影 | MUST | TODO | Native不重跑 |
| Q25E-M4-03 | M4 | 外部完整推理 | Specific Final | VL-RewardBench 1247×K3 | logical predictions | MUST | TODO | sample-major available-slot |
| Q25E-M4-04 | M4 | 技术失败恢复 | Specific Final | VL-RewardBench | recovery/still-failed/coverage | MUST | TODO | 不改变有效预测 |
| Q25E-M4-05 | M4 | 主结果与配对报告 | Initial / Transferred E4 / Specific Final | VL-RewardBench | Overall/Macro/Strict、McNemar | MUST | TODO | benchmark选择禁止 |
| Q25E-M5-01 | M5 | 机制与失败诊断 | Qwen2.5 vs Qwen3 | Offline | wrong overlap、root/node、position/cost | MUST | TODO | 不增加模型请求 |

## Implementation status

- [x] 新增独立 Phase18 Qwen2.5 evolution runner 与七个 stages。
- [x] Worker model、Prompt v2、temperature/max_tokens 和双端点 pool 使用实验局部配置，不修改旧协议。
- [x] ErrorSignature、cluster、candidate、prediction 与 failure history 均写入独立目录；该协议显式关闭 legacy/v1 signature cache 复用，Qwen3 evolution artifact 只允许作为报告对照。
- [x] 新增 heldout-500 Initial / transferred E4 / Qwen2.5-specific Final 三方诊断。
- [x] 新增 VL-RewardBench K=3 runner；只读复用已有 Qwen2.5 transferred E4 Control，不重复 Native。
- [x] 新增 focused tests、CLI stages 与冻结配置块。
- [ ] 实际执行 freeze/audit/smoke 和正式实验；上表状态在产物生成后更新。

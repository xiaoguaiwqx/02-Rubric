# Unified-Subtree + Global-Arbiter 对齐演化实验 Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R001 | M0 | 冻结协议与身份 | Phase19 aligned evolution | offline | hashes, request specs, heldout access | MUST | READY | 离线 `freeze` 已通过；正式运行会复用冻结身份 |
| R002 | M0 | 重放正式执行器 | Phase17 E4 Clean S5-v2 | VL-RB cached | exact prediction match | MUST | READY | Phase19 离线 audit 已通过；VL-RB audit 将执行 control exact replay |
| R003 | M1 | 端到端增量 smoke | one-root candidate | Discovery20 | parse, cache reuse, corrected/harmed | MUST | TODO | `aligned-evolution-smoke` |
| R004 | M2 | 完整 aligned evolution | Initial → 3–5 epochs | Discovery100 | Strict, accepted ops, attribution | MUST | TODO | `aligned-evolution-run` |
| R005 | M2 | 冻结 discovery 结果 | final aligned Rubric | Discovery100 | final Strict, trajectory | MUST | TODO | `aligned-evolution-report` |
| R006 | M2 | 每 epoch 独立诊断 | aligned trajectory | Dev150 | Overall/Macro/domain, corrected/harmed | MUST | TODO | run 内自动执行；禁止选择 |
| R007 | M3 | 内部 exploratory 泛化 | aligned final | heldout-500 | Strict, Coverage, paired delta | MUST | TODO | `aligned-evolution-heldout` |
| R008 | M4 | 外部协议冻结与审计 | aligned final vs controls | VL-RB | identities, K=3 schedule | MUST | TODO | `vlrb-aligned-evolution-freeze/audit` |
| R009 | M4 | 外部 smoke | aligned final | VL-RB 20 | parse, endpoint pool, report count | MUST | TODO | `vlrb-aligned-evolution-smoke` |
| R010 | M4 | 外部主运行 | aligned final | VL-RB 1,247 × K3 | Strict/Overall/Macro/Coverage | MUST | TODO | `vlrb-aligned-evolution-run` |
| R011 | M4 | 技术失败恢复 | same Prompt retry | failed calls only | unresolved failures | MUST | TODO | `vlrb-aligned-evolution-retry` |
| R012 | M4 | 主结果与 paired statistics | Initial / Phase17 E4 / aligned final | VL-RB | CI, McNemar, category/source | MUST | TODO | `vlrb-aligned-evolution-report` |
| R013 | M5 | rejected operator 归因 | all attempts | Discovery100 | failure types, transition chain | MUST | TODO | report 内离线分析 |
| R014 | M5 | joint interaction 诊断 | joint failures | Discovery100 | leave-one-root-out | MUST if triggered | TODO | 不用于子集搜索 |
| R015 | M5 | child leave-one-out 案例 | unexplained harmed cases | small subset | qualitative attribution | NICE | TODO | 不影响接受/选择 |

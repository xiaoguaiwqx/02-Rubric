# Visual Grounding Gate-only Experiment Tracker

Protocol v2 amendment: Gate inference uses the equivalent-model endpoint pool
`vllm-8000 + vllm-8001` with available-slot dynamic scheduling (20 slots per
endpoint, 40 global). The earlier single-endpoint v1 smoke remains isolated and
is not reused by v2.

Protocol v3 amendment: Gate JSON follows the Pairwise Worker mechanism: the
prompt requests JSON, the server performs ordinary decoding without
`response_format`, and the existing strict local parser/retry/fallback policy
enforces the routing contract. Dual-endpoint scheduling remains unchanged;
v1/v2 caches are not reused.

Exploratory heldout authorization: the frozen discovery gate remains failed
(`exact vector=0.70`; visual coverage `86/90` versus `87/90`), while the user
approved a separately labelled heldout diagnostic because mean per-child swap
agreement is `0.9125` and coverage differs by only one sample. Heldout results
must not be reported as a frozen-protocol pass.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| VG-G001 | M0 | 冻结 Phase10 Rubric、Prompt v2 votes 和 routing contracts | offline freeze | discovery-90 + heldout manifest | hash/identity completeness | MUST | DONE | `visual-gate-freeze`；未读取 heldout metrics |
| VG-G002 | M0 | 验证 all-children 离线重放 | frozen Control | discovery-90 | exact prediction replay | MUST | DONE | 90/90逐样本一致；Rubric hash=`17ad7a0b...` |
| VG-G003 | M1 | 验证 Gate schema 与 same-root 限制 | compact System Gate | discovery 20 samples | valid rate, invalid IDs, fallback | MUST | TODO | 无 A/B preference 输出 |
| VG-G004 | M2 | 运行 discovery Gate | Dynamic Gate | discovery-90 | subtree/full-M1 ACC, activation, conflicts | MUST | TODO | 90 个 Gate 请求 |
| VG-G005 | M2 | A/B 交换一致性 | Dynamic Gate swapped | frozen 20 samples | exact set match, status agreement | MUST | TODO | 不改变其他输入 |
| VG-G006 | M3 | 固定 subset 与 Gate 对照报告 | parent/all/best-fixed/gate/oracle | discovery-90 | ACC, coverage, corrected/harmed | MUST | TODO | 冻结 best subset 后才能 heldout |
| VG-G007 | M4 | 一次性 Gate 泛化 | same frozen variants | heldout-500 | subtree ACC, full M1, CI, McNemar | MUST | TODO | exploratory；500 个 Gate 请求 |
| VG-G008 | M5 | 最终机制与效率报告 | paired comparison | discovery + heldout | routing gain, conflict reduction, req/min | MUST | TODO | 禁止依据 heldout 改策略 |

# Shared-Output Pool v1 实验摘要

## 实验身份

本实验使用同一 backend pool 生成冻结的 Pairwise、Gate 和 Root Router artifact，所有系统准确率只由离线 shared-output replay 计算。heldout-500 在本阶段属于工程评估集，不再解释为未见测试集。

- Criteria source SHA-256：`20FA11AB4516CBC510E2AE65D5605137D086676F7821D6A65EC20C1355C7F0ED`
- Dataset SHA-256：`FA57CA429469D3C2C014E7AEF42801DE43BEFAD3FC600738DA7C9FB12FD312C7`
- Rubric SHA-256：`71ead5d87681f6fe6cc1e0a7472db7ab4270512dc6c58000f30589ba8c8d006c`
- Config SHA-256：`55fc716efa42adfbec600c75868c904a5bc44e94a74640cc02461beeb8293708`
- Backend pool 逻辑身份：`qwen3vl8b-vllm-pool-v1`，共同模型为 `Qwen/Qwen3-VL-8B-Instruct`
- Endpoint equivalence：Pairwise agreement `93.33%`，Gate edge-activation agreement `100%`，Router selection agreement `100%`

## heldout-500 结果

| Pairwise | 系统 | Accuracy | Coverage |
|---|---|---:|---:|
| P05 | B1 Flat | 0.688 | 0.966 |
| P05 | H1 Hierarchical-only | 0.692 | 0.970 |
| P05 | G1 Gating-only | 0.690 | 0.970 |
| P05 | M1 All-roots Cascade | 0.692 | 0.972 |
| P05 | M2 Routed-roots Cascade | 0.674 | 0.966 |
| P00 | B1 Flat | 0.682 | 0.974 |
| P00 | H1 Hierarchical-only | 0.680 | 0.966 |
| P00 | G1 Gating-only | 0.678 | 0.972 |
| P00 | M1 All-roots Cascade | 0.674 | 0.964 |
| P00 | M2 Routed-roots Cascade | 0.664 | 0.960 |

P05 成功恢复到接近 exp4 的 `0.696` 基线；静态 strict forest 的 H1/M1 只带来 `+0.4` percentage point，尚不足以支持当前 Cascade 结构有效。M2 相对 B1 下降 `1.4` points，说明当前 Root Router 的裁剪会遗漏有用 criteria。

## reserve-50 的 P05 随机方差

| 系统 | 三次平均 Accuracy | 样本标准差 |
|---|---:|---:|
| B1 Flat | 0.653 | 0.031 |
| H1 Hierarchical-only | 0.660 | 0.040 |
| G1 Gating-only | 0.653 | 0.031 |
| M1 All-roots Cascade | 0.653 | 0.031 |
| M2 Routed-roots Cascade | 0.653 | 0.012 |

三次采样波动说明小幅单次增益不能独立作为方法证据；后续演化应使用 paired shared-output 反馈，并报告重复运行或置信区间。

## 结论

- `PASS_BASELINE_RECOVERY`：Pairwise P05 B1 达到 `0.688`，双通道成功消除了 Structured Worker 导致的主要语义漂移。
- `REVISE_CASCADE`：当前手工静态 forest 只产生很小的增益，需要由演化过程优化结构、边条件或 criteria，而不是宣称静态结构已经有效。
- `REVISE_ROOT_ROUTER`：M2 明显低于 M1/B1。Router 对部分低选择率但有判别力的 roots 存在 omission，后续应改进 root descriptions、训练/反馈信号或路由回退策略。
- Pairwise、Gate、Router、Rubric Forest、Cascade Executor、cache、trace 与 backend pool 已形成可复用验证基础设施，可以为下一阶段 Rubric evolution loop 提供离线反馈信号。

## Artifact 归档

完整 prediction、cache、trace、telemetry 和 manifest 不进入 Git，已归档到：

```text
D:\3-Work\02-DD-LLM\CritiQ-experiment-archive\evolving-structured-rubrics\2026-08-03\
```

归档清单：`archive_manifest.json`。归档前后文件数、总字节数和关键 artifact SHA-256 均已核对一致。

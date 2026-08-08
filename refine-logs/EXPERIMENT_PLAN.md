# Experiment Plan

**Problem**: ErrorSignature 中的视觉事实和 A/B 偏好方向错误可能污染 Split 的聚类与 child generation。
**Method Thesis**: 使用更强的多模态 Qwen3.5-397B 生成 ErrorSignature，应先提升 paired signature 质量，再验证其是否改善完整 Split。
**Date**: 2026-08-06

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: 397B 生成的 ErrorSignature 比 8B 更可靠 | 签名是聚类和 child generation 的上游表示 | 同一 27 样本上，方向、视觉事实、内部一致性、可执行性四项人工 paired audit 明显更优 | B1 |
| C2: 更可靠签名能改善 Split | 验证签名质量不是装饰性改进 | 在 B1 通过后，完整 Split 的 target-cluster accuracy、跨簇错误和 collective Fitness 优于原流程 | B2 |

Anti-claim：最终变化只是下游随机 Pairwise Worker 噪声，而非 ErrorSignature 模型贡献。

## Paper Storyline

- Main paper must prove: 强模型改善上游错误表示，并将收益传递到 Split。
- Appendix can support: 逐样本 signature qualitative cases 与 token/latency 成本。
- Experiments intentionally cut: 在 B1 失败前调整聚类、child prompt 或 Fitness。

## Experiment Blocks

### Block 1: Paired ErrorSignature model ablation

- Claim tested: C1。
- Dataset / split / task: discovery-90 中冻结的 27 个 parent decisive-wrong 样本。
- Compared systems: Qwen3-VL-8B-Instruct vs Qwen/Qwen3.5-397B-A17B。
- Controlled: image、question、A/B、gold、parent vote/thought、prompt、temperature=0.2、seed=42、input_mode=multimodal。
- Treatment: 仅模型和服务后端身份变化。
- Metrics: valid rate；人工审计 direction consistency、visual fact consistency、internal consistency、actionability；preferred-signature win rate。
- Success criterion: 397B 在四项错误率中均不劣，且 preferred-signature 胜率明显超过 50%；contextual 六样本不再出现系统性方向矛盾。
- Failure interpretation: 大模型替换不能修复表示设计，应修改 ErrorSignature schema/prompt。
- Priority: MUST-RUN。

### Block 2: Gated end-to-end Split propagation

- Claim tested: C2。
- 前置门槛: B1 通过后才实现并运行。
- Compared systems: 已完成的 8B-signature Split vs 397B-signature Split。
- Metrics: target-cluster correct/support、non-target wrong、平均 child coverage、重叠激活、collective Fitness、M1 ACC。
- Setup: downstream cluster/child/Pairwise/Fitness 配置保持当前版本；建议 Pairwise 至少 3 seeds 或 temperature=0 复核，避免单次 temperature=0.5 噪声。
- Success criterion: collective Fitness 提升，且收益来自 target correctness 增加而非 coverage 膨胀；M1 不退化。
- Failure interpretation: 瓶颈转向 cluster、child applicability 或 Pairwise Worker。
- Priority: MUST-RUN after gate。

### Block 3: One-shot heldout Visual-subtree generalization

- Claim tested: discovery-90 上 Visual grounding 子树优于完整 M1 的现象能否泛化到 heldout-500。
- Prerequisite: R003-R006 已完成；candidate、children、Pairwise prompt 与聚合规则全部冻结。
- Primary variant: parent + 全部四个 397B-signature children。
- Pre-registered comparators: parent-only、discovery 上预先选定的 spatial+direct 子集、原完整 M1。
- Inference: 仅四个新 children 请求本地 vllm-8001；复用已有 heldout parent/root Pairwise artifact。
- Metrics: ACC、coverage、Wilson 95% CI、paired corrected/harmed、exact McNemar、逐 child ACC/coverage、激活重叠与冲突。
- Data policy: heldout 仅查看一次；报告生成后禁止依据结果筛选、改写或调参。后续开发必须使用新的 validation/test split。
- Priority: MUST-RUN one-shot final diagnostic。
## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 配置和冻结身份检查 | CLI preflight | prompt/decoding/sample IDs 完全一致 | 无模型推理 | API key/模型多模态兼容 |
| M1 | 生成397B paired signatures | R001 | 27/27 valid | 27 次串行远程请求 | 成本和中断；支持 shard 续跑 |
| M2 | 生成 paired report | R002 | 完成人工四项审计 | 无模型推理 | 主观判断；保留逐项 notes |
| M3 | 完整 Split treatment | R003-R006 | 仅在 M2 通过后启动 | 聚类、child、90样本 Pairwise | 下游采样噪声 |
| M4 | 一次性 Visual heldout 泛化 | R007 | 运行前冻结全部版本 | 4×500 child Pairwise，仅8001 | heldout 泄漏；生成报告后禁止调参 |

## Compute and Data Budget

- R1: 27 次 Qwen3.5-397B 多模态请求，远程并发固定为 1。
- R2: 本地离线报告，无 GPU/API 请求。
- Heldout-500: 仅允许 R007 一次性最终诊断；报告生成后禁止用于选择或调参。
- Biggest bottleneck: paired signature 人工视觉审计。

## Risks and Mitigations

- Pairwise temperature=0.5 的单次评估噪声：B2 使用多 seed 或确定性复核。
- 397B 过拟合 gold：审计可执行性，禁止把 “human-preferred” 当作决策规则。
- 旧结果被覆盖：新结果固定写入 `phase6_split_signature_qwen35_397b`。

## Final Checklist

- [x] Novelty/model contribution isolated in B1
- [x] Must-run and gated downstream runs separated
- [x] Existing outputs protected
- [x] B1 manual audit completed
- [x] B2 implemented after gate
- [x] B2 execution completed
- [x] R007 variants and heldout policy frozen in code
- [ ] R007 one-shot heldout execution completed
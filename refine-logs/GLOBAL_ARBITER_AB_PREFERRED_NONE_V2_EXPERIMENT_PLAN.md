# Global Arbiter A/B-preferred + None-tolerant v2 实验计划

**问题**：旧 S5 的提示词要求 A/B，但少数技术失败曾进入额外 tie-break rescue，因而无法严格判断“同一提示词下允许模型语义性输出 `None`”是否改变结果。

**方法主张**：保持 Arbiter 模型输入和解码协议不变，仅把 `None` 从解析失败改为有效弃权，并完全移除专用 rescue，即可得到可解释、可复现的 Strict ACC、Coverage 与 OverallAcc。

## Claim Map

| Claim | 最小可信证据 |
|---|---|
| C1：旧 S5 的高覆盖是否来自模型自身，而非 rescue | 全新生成全部 3,741 个 Arbiter 输出；无 rescue；统计原生 `None` 数量与 Coverage |
| C2：解析语义变化是否影响最终性能 | 与旧 S5、S4、S3、S0 在同一 1,247 样本和冻结 K=3 schedule 上做 paired comparison |

## 冻结协议

- 数据：VL-RewardBench 1,247 对。
- Rubric：Phase17 E4。
- 子树证据：只读复用 S3 的 18,705 份报告（1,247 × 3 × 5）。
- Arbiter：每个 replicate 输入同 replicate 的五份子树报告，再对三个 Arbiter 判断做 K=3 多数聚合。
- System/User Prompt：与旧 S5 字节一致，仍明确偏好输出 A/B。
- 解析：`A`、`B`、`None` 均为合法语义输出。
- 重试：仅 transport、空输出、非法 JSON、非法 label；同一提示词最多额外 10 次。
- 禁止：`None` 不触发重试；不使用 tie-break/rescue prompt；不使用 benchmark 结果选择变体。
- 解码：temperature=0.5，max_tokens=2048，generation seed 不设置。
- 调度：两个 endpoint，sample-bundle available-slot affinity。
- 隔离：新目录、新 protocol/request kind、新 cache namespace；旧 Arbiter cache 不复用。

## 对照与指标

| 系统 | 作用 |
|---|---|
| S0 Explicit Recursive | 原始全节点递归投票 |
| S3 Unified Subtree | 五子树报告等权聚合 |
| S4 Global Arbiter (`None` allowed prompt) | 允许弃权的旧全局仲裁 |
| Legacy S5 | 旧 A/B-only + 历史 rescue 结果，只读参考 |
| Clean S5-v2 | 本实验：A/B-preferred prompt + 原生 `None` 解析 |

主要报告 Strict ACC、OverallAcc（覆盖内）、MacroAcc、Coverage、correct count、A/B/None 分布、paired corrected/harmed/net 和 Exact McNemar p。Strict ACC 是论文主指标。

## 执行顺序

1. freeze：冻结数据、Rubric、schedule、提示词 hash、S3/S4/Legacy S5 artifact hash。
2. audit：离线验证请求数、提示词、解析器、子树复用和对照身份。
3. smoke：20 样本 × 3 replicate。
4. run：完整 1,247 × 3；全新 Arbiter 请求。
5. retry：只补技术失败，成功项从本实验 cache 读取。
6. report：生成最终 JSON/Markdown 和 paired analysis。

预计完整 Arbiter 推理约 35–45 分钟；实际取决于两个服务的尾部生成。


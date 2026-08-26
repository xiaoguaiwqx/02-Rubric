# Global Arbiter A/B-only v1：VL-RewardBench 实验计划

**日期**：2026-08-26  
**实验性质**：exploratory aggregation ablation  
**问题**：S4 Global Arbiter 的覆盖内准确率达到 71.64%，但 73 个最终 `None` 将 Coverage 降至 94.15%，Strict ACC 仅为 67.44%。本实验检验去掉语义 abstention 后，Global Arbiter 的真实二元偏好能力能否超过 S0 Explicit Recursive。

## 1. Claim map

| Claim | 最小可信证据 |
|---|---|
| C1：S4 的主要瓶颈是过度输出 `None`，而不是全局仲裁能力不足 | S5 在 100% Coverage 下显著提高 Strict ACC，并接近或超过 S0 的 70.01% |
| C2：A/B-only 的收益来自新的二元仲裁策略，而不是对历史 `None` 的后处理 | 全部 3,741 个 replicate-level Arbiter 请求使用新 Prompt 重新生成，不只补跑旧 `None` |
| Anti-claim：覆盖内指标因选择性拒答虚高 | S5 parser 只接受 A/B；技术失败恢复后最终语义 `None` 数必须为 0 |

## 2. 冻结对照与唯一变量

| 系统 | 聚合方式 | 本轮请求 |
|---|---|---:|
| S0 Explicit Recursive | 27 nodes递归聚合后K=3 | 只读复用 |
| S3 Unified Subtree | 五root等权聚合后K=3 | 只读复用 |
| S4 Global Arbiter | 五份同replicate subtree reports到可 abstain Arbiter，最后K=3 | 只读复用 |
| S5 Global Arbiter A/B-only | 五份同replicate subtree reports到强制二元 Arbiter，最后K=3 | 3,741 |

冻结：VL-RewardBench 1,247条、Phase17 E4 Rubric、S3完整reports、K=3 schedule、Qwen3-VL-8B-Instruct、temperature=0.5、max_tokens=2048、不设置generation seed、两个available-slot endpoints。唯一改变是 Arbiter 的最终决策空间由 `A/B/None` 改为 `A/B`。

## 3. Prompt 与执行顺序

System Prompt 保留 S4 的相关证据综合、视觉事实优先和禁止机械计票要求，并将决策规则改为：

> Return a relative preference for every pair. If both responses are imperfect or the evidence is limited, select the response with stronger support and the less severe error. Do not abstain.

JSON schema 保持四字段，只允许：

```json
{
  "analysis_a": "Analyze A using the image and subtree evidence.",
  "analysis_b": "Analyze B using the image and subtree evidence.",
  "thought": "Integrate the evidence into one relative preference.",
  "answer": "A / B"
}
```

正式顺序保持：

```text
replicate k: five same-replicate reports -> A/B-only Arbiter -> decision k
final: K=3(decision 1, decision 2, decision 3)
```

不允许先进行root K=3、五-root多数投票或只补跑S4的None。

## 4. Stages 与 artifacts

```text
vlrb-global-arbiter-ab-only-freeze
vlrb-global-arbiter-ab-only-audit
vlrb-global-arbiter-ab-only-smoke
vlrb-global-arbiter-ab-only-run
vlrb-global-arbiter-ab-only-retry
vlrb-global-arbiter-ab-only-report
```

输出：

```text
output/evolving_structured_rubrics/vl_rewardbench_global_arbiter_ab_only_v1/
```

除常规 manifest、prediction、parse failure、efficiency 和 final report 外，必须输出：

- `former_s4_none_subset.json`：S4最终73个None在S5中的准确率与类别分布；
- `s4_to_s5_transition.json`：correct/wrong/None到S5 correct/wrong的转移；
- `root_majority_override_audit.json`：仅离线诊断S5如何处理root冲突；
- `paired_comparison.json`：S5分别对S4、S3、S0的McNemar比较。

## 5. 指标与解释标准

主要指标：Strict ACC、绝对正确数、Macro Strict ACC、Coverage。S5最终技术失败为0时，OverallAcc应等于Strict ACC，Coverage应为100%。

关键阈值：

- S4当前841条正确；S0当前873条正确；
- 原S4的73条None中，S5至少判断正确32条可追平S0；
- 判断正确33条，即45.2%，可超过S0；
- S5超过S0且paired检验不显示明显退化：支持A/B-only Global Arbiter；
- S5高于S4但低于S0：覆盖问题被修复，但仲裁仍不及Explicit Recursive；
- S5不高于S4：强制二元决策改变了已有A/B判断，Prompt本身引入新的错误。

## 6. 测试与预算

必须验证：parser拒绝`None`；每条样本恰好三个新请求；成功调用均为A/B；S0/S3/S4 hashes与结果精确复现；S5正式路径不调用五-root聚合；失败最多重试10次；report前技术失败必须为0；Coverage为100%；full unittest与compile通过。

预计请求3,741个，按S4实测约109.8请求/分钟，预计完整run约34分钟，smoke约1分钟，freeze/audit/report为离线阶段。

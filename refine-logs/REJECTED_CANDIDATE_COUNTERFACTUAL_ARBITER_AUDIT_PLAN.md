# Phase21 Rejected Candidate Counterfactual Arbiter Audit

## 1. 研究问题

Phase21 的 25 个 Split bundle 都被 root-scope Strict ACC 规则拒绝，但部分候选在把
`None` 视为选择性弃权后具有更高的选择性效用。本实验验证：

> root-local 选择性效用是否能够转化为五棵子树经过 Global Arbiter 后的系统收益？

这是 Discovery100 上的机制审计，不是 heldout 泛化实验，也不用于从 25 个候选中选择
新的正式 checkpoint。

## 2. 唯一反事实改动

对每个 rejected candidate：

1. 读取该候选已经缓存的完整 Unified-Subtree report；
2. 读取同一 epoch 的五棵 epoch-start baseline reports；
3. 只替换候选所属 root，另外四棵保持不变；
4. 使用冻结的 Clean S5 Global Arbiter 重新生成最终判断；
5. 使用 Discovery100 的人工 A/B gold 计算系统 Strict ACC。

不重新生成 Unified-Subtree report，不调用 Split/Refine Manager，不修改候选 Rubric。

## 3. 两阶段运行

### Stage 1：预冻结正负候选

| 角色 | 候选 |
|---|---|
| positive | Clarity E1 |
| positive | Visual Grounding E2 |
| borderline | Factuality E1 |
| negative | Completeness E4 |
| negative | Visual Grounding E5 |
| negative | Creativity E5 |

共 `6 × 100 = 600` 次新 Arbiter 请求。名单在看到结果前冻结，Stage 1 只验证方向和
整条反事实管线，不作为是否报告 Stage 2 的性能门槛。

### Stage 2：补齐全部候选

对剩余 19 个候选生成 `19 × 100 = 1,900` 次 Arbiter 结果。随后对全部 25 个候选
离线计算相关性，不再产生模型调用。

## 4. 指标

在每个 root 的 frozen baseline scope 上定义：

$$
U_r
=
\frac{N_{\mathrm{correct}}-N_{\mathrm{wrong}}}{|\mathcal S_r|},
$$

其中正确 A/B 为 `+1`，错误 A/B 为 `-1`，`None` 为 `0`。主要自变量为候选相对
baseline 的 $\Delta U_r$，主要因变量为：

$$
\Delta\operatorname{StrictAcc}_{\mathrm{system}}
=
\operatorname{StrictAcc}(\text{one-root counterfactual})
-
\operatorname{StrictAcc}(\text{common Phase21 epoch-0 control}).
$$

同时报告：

- 当前 Phase21 formal Strict net corrected；
- covered ACC 与 Coverage 变化；
- full-data 选择性效用；
- 系统 corrected、harmed、net、A/B/None 分布；
- Pearson、Spearman、Kendall tau-b；
- root fixed-effect slope/correlation；
- 按 root 分层 bootstrap 的 Spearman 95% CI；
- 请求量、token、错误和运行时间。

## 5. 输出与结论边界

输出目录：

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase21_rejected_candidate_counterfactual_arbiter_audit_v1/
```

重要文件：

- `frozen_manifest.json`
- `offline_audit.json`
- `predictions/<candidate_id>.json`
- `candidate_results.csv`
- `final_report.json`
- `final_report.md`

若选择性效用比 formal Strict net 更稳定地预测系统收益，则后续 Phase22 可以把选择性
效用作为 root-local acceptance 候选；若无相关性，则应优先重新设计 root-local
supervision，而不是继续调整接受阈值。

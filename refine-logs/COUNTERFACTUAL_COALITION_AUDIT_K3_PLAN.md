# K=3 Counterfactual Coalition Audit

## 目的

本实验只回答一个机制问题：Phase21 中由单棵 root 的 Unified-Subtree
局部收益定义的候选效用，能否在五棵子树共同进入 Global Arbiter 后转化为
系统收益，以及多个 root 候选之间是否存在明显的互补或冲突。

这是 Discovery100 上的缓存级、同样本 counterfactual audit。结果只用于诊断
接受目标，不用于事后选择 Rubric、checkpoint 或正式测试配置。

## 冻结协议

- 数据：Discovery100 的相同 100 条样本与顺序。
- 候选：Phase21 五个 epoch、每轮五个 root 的 25 个候选。
- 输入：只复用 Phase21 已缓存的完整 Unified-Subtree reports；不生成子树报告。
- Arbiter：Clean S5-v2 的相同模型、prompt、parser、温度和解码设置。
- 每个系统运行三次，逐样本以 `A/B` 至少两票形成 K=3 结果；其余情况为
  `None`，并按 Strict ACC 计错。
- 旧 K=1 baseline 和 25 个 singleton 调用严格作为 replicate 0 复用，新增
  replicate 1、2。

## 两阶段运行

### 1. K=3 singleton 稳定性

先把共同 baseline 与全部 25 个单-root counterfactual 升级到 K=3，重新计算：

- 每个候选的 corrected、harmed、net corrected；
- K=1 与 K=3 的收益符号一致率、Pearson 和 Spearman；
- 原局部选择性效用与 K=3 系统 Strict ACC 增量的相关性。

### 2. E1/E5 全 coalition 穷举

在 Epoch 1 和 Epoch 5 分别穷举五个 root 候选的全部 $2^5=32$ 个子集。
空集是共同 baseline，单元素集合复用 singleton K=3，新增运行 26 个多元素
coalitions。

对每个子集 $S$ 定义：

$$
v(S)=N_{\mathrm{correct}}(S)-N_{\mathrm{correct}}(\varnothing).
$$

报告：

- 全 32 个子集的 Strict ACC、Coverage、corrected、harmed 和净收益；
- 精确最优子集（并列时优先更小子集，再按固定 mask）；
- 所有正 singleton 的联合提交结果；
- exact Shapley value；
- 两两交互 $v(\{i,j\})-v(\{i\})-v(\{j\})+v(\varnothing)$；
- 最优子集的 paired bootstrap 稳定性。

## 调用预算

| 部分 | 逻辑请求 | 复用 | 新 Arbiter 请求 |
|---|---:|---:|---:|
| baseline + 25 singleton，K=3 | 7,800 | 2,600 | 5,200 |
| E1/E5 各 26 个多-root coalition，K=3 | 15,600 | 0 | 15,600 |
| 合计 | 23,400 | 2,600 | 20,800 |

新增 Unified-Subtree 请求恒为 0。Smoke 使用正式 cache namespace，成功请求可被
后续全量直接复用。

## 输出

目录：

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase21_counterfactual_coalition_audit_k3_v1/
```

主要文件：

- `frozen_manifest.json`：样本、候选、源报告和调用预算身份；
- `k3_singleton_results.csv`：全部 25 个候选的 K=1/K=3 对照；
- `k3_local_predictor_correlations.json`：局部指标到 K=3 系统收益的相关性；
- `epoch_01/coalition_results.csv`、`epoch_05/coalition_results.csv`：全子集；
- `shapley_values.json`、`pairwise_interactions.csv`：贡献与交互；
- `final_report.json`、`final_report.md`：最终机制审计报告。

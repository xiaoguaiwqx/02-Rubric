# Phase22 全样本净收益与自适应重新聚类实验计划

> 冻结协议，保留实验前设计，不作为当前运行进度。实验已完成；结果见[实验索引](../README.md)及主计划18.7节。文中的计划阶段、预算与验收条目是历史约束，不代表仍待执行。

**Problem**：Phase21 已以完整 root subtree 为 Split/Refine 原子候选，但只在 baseline 输出 A/B 的 selective scope 上计算收益，而且 Split 失败后固定复用原 clusters。Phase22 直接组合两项变化：所有候选使用全样本 +1/-1/0 净收益接受；Split 竞争失败后，根据失败归因选择复用 clusters 或重新聚类。

**Method Thesis**：每个完整 root subtree 只与自己的旧版本竞争。在全部优化样本上，旧错新对记 $+1$、旧对新错记 $-1$、其余记 $0$，总收益严格大于 $0$ 才原子接受；若 Split 被拒且失败来自语义分解，则下一次尝试重新聚类并重新生成完整 child bundle。

**Date**：2026-09-02
**Experiment ID**：phase22_all_sample_subtree_adaptive_recluster_evolution_v1
**Source protocol**：phase21_unified_subtree_bundle_evolution_v1

## 1. 实验定位

这是一个单一组合实验，不再拆成 MetricOnly A 和 Recluster B 两条 trajectory：

~~~text
Phase21
  + all-sample net-gain acceptance
  + failure-aware Split reclustering
  = Phase22
~~~

Phase22 从与 Phase21 相同的 Initial rubric 开始独立演化，不从其他 Phase22 分支的 final rubric 继续。

相对 Phase21，Phase22 同时改变：

1. candidate acceptance metric；
2. rejected Split 的 retry cluster policy。

因此最终结果评价的是**组合方案整体效果**。如果 Phase22 优于或劣于 Phase21，不能严格把差异单独归因给其中一个机制。第一轮不额外运行因子化消融。

## 2. 工作假设与主张

对 roots $T_1,\ldots,T_R$，采用近似局部可分解假设：

$$
J_{\mathrm{system}}(T_1,\ldots,T_R)
\approx
\sum_{r=1}^{R}w_rJ_r(T_r)+\epsilon.
$$

其中 $J_r(T_r)$ 是 root-local Unified-Subtree 正确性，$\epsilon$ 表示 root 间交互、Global Arbiter 行为及数据分布影响。本实验不假设 $\epsilon=0$，但暂不让它进入局部候选接受。

### Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Blocks |
|---|---|---|---|
| C1：全样本净收益可作为统一的 Split/Refine 接受指标 | 它完整统计 baseline None 在内的所有旧错新对和旧对新错 | 每个候选都有全样本 ledger；所有 accepted operations 均满足 $G_r>0$；结果可从 sample artifacts 重算 | B1、B2 |
| C2：当失败来自错误分解时，重新聚类可突破固定 cluster retry 的搜索限制 | 如果 cluster 本身错误，只重写 children 无法修复分解 | 所有重新聚类均由冻结规则触发，cluster membership 确实变化，并产生可解释的后续 candidate outcome | B1、B2、B4 |

### Anti-claims

本实验不声称：

- 每个局部正收益一定导致全局正收益；
- 全样本 metric 消除了数据量或数据分布问题；
- 自适应重新聚类一定优于固定 clusters；
- Global Arbiter 可以被 root-local accuracy 完全替代；
- Phase22 相对 Phase21 的差异能被严格拆分成 metric effect 与 reclustering effect；
- 先前事实实验可直接决定本次候选或 retry action。

先前实验结果不进入 trigger、候选生成、retry action、接受、早停或 checkpoint 选择。

## 3. 保持不变的 Phase21 机制

| Component | Phase22 |
|---|---|
| Initial rubric 与 Discovery100 | 与 Phase21 相同 |
| Dev150、heldout500、VL-RewardBench | 相同数据与数据角色 |
| Unified-Subtree Worker | 相同 model、prompt、parser、temperature、max tokens |
| Manager 与 Global Arbiter | 相同 model 与 runtime |
| Split/Bundle Refine trigger | 与 Phase21 相同 |
| ErrorSignature 来源 | 与 Phase21 相同，全部来自 Unified-Subtree Worker |
| 首次 Split clustering | 与 Phase21 相同 |
| child bundle generation | 与 Phase21 相同 |
| Bundle Refine proposal 与 mutation scope | 与 Phase21 相同 |
| strong child locking / partial acceptance | 均为 False |
| candidate unit | complete root subtree |
| candidate commit | 所有独立通过 roots 同步提交 |
| Global Arbiter | post-commit diagnostic only |
| epochs | 最少 $3$、最多 $5$ |
| heldout/VL-RB | 不参与选择 |

Phase22 只额外启用 all-sample acceptance 和 failure-aware reclustering。

## 4. 全样本收益定义

对 root $r$、全部 Discovery 样本 $D=\{x_i\}_{i=1}^{N}$，baseline prediction 为 $p_i^0$，candidate prediction 为 $p_i^1$，gold 为 $y_i$。

逐样本收益：

$$
g_i
=
\mathbf{1}[p_i^1=y_i]
-
\mathbf{1}[p_i^0=y_i].
$$

等价地：

$$
g_i=
\begin{cases}
+1, & p_i^0\neq y_i\ \land\ p_i^1=y_i,\\
-1, & p_i^0=y_i\ \land\ p_i^1\neq y_i,\\
0, & \text{otherwise}.
\end{cases}
$$

完整 root candidate 的收益：

$$
G_r
=
\sum_{i=1}^{N}g_i
=
N_{\mathrm{corrected}}-N_{\mathrm{harmed}}.
$$

Split 和 Refine 使用同一接受规则：

$$
\operatorname{Accept}(C_r)
\iff
G_r>0.
$$

语义冻结：

- A/B 与 gold 相同为 correct；
- A/B 与 gold 不同为 wrong；
- None 为 wrong；
- baseline None → candidate gold 为 $+1$；
- baseline gold → candidate None 为 $-1$；
- wrong → wrong、correct → correct 均为 $0$；
- $G_r=0$ 拒绝；
- unresolved technical failure 不计作 wrong，不产生科学决定，沿用 Phase21 pause-without-history-commit。

Phase22 acceptance support 为全部 Discovery rows：

$$
S_{\mathrm{accept}}=D_{\mathrm{Discovery}}.
$$

Phase21 selective-scope metric 可从相同 reports 计算，但只标记 diagnostic-only，不能改变 Phase22 decision。

Phase21 root evaluator 已生成全部 rows 的 reports，因此 metric 本身新增 $0$ 次 Worker 推理。trajectory 因接受或 retry 决策改变后，后续 epoch 才产生新调用。

## 5. Split 流程

### 5.1 首次 Split

首次尝试与 Phase21 相同：

1. 从 epoch-start Unified-Subtree baseline 构造 root ErrorSignatures；
2. 对错误 signatures 聚类；
3. 为全部 clusters 生成完整 child bundle；
4. 形成完整 candidate subtree；
5. 用 Unified-Subtree Worker 推理全部 Discovery samples；
6. 计算 all-sample $G_r$；
7. $G_r>0$ 则整包接受；
8. 否则整包拒绝并执行 failure attribution。

不锁定单个 child，不部分保留，不使用 Specialized ACC 或 Global Arbiter 推翻决定。

### 5.2 Split 失败归因

科学拒绝后的 attribution 输入包括：

- baseline/candidate 完整 subtree；
- root ErrorSignatures；
- 原 ClusterProposal 与完整 child bundle；
- all-sample corrected/harmed IDs；
- before/after Unified reports；
- harmed/corrected 代表样本；
- 历史失败与 cluster lineage。

沿用 primary failure types：

- no_effect
- missed_correction
- overcorrection_on_harmed
- coverage_loss_to_none
- sibling_boundary_conflict
- evidence_misuse
- cluster_or_decomposition_error
- mixed_or_inconclusive

### 5.3 Retry action

retry action 由代码根据 validated primary_failure_type 确定：

| Primary failure type | Next Split action |
|---|---|
| cluster_or_decomposition_error | recluster_regenerate_complete_bundle |
| 其他科学失败 | reuse_clusters_regenerate_complete_bundle |
| technical / transport / parse failure | 技术重试，不执行语义 retry action |

建议实现：

~~~python
def split_retry_action(primary_failure_type: str) -> str:
    if primary_failure_type == "cluster_or_decomposition_error":
        return "recluster_regenerate_complete_bundle"
    return "reuse_clusters_regenerate_complete_bundle"
~~~

### 5.4 重新聚类

当 action 为 recluster_regenerate_complete_bundle：

1. 复用同一 baseline 对应的 Unified ErrorSignatures；
2. 不复用旧 ClusterProposal；
3. Cluster Manager 读取旧 cluster、failure attribution、all-sample harmed/corrected evidence 与 retry history；
4. 生成新 ClusterProposal；
5. 重新生成全部 children；
6. 形成新完整 root subtree candidate；
7. 在全部 Discovery samples 上重新推理并计算 $G_r$；
8. 仍仅按 $G_r>0$ 原子接受。

约束：

- max_children=5 与 minimum cluster size 沿用 Phase21；
- 不保留旧 strong child；
- 不做 cluster/child partial acceptance；
- 所有 children 全部重新生成；
- 总 attempt 数仍受 Phase21 epoch/attempt 上限控制；
- 每次 retry 保存 old/new cluster hash、membership 与 lineage。

### 5.5 复用聚类

若失败原因不指向分解：

- 复用旧 ClusterProposal；
- 利用失败证据重新生成完整 child bundle；
- 所有 children 重新生成；
- 仍执行完整 subtree all-sample 原子竞争。

## 6. Refine 流程

Bundle Refine 保持 Phase21 proposal 与 mutation scope，仅替换接受指标：

1. 从 Unified root errors 构造 evidence；
2. Manager 生成 atomic Refine patch；
3. 应用于完整 root subtree；
4. 对全部 Discovery samples 推理；
5. 计算 $G_r$；
6. $G_r>0$ 整包接受，否则整包拒绝。

Refine 不涉及 clustering，因此不执行 recluster retry。

## 7. 多 root 提交

每个 root 只与自己的 epoch-start baseline 比较。所有 $G_r>0$ candidates 同步提交：

~~~text
root 1: baseline ↔ candidate
root 2: baseline ↔ candidate
...
all positive candidates → synchronous commit
~~~

不选跨-root winner，不做 subset search。提交时 accepted root 直接复用 candidate report，commit-time root reinference count 必须为 $0$。Global Arbiter 仅作 post-commit diagnostic，即使 system metric 下降也不回滚局部接受。

## 8. 实现设计

### 8.1 Runtime scorer

修改：

- experiments/evolving_structured_rubrics/aligned_system_runtime.py

新增 paired_root_all_samples(before, after, rows)，输出：

- all_sample_support
- corrected/harmed/unchanged IDs 与 counts
- all_sample_net_gain
- Strict ACC before/after/delta
- None transition IDs
- technical failure counts
- sample-level gain ledger
- gain_ledger_sha256

保留现有 paired_root() 不变。

### 8.2 Phase22 runner

新增：

- experiments/evolving_structured_rubrics/all_sample_adaptive_recluster_evolution.py

身份：

- EXPERIMENT_DIR = phase22_all_sample_subtree_adaptive_recluster_evolution_v1
- PROTOCOL_VERSION = all-sample-root-subtree-adaptive-recluster-evolution-v1

stages：

- all-sample-adaptive-evolution-freeze
- all-sample-adaptive-evolution-audit
- all-sample-adaptive-evolution-smoke
- all-sample-adaptive-evolution-run
- all-sample-adaptive-evolution-report
- all-sample-adaptive-evolution-heldout
- all-sample-adaptive-evolution-final-report

实现要求：

- 复用 Phase21 Manager、candidate preparation、merge、commit 和 report helpers；
- all-sample metric 是唯一 acceptance source；
- selective metric 只读；
- retry action 由 validated attribution 派生；
- accepted root reports 直接提交；
- attempts 保存完整 metric 与 cluster lineage。

### 8.3 Config / CLI

修改：

- experiments/evolving_structured_rubrics/configs/rubric_evolution_phase5.example.json
- experiments/evolving_structured_rubrics/run_rubric_evolution.py

新增 all_sample_adaptive_recluster_evolution_v1_experiment。关键设置：

- acceptance_metric = all_sample_unified_net_gain_gt_0
- split_acceptance = all_sample_unified_net_gain_gt_0
- refine_acceptance = all_sample_unified_net_gain_gt_0
- split_retry = failure_aware_recluster_or_reuse_clusters
- recluster_failure_types = [cluster_or_decomposition_error]
- partial_acceptance = false
- global_arbiter_role = post_commit_diagnostic_only

其余字段与 Phase21 对应值一致。

### 8.4 VL-RewardBench

新增：

- experiments/evolving_structured_rubrics/vl_rewardbench_all_sample_adaptive_recluster_evolution.py

比较 Initial、Phase21 final、Phase22 final；Phase17 E4 可作历史强 control。使用与 Phase21 相同的 $K=3$、A/B schedule、runtime 与 retry。benchmark 不参与 rubric 选择。

### 8.5 Tests

新增：

- tests/structured/test_all_sample_adaptive_recluster_evolution.py
- tests/structured/test_vl_rewardbench_all_sample_adaptive_recluster_evolution.py

必须覆盖：

1. +1/-1/0 与 None 语义；
2. 全 rows support；
3. sample identity drift；
4. technical failure pause；
5. net $-1/0/+1$ → reject/reject/accept；
6. Split/Refine 使用同一 gate；
7. trigger/error IDs 与 Phase21 相同；
8. cluster_or_decomposition_error → recluster；
9. 其他 failures → reuse clusters；
10. technical failure 不触发 recluster；
11. recluster 不复用旧 ClusterProposal；
12. recluster 后全部 children 重新生成；
13. reuse action 保留 cluster hash；
14. 无 locked child / partial acceptance；
15. max attempts 不增加；
16. commit 零 root reinference；
17. ledger/action 可重算；
18. VL-RB selection forbidden。

## 9. Artifact contract

每个 attempt 保存：

- candidate_rubric.json
- root_unified_subtree.json
- all_sample_competition.json
- all_sample_gain_ledger.json
- all_sample_evidence.json
- phase21_selective_metric_diagnostic.json
- failure_attribution.json（若 rejected）
- retry_action.json（若有下一次科学 retry）

每个 Split retry 额外保存：

- source_error_signatures.json
- previous_cluster.json
- new_cluster.json 或 reused_cluster.json
- cluster_diff.json
- complete_child_bundle/

cluster_diff.json 包含 old/new SHA、cluster count、sample mapping、moved IDs、split/merged summary、failure evidence hash、Manager metrics、complete_child_regeneration=true、partial_acceptance=false。

## 10. Experiment Blocks

### B1：协议、指标与 retry 审计

- Claims：C1、C2。
- Data：fixtures + Discovery manifest。
- Metrics：gain ledger、trigger/error identity、failure/action mapping、cluster lineage。
- Success：公式重算一致；只有计划中的 metric 与 retry policy 相对 Phase21 改变；smoke 覆盖 reuse/recluster。
- Priority：MUST-RUN。

### B2：Discovery100 完整演化

- Systems：Phase21 frozen report control；Phase22 treatment。
- Setup：相同 Initial rubric、Discovery100、$K=1$、temperature $0.5$、3–5 epochs。
- Primary：attempt corrected/harmed/net、decision、retry action、recluster count、cluster changes、final Discovery system Strict ACC。
- Secondary：selective/all-sample disagreement、None transitions、calls/tokens/latency、post-commit metrics。
- Success：accepted 全部满足 $G_r>0$；recluster 全部由冻结 failure type 触发；trajectory 可重放。
- Negative outcomes：全拒、无 recluster、recluster 后仍拒绝，均作为有效结果报告。
- Priority：MUST-RUN。

### B3：独立数据验证

- Data：Dev150、heldout500、VL-RB1247。
- Systems：Initial、Phase21 final、Phase22 final；Phase17 E4 辅助。
- Metrics：Strict/Overall/Macro/Coverage、paired corrected/harmed、McNemar、bootstrap CI。
- Dev 逐 epoch 只读；heldout final-only；VL-RB final $K=3$。
- Priority：MUST-RUN。

### B4：重新聚类与局部到全局分析

- Units：rejected Split、retry、accepted operation、epoch commit。
- Metrics：cluster membership 变化、retry candidate net、local/global direction quadrants、root/operator/None 分层。
- Priority：MUST-RUN。

### Intentionally cut

第一版不做：

- 单独 MetricOnly trajectory；
- factorized A/B 消融；
- $K=3$ candidate gate；
- D_gen/D_accept 再切分；
- Global Arbiter gate；
- cross-root subset search；
- best-of-N；
- partial cluster/child acceptance；
- Refine mutation scope 扩展；
- 根据旧事实实验挑 root/checkpoint。

## 11. Run Order

| Milestone | Goal | Runs | Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 冻结组合协议 | config/schema/manifest | 只有计划中的两项变化 | 离线 | 隐式漂移 |
| M1 | scorer + mapper | fixtures | focused tests通过 | 0.5–1天 | None/action错误 |
| M2 | smoke | Split、Refine、reuse、recluster | support正确、两条retry覆盖 | 每candidate约20 root calls | 难构造cluster_error |
| M3 | full evolution | E1–E5 | 按冻结规则完成 | Phase21同量级+Manager calls | trajectory分叉 |
| M4 | internal eval | Dev + heldout | 只读 | Dev约900/checkpoint，heldout约3000 | leakage |
| M5 | external eval | VL-RB $K=3$ | final only | 22446 calls/system | 技术失败 |
| M6 | report | metric/cluster/local-global | 表格可重算 | 离线 | 过度归因 |

M3 即使 ACC 低、全拒或无 recluster，也完成并报告。Dev/heldout/VL-RB 不触发 rerun 或协议修改。

## 12. Compute budget

- scorer：$0$ GPU；
- Discovery candidate：每次约 $100$ root calls；
- recluster 增量：每次比 fixed retry 多 $1$ 次 Cluster Manager call；
- Dev150：约 $900$ logical calls/system/checkpoint；
- heldout500：约 $3000$ calls/system；
- VL-RB：$22446$ calls/system；
- 新数据与人工标注：$0$。

## 13. Risks

- **两项变化无法拆分**：明确只评价组合方案，不做单因素结论。
- **Manager 错判 decomposition error**：保存 evidence；action 由代码映射验证。
- **反复 recluster**：总 attempts 不增加，lineage 可审计。
- **None 语义混合**：统一按 wrong，单独报告 transitions。
- **technical failure 污染**：任一 unresolved failure 均暂停。
- **局部正全局负**：记录为假设边界，不回滚局部门。
- **数据分布影响**：Discovery演化，独立集只读。
- **旧 output 污染**：独立目录、manifest、cache namespace。

## 14. Implementation work packages

1. WP1：实现 paired_root_all_samples() 与 tests。
2. WP2：实现 failure type → retry action 与 cluster lineage。
3. WP3：实现 Phase22 runner。
4. WP4：新增 config、CLI、audit、smoke。
5. WP5：运行完整 evolution 与 report。
6. WP6：运行 Dev、heldout、VL-RB。
7. WP7：完整验证。

验证命令：

~~~text
conda activate critiq
python -m py_compile <Phase22 modules>
python -m unittest tests.structured.test_all_sample_adaptive_recluster_evolution
python -m unittest tests.structured.test_vl_rewardbench_all_sample_adaptive_recluster_evolution
python -m unittest discover -s tests -p "test_*.py"
git diff --check
~~~

## 15. Final checklist

- [ ] Phase22 是单一组合 trajectory
- [ ] all-sample $G_r>0$ 是 Split/Refine 唯一 gate
- [ ] 全部 Discovery rows 进入 ledger
- [ ] None=wrong，technical failure=pause
- [ ] tie拒绝
- [ ] 错误证据全部来自 Unified Worker
- [ ] cluster_error 才 recluster
- [ ] 其他失败复用 clusters
- [ ] recluster 后完整 children 全重生成
- [ ] 无 locked child / partial acceptance
- [ ] max attempts 不增加
- [ ] Global Arbiter 只读
- [ ] commit 零 root 重推
- [ ] independent eval 不参与选择
- [ ] artifacts 可重放
- [ ] tests/compile/diff check通过

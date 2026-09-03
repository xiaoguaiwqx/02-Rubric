# Phase21：Unified Root-Subtree Bundle Evolution

> 冻结协议，保留实验前设计。该实验已完成且25个候选全部拒绝；当前结果入口见[实验索引](../README.md)。

## 1. 研究问题

Phase17 以 criterion/node 为优化单位，Phase20 又混用了 Pairwise root scope、
Specialized ACC 与 Unified-Subtree 竞争。因此，算子看到的错误、候选接受语义和
最终推理语义并不完全一致。Phase21 只改变这一点：**完整 root subtree 是 Split
和 Refine 的最小优化与提交单位，Unified-Subtree Worker 是局部演化的唯一评分者。**

本实验不让 Global Arbiter 指导局部接受；它只在一次 epoch 同步提交完成后报告
五棵子树联合作用的系统指标。criterion-level Pairwise/Specialized 结果也只在提交
后生成，用于诊断，不能进入调度、Manager 上下文、错误选样或接受判断。

## 2. 一轮演化的数据流

```text
epoch-start Rubric R_t
        │
        ▼
5 × Unified-Subtree baseline（每个 root 一份完整 report）
        │
        ├─ scope：baseline 输出 A/B 的样本
        ├─ error：scope 内判断不等于 gold 的样本
        └─ root-level ErrorSignatures
        │
        ▼
┌──────────────────────────┬────────────────────────────┐
│ Split                    │ Subtree Bundle Refine      │
│ 创建整套新 children      │ 联合小修部分现有 children │
│ 不锁强孩子、不部分接受   │ ID/name/edges 均不改变    │
└──────────────────────────┴────────────────────────────┘
        │
        ▼
candidate root Unified-Subtree report
        │
        ▼
同一 frozen scope 上的 paired corrected / harmed
        │
        ├─ corrected − harmed > 0：原子接受
        └─ 否则：原子拒绝并生成 bundle failure attribution
        │
        ▼
所有独立通过的不同 roots 同步提交
        │
        ├─ accepted root 直接复用 candidate report
        └─ unchanged/rejected root 复用 epoch-start report
        │
        ▼
Global Arbiter 提交后诊断 → Dev150 只读诊断 → Pairwise 只读诊断
```

## 3. Epoch baseline、scope 与错误经验

对 root $r$ 和样本 $x$，Unified-Subtree Worker 输出：

$$
U_r(R_t,x)\in\{A,B,\mathrm{None}\}.
$$

本轮冻结 scope：

$$
\mathcal S_r^t=\{x\mid U_r(R_t,x)\in\{A,B\}\}.
$$

错误集合只取 scope 内的 Unified mismatch：

$$
\mathcal E_r^t
=\{x\in\mathcal S_r^t\mid U_r(R_t,x)\ne y_x\}.
$$

baseline 的 `None` 不进入 scope，但计入全数据 Coverage；candidate 在 frozen scope
中输出 `None` 时按错误计。candidate 无权改变本轮 scope。任何 baseline/candidate
技术失败都会暂停当前 epoch，并且不追加科学历史。

每个被调度 root 都保存：

```text
epochs/epoch_XX/roots/<root_shard>/baseline/error_signatures.json
```

每条 root-level ErrorSignature 包含 `sample_id`、`task_pattern`、`visual_focus`、
`candidate_difference`、`subtree_failure` 和 `suggested_subdomain`。Split retry 复用
上一轮已冻结 signatures 与 clusters，但在当前 epoch 写出带来源和 hash 的物理副本。

## 4. 原子 Split

Split 使用 Unified baseline 的 Coverage、scope Strict ACC、support 和 mismatch 数触发。
首次尝试把 ErrorSignatures 聚类，每个 cluster 生成一个 child，然后把全部 children
一起应用到 root，形成一个候选 subtree。retry 不重新聚类、不锁定任何 child，而是
利用上轮 bundle failure attribution 重新生成整套 children。

取消的旧机制包括：strong-child/locked-child、单 child 保留、partial acceptance、
Specialized ACC acceptance 和 `single_child_specialized_accuracy` 决策。

接受分数为：

$$
\Delta_r^{\mathrm{split}}
=N_{\mathrm{corrected}}-N_{\mathrm{harmed}}.
$$

仅当 $\Delta_r^{\mathrm{split}}>0$ 时整套 children 原子接受；否则全部拒绝。

## 5. Subtree Bundle Refine

Bundle Refine 面向已有 children 的完整 root subtree。Manager 同时看到当前 root 与
全部 children、root-level ErrorSignatures、最多 6 条确定性代表错误样本及图像、
Rubric memory 和该 root 的历史失败归因。

Manager 可自主选择只修改真正有问题的 children，而不是机械重写全部节点。约束为：

- 至少修改一个 child；
- root、node ID、criterion name、score、examples、edges 和拓扑不变；
- 仅允许修改 child description 的 focus、适用边界和 decision rule；
- 未编辑 children byte-identical；
- 多个 edits 组成一个原子 patch，不允许分别接受。

接受分数为：

$$
\Delta_r^{\mathrm{refine}}
=N_{\mathrm{corrected}}-N_{\mathrm{harmed}}.
$$

同样仅当 $\Delta_r^{\mathrm{refine}}>0$ 时整套 patch 接受。

## 6. 失败归因与重试

拒绝后的 attribution 同时读取 baseline/candidate subtree、root ErrorSignatures、
before/after 指标、corrected/harmed IDs、两侧 Unified reports、最多 6 个 harmed 与
3 个 corrected 图像样本、候选 bundle 和历史 retry feedback。

一级类型固定为：`no_effect`、`missed_correction`、
`overcorrection_on_harmed`、`coverage_loss_to_none`、
`sibling_boundary_conflict`、`evidence_misuse`、
`cluster_or_decomposition_error`（Split only）、
`bundle_edit_interaction`（Refine only）和 `mixed_or_inconclusive`。

归因只进入同一 root 下一轮的 Manager context，不修改当轮 Rubric，也不能触发部分接受。

## 7. 多 root 提交语义

同一个 root 每轮最多产生一个候选：未完成 Split 时 Split 优先，Split 接受后的后续
epoch 才允许 Bundle Refine。不同 roots 分别与各自 epoch-start baseline 竞争；所有
满足 `net_corrected > 0` 的 roots 一次同步提交，不选“提升最大的 root”，不进行 root
subset search，也不让 Global Arbiter 决定接受集合。

代码保存每个 committed root 的 scientific-call hash。accepted root 的 candidate hash
必须与 committed hash 完全一致；提交阶段的新增 Unified root 调用数必须为 0。
Global Arbiter 只重新读取五份已冻结报告并生成系统诊断，即使全局指标下降也不能回滚
局部接受。

## 8. 数据与评测

- Discovery100：演化和错误经验；`K=1`，不 swap。
- Dev150：每个 epoch 提交后诊断，不进入 Manager、不早停、不选 checkpoint。
- RLHF-V heldout500：final Rubric 一次性探索性测试。
- VL-RewardBench：1,247 条，`K=3`；每 replicate 为 5 个 Unified-Subtree reports
  加 1 个 Global Arbiter。
- 解码：`temperature=0.5`、`max_tokens=2048`、不设置 generation seed；同 Prompt
  parser 重试最多 10 次；两个 available-slot endpoints。
- epoch：最少 3、最多 5；第 3 轮以后只有在不存在 Split retry 和 Bundle Refine
  candidate 时才可提前结束。

VLRB 主要比较 Initial five-root、Phase17 E4 strong Control 与 Phase21 final。
Phase20 只作为带已知协议混用问题的诊断参考，不进入主要结论。Strict ACC 为主指标，
同时报告 OverallAcc、Coverage、MacroAcc、类别 ACC、paired bootstrap 95% CI 和
McNemar exact two-sided $p$。

## 9. 目录、配置与 CLI

- 演化目录：`phase21_unified_subtree_bundle_evolution_v1`
- 演化配置：`unified_subtree_bundle_evolution_v1_experiment`
- VLRB 目录：`vl_rewardbench_unified_subtree_bundle_evolution_v1`
- VLRB 配置：`vlrb_unified_subtree_bundle_evolution_v1_experiment`

演化 stages：

```text
subtree-bundle-evolution-freeze
subtree-bundle-evolution-audit
subtree-bundle-evolution-smoke
subtree-bundle-evolution-run
subtree-bundle-evolution-report
subtree-bundle-evolution-heldout
subtree-bundle-evolution-final-report
```

VLRB stages：

```text
vlrb-subtree-bundle-freeze
vlrb-subtree-bundle-audit
vlrb-subtree-bundle-smoke
vlrb-subtree-bundle-run
vlrb-subtree-bundle-retry
vlrb-subtree-bundle-report
```

## 10. 实现验收

必须满足：100% accepted operations 有同 scope 的 Unified paired evidence；0 个接受依赖
Specialized ACC；0 个 Split 使用 locked child；0 个 Refine 部分提交；0 个 accepted
root 在 commit 时重推；每个 scheduled root 有本 epoch 的本地 ErrorSignature artifact；
heldout/VLRB 无 unresolved technical failures。

方法主张得到支持的条件是：Phase21 的 VLRB Strict ACC 优于 Phase17 E4，或者相差不
超过 1 个百分点且消除了局部语义错位，并在 Dev150/heldout500 上给出稳定净修正。
无论性能是否提高，都完整报告 root-local 收益与 Global Arbiter interaction cases。

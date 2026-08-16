# Root Boundary Pre-Refine → Split+Refine Experiment Plan

**Problem**: 五个初始 roots 的覆盖率普遍较高，它们可能在同一偏好对上同时投票并产生跨 root 语义重叠、冲突和错误多数票稀释。

**Method thesis**: 在不使用 Gate Worker、不改变 M1 投票和现有 Split/Refine 竞争规则的前提下，先将五个 roots Refine 为边界明确的软路由专家，再运行冻结的 Locked-Split + Role-aware Refine，检验更干净的 root 错误区域是否产生更专业的 children 并提高最终 M1。

**Date**: 2026-08-16

**Protocol**: `root-boundary-pre-refine-split-refine-v3`

**Output**: `output/evolving_structured_rubrics/rubric_evolution_phase5/phase15_root_boundary_pre_refine_split_refine_v3/`

## 1. Claim map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|---|---|---|---|
| C1: Root 预 Refine 能形成有效的软路由边界 | 说明高 root coverage 与冲突可以通过 criterion 自身的适用性语义改善，而不必立即引入 Root Gate | 至少有 roots 通过原 Refine 竞争；root 重叠/冲突减少；Coverage 下降主要来自抑制错误决定，而非丢失正确决定 | B1, B2 |
| C2: Root 边界化是 Split+Refine 的有效预处理 | 这是完整方法上是否保留该阶段的决定依据 | 最终 treatment 在同一 heldout-500 和同一 Worker prompt 下高于 Phase 10 control，corrected > harmed，且 root 冲突不反弹 | B3, B4 |

**Anti-claim to rule out**: 改善不能只是 Root Refine 通过过度输出 `None` 筛掉难样本造成的选择性 ACC 上升；也不能把 Pairwise Worker Prompt v2 本身的收益错归因于 Root 预 Refine。

## 2. Frozen experimental scope

### 2.1 Starting point

- 从 Phase 5 冻结的五个初始 roots 和 discovery-90 开始，不从 Phase 10 final rubric 继续。
- Root 预 Refine 开始时 rubric 中只有五个 roots，没有 children。
- 五个 roots 共同视为同级专家；不新增虚拟 parent node，不改变 topology。

### 2.2 Frozen components

- **No Gate**: 不运行 Root Gate 或 Child Gate；五个 roots 始终按原 M1 规则参与投票。
- **Manager**: `Qwen/Qwen3.5-397B-A17B`，seed=42，每轮使用 epoch-start `global_rubric_v1`。
- **Worker**: Qwen3-VL-8B-Instruct，P05，单 replicate，`max_tokens=2048`。先在 discovery-90 上以 capped identity 重新生成五-root基线，确保 old/new competition 完全对称；实际请求由 `vllm-8000 + vllm-8001` available-slot pool 调度，总并发40、每端口上限20。
- **Split**: 复用 Locked-Split v2 的触发、ErrorSignature、聚类、强 child 锁定、整组 Specialized Accuracy 竞争和失败历史。
- **Child Refine**: 复用 role-aware 条件 `0.5 < ACC < 0.80`、`support >= 15`、`wrong >= 5`。
- **Root Refine competition**: 复用 Refine v1 的自竞争：新 criterion 在自身 A/B support 上 ACC 严格提升且 support `>=15` 才接受。
- 不修改投票权重、M1 聚合、Split/Refine fitness、children schema 或 failure-attribution 语义。

### 2.3 Prompt-version separation

Pairwise prompt 不能与 Root 预 Refine 同时成为因果变量：

1. **Operator-selection track**: Root 预 Refine 和后续 Split+Refine 使用 Phase 10 冻结的 Worker request identity，与历史 control 直接对齐。
2. **Deployment diagnostic**: 全部 discovery 决策和 final rubric hash 冻结后，可另行使用 Pairwise Worker Prompt v2 评估 control/treatment；该结果不得回流选择 candidates、epoch 或 checkpoint。

## 3. Experiment blocks

### B0 — Freeze and offline audit

- **Claim tested**: 后续差异只来自 Root 预 Refine。
- **Actions**:
  - 冻结初始 rubric/data/prediction/feedback hash，五个 root IDs，Manager/Worker request specs，Split/Refine thresholds 和 heldout 隔离。
  - 验证 Phase 10 control 完整，并记录其 discovery/heldout rubric 与 prediction hashes。
  - 计算初始 roots 的 ACC、support、Coverage、两两 decisive-overlap 和投票冲突。
  - 演化阶段禁止读取 heldout dataset、predictions 和 metrics。
- **Success criterion**: 所有 identity 逐项匹配，离线测试通过。
- **Priority**: MUST-RUN.

### B1 — Forced Root Boundary Refine

- **Claim tested**: C1.
- **Scheduling**:
  - 五个 roots 全部作为 `forced_root_boundary_refine=true` 候选；forced 只绕过自动 trigger，不绕过 parser、provenance、Pairwise 评估和 self-competition。
  - 最少 1 轮、最多 3 轮。接受的 root 在该阶段锁定；拒绝的 root 必须先完成 397B 失败归因，再携带历史重试。
  - 同一轮所有 root candidates 基于同一 epoch-start rubric 和 Global Memory 独立生成，轮末同步提交。
- **Candidate contract**:
  - 仅允许改写 description；node ID、criterion name、score、edges 和 topology 不变。
  - 必须包含 `Criterion focus`、`Applicable only when`、`Not applicable when`、`Decision rule` 四段。
  - Manager 必须参考其他四个 roots，指定当前 root 独有的证据范围、与相邻 roots 的边界以及应当 `None` 的情形；不得仅换名或泛化为总体回答质量。
- **Acceptance**:
  \[
  \operatorname{Acc}(c')>\operatorname{Acc}(c),\qquad |\mathcal S(c')|\ge15.
  \]
  Coverage 下降、root overlap/conflict 和完整 M1 只是诊断，不修改原 Refine 决策。
- **Failure interpretation**:
  - 如果某个 root 三轮都失败，保留原 description；不为了“凑齐四段”强制接纳退化候选。
  - 若 self ACC 上升但 M1 下降，说明局部 self-competition 不足以保证 root ensemble 改善。
- **Priority**: MUST-RUN.

### B2 — Root-only checkpoint diagnosis

- **Claim tested**: C1 及 anti-claim。
- **Compared systems**:
  1. Initial five roots.
  2. Accepted Root-pre-refined checkpoint.
- **Discovery metrics**:
  - 每个 root 的 own-support ACC/support/Coverage/wrong。
  - old/new fixed-scope 准确性，corrected/harmed，`A/B→None`、`None→A/B` 和 retained-support ACC。
  - 被抑制样本中 old-correct 与 old-wrong 数，用于区分正确边界化与选择性逃避。
  - 每样本激活 root 数、两两 activation Jaccard、A/B 冲突数和错误多数票稀释数。
  - 完整 M1 ACC/Coverage/corrected/harmed。
- **Success criterion**: 不强制所有五个 roots 都接受；至少有一个有效边界化案例，且总体冲突下降不以大量丢失 old-correct 为代价。
- **Table target**: `Initial roots vs Root-pre-refined` 的 root-level 与 ensemble-level 对照表。
- **Priority**: MUST-RUN.

### B3 — Locked-Split + Role-aware Refine from the root checkpoint

- **Claim tested**: C2.
- **Execution**:
  - 以 B1 最终 committed roots 为新的初始 rubric。
  - 按 Phase 10 协议运行最少 3、最多 5 epochs：Split 只调度初始 roots，Refine 调度已提交 children。
  - Split 失败后按冻结规则锁定最多一个强 child，其余 children 重新生成；不允许部分 Split 提交。
  - children 提交后从下一 epoch 才进入 Global Memory 和 role-aware Refine。
  - Root 预 Refine 阶段结束后，不再额外强制 roots Refine；后续只执行 Phase 10 冻结调度。
- **Early stop**: 第 3 轮后无 eligible/retryable operation 则停止，否则最多到第 5 轮。
- **Priority**: MUST-RUN.

### B4 — Final discovery and one heldout-500 comparison

- **Claim tested**: C2.
- **Compared systems**:

| System | Purpose |
|---|---|
| Initial five-root M1 | initialization baseline |
| Root-pre-refined only | isolate root boundary preconditioning |
| Phase 10 Locked-Split + Role-aware Refine | no-root-pre-refine control |
| Root-pre-refined → Locked-Split + Role-aware Refine | treatment |

- **Primary metrics**: heldout M1 ACC、correct count、Coverage、Wilson 95% CI、paired corrected/harmed、net corrected 和 exact McNemar。
- **Secondary metrics**:
  - discovery M1 与 discovery→heldout delta；
  - 每个 final root subtree ACC/Coverage；
  - root overlap/conflict、错误多数票稀释；
  - Split 触发/接受轮次、强-child锁定、children 数、语义近重复；
  - Manager/Worker requests、tokens、wall time 和 rubric 增长。
- **Heldout discipline**:
  - 在访问 heldout 前同时冻结 Root-only 和 final treatment rubric hashes、control hashes、dataset hash 与所有指标。
  - 只访问一次 heldout-500，可在同一 stage 评估多个已冻结 checkpoints；不得根据 heldout 重选 root proposals、children 或 epoch。
  - Prompt-v2 deployment diagnostic 必须独立标记，只与相同 Prompt v2 下的 control 比较。
- **Success criterion**:
  - 强支持：treatment heldout M1 高于同 prompt 的 Phase 10 control，corrected > harmed，Coverage 无实质下降，且 root conflict 减少。
  - pilot evidence：ACC 上升但 McNemar 不显著。
  - 否定结果：Root-only 或 final treatment 下降，尤其是 harmed 主要来自被转为 `None` 的 old-correct 样本。
- **Priority**: MUST-RUN.

### B5 — Optional boundary-prompt ablation

- **Claim tested**: 如果 B1 有收益，区分“额外 Refine compute”与“跨-root边界指导”。
- **Variant**: 对相同五个 roots 使用普通 Refine v1 prompt，不加跨-root边界指令；Manager、evidence、seed 和竞争相同。
- **When to run**: 仅在 B1/B4 展示正向收益后运行，不阻塞主实验。
- **Priority**: NICE-TO-HAVE.

## 4. Artifact and stage design

Proposed CLI stages:

```text
root-pre-refine-freeze
root-pre-refine-baseline
root-pre-refine-audit
root-pre-refine-smoke
root-pre-refine-run
root-pre-refine-report
root-pre-refine-split-refine-run
root-pre-refine-split-refine-report
root-pre-refine-heldout
root-pre-refine-final-report
```

Directory layout:

```text
phase15_root_boundary_pre_refine_split_refine_v3/
  frozen_manifest.json
  offline_audit.json
  root_pre_refine/
    epochs/epoch_01..03/
      rubric_memory.json
      roots/<root_id>/attempt_NN/
      rubric_committed.json
      combined_pairwise.json
      summary.json
    final/
      rubric.json
      discovery_report.json
  evolution/
    evolution_history.json
    epochs/epoch_01..05/
    final/
      rubric.json
      discovery_report.json
  heldout500/
    frozen_manifest.json
    root_pre_refine/
    final_treatment/
    report.json
  final_report.json
  final_report.md
```

Every Root Refine attempt stores:

```text
trigger.json
rubric_memory_ref.json
evidence.json
proposal.json
candidate_rubric.json
candidate_pairwise.json
combined_pairwise.json
node_evaluation.json
root_boundary_diagnostic.json
m1_diagnostic.json
failure_attribution.json
history_projection.json
```

## 5. Run order and decision gates

| Milestone | Goal | Runs | Decision gate | Estimated cost | Main risk |
|---|---|---|---|---|---|
| M0 | Offline correctness | tests + freeze + audit | identities and heldout isolation pass | minutes, no model calls | historical artifact drift |
| M1 | Root boundary preconditioning | 5 roots, 1–3 attempts each | at least one accepted root; all rejections have attribution | 450 Worker calls per full attempt wave + Manager calls | selective abstention |
| M2 | Root-only diagnosis | offline report | fixed-scope and `None` transition accounting complete | minutes | misreading lower Coverage as success |
| M3 | Full discovery evolution | 3–5 epochs | final rubric/report frozen | comparable to Phase 10 plus preconditioning cost | compounding local decisions |
| M4 | One heldout evaluation | four frozen systems | no post-heldout selection | changed descriptions × 500 | exploratory reuse of heldout |
| M5 | Optional prompt-v2 deployment replay | control/treatment under v2 | same request identity on both systems | changed descriptions × 500 | prompt gain confounded with method gain |

### Stop/go policy

- B1 全部拒绝：停止 B3，说明现有 Refine self-competition 无法产生可接受的 root boundary candidate。
- B1 有接受但 root-only M1 下降：仍可以运行 B3 作为机制诊断，但必须预先标记为 exploratory，不能把后续最佳 checkpoint 当作选择结果。
- B1 有接受且 root-only 冲突减少、M1 不降：正常进入 B3。

## 6. Required tests and acceptance checklist

- [ ] 准确调度五个初始 roots，不调度 children 进入 Root 预 Refine。
- [ ] `forced=true` 只绕过 trigger，不绕过 parser、provenance、self-competition 或 support gate。
- [ ] Root candidate 只改 description，且四段完整；ID/name/score/topology 不变。
- [ ] 同一 Root-pre-refine epoch 的五个 candidates 看到相同 Global Memory hash，同步提交与遍历顺序无关。
- [ ] 已接受 root 不再重试；拒绝 root 只在归因成功后进入下一轮。
- [ ] 原 Refine 严格 self-ACC 接受与 support `>=15` 逻辑不变。
- [ ] Coverage、overlap、conflict 和 M1 只是诊断，不能改写 Root Refine 决定。
- [ ] 报告区分 old-correct 被抑制与 old-wrong 被抑制，不把 Coverage 下降自动表述为改善。
- [ ] B3 完整复用 Locked-Split/Role-aware Refine 协议，不引入 Gate、新权重或新 fitness。
- [ ] Phase 10 control 和 treatment 的主对照使用同一 Pairwise prompt/request identity。
- [ ] discovery 阶段无法读取 heldout；heldout 前冻结 Root-only/final/control hashes。
- [ ] 中断恢复不重新生成已完成的 candidate 或 Pairwise shard。
- [ ] 完整 `unittest`、Root-pre-refine focused tests、Split/Refine regression 与 compile checks 通过。

## 7. Paper placement

- **Main paper candidate**: 若 C2 成立，用一张 `Initial → Root-pre-refined → Full evolution` 阶段表和一张 root activation/conflict 图支撑“先边界化、再专业化”的演化叙事。
- **Appendix**: 每个 root 的 description diff、`None` 转移、失败归因、全部 Split/Refine trajectories。
- **Intentionally cut**: Root Gate、Child Gate、权重搜索、新聚合器和新 discovery 数据混合；这些都会破坏本次 Root 预 Refine 的单变量解释。

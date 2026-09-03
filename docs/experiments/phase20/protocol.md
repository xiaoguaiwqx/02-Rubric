# Local Unified-Subtree 竞争演化实验计划

> 历史冻结协议，不作为当前默认入口或实时状态。后续协议仍引用其 Initial 基线产物，因此保留设计和实现。当前入口见[实验索引](../README.md)。

**Problem**：Phase17 使用显式递归多数票（children 多数、平票回退 parent）计算 Split 的 SpecializedAcc；Phase19 又同时改变了错误归因、候选接受、同-root竞争和跨-root提交，无法单独判断新聚合语义对演化的影响。
**Method Thesis**：保留 Phase17 的 Split/Refine 生成、触发、局部错误经验、retry 和同步提交，只将 Split 的竞争指标替换为冻结 root scope 上的 Unified-Subtree Strict ACC，并为 Refine 增加同一 root-level 非退化约束，可以在不引入 Global-Arbiter credit assignment 的前提下，使演化选择更贴近新的子树执行语义。
**Date**：2026-08-29

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：Unified-Subtree 能作为 Split 的直接竞争器 | Split 的对象是整棵细粒度专家子树，而非 children 的显式多数票 | Split 前后在同一冻结 root scope 上做 paired comparison；被接受 Split 的 root-scope net corrected 均为正，并报告与旧 SpecializedAcc 的选择分歧 | B1、B2 |
| C2：Refine 的 node-level 改善可以在不损害 root 子树的情况下保留 | 只看 node 可能产生 sibling/子树交互退化；只看全局又会过滤被其他 roots 掩盖的专家收益 | Refine 必须 node-level 通过且 root-scope net corrected 非负；外部结果不因 Coverage 坍缩获得表面收益 | B1、B3 |

**Anti-claim**：收益来自新的 ErrorSignature 过滤、Global Arbiter 参与接受、同-root winner、跨-root联合回退、checkpoint 后验选择或更换模型/Prompt。

## Paper Storyline

- Main paper must prove：把 Split 的竞争聚合从显式多数票替换为 Unified-Subtree 后，是否能选择出更符合最终子树执行语义的结构；Refine 的 root guard 是否能阻止局部改善破坏整棵子树。
- Supporting evidence：旧 SpecializedAcc 与新 root-scope 指标的 disagreement、corrected/harmed 案例、固定失败阶段与 retry 轨迹。
- Appendix can support：完整 child 指标、sibling conflict、同-root多候选提交后的组合交互、token/latency。
- Experiments intentionally cut：Global Arbiter 过滤错误经验、同-root子集搜索、跨-root winner/joint gate、Create/Merge/Drop、Root Gate、权重搜索、跨模型迁移和 Dev checkpoint 选择。

## 1. 冻结协议与唯一变量

实验从与 Phase17 相同的五个 Initial roots 独立开始。

| 组成 | 处理 |
|---|---|
| Discovery100 / Dev150 / RLHF-V heldout-500 / VL-RewardBench | 完全复用 |
| Qwen3-VL-8B-Instruct、Pairwise Prompt v2、Unified-Subtree Prompt、Global-Arbiter Prompt | 完全复用 |
| Split/Refine trigger、Manager Prompt、Rubric Memory、Locked-child retry、3–5 epoch | 完全复用 Phase17 |
| ErrorSignature | 使用 root/node 自身错误；不与 Global Arbiter wrong 取交集 |
| 多候选提交 | Phase17 同步提交；不选同-root winner，不做跨-root joint gate |
| Dev/heldout/VL-RB | 只诊断和测试，不参与接受或 checkpoint 选择 |
| 唯一方法变化 | Split 用 root Unified-Subtree 直接竞争；Refine 增加 root Unified-Subtree 非退化门槛 |

## 2. 冻结 root scope 与计分

第 $t$ 轮开始时，对 root $r$ 冻结：

$$
\mathcal S_r^t
=
\{x_i:\text{epoch-start root local Pairwise vote 为 A 或 B}\}.
$$

本轮所有属于 root $r$ 的候选都使用同一个 $\mathcal S_r^t$。candidate 不能改变评估分母。

为避免符号歧义，后文统一使用以下记号：

- $R_t$ 表示第 $t$ 轮开始时冻结的**完整 Rubric**，其中包含全部 roots 及其 children；它不是某一个 root。
- $U_r(R)$ 表示从 Rubric $R$ 中取出以 root $r$ 为根的完整子树，并使用 Unified-Subtree Worker 对该子树进行一次整体判断。
- Split 或 Refine 候选先通过显式的 `ApplySplit` 或 `ApplyRefine` 操作生成一个候选 Rubric，再与 $R_t$ 比较；下文不再使用容易被误解为数值加法的“$R_t+C$”写法。

对 Unified-Subtree 输出 $\hat y_r(x)\in\{A,B,\mathrm{None}\}$，定义：

$$
\operatorname{StrictAcc}_{\mathcal S_r^t}(U_r)
=
\frac{1}{|\mathcal S_r^t|}
\sum_{x_i\in\mathcal S_r^t}
\mathbf 1[\hat y_r(x_i)=y_i].
$$

scope 内的 `None` 按错误计；解析失败属于技术失败；scope 外只报告激活和 Coverage，不进入主接受指标。实现与报告字段固定为：

- `root_scope_support`
- `root_scope_strict_accuracy_before/after`
- `root_scope_coverage_before/after`
- `root_scope_corrected_sample_ids`
- `root_scope_harmed_sample_ids`
- `root_scope_net_corrected`

## 3. 算子竞争与接受

### 3.1 Split

Split 保持原来的 ErrorSignature、聚类、children 数量、结构合法性、强-child锁定与 retry。对 root $r$ 产生 Split 候选 $C_r^{\mathrm{split}}$ 后，先将其应用到完整 Rubric：

$$
R_{t,r}^{\mathrm{split}}
=
\operatorname{ApplySplit}
\left(R_t,C_r^{\mathrm{split}}\right).
$$

其中，$R_{t,r}^{\mathrm{split}}$ 表示“只对 root $r$ 应用本次 Split、其余 roots 保持不变”得到的候选 Rubric。竞争时只比较该操作直接改变的 root $r$ 子树：

$$
\Delta_r^{\mathrm{split}}
=
\operatorname{StrictAcc}_{\mathcal S_r^t}
\left(U_r(R_{t,r}^{\mathrm{split}})\right)
-
\operatorname{StrictAcc}_{\mathcal S_r^t}
\left(U_r(R_t)\right).
$$

接受条件：

$$
\boxed{\operatorname{root\_scope\_net\_corrected}>0}
$$

并要求 Unified-Subtree 技术失败为零。旧 `parent_accuracy`、`specialized_accuracy`、child ACC/support/coverage、sibling conflict 和 leave-one-out 继续保存，但全部为诊断，不能推翻新决定。

### 3.2 Refine

触发条件保持：

$$
0.5 < \operatorname{Acc}(c)<0.80,\qquad
|\mathcal S(c)|\ge15,\qquad
\operatorname{Wrong}(c)\ge5.
$$

候选先通过 Phase17 原 node-level acceptance。对属于 root $r$ 的 child $c$ 产生 Refine 候选 $C_c^{\mathrm{refine}}$ 后，先将其应用到完整 Rubric：

$$
R_{t,c}^{\mathrm{refine}}
=
\operatorname{ApplyRefine}
\left(R_t,C_c^{\mathrm{refine}}\right).
$$

这里，$R_{t,c}^{\mathrm{refine}}$ 表示“只替换 child $c$ 的 criterion 定义、其余结构与节点保持不变”得到的候选 Rubric。随后重新运行 child $c$ 所属的整棵 root 子树，并与该 root 在轮次开始时的版本比较：

$$
\Delta_r^{\mathrm{refine}}
=
\operatorname{StrictAcc}_{\mathcal S_r^t}
\left(U_r(R_{t,c}^{\mathrm{refine}})\right)
-
\operatorname{StrictAcc}_{\mathcal S_r^t}
\left(U_r(R_t)\right).
$$

接受条件：

$$
\boxed{
\Delta_{\mathrm{child}}>0
\quad\land\quad
\operatorname{root\_scope\_net\_corrected}\ge0
}
$$

局部改善、root 持平允许接受；root 下降则拒绝。

### 3.3 同步提交

同一 root 的多个 Refine 和不同 roots 的多个 Split/Refine，只要各自通过就全部同步提交。v1 不做候选子集搜索、不选最佳赢家、不使用 Global Arbiter 回滚。提交后重新生成合并后的五份 root 报告和 Global-Arbiter 结果，仅作为 epoch 诊断，并记录同-root/跨-root interaction regression。

## 4. 失败归因与 retry

### 4.1 固定失败阶段（代码决定）

- `node_local_regression`
- `root_unified_subtree_regression`
- `technical_failure`

技术失败只进入原 Prompt 重试，不做语义归因。

### 4.2 半固定语义原因（Manager 选择）

- `applicability_overreach`
- `excessive_abstention`
- `decision_rule_regression`
- `sibling_scope_conflict`
- `subtree_reasoning_interference`
- `unstable_root_integration`
- `other`
- `uncertain`

Manager 还必须输出 confidence、来自 supplied evidence 的 sample IDs、自然语言 summary 和 recommended revision。原因标签只用于反馈和统计，不参与接受。

当 Refine node-level 通过但 root-level 下降时，反馈包含完整 corrected/harmed IDs，以及最多6条 harmed、3条 corrected 代表样本；每条包含图像、问题、A/B、gold、目标 child 前后 vote/thought、root Unified-Subtree 前后答案与分析、parent/sibling descriptions。

## Experiment Blocks

### Block 1：离线语义与指标审计
- Claim tested：新指标只改变聚合，不改变数据范围和算子合法性。
- Dataset / split：构造离线 fixtures + Discovery100 manifest。
- Compared systems：旧 SpecializedAcc replay；新 root-scope evaluator。
- Metrics：scope identity、paired corrected/harmed、None/技术失败处理、deterministic replay。
- Success criterion：同一 epoch/root/candidate 的 scope hash 完全一致；公式与离线重算一致。
- Priority：MUST-RUN。

### Block 2：Discovery100 完整演化
- Claim tested：新 Split 竞争与 Refine 双 gate 能稳定运行并保留局部专家。
- Compared systems：Phase17 历史 Control；新 treatment。
- Metrics：每轮 schedule/accept/reject、root-scope Strict ACC、旧新竞争器 disagreement、Rubric size、Global-Arbiter epoch Strict ACC。
- Success criterion：所有正式接受满足冻结规则；错误经验不被 Global Arbiter 过滤；没有后验 checkpoint 选择。
- Priority：MUST-RUN。

### Block 3：独立与外部评测
- Dataset：Dev150（逐 epoch 诊断）、heldout-500（最终 E5）、VL-RewardBench 1,247（K=3）。
- Compared systems：Initial、Phase17 E4/E5（复用）、treatment E5。
- Primary metrics：VL-RB Strict ACC、OverallAcc、MacroAcc、Coverage、paired corrected/harmed、McNemar exact p。
- Secondary metrics：Discovery/Dev/heldout Strict ACC、分类结果、token、调用数和 latency。
- Success criterion：不依赖 Coverage 下降；相对 Phase17 的差异由 paired 结果报告，不以点估计宣称显著。
- Priority：MUST-RUN。

### Block 4：机制与失败诊断
- Claim tested：新聚合器是否真的改变 Split 选择，以及 root guard 拒绝了哪些 node-level 正收益 Refine。
- Metrics：SpecializedAcc/New-guard 四象限、failure taxonomy、same-root interaction、代表案例。
- Success criterion：每个分歧候选都能追溯到固定 scope、报告和 sample IDs。
- Priority：MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | 增量成本 | 风险 |
|---|---|---|---|---|---|
| M0 | freeze + audit | manifest、scope、request identities | 全部 identity/scope checks 通过 | 离线 | scope 定义漂移 |
| M1 | smoke | 20条，至少一个 Split 和一个 Refine | parse/technical valid rate=100%，不设 ACC go gate | 每候选约20次 root 请求 | Unified 输出解析 |
| M2 | full evolution | Discovery100，固定5 epochs | 只按冻结接受规则执行 | 初始 root 报告约500次；每候选新增约100次 root 请求；每 epoch 约100次 Arbiter 诊断 | API 成本和同-root交互 |
| M3 | internal evaluation | Dev150 + heldout-500 | 只报告，不选择 | 每个完整系统分别约 $6N$ 次请求 | 误用诊断集选择 |
| M4 | external evaluation | VL-RB K=3 | 完成预注册比较 | $1247\times3\times(5+1)=22{,}446$ 次请求 | 长尾技术失败 |
| M5 | report | paired stats + mechanism table | 数字可从 sample-level artifact 重算 | 离线 | 过度解读点估计 |

实际 wall time 以 M1 双端点吞吐外推，不预先写死；请求数和 token 必须在报告中实测。预计主要成本来自 M2 候选 root 报告和 M4 的22,446次正式请求。

## Risks and Mitigations

- **Unified-Subtree 随机波动**：冻结 baseline 报告、样本顺序、temperature/max_tokens/parser；所有候选与同一 epoch baseline 配对。
- **scope 奖励过宽或过窄**：scope 由 epoch-start root local Pairwise decisive outputs 冻结，报告 scope 外激活但不改变分母。
- **同-root候选独立通过、合并后退化**：v1 不回滚，只保存合并后诊断；若频繁发生，才单独设计子集竞争实验。
- **taxonomy 强迫错误归因**：保留 `other` / `uncertain` 和自由文本；固定标签不进入接受公式。
- **新旧轨迹随机差异**：相同状态与 request identity 时优先复用冻结候选/预测；轨迹分叉后记录 fresh generation，不把差异全部归因于聚合器。
- **结构膨胀**：继续使用 Phase17 的 root child 容量和 Split retry 上限；不额外放宽。

## Compute and Data Budget

- 数据准备：无新增数据。
- 人工标注：无。
- Manager 调用：与 Phase17 同量级。
- 新增主要调用：每个候选一组100条 root Unified-Subtree；每轮100条 Global-Arbiter 诊断。
- 最终 VL-RB：22,446 个逻辑请求，两个 available-slot endpoints。
- 最大瓶颈：Unified-Subtree 报告生成与长尾重试，而非离线指标计算。

## Final Checklist

- [ ] root scope 在 epoch 开始冻结并带 hash
- [ ] Split 不再由 SpecializedAcc 决定
- [ ] Refine 保留 node gate，并增加 root non-regression
- [ ] ErrorSignature 不与 Global-Arbiter wrong 取交集
- [ ] 多候选保持 Phase17 同步提交
- [ ] Global Arbiter 不参与 candidate acceptance
- [ ] Dev/heldout/VL-RB 不参与选择
- [ ] 失败阶段、半固定原因和证据 schema 可重放
- [ ] focused unittest、完整 unittest、compile 和 diff check 通过
- [ ] sample-level paired statistics 可从 artifact 重算

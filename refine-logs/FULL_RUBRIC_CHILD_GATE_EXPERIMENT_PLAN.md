# Full-Rubric Child-Gate 实验计划

**问题**：Visual-only Gate 已证明能够学习稀疏且高精度的局部适用域，但单-root候选空间无法覆盖 criterion 中指向其他 roots 的适用关系。

**方法主张**：保持五个 roots 全部参与 M1，只在每个 root 内用 Gate 选择适用 children，可以在不修改 Rubric、criterion 判断和聚合规则的前提下，减少无关投票与 sibling conflict，并检验完整 Rubric 上的最终 ACC。

**日期**：2026-08-16

## 1. Claim Map

| Claim | 为什么重要 | 最小可信证据 |
|---|---|---|
| C1：Gate 能在完整 Rubric 中形成稀疏、高精度的 child 激活域 | 验证 applicability 文本可作为可执行边条件，而不只是自然语言说明 | 五个 roots 均产生非平凡稀疏路由；激活域 ACC、冲突和有效率可审计 |
| C2：完整 Rubric Child-Gate 能保持或提高五-root M1 | 回答单-root Gate 的负向 ACC 是否来自候选空间不完整 | heldout-500 上相对 all-children 的逐样本 M1 比较和 McNemar |
| Anti-claim：收益只是重新运行 Pairwise Worker 或修改投票规则造成 | 保证这是干净的 Gate ablation | 完全复用 Phase 10 Prompt-v2 的22节点预测；只新增 Gate 请求 |

## 2. 冻结协议

### 2.1 固定来源

- Rubric：`phase10_five_root_locked_split_refine_v1/final/rubric.json`。
- 结构：5个 roots、17个 direct children，共22个 nodes；当前没有更深层级。
- Pairwise：只读复用 `pairwise_worker_cache_prompt_ablation_v1` 的 `s3_prompt_v2_dynamic` discovery-90 与 heldout-500 预测。
- 不运行 Split、Refine、Root Gate、权重搜索或新的 Pairwise Worker。
- 五个 roots 始终参与最终等权 M1，不允许 Gate 关闭 root。

### 2.2 每样本路由与聚合

对每个样本和每个 root 分别运行一次 sibling Gate：

```text
sample
  ├─ completeness Gate   → 0–3 children
  ├─ visual Gate         → 0–4 children
  ├─ factuality Gate     → 0–4 children
  ├─ creativity Gate     → 0–3 children
  └─ clarity Gate        → 0–3 children
```

- `applicable`、`uncertain`：激活 child。
- `not_applicable`：关闭 child。
- root 内先聚合激活 children；children 平票、全部 `None` 或没有 child 激活时回退该 root 的 parent vote。
- 五个 root 子树结果继续使用现有等权 M1 聚合。
- 某 child 文本指向其他 root 时，本版不做显式跨 root 跳转；因为五个 root 的 Gate 都会运行，目标 root 仍有机会独立激活相应 child。

### 2.3 Gate Worker

- 模型：`Qwen/Qwen3-VL-8B-Instruct`。
- API：`vllm-8000` 与 `vllm-8001`，available-slot动态调度。
- `temperature=0.2`，`max_tokens=2048`，`seed=42`。
- 每个root使用固定System Prompt：parent信息、direct children及其 Focus/Apply/Exclude。
- User Prompt包含图像、question、A/B，末尾固定加入：

```text
Return JSON only. For every child, output an object containing both
"status" and "reason"; never output a bare status string.
```

- `status`必须属于 `applicable / not_applicable / uncertain`。
- `reason`只用于诊断，不校验内容、类型或长度。
- 格式失败后最多重试5次；最终失败记录 `parse_error`，该root临时回退 all children，任务不中断。
- 每个split报告 parse-valid rate、失败ID和fallback数量；目标为100%。

### 2.4 调度

- 按 root-major 顺序成批运行：同一root的全部样本共享相同System Prompt，优先利用vLLM prefix cache。
- 同一root内使用双端口available-slot调度，不做静态50/50分片。
- 缓存键包含 root、routing contract、prompt、sample fingerprint、模型与解码配置。
- 成功缓存复用；失败缓存下次只重跑对应 `(sample, root)` shard。

## 3. 对照与指标

### 3.1 主要系统

| 系统 | 作用 |
|---|---|
| Five-root parent-only | 无children基线 |
| Five-root all-children | Phase 10完整Rubric主基线 |
| Visual-only Gate | 已完成的局部Gate诊断 |
| Full-Rubric Child-Gate | 本实验主要系统 |
| Oracle child-routing upper bound | 使用gold搜索各root可达投票组合的诊断上界，不可部署 |

不使用 discovery 搜索出的固定子集作为主要方法，也不根据 heldout 选择任何Gate方案。

### 3.2 主要指标

- 完整五-root M1：ACC、Coverage、correct count、Wilson 95% CI。
- Full Gate vs all-children：corrected、harmed、net corrected、exact McNemar。
- 每个root：subtree ACC、Coverage、平均激活children、空路由率、sibling conflict。
- 每个child：activation、decisive support、激活域ACC、错误数。
- 全局稀疏性：平均激活children总数，相对固定17 children的减少比例。
- Gate稳定性：parse-valid rate、fallback、重试、A/B交换后的status一致率。
- 成本：Gate请求、token、墙钟时间、端口分布；另报告理论上可跳过的Pairwise node执行量。

注意：本实验复用已生成的22节点Pairwise预测，因此实际墙钟时间只测量Gate开销；“节省的Worker推理”是由激活矩阵计算的反事实估计，不能伪装成实测加速。

## 4. 实验阶段

新增独立目录：

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase14_full_rubric_child_gate_v1/
```

新增CLI stages：

```text
full-child-gate-freeze
full-child-gate-smoke
full-child-gate-discovery
full-child-gate-report
full-child-gate-heldout-exploratory
full-child-gate-final-report
```

### M0：Freeze 与离线审计

- 校验 Phase 10 rubric hash、5/17/22结构、每个child四段description和所有Pairwise prediction hashes。
- 冻结五个routing contracts、prompt hashes、双端口身份、数据hash和聚合规则。
- 证明freeze/discovery阶段不能读取heldout labels或报告。

### M1：Smoke

- 固定20个discovery样本，五个roots均运行，共100个Gate判断。
- 验证双端口、root-major调度、JSON parser、缓存和恢复。
- 门槛：100/100 parse-valid；每个root恰有20条路由；离线M1可完整重放。

### M2：Discovery-90

- 主路由：`90 × 5 = 450`次Gate判断。
- 位置审计：冻结20个样本，对A/B交换后再运行五个roots，共100次。
- 输出完整系统表、逐root/child诊断、冲突和稀疏性。
- Discovery不用于修改prompt、选择children或调整聚合；它只确认技术有效并形成预注册诊断。

### M3：Heldout-500 exploratory

- 在协议和discovery报告冻结后运行 `500 × 5 = 2500`次Gate判断。
- 复用同一heldout-500，明确标记为 exploratory paired diagnostic。
- 不因heldout结果重新选择root、child、prompt或路由策略。

### M4：Final report

- 汇总 discovery/heldout 的主要系统、逐root、逐child、配对统计、解析率、位置一致性和成本。
- 明确区分：局部语义路由证据、完整M1结果、Oracle诊断上界、反事实计算节省。

## 5. Success 与失败解释

### 正向成功

- Gate解析率100%，无技术fallback。
- 平均激活children显著低于17，且五个roots均形成可解释的局部激活域。
- sibling conflict下降，同时heldout完整M1不低于all-children；若高于all-children且 corrected > harmed，则支持性能收益。
- 最好获得显著McNemar；不显著时仅表述为exploratory evidence。

### 结果解释

- **稀疏且M1提高**：支持完整Rubric applicability routing。
- **稀疏且M1持平**：支持计算/解释性收益，但不支持准确率收益。
- **激活域高精度但M1下降**：说明路由召回或聚合仍有问题，不等于Gate不能理解适用性。
- **roots间表现差异大**：说明部分子树的Applicable/Not applicable质量不足，可为后续Refine edge文本提供证据。
- **大量空路由或低交换一致性**：优先修复Gate calibration，而不是修改criterion投票。

## 6. 必须覆盖的测试

- 五个roots分别得到且只得到自己的direct children；禁止跨root错误ID。
- 每个样本五个Gate结果齐全，输出顺序不影响聚合结果。
- 没有child激活时只回退对应root，不关闭该root。
- `uncertain`激活；`not_applicable`不激活。
- reason完全不影响有效性；status和child集合严格校验。
- 失败最多重试5次，最终失败被标记但不中断；重跑只补失败shard。
- 成功缓存按root隔离，不能把一个root的Gate结果复用于另一个root。
- all-children与parent-only离线重放必须与冻结基线完全一致。
- Gate不能改变任何Pairwise node output、Rubric或M1聚合逻辑。
- discovery阶段无法访问heldout；heldout报告生成后禁止选择。
- focused tests、完整unittest、compileall和diff check通过。

## 7. 预计成本

| 阶段 | Gate请求 | 预计时间 |
|---|---:|---:|
| Smoke | 100 | 约1分钟 |
| Discovery主路由 | 450 | 约3–6分钟 |
| Discovery位置审计 | 100 | 约1–2分钟 |
| Heldout | 2500 | 约15–30分钟 |

最大风险不是推理成本，而是五个roots的routing contract质量不同。第一版坚持不Refine roots，以保证结果只归因于Child Gate；Root Gate与root Refine留作后续独立实验。

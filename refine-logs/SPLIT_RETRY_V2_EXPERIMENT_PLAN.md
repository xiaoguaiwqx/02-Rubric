# Split-retry v2：强 Child 锁定与样本驱动修复计划

**日期**：2026-08-10

**定位**：一个最小机制实验。先在 Visual Grounding 已知失败案例上验证，不改 Split trigger、Specialized 投票、完整集合竞争或 Worker。

## 1. 问题与研究主张

当前 Split 在整组 children 竞争失败后，会重新聚类并重新生成全部 children。这样会出现两个问题：

1. 已经有正向贡献的 child 会在下一轮被改名或改写，搜索过程遗忘已有成果；
2. 失败通常由少数过宽或互相冲突的弱 children 导致，但下一轮仍从头搜索全部语义划分。

本实验只验证两个主张：

| ID | 主张 | 最小证据 |
|---|---|---|
| C1 | 按相对父准则贡献锁定强 child，可以避免 retry 遗忘已有局部专家。 | 锁定 child 的 criterion hash 与 Pairwise prediction hash 跨轮完全一致，且其单独与 parent 组合时在固定 parent scope 上有稳定正增益。 |
| C2 | 不重新聚类，仅用少量边界样本改写或替换弱 child，可以提高完整 child 集合的 Specialized ACC。 | Visual Grounding 在至多两轮修复后达到完整接受；若仍未达到，则至少安全保留锁定集合。 |

**反主张**：提升来自 heldout 选择、重新聚类、改变投票规则，或重新评估锁定 child。上述行为在本实验中全部禁止。

## 2. 强 Child 的定义

强弱必须相对当前 parent 定义，不能统一要求 child ACC 超过某个绝对值。对于强 Visual Grounding parent，child 达到相同绝对 ACC 更困难；对于较弱 parent，普通 child 很容易超过父准则。

令父准则固定有效区域为 \(\mathcal S(p)\)。单个 child 与 parent 的回退组合为：

$$
\hat y_{p\oplus c}(x)=
\begin{cases}
c(x), & c(x)\in\{A,B\},\\
p(x), & c(x)=\mathrm{None}.
\end{cases}
$$

定义：

$$
\Delta_{\mathrm{single}}(c)
=
\operatorname{Acc}_{\mathcal S(p)}(p\oplus c)
-
\operatorname{Acc}_{\mathcal S(p)}(p),
$$

$$
G(c)=N_{\mathrm{corrected}}(c)-N_{\mathrm{harmed}}(c).
$$

一个 child 成为**强 child 候选**需要同时满足：

1. 在 $ \mathcal S(p) $ 上 decisive support $\ge 15 $；
2. $ \Delta_{\mathrm{single}}(c)>0 $ ；
3. $ G(c)\ge 3 $，排除只偶然多纠正一个样本的弱证据；
4. 在其目标 cluster 样本上 `target_corrected >= target_harmed`，避免锁定一个偏离原语义任务的 child。

### 多个强 child 的兼容性

多个 individually strong children 可能互相冲突，因此不做组合枚举搜索，而采用确定性贪心：

1. 按 `G(c)` 降序、support 降序、cluster ID 升序排序；
2. 从空锁定集合开始，仅当加入该 child 后，锁定集合的 Specialized ACC 严格提高时才加入；
3. 未通过兼容性检查的 child 转入待修复集合。

最终得到兼容锁定集合 $ \mathcal L $。这个定义保证锁定集合本身相对 parent 是正向的，又避免使用一个对所有 roots 不公平的绝对 ACC 门槛。

已完成的 Visual Grounding artifact 可作为离线验收样例：`peripheral_detail_verification_accuracy` 单独与 parent 组合后由 `62/89` 提升至 `68/89`，净纠正 6 个样本，应被 v2 稳定识别并锁定。

## 3. 简化后的 Retry 流程

### 3.1 首次 Split

首次 Split 保持现有逻辑：

```text
ErrorSignatures
→ clustering
→ 每个 cluster 生成一个 child
→ children 多数投票，平票/全 None 回退 parent
→ 在固定 parent scope 上比较 Specialized ACC 与 Parent ACC
```

若完整集合通过，正常接受，不进入 v2 retry。

### 3.2 首次失败后的冻结

若完整集合失败：

1. 保存每个 child 的 ACC、support、corrected/harmed、target/non-target 表现和 sibling conflict；
2. 按第 2 节规则得到兼容锁定集合；
3. 冻结原始 cluster assignment，不再次调用 clustering Manager；
4. 对锁定 child 冻结 `name + description + node ID + criterion hash + Pairwise prediction hash`；
5. 失败归因明确指出哪些 child 被锁定、哪些 child 需要改写或替换。

### 3.3 用小样本包修复其他 children

每个未锁定 cluster 构建一个最多 8 条的确定性样本包：

| 样本类型 | 最多数量 | 作用 |
|---|---:|---|
| target wrong / None | 3 | 告诉 Manager 当前 cluster 中尚未解决的模式 |
| target correct | 2 | 保留已有正确边界 |
| non-target harmed | 3 | 收紧适用条件，减少过宽激活 |

样本按冻结 discovery 顺序选择，不足时留空，不从其他类别补齐。Manager 可读取 question、A/B、gold、ErrorSignature、原 vote/thought 和图片；Pairwise Worker 仍只读取 criterion description，不读取样本、gold 或失败历史。

Manager 对每个未锁定 child 二选一：

- `rewrite`：cluster 语义正确，但描述过宽或决策边界不清；保持 cluster，重写 description；
- `replace`：child 的目标方向错误或无法表示该 cluster；保持原 cluster 样本，生成新的 criterion 替代。

锁定 child 不发送生成请求，也不发送新的 Worker 请求；直接复用原 Pairwise predictions。每轮只重新评估被改写或替换的 children。

### 3.4 竞争和停止

- 最多执行 **2 轮**固定聚类修复；
- 每轮仍将 `锁定 children + 新修复 children` 作为完整集合，用原 Specialized 投票和原接受条件竞争；
- 若

$$
\operatorname{Acc}_{\mathrm{spec,full}}
\ge
\operatorname{Acc}_{\mathrm{parent}},
$$

则决策为 `full_accept`，完整集合提交。

如果两轮完整集合都失败，则只评估已经冻结的兼容锁定集合：


$$
\operatorname{Acc}_{\mathrm{spec,locked}}
\ge
\operatorname{Acc}_{\mathrm{parent}}.
$$

- 满足时决策为 `partial_accept_locked`：保留 parent 和锁定 children，其他 cluster 标记为 unresolved；
- 不满足或没有锁定 child 时决策为 `reject_parent_only`。

`partial_accept_locked` 必须与完整 Split 成功分开记录；它只说明搜索发现了可安全保留的局部专家，不能声称完整语义拆分成功。

## 4. 最小实验设计

输出目录建议：

```text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase8_visual_split_retry_locked_v2/
```

新增 stages：

```text
split-retry-v2-freeze
split-retry-v2-audit
split-retry-v2-run
split-retry-v2-report
```

### Run A：离线审计

- 读取已完成的 Visual Grounding rejected attempt；
- 重算单 child 贡献、兼容锁定集合和 locked-only Specialized ACC；
- 预期识别 `peripheral_detail_verification_accuracy`；
- 不调用 Manager、Worker 或 heldout。

### Run B：单 root 修复

- 固定 Visual Grounding parent、discovery-90、ErrorSignatures、原 clustering 和 seed=42；
- 最多两轮 sample-driven rewrite/replace；
- 397B Manager 使用 Global Rubric Memory；
- Worker 使用冻结的 8000/8001 同模型服务之一，运行前固定 endpoint identity；
- 禁止访问 heldout-500。

### Baselines

| 系统 | Parent / best Specialized ACC | 作用 |
|---|---:|---|
| Parent-only | `62/89 = 69.66%` | 固定竞争基线 |
| 当前 child-feedback retry | 最好 `61/89 = 68.54%` | v1 retry 对照 |
| Archived single strong child + parent | `68/89 = 76.40%` | 锁定机制离线 sanity check，不作为完整 Split 结果 |

## 5. 指标、判定与后续分支

主要报告：

- full / locked-only Specialized ACC；
- corrected、harmed、net corrected；
- 每个 child 的 support、ACC、target/non-target 表现；
- sibling conflict；
- lock hash 是否完全一致、Pairwise 是否确实复用；
- clustering 请求数、child-generation 请求数和 Worker 请求节省量。

结果解释：

| 结果 | 结论 | 下一步 |
|---|---|---|
| `full_accept` | C1、C2 均得到初步支持 | 再做 3 seeds，之后接入完整 Split+Refine |
| 仅 `partial_accept_locked` | C1 支持；sample-based 弱 child 修复仍不足 | 保留强 child 机制，下一步研究 residual re-clustering 或 Refine-parent→Split |
| `reject_parent_only` | 当前强 child 定义或 archived signal 不稳定 | 停止集成，检查锁定阈值与 prediction identity |

本轮不做多 seed、不做 heldout、不做五-root。只有 discovery 机制成立并冻结 candidate 后，才运行一次 exploratory heldout。

## 6. 实现与测试门槛

- 新模式关闭时，冻结 Split v1 的 request identity、history artifact 和输出不变；
- 强 child 分类边界覆盖 support=15、net gain=3、target guard 和平局；
- Visual archived strong child 必须被锁定，而仅净提升 1 个样本的 child 不锁定；
- 多个 strong children 的兼容选择确定且不依赖遍历顺序；
- retry 不产生 clustering 请求；
- 锁定 child 的 criterion/prediction hash 跨轮完全一致；
- rewrite/replace child 产生新的 request identity；
- full acceptance 仍只由原 Specialized ACC 条件决定；
- partial acceptance 仅能提交兼容锁定集合；
- transport pause、恢复、归因必要收尾和缓存原子写行为保持不变；
- discovery 阶段不能读取 heldout；
- focused tests、Split regression、完整 `unittest` 和 `compileall` 通过。

## 7. 明确延后

- 对 residual errors 重新聚类：若简单的固定 cluster + sample 修复失败，再作为 v3；
- 先 Refine parent 再 Split：保持为独立调度消融；
- 递归 Split、Drop、Merge、权重搜索和 heldout-driven 选择：不进入本实验。

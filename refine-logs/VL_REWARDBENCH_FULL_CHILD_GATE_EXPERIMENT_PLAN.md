# VL-RewardBench Full Child-Gate 实验计划

**问题**：Full Child-Gate 在 RLHF-V heldout 上实现了显著稀疏化，但未提升 M1；需要检验该路由机制能否迁移到包含 General、Hallucination 与 Reasoning 的 VL-RewardBench。

**方法假设**：固定 Phase10 Rubric 与 Prompt v2 Pairwise 判断，仅在每个 root 内动态选择 direct children，可以减少无关或冲突投票，并在不改变五-root聚合的情况下改善或保持外部分布性能。

## 1. 冻结对照

| 项目 | 冻结设置 |
|---|---|
| Dataset | VL-RewardBench 1,247 pairs |
| Rubric | Phase10 final，5 roots / 17 children / 22 nodes |
| Pairwise source | 已完成的 Prompt v2 K=3 三组预测，禁止重新生成 |
| Root policy | 五个 roots 始终全部参与等权 M1，不使用 Root Gate |
| Child routing | 每个 root 独立 Gate，仅能激活该 root 的 direct children |
| Gate model | Qwen3-VL-8B-Instruct，双端口 8000/8001 |
| Gate decoding | temperature=0.2，max_tokens=2048，replicate seeds=42/43/44 |
| Invalid route | 该 root 回退到 all children，并记录解析失败 |
| K=3 | 三次独立 Gate 推理；禁止跨 replicate 复用模型输出 |

缓存仅用于同一个 replicate 的中断恢复。目录必须隔离：

```text
run/gate/replicate_01/cache/
run/gate/replicate_02/cache/
run/gate/replicate_03/cache/
```

即使 replicate 1 与 3 的 A/B 顺序相同，也必须独立请求模型。

## 2. 比较系统

| 系统 | 作用 |
|---|---|
| `parent_only` | 五个 roots，不启用 children |
| `all_children` | 已完成的 Phase10 Prompt v2 等权 M1；主要基线 |
| `full_child_gate` | 五个 roots 内分别动态路由 children；主要方法 |
| `single_root_gate_*` | 每次只对一个 root 使用 Gate，其余 roots 保留全部 children；离线归因 |
| `oracle_child_routing_upper_bound` | 使用 gold 搜索可正确聚合的子集；仅表示信号上限 |

`single_root_gate_*` 和 oracle 均复用同一批 Gate 与 Pairwise artifact，不产生额外模型请求。Oracle 严禁作为可部署方法或主要结果。

## 3. K=3 执行语义

对 replicate \(r\in\{1,2,3\}\)：

1. 读取该 replicate 已冻结的22-node Pairwise预测。
2. 按该 replicate 的 A/B counterbalance 顺序，为五个 roots 分别执行 Gate。
3. 在每个 root 内，仅聚合 Gate 激活的 children；无 child 或 child 平票时回退 parent。
4. 五个 root 仍按原始等权 M1 聚合。
5. 将显示位置 A/B 映射回原始回答索引。

最后对三个原始索引预测做 K=3 多数投票。Gate 不读取 Pairwise vote、gold、类别标签或其他 replicate 的结果。

## 4. 指标与诊断

主要指标：

- OverallAcc、MacroAcc；
- General、Hallucination、Reasoning 三类准确率；
- Coverage、Strict ACC；
- `all_children -> full_child_gate` 的 corrected、harmed、net corrected 与 exact McNemar。

路由诊断：

- 平均激活 children 数与激活削减率；
- 每个 root 的空路由率、sibling conflict before/after；
- 每个 child 在激活且 decisive 区域的 ACC/support；
- Gate parse-valid rate、重试数与失败样本；
- replicate 1/3 同顺序一致性，以及交换顺序后的状态一致性；
- 五个 `single_root_gate_*` 对最终 M1 的边际影响。

## 5. 成功与解释标准

主要成功标准：

```text
Full-Gate OverallAcc >= All-children OverallAcc
且 corrected > harmed
```

支持性标准：

- MacroAcc 不下降；
- Coverage 下降不超过1pp；
- parse-valid rate=100%；
- children 激活量减少至少70%。

若 ACC 持平但激活量显著下降，只能表述为“潜在效率收益”；本实验复用预计算 Pairwise 输出，不直接证明真实端到端加速。若 ACC 下降，则结合 single-root 与类别结果判断是路由错误、过度稀疏化，还是跨分布适用边界失效。

## 6. 运行阶段与成本

```text
vlrb-child-gate-freeze
vlrb-child-gate-audit
vlrb-child-gate-smoke
vlrb-child-gate-run
vlrb-child-gate-retry
vlrb-child-gate-report
```

请求预算：

- Smoke：\(20\times5\times3=300\) 个 Gate 请求；
- Full：\(1247\times5\times3=18,705\) 个独立 Gate 请求；
- 不新增 Pairwise Worker 请求。

按近期双端口 Full-Gate 吞吐估计，完整 Gate 约需1.5–3小时；实际时间受 VL-RewardBench 文本长度和异常长输出影响。

## 7. Artifact

```text
output/evolving_structured_rubrics/vl_rewardbench_phase14_full_child_gate_v1/
  frozen_manifest.json
  offline_audit.json
  smoke/gate/replicate_01..03/
  run/gate/replicate_01..03/
  retry/gate/replicate_01..03/
  final_report.json
  final_report.md
```

最终报告必须明确：这是复用同一 VL-RewardBench 的 exploratory transfer experiment，不用于重新选择或修改 Rubric。

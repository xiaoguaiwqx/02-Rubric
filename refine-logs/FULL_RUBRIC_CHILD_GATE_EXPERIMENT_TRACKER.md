# Full-Rubric Child-Gate Experiment Tracker

| Run ID | 阶段 | 目的 | Split | 请求量 | 关键验收 | 状态 |
|---|---|---|---|---:|---|---|
| FCG-001 | Freeze | 冻结Phase 10 Rubric、五个contracts和Pairwise hashes | Offline | 0 | 5 roots / 17 children / 22 nodes；无heldout访问 | TODO |
| FCG-002 | Smoke | 验证五root Gate、双端口、parser和缓存 | Discovery-20 | 100 | parse 100%；五root结果完整 | TODO |
| FCG-003 | Discovery | 生成五root主路由 | Discovery-90 | 450 | 无缺失shard；聚合可重放 | TODO |
| FCG-004 | Position audit | 检查Gate是否受A/B顺序影响 | Discovery-20 swapped | 100 | 报告逐root和逐child一致率 | TODO |
| FCG-005 | Discovery report | 冻结主指标、稀疏性和冲突诊断 | Discovery-90 | 0 | 不做后验选择 | TODO |
| FCG-006 | Heldout | 完整Rubric Child-Gate探索性验证 | Heldout-500 | 2500 | parse 100%；无协议漂移 | TODO |
| FCG-007 | Final report | all-children、Visual-only与Full Gate配对比较 | Both | 0 | ACC/Coverage/McNemar/成本完整 | TODO |

## Stop / Go

- Smoke出现缺失root、错误child ID、聚合漂移或解析率不足100%：先修实现，不进入Discovery。
- Discovery完成后冻结协议；不根据Discovery ACC修改prompt、选择children或聚合。
- Heldout为复用数据上的exploratory diagnostic，不作为新的无偏confirmatory test。

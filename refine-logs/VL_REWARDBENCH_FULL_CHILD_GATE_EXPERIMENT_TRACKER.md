# VL-RewardBench Full Child-Gate Tracker

| Run | 目的 | 请求量 | 状态 | 验收 |
|---|---|---:|---|---|
| Freeze | 冻结 Rubric、K=3 Pairwise、顺序和 Gate identity | 0 | DONE | 1,247样本、18,705逻辑请求、seeds=42/43/44 |
| Audit | 离线重放 all-children 基线 | 0 | DONE | OverallAcc=69.53%，MacroAcc=63.37%，逐票一致 |
| Smoke | 验证三组独立 Gate、双端口和聚合 | 300 | DONE | 三组cache hit均为0；8000/8001各150；parse=100% |
| Full | 完成1,247样本 K=3 Gate | 18,705 | TODO | 不跨 replicate 复用；安全续跑 |
| Retry | 仅重试解析失败 shard | 按失败数 | TODO | 保留失败标记，目标 parse=100% |
| Report | Overall/Macro/category/paired/routing | 0 | TODO | all/full/single-root/oracle 完整 |

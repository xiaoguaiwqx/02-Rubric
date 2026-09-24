# Hallucination100 本地 Split 初始化对照

## 冻结设计

从五个固定 root 出发，用 Hallucination100 生成错误签名、语义聚类和初始 children，再执行现有五轮逐例反思与局部 Strict ACC 接受。直接复用 `framework_v6.initialize`、`subtree_local_reflection.run` 和 `framework_v6.external`。

首轮 seed11，复用 `output/vlrb_hallucination100_transfer/seed11` 的配置和数据路径、100 条训练样本及 A/B 顺序、648 条 Hallucination 留出名单；一条近似图像仍排除。Manager Qwen3.5-27B、不思考；signature/case_reflection 并发均 6；Worker Qwen3-VL-8B；每根随机正确案例 5 条；其他算法与提示词不变。

区别仅为初始化来源：旧 seed11 手动装载历史 Init，本实验让现有 runner 在独立目录中调用初始化。初始化继续沿用原有生成规则，不增加新的接受门。

新 Init 与新 Final 均在完整 VLRB 1247 条上执行冻结 K=3 日程；Final 复用新 Init 中未改动 root 的报告。历史共同 Init 和旧 seed11 Final 均读取保存的预测，不重新推理。

主要比较：新 Init vs 历史 Init；新 Final vs 新 Init；新 Final vs 旧 seed11 Final。汇总完整 VLRB、训练100、未见 Hallucination648、General、Hallucination全组、Reasoning 的 Strict ACC、覆盖率、Covered ACC、纠正/伤害。主要迁移结论依据未见 Hallucination。完整 Hallucination/完整 VLRB 包含演化样本。

## 运行命令

```powershell
conda activate critiq
Set-Location 'D:\3-Work\02-DD-LLM\CritiQ-framework-v6'
$module = 'experiments.evolving_structured_rubrics.vlrb_hallucination_fresh_init'
python -m $module all --seed 11 --attempt-limit 10 --manager-attempt-limit 10
```

`all` 顺序执行初始化与演化、VLRB 新 Init/Final 推理、离线四版本汇总。输出为 `output/vlrb_hallucination100_fresh_init/seed11`。若已在后台运行，不要重复启动同一目录。

中断后同一命令恢复；如果达到累计 Worker 尝试上限，调高 `--attempt-limit` 后恢复，成功缓存复用。也可分别使用 `prepare`、`run`、`vlrb`、`report` 阶段。Manager 尝试上限独立指定。

## 状态

配置及冻结样本名单已准备，离线配置检查通过。真实进度以新输出目录的 `state.json`、`vlrb/report.json`、`transfer_report.json` 和日志为准。

# Hallucination100 三种子实验：运行指南

协议见 [plan.md](plan.md)。实验入口复用历史 `discovery100_27b_strict_preserve5` 的配置与 Init Rubric；演化集按冻结的 VLRB K=3 日程第一轮排列 A/B 并重标 Gold。**不对历史 Init 或 Discovery Final 发起新的 VLRB 调用**。每个新 Final 各跑完整 VLRB1247 官方 K=3，再离线切出 Hallucination 留出集。输出写入 `output/vlrb_hallucination100_transfer/seed11|seed29|seed47`，不覆盖旧运行。

在 PowerShell 中，从此仓库根目录执行：

```powershell
cd 'D:\3-Work\02-DD-LLM\CritiQ-framework-v6'
conda activate critiq
$module = 'experiments.evolving_structured_rubrics.vlrb_hallucination_transfer'

# 纯离线：三个种子的固定抽样与配置检查；不会调用模型。
foreach ($seed in 11, 29, 47) {
    python -m $module prepare --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "prepare failed for seed $seed" }
    python -m $module check --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "check failed for seed $seed" }
}

# 模型运行：每个 seed 复用同一 Init Rubric，在各自的 100 条上演化。
# vlrb 阶段仅对该 seed 的新 Final 做完整 1247 条 K=3 推理。
foreach ($seed in 11, 29, 47) {
    python -m $module run --seed $seed --manager-attempt-limit 10
    if ($LASTEXITCODE -ne 0) { throw "evolution failed for seed $seed" }
    python -m $module vlrb --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "VLRB evaluation failed for seed $seed" }
    python -m $module report --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "report failed for seed $seed" }
}

# 离线汇总三个 seed；报告每个 seed 和未见 Hallucination 留出集的增量均值/范围。
python -m $module summary
if ($LASTEXITCODE -ne 0) { throw 'three-seed summary failed' }
```

`prepare` 和 `check` 可重复执行；`run` 和 `vlrb` 使用现有缓存/状态续跑，失败后先检查相应 `seedXX` 目录并重复同一命令，不新建另一个输出身份。`run` 的 Manager 重试上限为 10，Worker 的 `--attempt-limit` 仍默认为 4；已写入的 Manager 成功响应直接复用，失败记录从下一次序号继续。若模型服务地址变化，应先核对历史配置并为整个实验使用一致的新配置，不在三个 seed 中途只修改其中一个。默认 Manager 读取历史配置所指向的 `D:\3-Work\02-DD-LLM\CritiQ\.env`。

每个 `seedXX/transfer_report.json` 给出全量、Hallucination 留出集及各来源统计；来源统计分别列出全量 `hallucination_sources` 与未见样本 `heldout_hallucination_sources`。`summary.json` 给出三种子对照。`seedXX/split.json` 冻结 100 条训练 ID、留出 ID 和重复剔除原因。完整 VLRB 结果包含演化用的 100 条，判断未见样本迁移必须查看留出结果。

# Phase22 运行与恢复指南

> 历史8B协议的复现入口，不是新模型评测配置。Discovery、heldout及VLRB已完成；实际状态以运行目录中的 `stage_status.json` 和最终报告为准。不要因本页列有命令而重新启动已完成的实验。

**Experiment**: `phase22_all_sample_subtree_adaptive_recluster_evolution_v1`\
**Config**: `experiments/evolving_structured_rubrics/configs/rubric_evolution_phase5.example.json`\
**Output root**: `output/evolving_structured_rubrics/rubric_evolution_phase5`

## 历史启动前验证

以下为原始交接时完成的前置阶段，不代表当前实验仅进行到这里：

- `all-sample-adaptive-evolution-freeze`
- `all-sample-adaptive-evolution-audit`
- `all-sample-adaptive-evolution-smoke`

The live smoke evaluated 20 Discovery samples, completed 20/20, used both
Unified-Subtree endpoints, and did not mutate the formal trajectory.

Verification before handoff:

- focused tests: 30/30 passed;
- full repository tests: 477/477 passed;
- fresh-agent review: GO, no blocking issues;
- frozen offline audit: all 16 checks passed.

## Shell setup

Run every command in the `critiq` Conda environment:

~~~powershell
conda activate critiq
~~~

The Manager profiles require `GUIJI_API_KEY`. Set it in the current shell;
do not write the secret into the repository:

~~~powershell
$env:GUIJI_API_KEY = "<your-key>"
if (-not $env:GUIJI_API_KEY) { throw "GUIJI_API_KEY is missing" }
~~~

Optional Worker endpoint preflight:

~~~powershell
$config = "experiments/evolving_structured_rubrics/configs/rubric_evolution_phase5.example.json"
$configValue = Get-Content $config -Raw | ConvertFrom-Json
foreach ($endpoint in $configValue.backend_pool.endpoints) {
    Invoke-WebRequest ($endpoint.base_url.TrimEnd('/') + '/models') -UseBasicParsing -TimeoutSec 5
}
~~~

## Common arguments

~~~powershell
$config = "experiments/evolving_structured_rubrics/configs/rubric_evolution_phase5.example.json"
$output = "output/evolving_structured_rubrics/rubric_evolution_phase5"
$runner = "experiments.evolving_structured_rubrics.run_rubric_evolution"
~~~

## 1. Formal Discovery evolution

~~~powershell
python -m $runner --config $config --output-dir $output all-sample-adaptive-evolution-run
~~~

The stage is resume-aware, but a process exit alone does not mean the
trajectory completed. Check the persisted status before running the report:

~~~powershell
$phase22 = Join-Path $output "phase22_all_sample_subtree_adaptive_recluster_evolution_v1"
$status = Get-Content (Join-Path $phase22 "stage_status.json") -Raw | ConvertFrom-Json
$history = Get-Content (Join-Path $phase22 "evolution_history.json") -Raw | ConvertFrom-Json
if ($status.run.status -eq "paused" -or -not $history.completed) {
    Write-Host "Phase22 is paused or incomplete; resume run before report." -ForegroundColor Yellow
}
if ($status.run.status -ne "passed" -or -not $history.completed) {
    return
}
~~~

If a technical failure pauses the run, inspect:

~~~text
output/evolving_structured_rubrics/rubric_evolution_phase5/
  phase22_all_sample_subtree_adaptive_recluster_evolution_v1/
    stage_status.json
    epochs/epoch_XX/pause.json
~~~

Fix the technical cause and run the same command again. Do not edit scientific
history or convert technical failures into wrong predictions.

Do not run the report while `stage_status.json` contains
`run.status = paused` or while `evolution_history.json` contains
`completed = false`.

## 2. Discovery report

After the run reaches a completed state:

~~~powershell
python -m $runner --config $config --output-dir $output all-sample-adaptive-evolution-report
~~~

Primary outputs:

~~~text
.../phase22_all_sample_subtree_adaptive_recluster_evolution_v1/
  evolution_history.json
  final/rubric.json
  final/discovery_report.json
  dev150_trajectory.json
~~~

Each candidate attempt contains the all-sample ledger, decision, Phase21
selective diagnostic, and—when applicable—failure attribution and retry action.
Split attempts additionally contain cluster lineage and a complete child bundle.

## 3. Final-only heldout evaluation

~~~powershell
python -m $runner --config $config --output-dir $output all-sample-adaptive-evolution-heldout
python -m $runner --config $config --output-dir $output all-sample-adaptive-evolution-final-report
~~~

Do not use heldout results to modify, rerun, select, or roll back the trajectory.

## 4. VL-RewardBench K=3

This stage requires the completed Phase21 VL-RewardBench artifacts as a frozen
control. They are expected at:

~~~text
output/evolving_structured_rubrics/
  vl_rewardbench_unified_subtree_bundle_evolution_v1/
~~~

Run in order:

~~~powershell
python -m $runner --config $config --output-dir $output vlrb-all-sample-adaptive-freeze
python -m $runner --config $config --output-dir $output vlrb-all-sample-adaptive-audit
python -m $runner --config $config --output-dir $output vlrb-all-sample-adaptive-smoke
python -m $runner --config $config --output-dir $output vlrb-all-sample-adaptive-run
python -m $runner --config $config --output-dir $output vlrb-all-sample-adaptive-retry
python -m $runner --config $config --output-dir $output vlrb-all-sample-adaptive-report
~~~

Final external report:

~~~text
output/evolving_structured_rubrics/
  vl_rewardbench_all_sample_adaptive_recluster_evolution_v1/
    final_report.json
    final_report.md
~~~

The report compares Initial, Phase17 E4, Phase21 final, and Phase22 final.
Benchmark outputs are selection-forbidden. Phase22 is a combined intervention,
so the report must not attribute the difference separately to the acceptance
metric or reclustering.

## Acceptance invariant

For every complete root-subtree candidate:

$$
g_i =
\mathbf{1}[p_i^1=y_i]
-
\mathbf{1}[p_i^0=y_i],
$$

and

$$
\operatorname{Accept}(C_r)
\iff
\sum_i g_i > 0.
$$

`None` is wrong. A technical failure pauses the attempt. Split and Refine use
the same strict-positive gate.

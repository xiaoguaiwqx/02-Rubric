# Phase 8 Split-retry / Refine-trigger Parallel Runbook

The two ablations share frozen discovery inputs and Manager settings, but use
separate output directories, caches, and Pairwise endpoints.  Run freeze and
offline audit serially before starting the online stages in parallel.

```powershell
$ErrorActionPreference = "Stop"
$python = "C:\Users\wenqx\miniconda3\envs\critiq\python.exe"
$module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
$config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
$output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

function Invoke-Phase8Stage([string]$stage) {
    & $python -m $module --config $config --output-dir $output $stage
    if ($LASTEXITCODE -ne 0) { throw "$stage failed with exit code $LASTEXITCODE" }
}

# Shared preparation: no heldout access and no Worker requests.
Invoke-Phase8Stage "split-retry-visual-freeze"
Invoke-Phase8Stage "split-retry-visual-audit"
Invoke-Phase8Stage "refine-role-freeze"
Invoke-Phase8Stage "refine-role-audit"

# Both local Worker services must be ready before launching.
Invoke-RestMethod "http://10.102.137.255:8000/v1/models" | Out-Null
Invoke-RestMethod "http://10.102.138.0:8000/v1/models" | Out-Null

# Run these two commands in separate PowerShell terminals.
Invoke-Phase8Stage "split-retry-visual-run"  # Worker: 8000
Invoke-Phase8Stage "refine-role-run"         # Worker: 8001

# Freeze discovery-selected results only after each run completes.
Invoke-Phase8Stage "split-retry-visual-report"
Invoke-Phase8Stage "refine-role-report"

# Optional exploratory heldout for the discovery-frozen Refine result.
Invoke-Phase8Stage "refine-role-heldout"
Invoke-Phase8Stage "refine-role-final-report"
```

The Visual retry pilot intentionally ends at discovery report.  Its current
strong-child handling is Manager preservation guidance, not hash-identical
criterion or Pairwise-cache reuse; reports must retain that distinction.

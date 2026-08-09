# Manager Global-Rubric Memory Ablation

## 1. Research question

Does exposing the latest committed Rubric to the Split clustering and child-generation Managers improve the final equal-weight five-root M1 accuracy?

This is a clean Manager-memory ablation. Control is the completed `phase6_split_only_evolution_v2`; Treatment is `phase6_split_only_evolution_global_memory_v1`. Define, trigger conditions, Specialized Accuracy, acceptance, voting, Manager/Worker models, seed, and epoch policy remain unchanged.

## 2. Frozen comparison

| Component | Control v2 | Global-memory Treatment |
|---|---|---|
| ErrorSignatures | Original v2 artifacts | Exact v2 artifacts, read-only; no online fallback |
| Clustering | No global context | Epoch-start committed Rubric |
| Child generation | Parent, cluster, signatures, siblings, history | Same inputs plus the same epoch snapshot |
| Failure attribution | Existing required attribution | Unchanged; no Rubric memory |
| Acceptance | `specialized_accuracy >= parent_accuracy` | Unchanged |
| Final aggregation | Equal-weight five-root M1 | Equal-weight five-root M1 |
| Heldout | Existing heldout-500 | Same heldout-500, exploratory paired comparison |

## 3. Rubric memory contract

Each epoch saves `epochs/epoch_NN/rubric_memory.json`. It contains only:

- committed Rubric hash and ordered root IDs;
- preorder nodes with `node_id`, criterion name, and description;
- deterministically sorted parent-child edges and conditions.

It contains no examples, predictions, labels, metrics, scores, or heldout information. All roots scheduled in one epoch receive the same snapshot. Accepted children become visible only in the following epoch.

## 4. Signature reuse and failure conditions

`split-memory-freeze` verifies the completed Control run, Phase-5 lineage, discovery data, roots, triggers, parent Pairwise predictions, each root's original ErrorSignature request identity, and every root/sample signature hash. Per-root source identity is intentional because historical backend concurrency changes altered transport hashes without changing the ErrorSignature prompt/model/decoding contract. Treatment records a source manifest and copies only exact v2 artifacts into its own cache with `generated=0` provenance.

Freeze fails if Control is incomplete or any signature is missing, invalid, or identity-mismatched. Clustering, children, candidate Pairwise results, and failure attributions are never copied from Control.

## 5. Run order

```powershell
$ErrorActionPreference = "Stop"
$python = "C:\Users\wenqx\miniconda3\envs\critiq\python.exe"
$module = "experiments.evolving_structured_rubrics.run_rubric_evolution"
$config = "experiments/evolving_structured_rubrics/configs/local/rubric_evolution_phase5_8001.json"
$output = "output/evolving_structured_rubrics/rubric_evolution_phase5"

function Invoke-MemoryStage([string]$stage) {
    & $python -m $module --config $config --output-dir $output $stage
    if ($LASTEXITCODE -ne 0) { throw "$stage failed: $LASTEXITCODE" }
}

Invoke-MemoryStage "split-memory-freeze"
Invoke-MemoryStage "split-memory-smoke"
Invoke-MemoryStage "split-memory-run"
Invoke-MemoryStage "split-memory-report"
Invoke-MemoryStage "split-memory-heldout"
Invoke-MemoryStage "split-memory-final-report"
```

The heldout stage must not run before the final discovery Rubric is frozen.

## 6. Reporting and interpretation

Primary evidence is Control versus Treatment heldout-500 equal-weight five-root M1 ACC. Reports also include discovery ACC/Coverage, exact paired corrected/harmed and McNemar statistics, root-local acceptance trajectories, Manager/Worker costs, child counts, activation/overlap diagnostics, memory growth, and lexical near-duplicate candidates.

- Treatment ACC above Control: evidence that global Rubric memory helps performance.
- Equal ACC: no observed performance benefit; semantic changes remain diagnostic.
- Lower ACC: evidence that global memory may suppress useful cross-root specialization.

Because heldout-500 was used in prior experiments, the result is exploratory paired evidence rather than a new unbiased confirmatory test.

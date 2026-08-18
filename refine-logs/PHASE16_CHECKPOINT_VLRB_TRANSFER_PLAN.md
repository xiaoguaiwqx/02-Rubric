# Phase16 Checkpoint VL-RewardBench Transfer Diagnostic

**Problem**: Phase16 Prompt-v2 evolution reached its best observed heldout-500
M1 at epochs 3--4 (78.2%), while its final epoch-5 Rubric performed worse than
the Phase10 Prompt-v2 control on VL-RewardBench.  It is unknown whether this
external drop was already present before the heldout peak.

**Method thesis**: Evaluate frozen epoch-2 and epoch-3 Rubrics with the same
Prompt-v2, K=3 counterbalanced VL-RewardBench protocol, then compare their
paired external transfer behaviour to the existing epoch-5 treatment and
controls.

**Date**: 2026-08-17

## Claim map

| Claim | Why it matters | Minimum evidence | Blocks |
|---|---|---|---|
| C1: Epoch-3 improvement transfers beyond RLHF-V heldout-500, or it does not | Separates a useful stopping point from in-domain/post-hoc adaptation | K=3 OverallAcc, MacroAcc and paired corrected/harmed versus epoch 5 | B1 |
| C2: Any transfer change is attributable to checkpoint descriptions, not prompt, data, order or decoding | Makes the checkpoint diagnosis interpretable | Frozen shared request identity and exact reuse only for identical descriptions | B1, B2 |

## Experimental storyline

- **Main diagnostic**: Epoch 2 vs Epoch 3 vs existing epoch-5 Rubric on the
  same external benchmark protocol.
- **Controls**: Read-only Initial five-root Prompt v2 and Phase10 final
  Prompt-v2 results, already completed under this protocol.
- **Deliberately excluded**: no new Manager calls, no Split/Refine, no gate,
  no root-weight search, and no selection of a new official final Rubric.

## Block 1: Epoch-2 / Epoch-3 external transfer

- **Claim tested**: Whether the heldout trajectory has a corresponding
  external-transfer optimum.
- **Data/task**: Frozen VL-RewardBench, 1,247 pairs; structured image/question/
  candidate-A/candidate-B judgement.
- **Systems**:
  1. Initial five-root Prompt v2 (read-only control);
  2. Phase10 final Prompt v2 (read-only control);
  3. Phase16 epoch 2 equal-vote M1;
  4. Phase16 epoch 3 equal-vote M1;
  5. Phase16 epoch 5 equal-vote M1 (read-only existing treatment).
- **Protocol**: K=3 balanced A/B/A or B/A/B ordering, seed and record order
  exactly matching `vl_rewardbench_phase16_prompt_v2_evolved_v2`; Pairwise
  Worker Prompt v2, temperature 0.5, max_tokens 2048, available-slot pool over
  vLLM 8000 and 8001.  No benchmark labels appear in model prompts.
- **Reuse rule**: For every `(sample, replicate, node_id, description hash)`,
  reuse only an existing epoch-5 output with identical request identity.
  Generate each missing historical description once per scheduled record and
  replicate; never reuse an epoch-5 output merely because its criterion name
  matches.
- **Primary metrics**: OverallAcc and MacroAcc using the existing 397B output
  parser; paired corrected/harmed and exact McNemar versus epoch 5.
- **Secondary diagnostics**: General/Hallucination/Reasoning category ACC,
  validity/technical-retry rate, prediction coverage, changed-node counts and
  endpoint request distribution.
- **Success interpretation**:
  - Epoch 3 > epoch 5 externally: evidence that later Refine caused external
    regression and motivates an early-stopping/safety study.
  - Epoch 3 <= epoch 5 externally: the 78.2% heldout peak is not sufficient
    evidence of broader transfer; distribution mismatch remains the leading
    explanation.
  - Epoch 2 > epoch 3 externally: additional epoch-3 specialization harms
    transfer even before final epoch-5 updates.
- **Status**: MUST-RUN exploratory diagnosis.  Epochs were already inspected
  on heldout-500, so VL-RewardBench may characterize but must not retroactively
  select a formal final checkpoint.

## Block 2: Protocol integrity and report audit

- **Claim tested**: Comparisons use one benchmark identity and one worker
  identity.
- **Checks**: checkpoint Rubric hashes/node sets; record and schedule hashes;
  K=3 balance; worker request spec; source-output provenance; no benchmark
  labels in prompts; complete retry accounting; exact reconstruction of the
  existing epoch-5 logical votes.
- **Failure interpretation**: Any identity drift invalidates comparison and
  requires a new experiment directory/freeze before inference.
- **Status**: MUST-RUN before B1.

## Run order

| Milestone | Goal | Decision gate | Cost |
|---|---|---|---|
| M0 Freeze + audit | Freeze two historical Rubrics and reusable shards | All hashes/specs match | Offline |
| M1 Smoke | 20 records for both epochs across both endpoints | parse-valid >=99%, both endpoints used | At most 2 x 20 x 3 x changed nodes |
| M2 Full run | Fill only missing epoch-2/3 description predictions | No silent fallback or identity drift | At most 2 x 1,247 x 3 x 23 = 172,086 logical requests; normally much lower through exact description reuse |
| M3 Retry + report | Retry technical failures and produce paired report | unresolved failures explicitly reported | Retry-only |

## Risks and mitigations

- **Post-hoc selection**: label all results exploratory and retain all three
  checkpoints; do not replace epoch 5 by the best VL-RewardBench result.
- **Historical-description mismatch**: key reuse by node ID plus description
  SHA-256 and full request identity, never by criterion name.
- **Benchmark overlap concern**: this is an external diagnostic only; the next
  discovery-data redesign must source-filter and deduplicate against this
  benchmark.
- **Cost inflation**: deduplicate missing descriptions across epoch 2 and 3
  before dispatch, then shard/resume by `(replicate, description hash)`.

## Final checklist

- [ ] Epoch 2 and 3 Rubric hashes frozen
- [ ] Prompt/decode/order/dataset identity matches prior Prompt-v2 VLRB run
- [ ] Identical outputs only are reused
- [ ] K=3 and parser protocol unchanged
- [ ] OverallAcc, MacroAcc, category metrics and paired results reported
- [ ] Results marked exploratory; no post-hoc checkpoint selection

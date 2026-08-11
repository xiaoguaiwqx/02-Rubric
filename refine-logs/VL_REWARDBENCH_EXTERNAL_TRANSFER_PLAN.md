# VL-RewardBench External Transfer Experiment Plan

**Date**: 2026-08-11

**Status**: planned

**Experiment ID**: `vl_rewardbench_external_transfer_v1`

## Goal

Test whether the frozen structured Rubric system transfers to an external
vision-language preference benchmark, relative to both its unevolved five-root
baseline and the same Qwen3-VL-8B-Instruct model used as a generic benchmark
judge.

This is an **exploratory external-transfer evaluation**. The selected Rubric
was chosen using the existing heldout-500 development results; therefore this
run must not be presented as a fresh confirmatory model-selection test.

## Frozen Systems

| Label | Frozen artifact | Nodes | Existing heldout-500 M1 | Role |
|---|---|---:|---:|---|
| `native_vlrb_prompt` | `data/VL_RewardBench/prompt.py` | 0 | N/A | Same-model generic judge baseline |
| `initial_five_root_m1` | Phase-5 initial five-root rubric | 5 | 65.0% | Structured baseline |
| `role_aware_refine_epoch_01` | `phase8_refine_role_aware_v2/epochs/epoch_01/rubric_committed.json` | 17 | 71.2% | Primary system |

`role_aware_refine_epoch_01` is the selected primary Rubric because it is a
complete, discovery-committed five-root Rubric and is the best complete
checkpoint in the Role-aware Refine experiment (356/500, 71.2%). The 72.4%
fixed-ensemble checkpoint diagnostic is excluded because it was an exploratory
heldout counterfactual rather than an independently committed evolution
checkpoint. The 71.8% Visual locked-child-only configuration is retained as a
separate mechanism ablation, not the canonical complete Split+Refine system.

## Dataset and Scope

- Input: `data/VL_RewardBench/data/test-00000-of-00001.parquet`.
- Expected size: 1,247 preference pairs.
- Images are read from the parquet `image.bytes` field; no image conversion,
  resizing policy, answer rewriting, or label normalization may be applied
  without recording it in the manifest.
- The preferred response is the response whose `human_ranking` value is `0`;
  the response whose value is `1` is rejected. This follows the repository's
  `inference_hf.py` / `cal.py` convention and is verified by an offline unit
  test.
- By explicit protocol choice, overlap detection with the development data is
  skipped. The manifest must record `overlap_audit: skipped_by_protocol`, and
  every report must avoid calling this a strictly disjoint external test.

## Counterbalanced K=3 Protocol

For every pair with original responses `(r0, r1)`, deterministically sort the
internal sample IDs by `SHA256("vlrb-k3-v1|42|sample_id")`, then alternate the
first order (b) along that ordering. The three replicates use
`(b, 1-b, b)`. Thus every pair is evaluated in both response orders, while the
doubled order is exactly balanced as far as an odd-size benchmark permits:
624 pairs use `A/B/A` and 623 pairs use `B/A/B`.

The released parquet contains nine duplicated original IDs. They remain
separate preference-pair rows; the runner appends a stable row suffix only to
its internal cache/sample ID and retains the original benchmark ID for group
reporting.

All model decisions are mapped back to the original response index before
scoring. The K=3 result uses a majority decision among valid mapped votes:

\[
\hat y(x)=
\begin{cases}
i, & \sum_{k=1}^{3}\mathbf 1[\hat y_k(x)=i]\ge2,\\
\mathrm{Tie}, & \text{otherwise.}
\end{cases}
\]

The third evaluation removes the ordinary two-vote tie while preserving
benchmark-level A/B balance. A residual `Tie`, `None`, or native-prompt parse
failure counts as incorrect for strict Overall Accuracy and is also exposed
through Coverage and order-consistency diagnostics.

## Inference Systems

### Native VL-RewardBench prompt

- Model: `Qwen3-VL-8B-Instruct`, routed only to port 8000.
- Prompt: byte-identical logical content of `data/VL_RewardBench/prompt.py`.
- Decoding: the benchmark script's `temperature=0.2`, `top_p=0.2`.
- Parse only the declared final `Overall Judgment: Answer X is better` form,
  while storing raw responses and parse status. Do not repair ambiguous prose
  into a vote.

This is the benchmark-native baseline. Its decoding differs from the frozen
structured Worker P05, so it is a system baseline, not a prompt-only ablation.

### Structured systems

- Model / Worker: the frozen Qwen3-VL-8B-Instruct P05 Worker, only port 8000.
- Prompt: each criterion's existing structured Pairwise prompt; no native
  generic Accuracy/Completeness/Clarity/Relevance prompt is appended.
- Aggregation: unchanged equal-root M1, including each system's original
  parent/child vote and fallback semantics.
- Manager: never called.
- Benchmark labels: held exclusively outside Worker prompts.

## Metrics

Primary metrics for each system:

1. **K=3 strict Overall ACC**: correct majority choices / 1,247.
2. **K=3 Macro ACC**: unweighted mean of official VL-RewardBench task-group
   strict ACCs, reconstructed from the benchmark's `cal.py` grouping.

Secondary diagnostics:

- K=3 Coverage and covered ACC;
- per-order ACC and directional position gap;
- order-disagreement, `Tie`/`None`, and native parser-invalid rates;
- official source/task-group ACCs;
- paired corrected, harmed, net corrected, and exact McNemar comparisons for
  `initial_five_root_m1 -> role_aware_refine_epoch_01` and
  `native_vlrb_prompt -> role_aware_refine_epoch_01`;
- structured node count, request count, valid-rate, and elapsed time.

No VL-RewardBench metric may be used to select a checkpoint, modify a
description, choose children, alter a vote weight, or trigger another epoch.

## Stages and Artifacts

Output directory:

```text
output/evolving_structured_rubrics/vl_rewardbench_external_transfer_v1/
  frozen_manifest.json
  dataset_manifest.json
  order_schedule.json
  smoke/{native,structured,combined}/
  run/
    native/
    structured/{initial_five_root_m1,role_aware_refine_epoch_01}/
    combined/
  report.json
  report.md
```

Proposed CLI stages:

```text
vlrb-freeze
vlrb-smoke
vlrb-run
vlrb-report
```

- `vlrb-freeze`: verifies source artifacts and hashes; freezes the parquet,
  benchmark prompt, response-label convention, 1,247 IDs, K=3 order schedule,
  model identities, decoding settings, and port-8000 route.
- `vlrb-smoke`: deterministic 20-pair shard across all three orders and all three
  systems. Verifies image transmission, native parsing, order remapping, and
  unchanged structured aggregation. It cannot create a report or select a
  system.
- `vlrb-run`: resumes independently at `(system, replicate, sample, node)`
  shards. It produces native outputs and node-level structured outputs, then
  combines each structured M1 without accessing other experiments' labels.
- `vlrb-report`: validates completion and hashes, computes frozen metrics, and
  writes the final comparison. It never runs inference.

## Validation Gates

- Source initial and Epoch-1 Rubric hashes match the frozen artifacts.
- Dataset count is exactly 1,247 and every row has two responses and image
  bytes.
- The K=3 schedule contains each ID three times, includes both orders, and its
  doubled order is counterbalanced by the frozen seed.
- Gold mapping from `human_ranking` agrees with the official evaluator on
  synthetic `[0,1]` and `[1,0]` examples.
- A/B swapping never changes the underlying image, question, or response text.
- Native prompt contains no structured criterion text; structured prompts
  contain no benchmark gold/ranking/rationale.
- Both structured systems use only port 8000 and their frozen P05 request
  identity; native uses only port 8000 and its declared benchmark decoding.
- K=3 aggregation is invariant to replicate ordering after logical vote mapping.
- Interrupted runs resume without reusing a prediction from a different
  description hash, system, order, or decoding identity.
- Full unittest, focused VL-RewardBench conversion tests, parser tests,
  counterbalance tests, and compile checks pass before `vlrb-run`.

## Cost and Decision Rule

The planned request count is:

\[
(5 + 17 + 1)\times1247\times3 = 86{,}043.
\]

At the observed 8000 throughput of about 4,500 Pairwise requests per 88
minutes, the expected wall-clock duration is roughly 27--33 hours, subject to
prompt-length and retry differences. The smoke is approximately 1,380 requests.

This run is worthwhile if it yields a reproducible external comparison among
the three systems. A positive main result is `role_aware_refine_epoch_01`
strict Overall ACC higher than both baselines, with no material loss of K=3
Coverage. With K=3 and a heldout-selected primary Rubric, any gain is reported
as exploratory evidence; no claim of official K=5 leaderboard comparability is
made.

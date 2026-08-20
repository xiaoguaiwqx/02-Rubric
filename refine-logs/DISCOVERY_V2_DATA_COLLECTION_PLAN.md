# Discovery-v2 Data Collection Plan

> Selection protocol update: the original root-specific `D_evolve/D_boundary`
> selection has been replaced by criterion-agnostic `Coverage70 + Hard30`.
> `D_dev150` remains isolated. The current executable design and quotas are
> documented in `docs/Discovery-v2 Data Collection.md`; the sections below
> preserve the initial proposal for provenance.

## Objective

Build a new, fully human-reviewed multimodal preference dataset with three
scientifically isolated roles:

| Split | Count | Manager-visible | Purpose |
|---|---:|---|---|
| D_evolve | 75 | yes | ErrorSignature, Split, Refine, competition |
| D_boundary | 25 | yes | applicability, abstention, conflict, boundary evidence |
| D_dev | 150 | no | fixed per-epoch diagnostics only |

D_evolve is balanced across visual factuality, multimodal reasoning, and
general preference/instruction following. D_boundary uses the frozen 10/8/4/3
boundary-type quota. D_dev contains 40 regular plus 10 boundary/conflict pairs
per domain.

## Frozen workflow

Install the collection-only dependencies with `python -m pip install -e ".[data]"`.
VisionArena additionally requires `HF_TOKEN` after accepting its data agreement;
397B adjudication requires `GUIJI_API_KEY`.

```text
HF immutable source lock
  -> normalize + image recovery + VL-RewardBench exclusion + global dedup
  -> freeze random D_dev candidates
  -> Prompt-v2 five-root original/swap screen (Discovery only)
  -> stratified difficulty ranking
  -> 397B multimodal double-order pre-adjudication
  -> blind human first pass
  -> disagreement-only reconciliation
  -> deterministic A/B balancing and strict quota finalize
  -> distribution/isolation report
```

Hardness is frozen as:

```text
3 * I[M1 wrong] + 2 * root swap inconsistency + normalized root conflict
```

The Worker is Qwen3-VL-8B-Instruct with Pairwise Prompt v2, temperature 0.5,
max_tokens 2048, and the configured vllm-8000/vllm-8001 available-slot pool.
The pre-adjudicator is Qwen/Qwen3.5-397B-A17B. No D_dev record is sent through
Worker screening or used for active selection, Manager context, early stopping,
or checkpoint selection.

## Human review contract

The first-pass queue hides source gold, 397B conclusions, and Worker outputs.
Reviewers provide A/B, confidence, evidence, rationale, error labels,
applicable roots, and boundary type. Only disagreements expose the hidden
signals during reconciliation. A disagreement cannot enter the final dataset
until `reconciled=true`.

## Acceptance gates

- exact counts 75/25/150 and exact domain/boundary quotas;
- all final records have readable images and reviewed A/B gold;
- no Discovery/Dev image, question group, source ID, or pair overlap;
- no exact VL-RewardBench image/question/pair overlap;
- no Discovery source above 35 records and RLHF-V at most 25;
- at least two source families per Discovery domain;
- deterministic JSONL and manifest hashes for the same lock/reviews/seed.

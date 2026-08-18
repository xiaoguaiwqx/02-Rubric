# Phase16 Prompt-v2 Checkpoint Heldout Diagnostic

## Objective

Measure the heldout-500 trajectory of the Prompt-v2-aligned five-root
Split+Refine run at committed epochs 0--5.  The diagnostic tests whether the
final heldout improvement appears early, persists through Refine, or reverses
after repeated local optimization.

## Frozen protocol

- Source: `phase16_prompt_v2_locked_split_refine_v2`.
- Checkpoints: committed Rubrics from epochs 0, 1, 2, 3, 4 and 5.
- Dataset and Pairwise Worker identity: the existing Prompt-v2 heldout-500
  manifest; temperature=0.5, max_tokens=2048, available-slot 8000+8001 pool.
- No Manager calls, Split, Refine, Rubric edits, new selection, or retry of
  evolution operators.
- The same heldout set was already accessed.  This is an exploratory,
  post-hoc trajectory diagnostic and may not select a replacement final model.

## Reuse and cost

Epoch 0 and the final epoch share their existing predictions.  Across epochs
1--4, only 14 unique node-description versions differ from the final Rubric.
The runner therefore generates exactly `14 * 500 = 7000` missing Pairwise
predictions and reuses all other final Prompt-v2 outputs.

## Outputs

`phase16_prompt_v2_locked_split_refine_v2/checkpoint_heldout500/` contains a
frozen manifest, per-description prediction shards, each checkpoint's combined
prediction and M1 execution, plus a final table with M1 ACC, coverage, correct
count, root-subtree metrics, and paired corrected/harmed comparisons against
epochs 1 and 5.

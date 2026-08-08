# Split-only Evolution Experiment Plan

## Goal

Run one frozen trajectory over the five initial roots. Only roots with discovery ACC < 0.70 and Coverage > 0.80 are eligible. Accepted roots are locked; rejected roots retry with complete structured and natural-language history.

## Frozen protocol

- Initial roots only; children are never recursively split.
- Synchronous epochs, seed 42, three to five epochs.
- 397B multimodal ErrorSignature → 397B clustering → 397B child generation.
- Pairwise P05, one replicate, endpoint `vllm-8001` only.
- Accept the complete child set iff local `specialized_accuracy >= parent_accuracy` on the parent's fixed decisive A/B scope.
- Discovery stages cannot read heldout-500. Heldout is frozen and accessed once after the final rubric/report.

## Run order

1. `split-evolution-freeze`: validate identities, trigger exactly four roots, write epoch 00.
2. `split-evolution-run`: synchronous attempts, history retry, commit accepted patches.
3. `split-evolution-report`: freeze final rubric and discovery tables.
4. `split-evolution-heldout`: generate only missing accepted-child predictions on 8001.
5. `split-evolution-final-report`: produce the five required tables.

## Acceptance and reporting

All discovery accepts must be locally non-degenerate. The main heldout comparison is Init five-root M1 versus Final split-evolved M1, with Wilson CI, corrected/harmed and exact McNemar. No post-heldout selection is allowed. If no natural reject→retry occurs, history support is described as test-only rather than empirically demonstrated.
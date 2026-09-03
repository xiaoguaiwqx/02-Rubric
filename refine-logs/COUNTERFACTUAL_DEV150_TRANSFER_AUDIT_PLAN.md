# Frozen Candidate Dev150 K=3 Counterfactual Transfer Audit

## Question

The Discovery100 K=3 coalition audit found several promising singleton and
coalition systems, but those systems were selected on Discovery100.  This audit
asks whether their gains transfer to the independently frozen Dev150 set.

The audit is evaluation-only.  It must not update a Rubric, feed a Manager,
choose a checkpoint, or search Dev150 for a better subset.

## Frozen systems

All systems are fixed before any Dev150 generation:

| System | Frozen candidate roots |
|---|---|
| baseline | none |
| e1_completeness | Epoch 1 Completeness |
| e1_cvf | Epoch 1 Completeness + Visual Grounding + Factuality |
| e1_positive_union | Epoch 1 Completeness + Visual Grounding + Clarity |
| e1_all | all five Epoch 1 candidates |
| e5_completeness | Epoch 5 Completeness |
| e5_clarity | Epoch 5 Clarity |
| e5_positive_union | Epoch 5 Completeness + Factuality + Clarity |
| e5_all | all five Epoch 5 candidates |

The labels describe their Discovery100 roles.  No role is recomputed on Dev150.

## Execution protocol

- Dataset: frozen `dev_150.jsonl`, 150 samples, original order.
- Model, prompt, parser, temperature, token limit and endpoint pool: identical
  to Phase21 and the Discovery100 K=3 audit.
- No A/B swap; generation seed remains unset.
- Each system uses three Global-Arbiter replicates.  Final prediction is A/B
  majority; lack of a majority is `None` and is wrong under Strict ACC.
- The Phase21 epoch-0 Dev150 subtree reports and Arbiter replicate 0 are reused
  only after exact report-bundle identity validation.
- Ten unique candidate roots are newly evaluated on Dev150 once each.  Their
  reports are then shared by every frozen system that uses that root.

## Stages

1. `counterfactual-dev-transfer-freeze`: freeze data, source hashes, candidate
   identities, nine system definitions and Discovery100 reference metrics.
2. `counterfactual-dev-transfer-audit`: offline identity and completeness audit.
3. `counterfactual-dev-transfer-smoke`: 12 source-stratified samples and three
   representative systems.
4. `counterfactual-dev-transfer-root-reports`: materialize all ten candidate
   root reports on Dev150.
5. `counterfactual-dev-transfer-run`: evaluate all nine systems at K=3.
6. `counterfactual-dev-transfer-retry`: extend the same-prompt retry budget for
   unresolved technical calls.
7. `counterfactual-dev-transfer-report`: paired transfer analysis.

## Metrics and claims

Primary metric is Dev150 Strict ACC.  For each treatment, report paired
corrected/harmed/net, 95% paired-bootstrap CI, exact two-sided McNemar p-value,
Coverage, None rate, per-source and per-domain results, and Discovery-to-Dev
gain/sign transfer.

The only supported claim is whether the already-frozen Discovery100 systems
transfer to Dev150.  The audit cannot claim that the best Dev150 system has
been found, because Dev150 is not searched and must not be used for selection.

## Request budget

- Candidate root reports: `10 * 150 = 1,500` new Unified-Subtree calls.
- Logical Arbiter calls: `9 * 150 * 3 = 4,050`.
- Reused Arbiter calls: 150 baseline replicate-0 calls.
- New Arbiter calls: 3,900.
- Expected wall time on both available-slot endpoints: about 1.5--2.5 hours.

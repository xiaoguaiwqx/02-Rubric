# MoE Routing Iteration Findings

Date: 2026-06-06

## Evidence Sources

- `moe_replay_results.json`: full heldout500 replay using cached best11 worker votes plus router weights.
- `moe_split_replay_results.json`: 250/250 dev-test split replay; dev selects routing hyperparameters, test verifies them.
- `moe_e2e_tuned20_results.json`: real end-to-end tuned MoE vs majority on 20 selected samples.

## Key Findings

1. Dynamic routing is the main source of improvement.
   - Majority best11 replay: `0.672 = 336/500`.
   - Static-only weighted replay: `0.672 = 336/500`, no gain.
   - Default router-only/full MoE replay: `0.684 = 342/500`, `+6` correct.
   - Interpretation: current static prior is too weak because checkpoint stats only provide score/accuracy and use `coverage=1.0`; routing is carrying the useful signal.

2. The original default MoE config was too conservative.
   - Default config: `routing_threshold=0.2`, `tie_epsilon=0.05`.
   - Default full MoE replay: `0.684`, 17 ties, avg active criteria `5.64`.
   - Top sweep configs use `tie_epsilon=0.0` and `routing_threshold=0.3` or `0.5`, reaching `0.702 = 351/500`.
   - Interpretation: soft-tie suppression was discarding useful small weighted margins; raising the routing threshold removes noisy medium-relevance criteria.

3. Split replay supports the strict routing candidate beyond same-set sweep.
   - Dev selected: `routing_threshold=0.5`, `tie_epsilon=0.0`, `fallback_weight=0.3`, `alpha:beta=0.7:0.3`.
   - Dev: selected MoE `0.716 = 179/250`, majority `0.676 = 169/250`.
   - Test: selected MoE `0.688 = 172/250`, majority `0.668 = 167/250`.
   - Test cost proxy: avg active criteria `5.04`, pruning `54.1%`.
   - Interpretation: selected strict routing still improves on the held-out half, so the gain is not purely an artifact of sweeping on all 500 samples.

4. Real end-to-end tuned20 is directionally positive but too small to claim robustness.
   - Majority e2e tuned20: `0.550 = 11/20`.
   - MoE e2e tuned20: `0.600 = 12/20`.
   - MoE active worker calls: `100/220`, avg active criteria `5.0`.
   - Gains: samples `rlhfv-002339`, `rlhfv-001906`; loss: `rlhfv-000912`.
   - Interpretation: real execution matches the replay direction, but n=20 is a sanity result, not a final performance claim.

## Claim Gate

Verdict: partial support.

Supported claim:

> In replay with fixed worker outputs, VLM soft routing improves accuracy over equal majority voting and reduces estimated worker calls by about half. A stricter route threshold and exact-tie-only aggregation are better than the initial soft-tie default.

Not yet fully supported:

> General end-to-end superiority of MoE routing on RLHF-V.

Missing evidence:

- Larger end-to-end subset or full heldout e2e run with tuned config.
- Actual wall-clock/API cost accounting, including router calls.
- Multiple seeds or repeated worker sampling if the VLM backend is stochastic.
- A true coverage signal for static weighting.

## Optimization Applied

Added and ran `split-replay` mode in `run_moe_routing_eval.py`.

Rationale:

- Avoid choosing hyperparameters on the same 500 samples used for reporting.
- Select the strict routing config on dev, then verify it on test using existing caches.

Selected config for next e2e runs:

```text
routing_threshold = 0.5
tie_epsilon = 0.0
fallback_weight = 0.3
static_alpha = 0.7
static_beta = 0.3
fallback_criteria = visual_grounding,factual_consistency
```

## Next Recommended Experiments

1. Run a larger tuned e2e subset.
   - Command target: `--mode e2e-subset --subset-size 100 --subset-hard-count 50`.
   - Success: MoE improves over majority by at least `+2` correct while keeping avg active criteria near `5`.

2. Add cost accounting to e2e output.
   - Track router calls, worker calls, avg active criteria, and wall-clock runtime.
   - Report actual cost proxy: `router_calls + worker_calls` and `worker_calls saved`.

3. Add coverage-aware static stats.
   - Use train-set or validation prediction logs to compute `coverage = applicable / total` for each criterion.
   - Re-run static-only and full MoE to test whether static priors begin contributing beyond router-only.

4. Investigate the e2e loss case `rlhfv-000912`.
   - Router activated only broad criteria plus specificity.
   - Check whether a pruned specialist criterion would have corrected the sample.


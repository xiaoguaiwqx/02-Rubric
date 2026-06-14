# MoE Routing Replay Report

- Created at: 2026-06-06T01:27:46.759268+08:00
- Experiment dir: `output\rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5`
- Heldout path: `data\RLHF-V\heldout_validation_500_pair.jsonl`
- Prediction cache: `output\rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5\final_ablation_predictions.json`
- Router cache: `output\rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5\moe_eval\router_cache_raw.json`
- Samples: 500
- Criteria: 11

## Main Results

| system | accuracy | correct/total | delta vs majority | ties | avg active | pruning | hard rescue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| router_only_weighted | 0.684 | 342/500 | +0.012 (+6) | 16 | 5.64 | 48.7% | 27 (19.7%) |
| full_moe | 0.684 | 342/500 | +0.012 (+6) | 17 | 5.64 | 48.7% | 27 (19.7%) |
| full_moe_no_fallback | 0.684 | 342/500 | +0.012 (+6) | 17 | 5.63 | 48.8% | 27 (19.7%) |
| majority_best11 | 0.672 | 336/500 | +0.000 (+0) | 23 | 11.00 | 0.0% | 0 (0.0%) |
| static_only_weighted | 0.672 | 336/500 | +0.000 (+0) | 23 | 11.00 | 0.0% | 0 (0.0%) |

## Top Sweep Configs

| rank | acc | correct | threshold | tie eps | fallback | alpha:beta | avg active | pruning | ties |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0.702 | 351/500 | 0.50 | 0.00 | 0.30 | 0.7:0.3 | 5.12 | 53.5% | 2 |
| 2 | 0.702 | 351/500 | 0.50 | 0.00 | 0.30 | 0.5:0.5 | 5.12 | 53.5% | 2 |
| 3 | 0.702 | 351/500 | 0.50 | 0.00 | 0.50 | 0.7:0.3 | 5.13 | 53.3% | 2 |
| 4 | 0.702 | 351/500 | 0.50 | 0.00 | 0.50 | 0.5:0.5 | 5.13 | 53.3% | 2 |
| 5 | 0.702 | 351/500 | 0.50 | 0.00 | 0.70 | 0.7:0.3 | 5.13 | 53.3% | 2 |
| 6 | 0.702 | 351/500 | 0.50 | 0.00 | 0.70 | 0.5:0.5 | 5.13 | 53.3% | 2 |
| 7 | 0.702 | 351/500 | 0.30 | 0.00 | 0.30 | 0.7:0.3 | 5.49 | 50.1% | 2 |
| 8 | 0.702 | 351/500 | 0.30 | 0.00 | 0.30 | 0.5:0.5 | 5.49 | 50.1% | 2 |
| 9 | 0.702 | 351/500 | 0.30 | 0.00 | 0.50 | 0.7:0.3 | 5.49 | 50.1% | 2 |
| 10 | 0.702 | 351/500 | 0.30 | 0.00 | 0.50 | 0.5:0.5 | 5.49 | 50.1% | 2 |
| 11 | 0.702 | 351/500 | 0.30 | 0.00 | 0.70 | 0.7:0.3 | 5.49 | 50.1% | 2 |
| 12 | 0.702 | 351/500 | 0.30 | 0.00 | 0.70 | 0.5:0.5 | 5.49 | 50.1% | 2 |
| 13 | 0.700 | 350/500 | 0.20 | 0.00 | 0.30 | 0.7:0.3 | 5.64 | 48.7% | 2 |
| 14 | 0.700 | 350/500 | 0.20 | 0.00 | 0.30 | 0.5:0.5 | 5.64 | 48.7% | 2 |
| 15 | 0.700 | 350/500 | 0.20 | 0.00 | 0.50 | 0.7:0.3 | 5.64 | 48.7% | 2 |
| 16 | 0.700 | 350/500 | 0.20 | 0.00 | 0.50 | 0.5:0.5 | 5.64 | 48.7% | 2 |
| 17 | 0.700 | 350/500 | 0.20 | 0.00 | 0.70 | 0.7:0.3 | 5.64 | 48.7% | 2 |
| 18 | 0.700 | 350/500 | 0.20 | 0.00 | 0.70 | 0.5:0.5 | 5.64 | 48.7% | 2 |
| 19 | 0.700 | 350/500 | 0.10 | 0.00 | 0.30 | 0.7:0.3 | 5.66 | 48.5% | 2 |
| 20 | 0.700 | 350/500 | 0.10 | 0.00 | 0.30 | 0.5:0.5 | 5.66 | 48.5% | 2 |

## Gain/Loss Vs Majority

### static_only_weighted

- gained: 0 (none)
- lost: 0 (none)

### router_only_weighted

- gained: 27 (rlhfv-003640, rlhfv-003332, rlhfv-003548, rlhfv-004577, rlhfv-003430, rlhfv-000309, rlhfv-000372, rlhfv-003252, rlhfv-003733, rlhfv-004786, rlhfv-003525, rlhfv-000102, rlhfv-002597, rlhfv-004669, rlhfv-003343, rlhfv-004751, rlhfv-002693, rlhfv-003794, rlhfv-000188, rlhfv-000614, rlhfv-001606, rlhfv-004825, rlhfv-002607, rlhfv-000587, rlhfv-000857, rlhfv-003101, rlhfv-003504)
- lost: 21 (rlhfv-000517, rlhfv-003493, rlhfv-001581, rlhfv-004946, rlhfv-001055, rlhfv-003732, rlhfv-003563, rlhfv-004945, rlhfv-001961, rlhfv-003825, rlhfv-001674, rlhfv-000456, rlhfv-000852, rlhfv-002152, rlhfv-004838, rlhfv-000310, rlhfv-000383, rlhfv-000532, rlhfv-002180, rlhfv-001458, rlhfv-000099)

### full_moe

- gained: 27 (rlhfv-003640, rlhfv-003332, rlhfv-003548, rlhfv-004577, rlhfv-003430, rlhfv-000309, rlhfv-000372, rlhfv-003252, rlhfv-003733, rlhfv-004786, rlhfv-003525, rlhfv-000102, rlhfv-002597, rlhfv-004669, rlhfv-003343, rlhfv-004751, rlhfv-002693, rlhfv-003794, rlhfv-000188, rlhfv-000614, rlhfv-001606, rlhfv-004825, rlhfv-002607, rlhfv-000587, rlhfv-000857, rlhfv-003101, rlhfv-003504)
- lost: 21 (rlhfv-000517, rlhfv-003493, rlhfv-001581, rlhfv-004946, rlhfv-001055, rlhfv-003732, rlhfv-003563, rlhfv-004945, rlhfv-001961, rlhfv-003825, rlhfv-001674, rlhfv-000456, rlhfv-000852, rlhfv-002152, rlhfv-004838, rlhfv-000310, rlhfv-000383, rlhfv-000532, rlhfv-002180, rlhfv-001458, rlhfv-000099)

### full_moe_no_fallback

- gained: 27 (rlhfv-003640, rlhfv-003332, rlhfv-003548, rlhfv-004577, rlhfv-003430, rlhfv-000309, rlhfv-000372, rlhfv-003252, rlhfv-003733, rlhfv-004786, rlhfv-003525, rlhfv-000102, rlhfv-002597, rlhfv-004669, rlhfv-003343, rlhfv-004751, rlhfv-002693, rlhfv-003794, rlhfv-000188, rlhfv-000614, rlhfv-001606, rlhfv-004825, rlhfv-002607, rlhfv-000587, rlhfv-000857, rlhfv-003101, rlhfv-003504)
- lost: 21 (rlhfv-000517, rlhfv-003493, rlhfv-001581, rlhfv-004946, rlhfv-001055, rlhfv-003732, rlhfv-003563, rlhfv-004945, rlhfv-001961, rlhfv-003825, rlhfv-001674, rlhfv-000456, rlhfv-000852, rlhfv-002152, rlhfv-004838, rlhfv-000310, rlhfv-000383, rlhfv-000532, rlhfv-002180, rlhfv-001458, rlhfv-000099)

## Router Summary

| criterion | mean raw | mean fallback | active rate | active count |
| --- | ---: | ---: | ---: | ---: |
| visual_grounding | 0.989 | 0.994 | 100.0% | 500 |
| factual_consistency | 0.993 | 0.996 | 100.0% | 500 |
| specificity | 0.776 | 0.776 | 93.2% | 466 |
| completeness | 0.632 | 0.632 | 84.2% | 421 |
| contextual_sensitivity | 0.345 | 0.345 | 58.0% | 290 |
| sensitivity_to_implied_meaning | 0.205 | 0.205 | 40.8% | 204 |
| visual_coherence_of_composition | 0.233 | 0.233 | 31.6% | 158 |
| adaptation_to_visual_quality | 0.129 | 0.129 | 20.8% | 104 |
| interpretive_accuracy_under_ambiguous_context | 0.099 | 0.099 | 17.6% | 88 |
| calibrated_uncertainty | 0.065 | 0.065 | 14.2% | 71 |
| temporal_consistency_in_motion | 0.026 | 0.026 | 4.0% | 20 |

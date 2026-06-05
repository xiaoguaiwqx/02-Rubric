# Final Ablation Report

- Experiment directory: `output\rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5`
- Prediction cache: `output\rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5\final_ablation_predictions.json`
- Heldout samples: 500
- Current cache best11: 0.672 = 336/500
- Exp3 final reference: 0.676 = 338/500 from `output\rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5\final_heldout_diagnostics.json`
- Note: current cache best11 differs from the original exp3 final; treat it as a separate inference pass.

## Criterion Groups

| group | n | criteria |
| --- | ---: | --- |
| best11 | 11 | visual_grounding, factual_consistency, specificity, completeness, calibrated_uncertainty, contextual_sensitivity, sensitivity_to_implied_meaning, visual_coherence_of_composition, adaptation_to_visual_quality, temporal_consistency_in_motion, interpretive_accuracy_under_ambiguous_context |
| top3_by_train | 3 | specificity, visual_coherence_of_composition, factual_consistency |
| top5_by_train | 5 | specificity, visual_coherence_of_composition, factual_consistency, calibrated_uncertainty, visual_grounding |
| top7_by_train | 7 | specificity, visual_coherence_of_composition, factual_consistency, calibrated_uncertainty, visual_grounding, completeness, temporal_consistency_in_motion |
| no_sensitivity_to_implied_meaning | 10 | visual_grounding, factual_consistency, specificity, completeness, calibrated_uncertainty, contextual_sensitivity, visual_coherence_of_composition, adaptation_to_visual_quality, temporal_consistency_in_motion, interpretive_accuracy_under_ambiguous_context |
| broad_only | 7 | visual_grounding, factual_consistency, specificity, completeness, contextual_sensitivity, adaptation_to_visual_quality, interpretive_accuracy_under_ambiguous_context |

## Results

| group | n | accuracy | correct/total | delta vs best11 | tie/None final | avg None votes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top7_by_train | 7 | 0.676 | 338/500 | +0.004 (+2) | 32 | 1.78 |
| broad_only | 7 | 0.674 | 337/500 | +0.002 (+1) | 17 | 0.49 |
| best11 | 11 | 0.672 | 336/500 | +0.000 (+0) | 23 | 2.43 |
| no_sensitivity_to_implied_meaning | 10 | 0.668 | 334/500 | -0.004 (-2) | 18 | 1.95 |
| top5_by_train | 5 | 0.658 | 329/500 | -0.014 (-7) | 43 | 1.15 |
| top3_by_train | 3 | 0.622 | 311/500 | -0.050 (-25) | 71 | 0.77 |

## Gain/Loss Vs Best11

### top3_by_train: gained 19, lost 44

- gained sample ids: rlhfv-000608, rlhfv-003332, rlhfv-004577, rlhfv-000019, rlhfv-001715, rlhfv-000990, rlhfv-000309, rlhfv-003125, rlhfv-003733, rlhfv-004786, rlhfv-001981, rlhfv-004669, rlhfv-003343, rlhfv-002693, rlhfv-003794, rlhfv-004825, rlhfv-000725, rlhfv-000669, rlhfv-003504
- lost sample ids: rlhfv-004591, rlhfv-001385, rlhfv-000914, rlhfv-001581, rlhfv-002855, rlhfv-001755, rlhfv-000930, rlhfv-003438, rlhfv-000489, rlhfv-002213, rlhfv-001055, rlhfv-003220, rlhfv-003732, rlhfv-000373, rlhfv-003396, rlhfv-003563, rlhfv-001961, rlhfv-001776, rlhfv-003769, rlhfv-003825, rlhfv-002268, rlhfv-002817, rlhfv-004744, rlhfv-003934, rlhfv-002152, rlhfv-004838, rlhfv-000064, rlhfv-004625, rlhfv-005019, rlhfv-002366, rlhfv-000623, rlhfv-000283, rlhfv-001872, rlhfv-001257, rlhfv-000258, rlhfv-000461, rlhfv-000892, rlhfv-002180, rlhfv-000132, rlhfv-001458, rlhfv-003557, rlhfv-000099, rlhfv-003151, rlhfv-000543

### top5_by_train: gained 16, lost 23

- gained sample ids: rlhfv-000608, rlhfv-003629, rlhfv-003332, rlhfv-004577, rlhfv-000990, rlhfv-000309, rlhfv-004786, rlhfv-002597, rlhfv-004669, rlhfv-002693, rlhfv-003794, rlhfv-004825, rlhfv-001032, rlhfv-001406, rlhfv-000587, rlhfv-003504
- lost sample ids: rlhfv-004591, rlhfv-000517, rlhfv-001581, rlhfv-001468, rlhfv-002213, rlhfv-003732, rlhfv-000373, rlhfv-003563, rlhfv-001961, rlhfv-001442, rlhfv-003825, rlhfv-003318, rlhfv-000852, rlhfv-002152, rlhfv-004838, rlhfv-000283, rlhfv-004889, rlhfv-004935, rlhfv-000461, rlhfv-002180, rlhfv-001458, rlhfv-003557, rlhfv-000099

### top7_by_train: gained 23, lost 21

- gained sample ids: rlhfv-000147, rlhfv-003332, rlhfv-003548, rlhfv-004577, rlhfv-001715, rlhfv-003141, rlhfv-000309, rlhfv-000372, rlhfv-002218, rlhfv-003252, rlhfv-004786, rlhfv-002597, rlhfv-004669, rlhfv-002693, rlhfv-003794, rlhfv-001664, rlhfv-001606, rlhfv-001032, rlhfv-001406, rlhfv-000587, rlhfv-000725, rlhfv-000669, rlhfv-003504
- lost sample ids: rlhfv-004591, rlhfv-000517, rlhfv-003732, rlhfv-003563, rlhfv-004945, rlhfv-001961, rlhfv-001442, rlhfv-003825, rlhfv-001674, rlhfv-000206, rlhfv-000852, rlhfv-002152, rlhfv-004838, rlhfv-004625, rlhfv-000623, rlhfv-000258, rlhfv-000461, rlhfv-000532, rlhfv-002262, rlhfv-001458, rlhfv-000099

### no_sensitivity_to_implied_meaning: gained 3, lost 5

- gained sample ids: rlhfv-003629, rlhfv-000372, rlhfv-000912
- lost sample ids: rlhfv-002213, rlhfv-003563, rlhfv-003318, rlhfv-001872, rlhfv-003867

### broad_only: gained 9, lost 8

- gained sample ids: rlhfv-003629, rlhfv-004577, rlhfv-000372, rlhfv-004786, rlhfv-003525, rlhfv-002693, rlhfv-001664, rlhfv-002607, rlhfv-000857
- lost sample ids: rlhfv-003493, rlhfv-003220, rlhfv-003563, rlhfv-003318, rlhfv-003839, rlhfv-001872, rlhfv-001906, rlhfv-000143

## Vote Distributions

### best11

| pattern | count |
| --- | ---: |
| `A=9|B=0|None=2|final=A|correct=True` | 24 |
| `A=0|B=10|None=1|final=B|correct=True` | 23 |
| `A=1|B=9|None=1|final=B|correct=True` | 23 |
| `A=0|B=8|None=3|final=B|correct=True` | 15 |
| `A=1|B=8|None=2|final=B|correct=True` | 13 |
| `A=8|B=1|None=2|final=A|correct=True` | 13 |
| `A=0|B=7|None=4|final=B|correct=True` | 9 |
| `A=0|B=11|None=0|final=B|correct=True` | 9 |
| `A=5|B=5|None=1|final=None|correct=False` | 9 |
| `A=8|B=0|None=3|final=A|correct=True` | 9 |
| `A=2|B=8|None=1|final=B|correct=True` | 9 |
| `A=4|B=5|None=2|final=B|correct=False` | 9 |
| `A=1|B=10|None=0|final=B|correct=True` | 9 |
| `A=3|B=5|None=3|final=B|correct=False` | 8 |
| `A=9|B=1|None=1|final=A|correct=True` | 8 |
| `A=4|B=4|None=3|final=None|correct=False` | 8 |
| `A=0|B=9|None=2|final=B|correct=True` | 8 |
| `A=5|B=3|None=3|final=A|correct=True` | 8 |
| `A=3|B=6|None=2|final=B|correct=False` | 7 |
| `A=1|B=8|None=2|final=B|correct=False` | 7 |
| `A=4|B=5|None=2|final=B|correct=True` | 7 |
| `A=0|B=10|None=1|final=B|correct=False` | 7 |
| `A=7|B=1|None=3|final=A|correct=True` | 7 |
| `A=1|B=7|None=3|final=B|correct=True` | 7 |
| `A=3|B=7|None=1|final=B|correct=True` | 7 |
| `A=2|B=7|None=2|final=B|correct=True` | 7 |
| `A=11|B=0|None=0|final=A|correct=True` | 6 |
| `A=8|B=2|None=1|final=A|correct=True` | 6 |
| `A=5|B=1|None=5|final=A|correct=True` | 6 |
| `A=10|B=0|None=1|final=A|correct=True` | 6 |
| `A=8|B=2|None=1|final=A|correct=False` | 6 |
| `A=3|B=6|None=2|final=B|correct=True` | 6 |
| `A=5|B=4|None=2|final=A|correct=True` | 6 |
| `A=7|B=2|None=2|final=A|correct=True` | 5 |
| `A=6|B=2|None=3|final=A|correct=True` | 5 |
| `A=7|B=0|None=4|final=A|correct=True` | 5 |
| `A=6|B=3|None=2|final=A|correct=False` | 5 |
| `A=5|B=2|None=4|final=A|correct=False` | 4 |
| `A=8|B=1|None=2|final=A|correct=False` | 4 |
| `A=6|B=1|None=4|final=A|correct=True` | 4 |
| ... 80 more patterns in JSON | |

### top3_by_train

| pattern | count |
| --- | ---: |
| `A=0|B=2|None=1|final=B|correct=True` | 82 |
| `A=0|B=3|None=0|final=B|correct=True` | 65 |
| `A=2|B=0|None=1|final=A|correct=True` | 62 |
| `A=1|B=1|None=1|final=None|correct=False` | 58 |
| `A=3|B=0|None=0|final=A|correct=True` | 36 |
| `A=0|B=2|None=1|final=B|correct=False` | 33 |
| `A=0|B=3|None=0|final=B|correct=False` | 22 |
| `A=1|B=2|None=0|final=B|correct=True` | 18 |
| `A=0|B=1|None=2|final=B|correct=True` | 17 |
| `A=2|B=1|None=0|final=A|correct=True` | 16 |
| `A=1|B=0|None=2|final=A|correct=True` | 15 |
| `A=2|B=0|None=1|final=A|correct=False` | 13 |
| `A=0|B=0|None=3|final=None|correct=False` | 13 |
| `A=1|B=2|None=0|final=B|correct=False` | 12 |
| `A=3|B=0|None=0|final=A|correct=False` | 11 |
| `A=2|B=1|None=0|final=A|correct=False` | 10 |
| `A=1|B=0|None=2|final=A|correct=False` | 10 |
| `A=0|B=1|None=2|final=B|correct=False` | 7 |

### top5_by_train

| pattern | count |
| --- | ---: |
| `A=0|B=4|None=1|final=B|correct=True` | 56 |
| `A=4|B=0|None=1|final=A|correct=True` | 50 |
| `A=0|B=5|None=0|final=B|correct=True` | 45 |
| `A=2|B=2|None=1|final=None|correct=False` | 26 |
| `A=0|B=3|None=2|final=B|correct=True` | 24 |
| `A=5|B=0|None=0|final=A|correct=True` | 21 |
| `A=3|B=1|None=1|final=A|correct=True` | 21 |
| `A=1|B=3|None=1|final=B|correct=True` | 19 |
| `A=3|B=0|None=2|final=A|correct=True` | 18 |
| `A=0|B=4|None=1|final=B|correct=False` | 14 |
| `A=2|B=1|None=2|final=A|correct=True` | 13 |
| `A=1|B=1|None=3|final=None|correct=False` | 13 |
| `A=1|B=3|None=1|final=B|correct=False` | 12 |
| `A=0|B=5|None=0|final=B|correct=False` | 11 |
| `A=1|B=4|None=0|final=B|correct=True` | 10 |
| `A=4|B=0|None=1|final=A|correct=False` | 9 |
| `A=3|B=1|None=1|final=A|correct=False` | 9 |
| `A=3|B=2|None=0|final=A|correct=True` | 9 |
| `A=0|B=3|None=2|final=B|correct=False` | 9 |
| `A=2|B=3|None=0|final=B|correct=False` | 9 |
| `A=2|B=3|None=0|final=B|correct=True` | 8 |
| `A=5|B=0|None=0|final=A|correct=False` | 8 |
| `A=0|B=2|None=3|final=B|correct=True` | 8 |
| `A=1|B=2|None=2|final=B|correct=False` | 8 |
| `A=2|B=1|None=2|final=A|correct=False` | 7 |
| `A=2|B=0|None=3|final=A|correct=True` | 6 |
| `A=4|B=1|None=0|final=A|correct=False` | 6 |
| `A=4|B=1|None=0|final=A|correct=True` | 6 |
| `A=3|B=2|None=0|final=A|correct=False` | 6 |
| `A=1|B=2|None=2|final=B|correct=True` | 6 |
| `A=1|B=0|None=4|final=A|correct=True` | 5 |
| `A=0|B=0|None=5|final=None|correct=False` | 4 |
| `A=0|B=1|None=4|final=B|correct=False` | 4 |
| `A=1|B=4|None=0|final=B|correct=False` | 4 |
| `A=0|B=1|None=4|final=B|correct=True` | 4 |
| `A=1|B=0|None=4|final=A|correct=False` | 4 |
| `A=2|B=0|None=3|final=A|correct=False` | 3 |
| `A=0|B=2|None=3|final=B|correct=False` | 3 |
| `A=3|B=0|None=2|final=A|correct=False` | 2 |

### top7_by_train

| pattern | count |
| --- | ---: |
| `A=0|B=6|None=1|final=B|correct=True` | 42 |
| `A=0|B=7|None=0|final=B|correct=True` | 30 |
| `A=0|B=5|None=2|final=B|correct=True` | 25 |
| `A=5|B=0|None=2|final=A|correct=True` | 25 |
| `A=6|B=0|None=1|final=A|correct=True` | 25 |
| `A=0|B=4|None=3|final=B|correct=True` | 18 |
| `A=1|B=5|None=1|final=B|correct=True` | 17 |
| `A=4|B=1|None=2|final=A|correct=True` | 16 |
| `A=2|B=2|None=3|final=None|correct=False` | 13 |
| `A=3|B=3|None=1|final=None|correct=False` | 13 |
| `A=1|B=4|None=2|final=B|correct=True` | 13 |
| `A=7|B=0|None=0|final=A|correct=True` | 12 |
| `A=2|B=3|None=2|final=B|correct=False` | 11 |
| `A=3|B=1|None=3|final=A|correct=True` | 11 |
| `A=1|B=4|None=2|final=B|correct=False` | 10 |
| `A=5|B=1|None=1|final=A|correct=True` | 10 |
| `A=0|B=6|None=1|final=B|correct=False` | 9 |
| `A=4|B=0|None=3|final=A|correct=True` | 9 |
| `A=3|B=2|None=2|final=A|correct=True` | 9 |
| `A=2|B=3|None=2|final=B|correct=True` | 9 |
| `A=0|B=5|None=2|final=B|correct=False` | 8 |
| `A=4|B=2|None=1|final=A|correct=True` | 8 |
| `A=2|B=4|None=1|final=B|correct=True` | 7 |
| `A=0|B=4|None=3|final=B|correct=False` | 7 |
| `A=5|B=1|None=1|final=A|correct=False` | 7 |
| `A=4|B=1|None=2|final=A|correct=False` | 6 |
| `A=4|B=2|None=1|final=A|correct=False` | 6 |
| `A=0|B=7|None=0|final=B|correct=False` | 6 |
| `A=6|B=1|None=0|final=A|correct=True` | 6 |
| `A=1|B=6|None=0|final=B|correct=True` | 6 |
| `A=1|B=5|None=1|final=B|correct=False` | 6 |
| `A=0|B=3|None=4|final=B|correct=True` | 6 |
| `A=2|B=1|None=4|final=A|correct=False` | 5 |
| `A=3|B=2|None=2|final=A|correct=False` | 5 |
| `A=2|B=1|None=4|final=A|correct=True` | 5 |
| `A=1|B=1|None=5|final=None|correct=False` | 4 |
| `A=7|B=0|None=0|final=A|correct=False` | 4 |
| `A=3|B=1|None=3|final=A|correct=False` | 4 |
| `A=2|B=4|None=1|final=B|correct=False` | 4 |
| `A=1|B=2|None=4|final=B|correct=True` | 4 |
| ... 27 more patterns in JSON | |

### no_sensitivity_to_implied_meaning

| pattern | count |
| --- | ---: |
| `A=0|B=9|None=1|final=B|correct=True` | 33 |
| `A=0|B=10|None=0|final=B|correct=True` | 23 |
| `A=1|B=8|None=1|final=B|correct=True` | 20 |
| `A=9|B=0|None=1|final=A|correct=True` | 20 |
| `A=8|B=0|None=2|final=A|correct=True` | 18 |
| `A=0|B=8|None=2|final=B|correct=True` | 17 |
| `A=0|B=7|None=3|final=B|correct=True` | 13 |
| `A=7|B=1|None=2|final=A|correct=True` | 13 |
| `A=10|B=0|None=0|final=A|correct=True` | 12 |
| `A=1|B=7|None=2|final=B|correct=True` | 11 |
| `A=2|B=7|None=1|final=B|correct=True` | 11 |
| `A=3|B=5|None=2|final=B|correct=False` | 10 |
| `A=1|B=9|None=0|final=B|correct=True` | 10 |
| `A=8|B=1|None=1|final=A|correct=True` | 9 |
| `A=6|B=2|None=2|final=A|correct=True` | 9 |
| `A=4|B=4|None=2|final=None|correct=False` | 9 |
| `A=4|B=5|None=1|final=B|correct=False` | 8 |
| `A=3|B=5|None=2|final=B|correct=True` | 8 |
| `A=0|B=9|None=1|final=B|correct=False` | 7 |
| `A=7|B=0|None=3|final=A|correct=True` | 7 |
| `A=6|B=1|None=3|final=A|correct=True` | 7 |
| `A=2|B=6|None=2|final=B|correct=False` | 6 |
| `A=0|B=8|None=2|final=B|correct=False` | 6 |
| `A=5|B=4|None=1|final=A|correct=True` | 6 |
| `A=0|B=10|None=0|final=B|correct=False` | 6 |
| `A=2|B=5|None=3|final=B|correct=False` | 6 |
| `A=4|B=3|None=3|final=A|correct=True` | 6 |
| `A=9|B=1|None=0|final=A|correct=True` | 5 |
| `A=4|B=1|None=5|final=A|correct=True` | 5 |
| `A=7|B=2|None=1|final=A|correct=True` | 5 |
| `A=5|B=3|None=2|final=A|correct=False` | 5 |
| `A=4|B=2|None=4|final=A|correct=True` | 5 |
| `A=1|B=8|None=1|final=B|correct=False` | 5 |
| `A=3|B=4|None=3|final=B|correct=False` | 5 |
| `A=7|B=2|None=1|final=A|correct=False` | 5 |
| `A=4|B=2|None=4|final=A|correct=False` | 4 |
| `A=5|B=2|None=3|final=A|correct=False` | 4 |
| `A=4|B=5|None=1|final=B|correct=True` | 4 |
| `A=6|B=3|None=1|final=A|correct=False` | 4 |
| `A=3|B=6|None=1|final=B|correct=False` | 4 |
| ... 60 more patterns in JSON | |

### broad_only

| pattern | count |
| --- | ---: |
| `A=0|B=7|None=0|final=B|correct=True` | 86 |
| `A=7|B=0|None=0|final=A|correct=True` | 52 |
| `A=1|B=6|None=0|final=B|correct=True` | 36 |
| `A=0|B=7|None=0|final=B|correct=False` | 22 |
| `A=6|B=1|None=0|final=A|correct=True` | 22 |
| `A=2|B=5|None=0|final=B|correct=True` | 20 |
| `A=2|B=5|None=0|final=B|correct=False` | 18 |
| `A=3|B=4|None=0|final=B|correct=False` | 17 |
| `A=4|B=3|None=0|final=A|correct=True` | 15 |
| `A=5|B=2|None=0|final=A|correct=False` | 13 |
| `A=5|B=2|None=0|final=A|correct=True` | 13 |
| `A=3|B=4|None=0|final=B|correct=True` | 13 |
| `A=3|B=3|None=1|final=None|correct=False` | 12 |
| `A=4|B=3|None=0|final=A|correct=False` | 11 |
| `A=0|B=6|None=1|final=B|correct=True` | 10 |
| `A=6|B=1|None=0|final=A|correct=False` | 10 |
| `A=5|B=1|None=1|final=A|correct=True` | 9 |
| `A=4|B=2|None=1|final=A|correct=True` | 9 |
| `A=6|B=0|None=1|final=A|correct=True` | 9 |
| `A=7|B=0|None=0|final=A|correct=False` | 9 |
| `A=1|B=6|None=0|final=B|correct=False` | 8 |
| `A=4|B=2|None=1|final=A|correct=False` | 7 |
| `A=5|B=0|None=2|final=A|correct=True` | 6 |
| `A=1|B=5|None=1|final=B|correct=True` | 6 |
| `A=2|B=4|None=1|final=B|correct=False` | 5 |
| `A=4|B=0|None=3|final=A|correct=True` | 4 |
| `A=4|B=1|None=2|final=A|correct=True` | 4 |
| `A=3|B=2|None=2|final=A|correct=False` | 4 |
| `A=2|B=4|None=1|final=B|correct=True` | 4 |
| `A=3|B=0|None=4|final=A|correct=True` | 3 |
| `A=5|B=1|None=1|final=A|correct=False` | 3 |
| `A=2|B=3|None=2|final=B|correct=False` | 3 |
| `A=0|B=5|None=2|final=B|correct=False` | 3 |
| `A=3|B=2|None=2|final=A|correct=True` | 3 |
| `A=2|B=2|None=3|final=None|correct=False` | 3 |
| `A=2|B=3|None=2|final=B|correct=True` | 3 |
| `A=3|B=0|None=4|final=A|correct=False` | 3 |
| `A=1|B=1|None=5|final=None|correct=False` | 2 |
| `A=3|B=1|None=3|final=A|correct=True` | 2 |
| `A=1|B=5|None=1|final=B|correct=False` | 2 |
| ... 12 more patterns in JSON | |

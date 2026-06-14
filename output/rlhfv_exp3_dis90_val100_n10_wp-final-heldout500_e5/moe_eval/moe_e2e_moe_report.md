# MoE Routing E2E Report

- Created at: 2026-06-06T11:36:16.636192+08:00
- Experiment dir: `output\rlhfv_exp3_dis90_val100_n10_wp-final-heldout500_e5`
- Heldout path: `data\RLHF-V\heldout_validation_500_pair.jsonl`
- Samples: 500
- Criteria: 11
- Config: threshold=0.5, tie_epsilon=0.0, fallback_weight=0.3, alpha:beta=0.7:0.3
- Cached majority accuracy: 0.672 = 336/500
- Real MoE accuracy: 0.684 = 342/500
- Delta vs cached majority: +0.012 (+6)
- Router calls: 500
- Active worker calls: 2578
- Full worker-call baseline: 5500
- Worker pruning: 53.1%
- Avg active criteria: 5.16
- Wall time seconds: 3656.1

## Sample Indices

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
... +400 more

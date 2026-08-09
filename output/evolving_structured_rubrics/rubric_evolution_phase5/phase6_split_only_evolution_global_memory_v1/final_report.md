# Split-only Evolution Final Report

Natural reject→retry observed; inspect retry table.

## 1. Epoch × root trajectory

| epoch | root | attempt | decision | parent ACC | specialized ACC | delta |
|---|---|---|---|---|---|---|
| 0 | init_01_completeness_and_coverage | None | eligible | None | None | None |
| 0 | init_02_visual_grounding_and_details | None | eligible | None | None | None |
| 0 | init_03_factuality_no_hallucination | None | eligible | None | None | None |
| 0 | init_04_creativity_and_expressiveness | None | eligible | None | None | None |
| 0 | init_05_clarity_and_coherence | None | eligible | None | None | None |
| 1 | init_01_completeness_and_coverage | 1 | competition_rejected | 0.6707317073170732 | 0.6219512195121951 | -0.04878048780487809 |
| 1 | init_02_visual_grounding_and_details | 1 | competition_rejected | 0.6966292134831461 | 0.651685393258427 | -0.0449438202247191 |
| 1 | init_03_factuality_no_hallucination | 1 | accepted | 0.6024096385542169 | 0.6987951807228916 | 0.09638554216867468 |
| 1 | init_04_creativity_and_expressiveness | 1 | competition_rejected | 0.627906976744186 | 0.6162790697674418 | -0.011627906976744207 |
| 1 | init_05_clarity_and_coherence | 1 | competition_rejected | 0.5632183908045977 | 0.5402298850574713 | -0.02298850574712641 |
| 2 | init_01_completeness_and_coverage | 2 | competition_rejected | 0.6707317073170732 | 0.6097560975609756 | -0.060975609756097615 |
| 2 | init_02_visual_grounding_and_details | 2 | competition_rejected | 0.6966292134831461 | 0.6292134831460674 | -0.0674157303370787 |
| 2 | init_03_factuality_no_hallucination | None | accepted_locked | None | None | None |
| 2 | init_04_creativity_and_expressiveness | 2 | competition_rejected | 0.627906976744186 | 0.6162790697674418 | -0.011627906976744207 |
| 2 | init_05_clarity_and_coherence | 2 | proposal_invalid | None | None | None |
| 3 | init_01_completeness_and_coverage | 3 | competition_rejected | 0.6707317073170732 | 0.6219512195121951 | -0.04878048780487809 |
| 3 | init_02_visual_grounding_and_details | 3 | competition_rejected | 0.6966292134831461 | 0.6629213483146067 | -0.03370786516853941 |
| 3 | init_03_factuality_no_hallucination | None | accepted_locked | None | None | None |
| 3 | init_04_creativity_and_expressiveness | 3 | accepted | 0.627906976744186 | 0.6395348837209303 | 0.011627906976744207 |
| 3 | init_05_clarity_and_coherence | 3 | accepted | 0.5632183908045977 | 0.5977011494252874 | 0.034482758620689724 |
| 4 | init_01_completeness_and_coverage | 4 | competition_rejected | 0.6707317073170732 | 0.6463414634146342 | -0.024390243902439046 |
| 4 | init_02_visual_grounding_and_details | 4 | competition_rejected | 0.6966292134831461 | 0.6179775280898876 | -0.07865168539325851 |
| 4 | init_03_factuality_no_hallucination | None | accepted_locked | None | None | None |
| 4 | init_04_creativity_and_expressiveness | None | accepted_locked | None | None | None |
| 4 | init_05_clarity_and_coherence | None | accepted_locked | None | None | None |
| 5 | init_01_completeness_and_coverage | 5 | accepted | 0.6707317073170732 | 0.6951219512195121 | 0.024390243902438935 |
| 5 | init_02_visual_grounding_and_details | 5 | competition_rejected | 0.6966292134831461 | 0.6292134831460674 | -0.0674157303370787 |
| 5 | init_03_factuality_no_hallucination | None | accepted_locked | None | None | None |
| 5 | init_04_creativity_and_expressiveness | None | accepted_locked | None | None | None |
| 5 | init_05_clarity_and_coherence | None | accepted_locked | None | None | None |

## 2. Final root local results

| root | criterion | status | attempts | children | parent ACC | specialized ACC |
|---|---|---|---|---|---|---|
| init_01_completeness_and_coverage | completeness_and_coverage | accepted_locked | 5 | verified_existence_over_hallucinated_volume, verified_fine_grained_coverage | 0.6707317073170732 | 0.6951219512195121 |
| init_02_visual_grounding_and_details | visual_grounding_and_details | exhausted | 5 |  | 0.6966292134831461 | 0.6292134831460674 |
| init_03_factuality_no_hallucination | factuality_no_hallucination | accepted_locked | 1 | visual_counting_and_presence_verification, visual_attribute_verification, spatial_relationship_and_perspective_verification, logical_consistency_and_prompt_adherence, grounded_specificity_verification | 0.6024096385542169 | 0.6987951807228916 |
| init_04_creativity_and_expressiveness | creativity_and_expressiveness | accepted_locked | 3 | verified_presence_and_quantity_in_articulation, spatial_accuracy_over_expressive_confidence, task_intent_alignment_in_expressiveness | 0.627906976744186 | 0.6395348837209303 |
| init_05_clarity_and_coherence | clarity_and_coherence | accepted_locked | 3 | spatial_and_quantitative_clarity, semantic_grounding_clarity | 0.5632183908045977 | 0.5977011494252874 |

## 3. Init vs Final M1

| split | system | ACC | Coverage | correct |
|---|---|---|---|---|
| discovery90 | Init five-root M1 | 0.6555555555555556 | 0.9666666666666667 | - |
| discovery90 | Final split-evolved M1 | 0.6333333333333333 | 1.0 | - |
| heldout500 | Init five-root M1 | 0.65 | 0.972 | 325 |
| heldout500 | Final split-evolved M1 | 0.7 | 0.986 | 350 |

## 4. Reject → retry

[
  {
    "root_id": "init_01_completeness_and_coverage",
    "attempts": 5,
    "decisions": [
      "competition_rejected",
      "competition_rejected",
      "competition_rejected",
      "competition_rejected",
      "accepted"
    ],
    "failure_reasons": [
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      null
    ],
    "metric_deltas": [
      -0.04878048780487809,
      -0.060975609756097615,
      -0.04878048780487809,
      -0.024390243902439046,
      0.024390243902438935
    ]
  },
  {
    "root_id": "init_02_visual_grounding_and_details",
    "attempts": 5,
    "decisions": [
      "competition_rejected",
      "competition_rejected",
      "competition_rejected",
      "competition_rejected",
      "competition_rejected"
    ],
    "failure_reasons": [
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      }
    ],
    "metric_deltas": [
      -0.0449438202247191,
      -0.0674157303370787,
      -0.03370786516853941,
      -0.07865168539325851,
      -0.0674157303370787
    ]
  },
  {
    "root_id": "init_04_creativity_and_expressiveness",
    "attempts": 3,
    "decisions": [
      "competition_rejected",
      "competition_rejected",
      "accepted"
    ],
    "failure_reasons": [
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      null
    ],
    "metric_deltas": [
      -0.011627906976744207,
      -0.011627906976744207,
      0.011627906976744207
    ]
  },
  {
    "root_id": "init_05_clarity_and_coherence",
    "attempts": 3,
    "decisions": [
      "competition_rejected",
      "proposal_invalid",
      "accepted"
    ],
    "failure_reasons": [
      {
        "code": "specialized_accuracy_below_parent",
        "stage": "competition",
        "details": {}
      },
      {
        "code": "proposal_invalid",
        "stage": "semantic_cluster",
        "details": {
          "stage": "semantic_cluster",
          "raw_response": "\n\n{\n  \"clusters\": [\n    {\n      \"cluster_id\": \"clarity_fluency_over_grounding\",\n      \"label\": \"Fluency and Confidence Bias Over Visual Grounding\",\n      \"shared_failure\": \"Evaluators prioritize linguistic fluency, narrative smoothness, or confident tone over visual faithfulness, incorrectly rating hallucinated but fluent responses as 'clearer' than accurate but less polished ones.\",\n      \"distinction\": \"Focuses on cases where the text quality (flow, confidence) masks visual errors, leading evaluators to mistake a 'smooth lie' for clear communication.\",\n      \"sample_ids\": [\n        \"rlhfv-002726\",\n        \"rlhfv-000510\",\n        \"rlhfv-002699\",\n        \"rlhfv-000277\",\n        \"rlhfv-002824\",\n        \"rlhfv-004635\",\n        \"rlhfv-002806\",\n        \"rlhfv-004995\",\n        \"rlhfv-001052\",\n        \"rlhfv-002080\",\n        \"rlhfv-001103\",\n        \"rlhfv-001528\",\n        \"rlhfv-000287\",\n        \"rlhfv-000449\",\n        \"rlhfv-001002\",\n        \"rlhfv-000628\",\n        \"rlhfv-000307\",\n        \"rlhfv-003560\",\n        \"rlhfv-002802\",\n        \"rlhfv-000042\",\n        \"rlhfv-002069\",\n        \"rlhfv-004571\",\n        \"rlhfv-000932\",\n        \"rlhfv-002738\",\n        \"rlhfv-002695\",\n        \"rlhfv-003046\",\n        \"rlhfv-000082\",\n        \"rlhfv-002182\",\n        \"rlhfv-000195\",\n        \"rlhfv-001347\",\n        \"rlhfv-000740\",\n        \"rlhfv-002821\",\n        \"rlhfv-004830\",\n        \"rlhfv-000541\",\n        \"rlhfv-003218\",\n        \"rlhfv-001449\",\n        \"rlhfv-000196\"\n      ]\n    }\n  ],\n  \"unclustered_sample_ids\": []\n}",
          "parse_error": "missing sample IDs: ['rlhfv-000478']",
          "attempt_count": 2,
          "metrics": {
            "logical_evaluations": 1,
            "api_attempts": 2,
            "parse_retries": 1,
            "input_tokens": 22722,
            "output_tokens": 18219,
            "total_tokens": 40941,
            "usage_complete": true,
            "call_latency_seconds": 158.35796290000144,
            "estimated_cost_usd": null,
            "cache_hits": 0,
            "cache_misses": 0,
            "error_count": 0
          }
        }
      },
      null
    ],
    "metric_deltas": [
      -0.02298850574712641,
      null,
      0.034482758620689724
    ]
  }
]

## 5. Cost and rubric growth

{
  "epochs": 5,
  "attempts": 17,
  "reused_v1_signature_artifacts": 0,
  "manager_api_attempts": 85,
  "manager_failed_api_attempts": 2,
  "manager_usage_incomplete_artifacts": 0,
  "manager_input_tokens": 871672,
  "manager_output_tokens": 442971,
  "manager_summed_latency_seconds": 4230.083295199996,
  "evolution_epoch_wall_seconds": 15022.280999999999,
  "pairwise_provenance_files": 16,
  "pairwise_api_attempts": 4926,
  "pairwise_usage_incomplete_calls": 0,
  "pairwise_input_tokens": 0,
  "pairwise_output_tokens": 0,
  "pairwise_summed_latency_seconds": 108933.66113300006,
  "reused_control_v2_signature_artifacts": 157,
  "initial_nodes": 5,
  "final_nodes": 17
}

## 6. Manager memory ablation

The same heldout-500 was used by prior experiments; this is an exploratory paired ablation, not a new unbiased confirmatory test.

| system | heldout ACC | Coverage | correct |
|---|---|---|---|
| Control v2 | 0.694 | 0.994 | 347 |
| Global-Rubric Memory | 0.7 | 0.986 | 350 |

Claim status: supports_memory_acc_gain; ACC delta=0.006000
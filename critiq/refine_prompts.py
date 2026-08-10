"""Versioned prompts for the Refine v1 operator."""

REFINE_GENERATION_PROMPT = """## Task
Refine exactly one existing pairwise judging criterion using its observed Worker
errors. Preserve the criterion's identity and intended semantic dimension. Make
its applicability boundary and decision rule more precise; do not create a new
criterion and do not solve unrelated Rubric dimensions.

## Current criterion
Node ID: {node_id}
Criterion name: {criterion_name}
Description:
{criterion_description}

## Current node evidence
{evidence_json}

## Previous failed Refine attempts for this node
{refine_failure_history_json}
Use the history to avoid repeating rejected descriptions. Current evidence is
authoritative.

## Current committed Rubric (global memory)
{rubric_memory_json}
Use this memory only to state a clean boundary with existing criteria and avoid
semantic duplication. Do not change any node, name, edge, score, or topology.

Return exactly one JSON object:
{{
  "criterion_name": "{criterion_name}",
  "description": "Criterion focus: ...\\n\\nApplicable only when: ...\\n\\nNot applicable when: ...\\n\\nDecision rule: ...",
  "failure_analysis": ["specific evidence-grounded cause of current errors"],
  "rationale": "why the revised wording should improve this same criterion",
  "representative_sample_ids": ["one to six IDs from the supplied evidence"]
}}

The description must contain exactly the four named sections in that order and
must be no longer than {max_description_chars} characters. The Decision rule
must explicitly state all of the following:
- Return None when the criterion is not applicable, visual evidence is
  insufficient, or neither candidate has a reliable advantage.
- Do not vote based on overall answer quality, fluency, completeness, or other
  criteria.
- Output A/B only from evidence relevant to this criterion.
Do not output Markdown or additional fields."""


REFINE_FAILURE_ATTRIBUTION_PROMPT = """## Task
Diagnose why a rejected Refine candidate failed to strictly improve the target
criterion's selective accuracy or violated its minimum support constraint.
Attribute the Refine operation, not overall answer quality.

## Original criterion
{original_criterion_json}

## Rejected proposal
{proposal_json}

## Refine evaluation
{evaluation_json}

## Changed predictions
{changed_predictions_json}

## Current committed Rubric (global memory)
{rubric_memory_json}

Return exactly one JSON object:
{{
  "summary": "concise natural-language explanation of why Refine failed",
  "failure_categories": ["lower_snake_case_category"],
  "details": ["specific evidence-backed detail"],
  "avoid_next_time": ["concrete instruction for the next Refine attempt"]
}}
Use categories such as applicability_still_too_broad, excessive_abstention,
visual_fact_incorrect, preference_direction_reversed, representative_overfit,
sibling_overlap, behavior_unchanged, insufficient_target_correction, or
non_target_harm when supported. Do not output Markdown or additional fields."""

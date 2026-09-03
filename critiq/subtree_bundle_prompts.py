"""Prompts for root-subtree atomic evolution (Phase21)."""

ROOT_ERROR_SIGNATURE_PROMPT = """## Task
Analyze one error made by a complete Unified-Subtree judge. Identify the
reusable judging failure, not merely the image topic. The complete subtree is
the optimization unit.

## Current root subtree
{subtree_json}

## Preference sample
Sample ID: {sample_id}
Question: {question}
Candidate A: {A}
Candidate B: {B}
Human preference: {gold}
Unified-Subtree answer: {subtree_answer}
Unified-Subtree report:
{subtree_report_json}

## Current committed Rubric
{rubric_memory_json}

Return exactly one JSON object with exactly these string fields:
{{
  "sample_id": "{sample_id}",
  "task_pattern": "...",
  "visual_focus": "...",
  "candidate_difference": "...",
  "subtree_failure": "...",
  "suggested_subdomain": "..."
}}
Do not output Markdown or additional fields."""


BUNDLE_REFINE_PROMPT = """## Task
Refine a complete root subtree as one coherent bundle. Select only the child
criteria that need a small boundary or decision-rule correction. Do not
mechanically rewrite every child. Preserve all node IDs, criterion names,
scores, examples, edges, and topology. Never edit the root criterion.

## Current root subtree
{subtree_json}

## Unified-Subtree error signatures
{signatures_json}

## Multimodal representative errors
The images are attached in the same sample order.
{representative_samples_json}

## Previous rejected bundle attempts
{failure_history_json}

## Current committed Rubric
{rubric_memory_json}

Only descriptions may change. Every edited description must contain exactly
these four sections in order: Criterion focus, Applicable only when, Not
applicable when, Decision rule. Keep each description within
{max_description_chars} characters. The Decision rule must return None when
the criterion is inapplicable or evidence is insufficient and must judge only
this criterion's evidence.

Return exactly one JSON object:
{{
  "root_id": "{root_id}",
  "original_subtree_sha256": "{original_subtree_sha256}",
  "edits": [
    {{
      "node_id": "an existing child ID",
      "criterion_name": "the unchanged criterion name",
      "original_description_sha256": "lowercase SHA-256",
      "description": "Criterion focus: ...\\n\\nApplicable only when: ...\\n\\nNot applicable when: ...\\n\\nDecision rule: ...",
      "failure_analysis": ["evidence-grounded cause"],
      "rationale": "why this child needs the small correction"
    }}
  ],
  "unchanged_node_ids": ["all child IDs not present in edits"],
  "bundle_rationale": "why these edits form one coherent subtree correction",
  "representative_sample_ids": ["one to six supplied sample IDs"]
}}
At least one child must be edited. Do not output Markdown or additional fields."""


BUNDLE_FAILURE_ATTRIBUTION_PROMPT = """## Task
Diagnose why an atomic {operator} candidate failed on the root's frozen
Unified-Subtree scope. Use the paired corrected and harmed evidence. Attribute
the complete bundle rather than selecting a child for partial acceptance.

## Baseline subtree
{baseline_subtree_json}

## Candidate subtree
{candidate_subtree_json}

## Error signatures
{signatures_json}

## Candidate construction
{candidate_json}

## Paired metrics and evidence
{paired_evidence_json}

## Previous retry history
{failure_history_json}

Allowed primary_failure_type values:
{allowed_types_json}

Return exactly one JSON object:
{{
  "primary_failure_type": "one allowed value",
  "implicated_child_ids": ["zero or more existing candidate child IDs"],
  "summary": "concise evidence-based diagnosis",
  "corrected_harmed_explanation": "what was corrected and what was harmed",
  "preserve_next_time": ["specific useful property to retain"],
  "change_next_time": ["specific bundle-level revision"]
}}
Do not output Markdown or additional fields."""


ROOT_ERROR_SIGNATURE_PROMPT_VERSION = "1.0.0"
BUNDLE_REFINE_PROMPT_VERSION = "1.0.0"
BUNDLE_FAILURE_ATTRIBUTION_PROMPT_VERSION = "1.0.0"
BUNDLE_PARSER_VERSION = "1.0.0"

"""Versioned Manager prompts for the Phase 6B Specialize operator."""

ERROR_SIGNATURE_PROMPT = """## Task
Analyze one decisive error made by the parent rubric criterion. Identify the
judging failure pattern that could recur across samples. Focus on why the
parent criterion failed to distinguish the human-preferred answer. Do not
cluster by the pictured object or scene alone.

## Parent criterion
Name: {criterion_name}
Description: {criterion_description}

## Preference pair
Sample ID: {sample_id}
Question: {question}
Candidate A: {A}
Candidate B: {B}
Human preference: {gold}
Parent vote: {parent_vote}
Parent worker thought: {thought}

Return exactly one JSON object with exactly these string fields:
{{
  "sample_id": "{sample_id}",
  "task_pattern": "...",
  "visual_focus": "...",
  "candidate_difference": "...",
  "parent_failure": "...",
  "suggested_subdomain": "..."
}}
Do not output Markdown or additional fields."""

SEMANTIC_CLUSTER_PROMPT = """## Task
Group the supplied parent-error signatures into reusable judging-failure
subdomains. A cluster must express a shared criterion failure, not merely a
shared image topic. Produce only meaningful clusters; do not force the maximum.

Parent criterion: {criterion_name}
Minimum samples per cluster: {min_cluster_size}
Maximum clusters allowed in this edit: {max_clusters}

Error signatures:
{signatures_json}

Previous failed Split attempts for this parent (may be empty):
{split_failure_history_json}
Use this history only to avoid repeating failed semantic partitions or criterion descriptions.
The current signatures remain authoritative.

Every sample ID must occur exactly once, either in one cluster or in
unclustered_sample_ids.

Return exactly one JSON object:
{{
  "clusters": [
    {{
      "cluster_id": "...",
      "label": "...",
      "shared_failure": "...",
      "distinction": "...",
      "sample_ids": ["..."]
    }}
  ],
  "unclustered_sample_ids": ["..."]
}}
Do not output Markdown or additional fields."""

SPLIT_FAILURE_ATTRIBUTION_PROMPT = """## Task
Diagnose why the complete child set from a failed Split reduced accuracy on the
parent criterion's frozen applicability domain. Attribute the operator failure,
not the overall response quality. Use the supplied corrected/harmed cases and
metrics. Do not propose deleting one child from the current set; the next Split
must regenerate a coherent complete set.

## Parent criterion
Name: {criterion_name}
Description: {criterion_description}

## Error signatures used by this Split
{signatures_json}

## Cluster proposal
{cluster_json}

## Generated children
{children_json}

## Local Split metrics
{metrics_json}

## Changed parent-domain predictions
{changed_predictions_json}

Return exactly one JSON object:
{{
  "summary": "a concise natural-language explanation of why this Split failed",
  "failure_categories": ["lower_snake_case_category"],
  "details": ["specific evidence-backed failure detail"],
  "avoid_next_time": ["concrete instruction for the next Split attempt"]
}}
Use categories such as cluster_semantically_mixed, child_too_broad,
visual_fact_incorrect, preference_direction_reversed, duplicate_children, or
unrelated_scene_activation when supported by the evidence. Do not output
Markdown or additional fields."""

CHILD_GENERATION_PROMPT_V1 = """## Task
Create one child criterion that specializes the parent for exactly the supplied
error cluster. The child must be narrower than the parent and should help a
pairwise judge distinguish the preferred answer in this recurring subdomain.

## Parent criterion
Name: {criterion_name}
Description: {criterion_description}

## Cluster
{cluster_json}

## Error signatures
{signatures_json}

## Existing sibling criteria
{siblings_json}

## Previous failed Split attempts for this parent
{split_failure_history_json}
Use this history only to avoid repeating failed criterion descriptions; the current cluster is authoritative.

Representative sample IDs that you may cite: {representative_sample_ids}

Representative sample text (image paths and image bytes are intentionally
excluded when a text-only Manager is used):
{representative_samples_json}

Return exactly one JSON object:
{{
  "criterion_name": "lower_snake_case_name",
  "description": "a precise standalone pairwise judging criterion",
  "rationale": "why this is narrower than the parent and targets the cluster",
  "representative_sample_ids": ["..."]
}}
Use 1-3 representative sample IDs from the supplied cluster. Do not output
Markdown or additional fields."""


CHILD_GENERATION_PROMPT_V2 = """## Task
Create one child criterion that specializes the parent for exactly the supplied
error cluster. The child must be narrower than the parent and should help a
pairwise judge distinguish the preferred answer in this recurring subdomain.

## Parent criterion
Name: {criterion_name}
Description: {criterion_description}

## Cluster
{cluster_json}

## Error signatures
{signatures_json}

## Existing sibling criteria
{siblings_json}

## Previous failed Split attempts for this parent
{split_failure_history_json}
Use this history only to avoid repeating failed criterion descriptions; the current cluster is authoritative.

Representative sample IDs that you may cite: {representative_sample_ids}

Representative sample text (image paths and image bytes are intentionally
excluded when a text-only Manager is used):
{representative_samples_json}

Return exactly one JSON object:
{{
  "criterion_name": "lower_snake_case_name",
  "description": "Criterion focus: ...\\n\\nApplicable only when: ...\\n\\nNot applicable when: ...\\n\\nDecision rule: ...",
  "rationale": "why this is narrower than the parent and targets the cluster",
  "representative_sample_ids": ["..."]
}}
The description must contain exactly these four sections in this order:
- Criterion focus: the specific, narrow visual dimension evaluated by this criterion.
- Applicable only when: the observable circumstances in which this criterion applies.
- Not applicable when: neighboring circumstances or evaluation dimensions outside its scope.
- Decision rule: how to compare A and B when applicable, and explicitly return None when
  neither candidate has a reliable advantage under this criterion.
Use 1-3 representative sample IDs from the supplied cluster. Do not output
Markdown or additional fields."""


# Current prompt. V1 remains available for exact comparison and rollback.
CHILD_GENERATION_PROMPT = CHILD_GENERATION_PROMPT_V2

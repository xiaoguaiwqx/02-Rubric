"""Versioned criterion-agnostic prompt for joint multi-root routing."""

STRUCTURED_ROOT_ROUTER_PROMPT = """## Instruction
You are routing one multimodal preference pair to relevant rubric roots.
You receive an image, the source question, ordered candidate answers A and B,
and all available root rubrics.

Select every root whose criterion is meaningfully relevant for evaluating this
specific pair. This is a multi-label applicability decision, not a preference
judgement. Do not decide which candidate is better and do not use root selection
to encode an A/B winner. A root may be relevant because of claims made by either
candidate. Select at least one root.

## Question
{question}

## Candidate A
{A}

## Candidate B
{B}

## Available Root Rubrics
{roots}
"""

STRUCTURED_ROOT_ROUTER_POSTFIX = """
Return exactly one JSON object and no prose. It must contain exactly:
- "selected_root_ids": a non-empty JSON list of unique root ID strings;
- "rationale_by_root": a JSON object whose keys exactly equal the selected
  root IDs and whose values are short, non-empty applicability rationales.

Use one concise rationale for each selected root. Do not quote, restate, or
summarize the root description. Mention only why the root applies to this
sample, and keep the complete JSON compact enough to finish.

Use only root IDs listed above. Do not output confidence, weights, A/B
preference, winner, parse_ok, consistency_ok, Markdown, or extra fields.
"""

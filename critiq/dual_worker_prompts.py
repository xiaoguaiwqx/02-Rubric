"""Frozen prompts for the decoupled pairwise-vote and gate-state workers."""

# This is deliberately byte-for-byte identical to demo_rlhfv.WORKER_PROMPT.
# Do not improve or reword it: the criteria in exp4 were evolved against this
# execution contract.
PAIRWISE_MULTIMODAL_WORKER_PROMPT = """## Instruction
You are judging an RLHF-V visual QA / image-text preference pair under one criterion. You are given the image, the source question, and two candidate answers.

Use the image when the criterion depends on visual evidence. If the criterion is not applicable to this pair, answer None.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}

Which candidate better matches the criterion and is more likely to align with human preference?"""

# This is deliberately byte-for-byte identical to the English exp4 postfix in
# critiq.i18n.  The parser only consumes answer, while the unchanged thought
# field keeps the actual request compatible with the criterion-evolution run.
PAIRWISE_WORKER_PROMPT_POSTFIX = """
Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A based on the given criterion.",
    "analysis_b": "Analyze B based on the given criterion.",
    "thought": "Compare A and B.",
    "answer": "A / B / None"
}
```
Return None if any of the following conditions are met:
- The criterion is not applicable to this pair of data pieces.
- They are of the same quality.
- You are unsure.
"""


GATE_STATE_WORKER_PROMPT = """## Instruction
You are checking one criterion for an RLHF-V visual QA / image-text pair.
Judge only whether each candidate satisfies this criterion. Do not choose a
winner and do not compare writing quality beyond what the criterion requires.

First decide whether the criterion is meaningfully applicable. If it is not
applicable or applicability is uncertain, both candidate statuses must be
uncertain. Otherwise apply the same absolute standard to A and B:
- pass: the candidate satisfies the criterion;
- fail: the candidate violates the criterion;
- uncertain: available evidence is insufficient.

Use the image whenever the criterion depends on visual evidence.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}
"""


# Experimental v2 keeps the v1 task and schema unchanged.  It adds only the
# minimum constraints needed to test whether absolute status judgements become
# less sensitive to the A/B slot; do not add criterion-specific definitions.
GATE_STATE_WORKER_PROMPT_V2 = """## Instruction
You are checking one criterion for an RLHF-V visual QA / image-text pair.
Judge only whether each candidate satisfies this criterion. Do not choose a
winner and do not compare writing quality beyond what the criterion requires.

First decide whether the criterion is meaningfully applicable. If it is not
applicable or applicability is uncertain, both candidate statuses must be
uncertain. Otherwise apply the same absolute standard to A and B:
- pass: the candidate satisfies the criterion;
- fail: the candidate violates the criterion;
- uncertain: available evidence is insufficient.

Judge each candidate against the image, question, and criterion on its own.
The status is absolute: it must not depend on whether the candidate is shown
as A or B, or on how good the other candidate is. Do not downgrade a candidate
merely because the other candidate is better, more detailed, or more precise.

Use the image whenever the criterion depends on visual evidence.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}
"""


# Experimental v2.1 changes only the decision procedure.  It keeps the same
# generic task, status definitions, and three-field output schema as v1/v2.
GATE_STATE_WORKER_PROMPT_V2_1 = """## Instruction
You are checking one criterion for an RLHF-V visual QA / image-text pair.
Judge only whether each candidate satisfies this criterion. Do not choose a
winner and do not compare writing quality beyond what the criterion requires.

First decide whether the criterion is meaningfully applicable. If it is not
applicable or applicability is uncertain, both candidate statuses must be
uncertain. Otherwise apply the same absolute standard to A and B:
- pass: the candidate satisfies the criterion;
- fail: the candidate violates the criterion;
- uncertain: available evidence is insufficient.

Judge each candidate against the image, question, and criterion on its own.
The status is absolute: it must not depend on whether the candidate is shown
as A or B, or on how good the other candidate is. Do not downgrade a candidate
merely because the other candidate is better, more detailed, or more precise.

Evaluate A and B in separate judgment passes. In each pass, use only that
candidate's own claims and the criterion, applying the same standard both
times. Use fail only when that candidate itself has a clear violation of the
criterion; use uncertain when the evidence is insufficient. After assigning
both statuses, do not revise either one merely to create a contrast.

Use the image whenever the criterion depends on visual evidence.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}
"""

GATE_STATE_WORKER_PROMPT_POSTFIX = """
Return exactly one JSON object and no prose. It must contain exactly these
three keys, and every value must be a JSON string:
- "applicable": exactly one of "yes", "no", or "uncertain"
- "status_a": exactly one of "pass", "fail", or "uncertain"
- "status_b": exactly one of "pass", "fail", or "uncertain"

If applicable is "no" or "uncertain", both statuses must be "uncertain".
Do not output a winner, pair preference, evidence, thought, Markdown, or any
additional field.
"""

"""Versioned English prompts for structured multimodal worker judgements."""

STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1 = """## Instruction
You are judging an RLHF-V visual QA / image-text preference pair under exactly one criterion. You receive the image, the source question, and candidate answers A and B.

First decide whether the criterion is meaningfully applicable to this pair. If it is applicable, assess A and B independently:
- pass: the candidate satisfies the criterion;
- fail: the candidate violates the criterion;
- uncertain: the available evidence is insufficient.

Then give a direct pairwise preference under this criterion. Pair preference is an explicit comparison and is not mechanically derived from pass/fail. When both candidates pass or both fail, you may still prefer A or B based on degree. Use tie only when they are genuinely equal under the criterion, and uncertain only when comparison is not reliable.

Use the image whenever the criterion depends on visual evidence. Keep evidence short and externally checkable; do not provide hidden chain-of-thought.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}
"""


STRUCTURED_WORKER_PROMPT_POSTFIX_V1 = """
Return exactly one JSON object and no prose. It must contain exactly these six
keys, and every value must be a JSON string:
- "applicable": exactly one of "yes", "no", or "uncertain"
- "status_a": exactly one of "pass", "fail", or "uncertain"
- "status_b": exactly one of "pass", "fail", or "uncertain"
- "pair_preference": exactly one of "A", "B", "tie", or "uncertain"
- "evidence_a": short, sample-specific evidence about the actual candidate A
- "evidence_b": short, sample-specific evidence about the actual candidate B

Consistency requirements:
- If applicable is no or uncertain, status_a, status_b, and pair_preference must all be uncertain.
- If status_a is pass and status_b is fail, pair_preference must be A.
- If status_a is fail and status_b is pass, pair_preference must be B.
- If both statuses are pass, both are fail, or either status is uncertain, pair_preference may be A, B, tie, or uncertain.

Do not output parse_ok, consistency_ok, local_decision, thought, Markdown, or
any additional field. Do not use generic placeholder evidence: both evidence
strings must be grounded in the current image, question, and candidates.
"""


# Phase 4 reliability revision. This keeps the generic V1 judgement semantics
# while making the applicability gate explicit and mechanically checkable.
STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1_1 = """## Instruction
You are judging an RLHF-V visual QA / image-text preference pair under exactly one criterion. You receive the image, the source question, and candidate answers A and B.

First make one binding applicability decision.
- If applicability is "yes", assess A and B independently as pass, fail, or uncertain, then compare them under the criterion.
- If applicability is "no" or "uncertain", do not assign pass or fail to either candidate. Both statuses and the pairwise preference must be "uncertain".

When applicable:
- pass: the candidate satisfies the criterion;
- fail: the candidate violates the criterion;
- uncertain: the available evidence is insufficient.

Pair preference is an explicit comparison and is not mechanically derived from pass/fail. When both candidates pass or both fail, you may still prefer A or B based on degree. Use tie only when they are genuinely equal under the criterion, and uncertain only when comparison is not reliable.

Use the image whenever the criterion depends on visual evidence. Keep evidence short and externally checkable; do not provide hidden chain-of-thought.

Before returning JSON, silently verify this invariant: applicability other than "yes" must never be combined with pass/fail or an A/B/tie preference.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}
"""


STRUCTURED_WORKER_PROMPT_POSTFIX_V1_1 = """
Return exactly one JSON object and no prose. It must contain exactly these six
keys, and every value must be a JSON string:
- "applicable": exactly one of "yes", "no", or "uncertain"
- "status_a": exactly one of "pass", "fail", or "uncertain"
- "status_b": exactly one of "pass", "fail", or "uncertain"
- "pair_preference": exactly one of "A", "B", "tie", or "uncertain"
- "evidence_a": short, sample-specific evidence about candidate A
- "evidence_b": short, sample-specific evidence about candidate B

Binding applicability gate:
- applicable="no" or applicable="uncertain" requires status_a="uncertain",
  status_b="uncertain", and pair_preference="uncertain".
- Never output pass, fail, A, B, or tie when applicable is not "yes".

Other consistency requirements:
- pass/fail requires pair_preference="A".
- fail/pass requires pair_preference="B".
- With equal statuses or an uncertain status, preference may be A, B, tie, or
  uncertain when applicable is "yes".

Do a final silent consistency check before emitting the JSON. Do not output
parse_ok, consistency_ok, local_decision, thought, Markdown, or extra fields.
Do not use generic placeholder evidence; ground both evidence strings in the
current image, question, criterion, and candidates.
"""


STRUCTURED_MULTIMODAL_WORKER_PROMPT_V2 = """## Instruction
You are judging an RLHF-V visual QA / image-text preference pair under exactly one criterion. You receive the image, the source question, and candidate answers A and B.

First decide whether the criterion is meaningfully applicable to this pair.

If it is applicable, assign an absolute status to each candidate before making
the pairwise comparison. Apply the same criterion-specific standard to both
candidates:
- pass: the candidate has no material violation of the criterion;
- fail: the candidate has at least one clear, verifiable, relevant, and
  material violation of the criterion;
- uncertain: the evidence needed to verify a key criterion-relevant claim is
  insufficient or ambiguous.

Candidate statuses are absolute and position-invariant. A candidate's status
must depend only on the image, question, criterion, and that candidate's own
content. It must not depend on whether the candidate appears as A or B, on the
quality or wording of the other candidate, or on which candidate you expect to
prefer. Being less detailed, less precise, or less well written than the other
candidate does not by itself cause failure. Do not downgrade one candidate
merely to justify preferring the other.

After assigning both statuses, compare the candidates directly under only the
given criterion. Pair preference is an explicit comparison and is not
mechanically derived from pass/fail. Both candidates may pass while one is
preferred because it has a reliable and meaningful advantage under the
criterion. Use tie when there is no reliable and meaningful difference under
the criterion; identical wording is not required. Use uncertain only when the
comparison itself cannot be made reliably.

When the criterion depends on visual evidence, silently verify the candidates'
criterion-relevant claims against the image. Each evidence string must directly
justify its corresponding status: for pass, identify the key supported claims;
for fail, identify the exact contradicted or unsupported claim; for uncertain,
identify what cannot be verified reliably. Do not add visual facts that are
unnecessary for the judgement. Keep evidence short and externally checkable;
do not provide hidden chain-of-thought.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}
"""


STRUCTURED_WORKER_PROMPT_POSTFIX_V2 = """
Return exactly one JSON object and no prose. It must contain exactly these six
keys, and every value must be a JSON string:
- "applicable": exactly one of "yes", "no", or "uncertain"
- "status_a": exactly one of "pass", "fail", or "uncertain"
- "status_b": exactly one of "pass", "fail", or "uncertain"
- "pair_preference": exactly one of "A", "B", "tie", or "uncertain"
- "evidence_a": short evidence that directly justifies status_a
- "evidence_b": short evidence that directly justifies status_b

Consistency requirements:
- If applicable is no or uncertain, status_a, status_b, and pair_preference must all be uncertain.
- If status_a is pass and status_b is fail, pair_preference must be A.
- If status_a is fail and status_b is pass, pair_preference must be B.
- If both statuses are the same, prefer A or B only for a reliable and meaningful difference in degree under the criterion; otherwise use tie.
- Never change an absolute candidate status merely to make it agree with the pairwise preference.

Do not output parse_ok, consistency_ok, local_decision, thought, Markdown, or
any additional field. Do not use generic placeholder evidence or merely
summarize a candidate: each evidence string must evaluate the actual candidate
in its current slot and justify its status using the current image, question,
and criterion.
"""

STRUCTURED_MULTIMODAL_WORKER_PROMPT_V2_1 = """## Task
Judge an RLHF-V visual QA / image-text preference pair under exactly one
criterion using the image, source question, and candidate answers A and B.

Follow this procedure exactly:
1. Decide whether the criterion is applicable.
2. Judge A while ignoring B.
3. Judge B while ignoring A.
4. Freeze both statuses; never revise them during comparison or evidence.
5. Compare A and B under only the given criterion.

For each candidate, classify its material criterion-relevant claims as
supported, contradicted, or not verifiable:
- pass: no material claim is contradicted and key claims are supported;
- fail: at least one material claim is clearly contradicted or fabricated;
- uncertain: no material contradiction is clear, but a key claim cannot be
  verified reliably from the available evidence.

For visual grounding, a material contradiction is a clearly wrong or
fabricated object, person, count, visible text, color, attribute, action, or
spatial relation that matters to the answer. Judge only claims the candidate
actually makes. Omitted visible details, minor wording differences, and
compatible coarse descriptions are not visual-grounding violations. Treat an
unresolvable fine detail as not verifiable instead of guessing.

Use one shared interpretation of the image for both candidates. Never treat
the same visual fact as present for one candidate and absent for the other.
Statuses must not depend on A/B position, the other candidate's quality, or
the expected pair preference.

After freezing statuses:
- prefer the passing candidate over a failing candidate;
- for equal statuses, prefer one only for a reliable, meaningful advantage;
- a more precise answer may be preferred without making the other fail;
- when both fail, compare violation severity when possible;
- use tie when evidence is sufficient and neither has a meaningful advantage;
- use uncertain only when evidence is insufficient for reliable comparison.

When applicable is yes, each evidence string must be one short sentence about
its own candidate. For fail or uncertain, identify one decisive contradicted
or not-verifiable key claim. For pass, identify one supported key claim and
state that no material contradiction was found. Do not summarize, discuss the
other candidate, add unnecessary facts, or narrate checking or self-correction.
If applicable is no or uncertain, use empty evidence strings.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}
"""


STRUCTURED_WORKER_PROMPT_POSTFIX_V2_1 = """
Return exactly one valid JSON object and no other text. Include exactly these
six keys, with JSON string values:
- "applicable": "yes", "no", or "uncertain"
- "status_a": "pass", "fail", or "uncertain"
- "status_b": "pass", "fail", or "uncertain"
- "pair_preference": "A", "B", "tie", or "uncertain"
- "evidence_a": one short sentence about A, or empty when not applicable
- "evidence_b": one short sentence about B, or empty when not applicable

Rules:
- If applicable is not "yes", both statuses and pair_preference must be
  "uncertain", and both evidence strings must be empty.
- pass/fail requires preference "A"; fail/pass requires preference "B".
- tie means sufficient evidence shows no meaningful difference.
- uncertain preference means the comparison lacks sufficient evidence.
- Each evidence must justify its own status, evaluate the correct candidate,
  and not contradict the other evidence about the same image fact.

Do not revise or debate the judgement. Do not output self-correction,
chain-of-thought, Markdown, parse_ok, consistency_ok, local_decision, thought,
or any additional field.
"""


# Active generic Phase 4 reliability revision. V1 remains archived above for
# exact comparison and rollback; V2/V2.1 remain separate robustness variants.
STRUCTURED_MULTIMODAL_WORKER_PROMPT = STRUCTURED_MULTIMODAL_WORKER_PROMPT_V1_1
STRUCTURED_WORKER_PROMPT_POSTFIX = STRUCTURED_WORKER_PROMPT_POSTFIX_V1_1

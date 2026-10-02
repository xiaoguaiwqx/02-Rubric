"""Frozen Worker and Global Arbiter prompts for the current method."""
from __future__ import annotations
import json
from typing import Any, Mapping, Sequence
from structured_rubrics.utils import parse_json


SUBTREE_PROMPT_VERSION = "implicit-unified-subtree-direct-judge-v1"


ARBITER_PROMPT_VERSION = "global-arbiter-evidence-synthesis-ab-only-v1"


UNIFIED_SUBTREE_SYSTEM_PROMPT = """## Instruction

You are judging a multimodal image-text preference pair under one structured rubric subtree. You are given the image, the source instruction or question, two candidate responses, and the rubric subtree.

The root defines the broad criterion, while its children provide specialized guidance for particular cases. Treat the entire subtree as one unified decision policy, not as a set of independent votes.

Use the image when the rubric depends on visual evidence. If the rubric subtree as a whole is not applicable to this pair, answer None.

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
- The rubric subtree as a whole is not applicable to this pair of data pieces.
- They are of the same quality.
- You are unsure.
"""


GLOBAL_ARBITER_AB_ONLY_SYSTEM_PROMPT = """## Instruction

You are the final decision arbiter for one structured multimodal preference system. You are given the image, the source instruction or question, two candidate responses, and five subtree assessments produced under complementary rubric dimensions.

Treat the subtree assessments as correlated evidence, not independent votes. Do not decide by counting their A/B labels. Independently verify the image, question, and candidate responses. A subtree assessment may be incorrect or inapplicable.

Prioritize verifiable visual and factual correctness. Consider completeness after factual validity; clarity and creativity may distinguish otherwise acceptable responses but cannot compensate for factual errors.

Return a relative preference for every pair. If both responses are imperfect or the evidence is limited, select the response with stronger support and the less severe error. Do not abstain.

Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A using the image and subtree evidence.",
    "analysis_b": "Analyze B using the image and subtree evidence.",
    "thought": "Integrate the evidence into one relative preference.",
    "answer": "A / B"
}
```
"""


def parse_subtree_response(raw: object) -> dict[str, str]:
    if not isinstance(raw, str):
        raise ValueError("Unified-Subtree response must be text")
    payload = parse_json(raw, allow_invalid_escapes=True)
    if not isinstance(payload, dict):
        raise ValueError("Unified-Subtree response must be one JSON object")
    answer = payload.get("answer")
    if answer not in {"A", "B", "None"}:
        raise ValueError("Unified-Subtree answer must be A, B, or None")
    return {
        "analysis_a": payload.get("analysis_a")
        if isinstance(payload.get("analysis_a"), str) else "",
        "analysis_b": payload.get("analysis_b")
        if isinstance(payload.get("analysis_b"), str) else "",
        "thought": payload.get("thought")
        if isinstance(payload.get("thought"), str) else "",
        "answer": str(answer),
    }


def subtree_user_prompt(row: Mapping[str, Any], rubric: Any, root_id: str) -> str:
    root = rubric.get_node(root_id)
    lines = [
        "## Question", str(row["question"]), "",
        "## Candidate A", str(row["A"]), "",
        "## Candidate B", str(row["B"]), "",
        "## Structured Rubric Subtree", "",
        "### Root Criterion",
        f"**{root.criterion.name}**: {root.criterion.description}", "",
        "### Specialized Child Criteria",
    ]
    children = rubric.children(root_id)
    if children:
        for index, child in enumerate(children, start=1):
            lines.extend([
                f"{index}. **{child.criterion.name}**",
                child.criterion.description,
            ])
    else:
        lines.append("None.")
    lines.extend([
        "", "Which candidate better follows this structured rubric subtree and is "
        "more likely to align with human preference?",
    ])
    return "\n".join(lines)


def parse_global_arbiter_ab_only_response(raw: object) -> dict[str, str]:
    if not isinstance(raw, str):
        raise ValueError("A/B-only Global-Arbiter response must be text")
    payload = parse_json(raw, allow_invalid_escapes=True)
    if not isinstance(payload, dict):
        raise ValueError("A/B-only Global-Arbiter response must be one JSON object")
    answer = payload.get("answer")
    if answer not in {"A", "B", "None"}:
        raise ValueError("Global-Arbiter answer must be A, B, or None")
    return {"answer": str(answer)}


def parse_global_arbiter_with_reason(raw: object) -> dict[str, str]:
    """Retain the emitted justification for reflection without changing votes."""
    result = parse_global_arbiter_ab_only_response(raw)
    payload = parse_json(raw, allow_invalid_escapes=True)
    for field in ("analysis_a", "analysis_b", "thought"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"Global-Arbiter {field} is missing")
        result[field] = payload[field]
    return result

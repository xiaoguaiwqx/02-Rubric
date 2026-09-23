"""Case-isolated feedback and one whole-child-group revision per root."""
from .framework_v6_manager import COMMON, PROMPTS, Manager, validate


LOCAL_PROMPTS = dict(PROMPTS, case_reflection=COMMON + """
The full current Rubric is shared background for scope and overlap awareness.
Only the target root's entire child group is eligible for improvement.
Inspect this ONE image/question/response pair and the target Worker's full report.
Global gold disagreement or None triggers review, not proof of a local error.
Explain whether the local scope, evidence, or rule execution needs improvement.
Execution difficulty can justify clearer instructions even if the current rule
is logically sound. Preserve valid scope boundaries and useful existing checks.
Return {"analysis": "Evidence-based local diagnosis", "critique": "A few
sentences of reusable child-group revision advice"}. If no justified actionable
change exists, return critique="" and explain in analysis. Never force agreement
with gold outside the root's scope. Do not repeat the whole input.
""", subtree_split=COMMON + """
Use the full current Rubric as background, and revise ONLY the target root's
complete child group using all supplied case critiques. Resolve conflicting or
redundant suggestions in this single call; do not blindly concatenate them.
Keep the root unchanged and preserve useful children and capabilities. You may
retain, rewrite, merge or add children. Return 2-5 children with short unique
snake_case names and complete, concise descriptions explaining applicability,
checks and local comparison. Avoid overlap with other roots. Do not add examples,
sample IDs, gold answers, or fixed A/B preferences to criteria.
Return {"children": [{"name": "...", "description": "..."}],
"change_summary": "Changes and retained capabilities"}.
""")


def validate_local(stage, result, payload):
    if stage == "case_reflection":
        if not isinstance(result, dict):
            raise ValueError("reflection must be an object")
        for field in ("analysis", "critique"):
            if not isinstance(result.get(field), str):
                raise ValueError(f"{field} must be text")
        if not result["analysis"].strip():
            raise ValueError("analysis must explain the case")
        return result
    return validate("children" if stage == "subtree_split" else stage, result, payload)


def make_manager(config, attempts, client=None):
    return Manager(config, attempts, client, prompts=LOCAL_PROMPTS,
                   validator=validate_local)

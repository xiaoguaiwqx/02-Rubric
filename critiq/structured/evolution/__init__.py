"""Pure, model-free primitives for structured-rubric evolution."""

from .feedback import criterion_fitness, extract_rubric_feedback
from .patch import apply_rubric_patch, diff_rubrics, plan_artifact_refresh
from .types import (
    ArtifactRefreshPlan,
    CandidateAcceptancePolicy,
    CandidateEvaluation,
    EditCandidate,
    ErrorSampleRef,
    EvolutionContext,
    EvolutionDecision,
    NodeFeedback,
    OperatorKind,
    RubricDiff,
    RubricFeedback,
    RubricPatch,
    evaluate_candidate,
)

__all__ = [
    "ArtifactRefreshPlan",
    "CandidateAcceptancePolicy",
    "CandidateEvaluation",
    "EditCandidate",
    "ErrorSampleRef",
    "EvolutionContext",
    "EvolutionDecision",
    "NodeFeedback",
    "OperatorKind",
    "RubricDiff",
    "RubricFeedback",
    "RubricPatch",
    "apply_rubric_patch",
    "criterion_fitness",
    "diff_rubrics",
    "evaluate_candidate",
    "extract_rubric_feedback",
    "plan_artifact_refresh",
]

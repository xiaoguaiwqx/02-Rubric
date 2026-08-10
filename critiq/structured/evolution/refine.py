"""Pure Refine v1 parsing, patching, prediction assembly and diagnostics."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from critiq.utils import PairData, parse_json

from ..aggregation import aggregate_flat_votes
from ..dual_worker import PairwisePredictionOutput
from ..judgement import FinalPreference, Vote
from ..schema import RubricCriterionSnapshot, RubricNode, StructuredRubric
from ..telemetry import ModelCallMetrics
from ..worker_output import StructuredCriterionSnapshot
from .patch import apply_rubric_patch
from .specialize import execute_offline_m1, project_pairwise_prediction
from .specialize_types import SpecializeManagerRequestSpec
from .refine_types import (
    RefineCandidate,
    RefineEvaluation,
    RefineNodeMetric,
    RefineProposal,
    RefineTriggerDecision,
)
from .types import EditCandidate, EvolutionContext, EvolutionDecision, OperatorKind, RubricPatch


class RefineParseError(ValueError):
    """A Refine Manager response violated the strict v1 schema."""


_SECTIONS = (
    "Criterion focus:",
    "Applicable only when:",
    "Not applicable when:",
    "Decision rule:",
)


def _parse(raw: object, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, str):
        raise RefineParseError(f"{label} response must be a string")
    try:
        value = parse_json(raw)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RefineParseError(f"failed to parse {label} JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RefineParseError(f"{label} JSON must be an object")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> Mapping[str, Any]:
    if set(value) != fields:
        raise RefineParseError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _validate_description(description: object, original: str, max_chars: int) -> str:
    if not isinstance(description, str) or not description.strip() or description != description.strip():
        raise RefineParseError("description must be a clean non-empty string")
    if description == original:
        raise RefineParseError("Refine proposal must change the description")
    if len(description) > max_chars:
        raise RefineParseError(f"description exceeds {max_chars} characters")
    positions = [description.find(section) for section in _SECTIONS]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise RefineParseError("description must contain the four required sections in order")
    if any(description.count(section) != 1 for section in _SECTIONS):
        raise RefineParseError("each required description section must occur exactly once")
    decision = description[positions[-1]:].lower()
    for phrase in ("none", "visual evidence", "insufficient",
                   "overall answer quality", "other criteria"):
        if phrase not in decision:
            raise RefineParseError(f"Decision rule must explicitly mention {phrase!r}")
    if not re.search(r"(?:do not|must not|never)[^.\n]*overall answer quality", decision):
        raise RefineParseError("Decision rule must forbid overall-answer-quality voting")
    if not re.search(r"(?:do not|must not|never)[^.\n]*other criteria", decision):
        raise RefineParseError("Decision rule must forbid voting from other criteria")
    if not re.search(r"(?:only|solely)[^.\n]*criterion", decision):
        raise RefineParseError("Decision rule must use only the current criterion")
    return description


def parse_refine_proposal_response(
    raw: object,
    *,
    node: RubricNode,
    allowed_representative_ids: Sequence[str],
    max_description_chars: int,
    attempt_count: int,
    metrics: ModelCallMetrics,
    request_spec: SpecializeManagerRequestSpec,
) -> RefineProposal:
    fields = {"criterion_name", "description", "failure_analysis", "rationale",
              "representative_sample_ids"}
    value = _exact(_parse(raw, "refine proposal"), fields, "refine proposal")
    if value["criterion_name"] != node.criterion.name:
        raise RefineParseError("Refine must preserve criterion_name")
    description = _validate_description(
        value["description"], node.criterion.description, max_description_chars)
    for field in ("failure_analysis", "representative_sample_ids"):
        if (not isinstance(value[field], list) or not value[field]
                or any(not isinstance(item, str) or not item.strip() for item in value[field])):
            raise RefineParseError(f"{field} must be a non-empty string list")
    representatives = tuple(item.strip() for item in value["representative_sample_ids"])
    if len(set(representatives)) != len(representatives) or not 1 <= len(representatives) <= 6:
        raise RefineParseError("representative_sample_ids must contain 1-6 unique IDs")
    if not set(representatives).issubset(allowed_representative_ids):
        raise RefineParseError("representative_sample_ids must come from supplied evidence")
    analysis = tuple(item.strip() for item in value["failure_analysis"])
    if len(set(analysis)) != len(analysis):
        raise RefineParseError("failure_analysis must not contain duplicates")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise RefineParseError("rationale must be non-empty")
    return RefineProposal(
        node.node_id,
        node.criterion.name,
        hashlib.sha256(node.criterion.description.encode("utf-8")).hexdigest(),
        description,
        analysis,
        rationale.strip(),
        representatives,
        raw.strip(),
        attempt_count,
        metrics,
        request_spec,
    )


def parse_refine_failure_attribution_response(raw: object) -> dict[str, Any]:
    fields = {"summary", "failure_categories", "details", "avoid_next_time"}
    value = _exact(_parse(raw, "refine failure attribution"), fields,
                   "refine failure attribution")
    result: dict[str, Any] = {}
    summary = value["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise RefineParseError("failure attribution summary must be non-empty")
    result["summary"] = summary.strip()
    for field in ("failure_categories", "details", "avoid_next_time"):
        items = value[field]
        if (not isinstance(items, list) or not items
                or any(not isinstance(item, str) or not item.strip() for item in items)):
            raise RefineParseError(f"{field} must be a non-empty string list")
        cleaned = [item.strip() for item in items]
        if len(set(cleaned)) != len(cleaned):
            raise RefineParseError(f"{field} must not contain duplicates")
        result[field] = cleaned
    return result


def detect_refine_trigger(
    context: EvolutionContext,
    node_id: str,
    thresholds: Mapping[str, Any],
    *,
    forced: bool = False,
) -> RefineTriggerDecision:
    node = context.rubric.get_node(node_id)
    feedback = context.feedback.nodes[node_id]
    if feedback.criterion_name != node.criterion.name:
        raise ValueError("Refine feedback criterion mismatch")
    required = {"tau_acc", "tau_refine", "tau_cov_high", "N_min_support"}
    if not required.issubset(thresholds):
        raise ValueError("Refine thresholds are incomplete")
    reasons: list[str] = []
    if feedback.accuracy <= thresholds["tau_acc"]:
        reasons.append("accuracy_not_above_tau_acc")
    if feedback.accuracy >= thresholds["tau_refine"]:
        reasons.append("accuracy_not_below_tau_refine")
    if feedback.coverage > thresholds["tau_cov_high"]:
        reasons.append("coverage_above_tau_cov_high")
    if feedback.support < thresholds["N_min_support"]:
        reasons.append("support_below_minimum")
    return RefineTriggerDecision(
        node_id=node_id,
        triggered=forced or not reasons,
        forced=forced,
        reasons=tuple(reasons),
        accuracy=feedback.accuracy,
        coverage=feedback.coverage,
        support=feedback.support,
        wrong=feedback.wrong,
    )


def build_refine_candidate(
    context: EvolutionContext,
    node_id: str,
    proposal: RefineProposal,
) -> RefineCandidate:
    node = context.rubric.get_node(node_id)
    expected_hash = hashlib.sha256(node.criterion.description.encode("utf-8")).hexdigest()
    if (proposal.node_id != node_id or proposal.criterion_name != node.criterion.name
            or proposal.original_description_sha256 != expected_hash):
        raise ValueError("Refine proposal does not match the current node")
    replacement = RubricNode(
        node_id=node.node_id,
        criterion=RubricCriterionSnapshot(
            node.criterion.name, proposal.description, node.criterion.score),
        examples=node.examples,
        lineage=node.lineage,
    )
    patch = RubricPatch(context.rubric.rubric_sha256, upsert_nodes=(replacement,))
    digest = hashlib.sha256(proposal.description.encode("utf-8")).hexdigest()[:12]
    edit = EditCandidate(f"refine-{node_id}-{digest}", OperatorKind.REFINE,
                         proposal.rationale, patch)
    after = apply_rubric_patch(context.rubric, patch)
    if (after.root_ids != context.rubric.root_ids or after.edges != context.rubric.edges
            or set(after.nodes) != set(context.rubric.nodes)):
        raise ValueError("Refine must not change Rubric topology")
    return RefineCandidate(node_id, edit, proposal)


def assemble_refined_pairwise_prediction(
    before_prediction: PairwisePredictionOutput,
    candidate_prediction: PairwisePredictionOutput,
    after_rubric: StructuredRubric,
    node_id: str,
) -> PairwisePredictionOutput:
    if (before_prediction.sample_ids != candidate_prediction.sample_ids
            or before_prediction.sample_fingerprints != candidate_prediction.sample_fingerprints
            or before_prediction.request_spec != candidate_prediction.request_spec):
        raise ValueError("Refine prediction identities do not match")
    target = after_rubric.get_node(node_id).criterion
    if (len(candidate_prediction.criteria) != 1
            or candidate_prediction.criteria[0].name != target.name
            or candidate_prediction.criteria[0].description != target.description):
        raise ValueError("candidate prediction must contain only the refined criterion")
    before_names = {item.name for item in before_prediction.criteria}
    if target.name not in before_names:
        raise ValueError("refined criterion is missing from baseline prediction")
    rows = []
    for old_row, new_row in zip(before_prediction.node_outputs,
                                candidate_prediction.node_outputs):
        row = dict(old_row)
        row[target.name] = new_row[target.name]
        rows.append(row)
    ordered = tuple(after_rubric.get_node(node_id).criterion
                    for node_id in after_rubric.preorder_node_ids())
    names = tuple(item.name for item in ordered)
    rows = tuple({name: row[name] for name in names} for row in rows)
    answers = tuple(aggregate_flat_votes(item.vote for item in row.values()) for row in rows)
    return PairwisePredictionOutput(
        before_prediction.sample_ids,
        before_prediction.sample_fingerprints,
        tuple(StructuredCriterionSnapshot(item.name, item.description) for item in ordered),
        rows,
        answers,
        before_prediction.request_spec,
    )


def _metric(outputs, gold: Sequence[str]) -> RefineNodeMetric:
    decisive = [item.parse_ok and item.answer_valid and item.vote in {Vote.A, Vote.B}
                for item in outputs]
    correct = [decisive[index] and item.vote.value == gold[index]
               for index, item in enumerate(outputs)]
    support = sum(decisive)
    return RefineNodeMetric(
        support=support,
        correct=sum(correct),
        wrong=support - sum(correct),
        accuracy=sum(correct) / support if support else 0.0,
        coverage=support / len(outputs),
        valid_rate=sum(item.parse_ok and item.answer_valid for item in outputs) / len(outputs),
    )


def _accuracy(values, gold: Sequence[str], scope: Sequence[bool] | None = None) -> float:
    if scope is None:
        scope = [True] * len(gold)
    support = sum(scope)
    if not support:
        return 0.0
    return sum(active and value.value == target
               for active, value, target in zip(scope, values, gold)) / support


def _coverage(values) -> float:
    return sum(value.value in {"A", "B"} for value in values) / len(values)


def _root_subtree_rubric(rubric: StructuredRubric, root_id: str) -> StructuredRubric:
    node_ids = tuple(node_id for node_id in rubric.preorder_node_ids()
                     if rubric.root_id_for(node_id) == root_id)
    return StructuredRubric(
        nodes={node_id: rubric.get_node(node_id) for node_id in node_ids},
        edges=tuple(edge for edge in rubric.edges
                    if edge.parent_id in node_ids and edge.child_id in node_ids),
        root_ids=(root_id,),
    )


def evaluate_refine_candidate(
    *,
    before_rubric: StructuredRubric,
    after_rubric: StructuredRubric,
    before_prediction: PairwisePredictionOutput,
    combined_prediction: PairwisePredictionOutput,
    candidate_prediction: PairwisePredictionOutput,
    dataset: Sequence[PairData],
    node_id: str,
    min_support: int,
) -> tuple[RefineEvaluation, Any, Any]:
    if min_support < 1:
        raise ValueError("min_support must be positive")
    before_prediction = project_pairwise_prediction(before_prediction, before_rubric)
    combined_prediction = project_pairwise_prediction(combined_prediction, after_rubric)
    if tuple(str(row["sample_id"]) for row in dataset) != before_prediction.sample_ids:
        raise ValueError("dataset and Refine prediction sample IDs differ")
    before_node = before_rubric.get_node(node_id)
    after_node = after_rubric.get_node(node_id)
    if (before_node.criterion.name != after_node.criterion.name
            or before_node.criterion.score != after_node.criterion.score
            or before_rubric.edges != after_rubric.edges
            or before_rubric.root_ids != after_rubric.root_ids
            or set(before_rubric.nodes) != set(after_rubric.nodes)):
        raise ValueError("Refine candidate changed node identity or topology")
    name = before_node.criterion.name
    old_outputs = tuple(row[name] for row in before_prediction.node_outputs)
    new_outputs = tuple(row[name] for row in combined_prediction.node_outputs)
    gold = tuple(str(row["answer"]) for row in dataset)
    sample_ids = before_prediction.sample_ids
    old_metric = _metric(old_outputs, gold)
    new_metric = _metric(new_outputs, gold)
    old_correct = [item.parse_ok and item.answer_valid and item.vote.value == target
                   for item, target in zip(old_outputs, gold)]
    new_correct = [item.parse_ok and item.answer_valid and item.vote.value == target
                   for item, target in zip(new_outputs, gold)]
    corrected = tuple(sample_id for sample_id, old, new in zip(sample_ids, old_correct, new_correct)
                      if not old and new)
    harmed = tuple(sample_id for sample_id, old, new in zip(sample_ids, old_correct, new_correct)
                   if old and not new)
    transitions = Counter(f"{old.vote.value}->{new.vote.value}"
                          for old, new in zip(old_outputs, new_outputs))
    old_decisive = [item.parse_ok and item.answer_valid and item.vote in {Vote.A, Vote.B}
                    for item in old_outputs]
    new_decisive = [item.parse_ok and item.answer_valid and item.vote in {Vote.A, Vote.B}
                    for item in new_outputs]
    expansion = [not old and new for old, new in zip(old_decisive, new_decisive)]
    expansion_wrong = sum(active and not new_correct[index]
                          for index, active in enumerate(expansion))

    before_execution, before_answers = execute_offline_m1(
        before_rubric, before_prediction, dataset)
    after_execution, after_answers = execute_offline_m1(
        after_rubric, combined_prediction, dataset)

    root_id = before_rubric.root_id_for(node_id)
    before_subtree = _root_subtree_rubric(before_rubric, root_id)
    after_subtree = _root_subtree_rubric(after_rubric, root_id)
    before_sub_prediction = project_pairwise_prediction(before_prediction, before_subtree)
    after_sub_prediction = project_pairwise_prediction(combined_prediction, after_subtree)
    before_sub_execution, _ = execute_offline_m1(before_subtree, before_sub_prediction, dataset)
    after_sub_execution, _ = execute_offline_m1(after_subtree, after_sub_prediction, dataset)
    old_sub_votes = tuple(trace.final_preference for trace in before_sub_execution.traces)
    new_sub_votes = tuple(trace.final_preference for trace in after_sub_execution.traces)
    subtree_scope = [vote in {FinalPreference.A, FinalPreference.B} for vote in old_sub_votes]
    subtree_support = sum(subtree_scope)
    if subtree_support == 0:
        raise ValueError("target root subtree has zero frozen support")
    old_sub_correct = [active and vote.value == target
                       for active, vote, target in zip(subtree_scope, old_sub_votes, gold)]
    new_sub_correct = [active and vote.value == target
                       for active, vote, target in zip(subtree_scope, new_sub_votes, gold)]
    subtree_corrected = tuple(sample_id for sample_id, old, new in
                              zip(sample_ids, old_sub_correct, new_sub_correct)
                              if not old and new)
    subtree_harmed = tuple(sample_id for sample_id, old, new in
                           zip(sample_ids, old_sub_correct, new_sub_correct)
                           if old and not new)

    m1_old_correct = [value.value == target for value, target in zip(before_answers, gold)]
    m1_new_correct = [value.value == target for value, target in zip(after_answers, gold)]
    m1_corrected = tuple(sample_id for sample_id, old, new in
                         zip(sample_ids, m1_old_correct, m1_new_correct) if not old and new)
    m1_harmed = tuple(sample_id for sample_id, old, new in
                      zip(sample_ids, m1_old_correct, m1_new_correct) if old and not new)

    parent_id = before_rubric.parent_id(node_id)
    siblings = (() if parent_id is None else
                tuple(child for child in before_rubric.children(parent_id)
                      if child.node_id != node_id))
    joint = conflict = 0
    for sibling in siblings:
        sibling_outputs = tuple(row[sibling.criterion.name]
                                for row in before_prediction.node_outputs)
        for target_output, sibling_output in zip(new_outputs, sibling_outputs):
            if (target_output.parse_ok and target_output.answer_valid
                    and target_output.vote in {Vote.A, Vote.B}
                    and sibling_output.parse_ok and sibling_output.answer_valid
                    and sibling_output.vote in {Vote.A, Vote.B}):
                joint += 1
                conflict += target_output.vote is not sibling_output.vote

    reasons: list[str] = []
    if new_metric.accuracy <= old_metric.accuracy:
        reasons.append("node_accuracy_not_strictly_improved")
    if new_metric.support < min_support:
        reasons.append("support_below_minimum")
    decision = EvolutionDecision.REJECT if reasons else EvolutionDecision.ACCEPT
    evaluation = RefineEvaluation(
        node_id=node_id,
        root_node_id=root_id,
        old_node=old_metric,
        new_node=new_metric,
        node_accuracy_delta=new_metric.accuracy - old_metric.accuracy,
        corrected_sample_ids=corrected,
        harmed_sample_ids=harmed,
        transition_counts=dict(sorted(transitions.items())),
        new_scope_expansion=sum(expansion),
        new_scope_expansion_wrong=expansion_wrong,
        subtree_scope_support=subtree_support,
        old_subtree_accuracy=sum(old_sub_correct) / subtree_support,
        new_subtree_accuracy=sum(new_sub_correct) / subtree_support,
        subtree_accuracy_delta=(sum(new_sub_correct) - sum(old_sub_correct)) / subtree_support,
        old_subtree_coverage=_coverage(old_sub_votes),
        new_subtree_coverage=_coverage(new_sub_votes),
        subtree_corrected_sample_ids=subtree_corrected,
        subtree_harmed_sample_ids=subtree_harmed,
        old_m1_accuracy=_accuracy(before_answers, gold),
        new_m1_accuracy=_accuracy(after_answers, gold),
        m1_accuracy_delta=_accuracy(after_answers, gold) - _accuracy(before_answers, gold),
        old_m1_coverage=_coverage(before_answers),
        new_m1_coverage=_coverage(after_answers),
        m1_corrected_sample_ids=m1_corrected,
        m1_harmed_sample_ids=m1_harmed,
        sibling_joint_decisive=joint,
        sibling_conflict_count=conflict,
        decision=decision,
        reasons=tuple(reasons),
    )
    return evaluation, before_execution, after_execution

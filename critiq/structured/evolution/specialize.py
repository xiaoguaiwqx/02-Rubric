"""Pure Specialize semantics, parsing, patching, artifact assembly and diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from critiq.utils import PairData, parse_json

from ..aggregation import aggregate_child_subtrees, aggregate_flat_votes
from ..dual_backend import OfflinePairwiseVoteBackend
from ..dual_executor import DualCascadeExecutor, replay_dual_execution_trace
from ..dual_worker import PairwisePredictionOutput
from ..executor import ExecutionConfig, StructuredSystemVariant
from ..judgement import FinalPreference, Vote
from ..schema import RubricCriterionSnapshot, RubricEdge, RubricNode, StructuredRubric
from ..semantics import EdgeCondition
from ..telemetry import ModelCallMetrics
from ..worker_output import StructuredCriterionSnapshot
from .patch import apply_rubric_patch, diff_rubrics, plan_artifact_refresh
from .specialize_types import (
    ChildCriterionProposal,
    ChildDiagnostic,
    ClusterProposal,
    ErrorSignature,
    SemanticCluster,
    SpecializeCandidate,
    SpecializeEvaluation,
    SpecializeManagerRequestSpec,
    SpecializeTriggerDecision,
    SubtreeDiagnostic,
)
from .types import (
    CandidateAcceptancePolicy,
    CandidateEvaluation,
    EditCandidate,
    EvolutionContext,
    EvolutionDecision,
    OperatorKind,
    RubricPatch,
    evaluate_candidate,
)


class SpecializeParseError(ValueError):
    """A Specialize Manager response violated its strict schema."""


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SpecializeParseError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _parse(raw: object, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, str):
        raise SpecializeParseError(f"{label} response must be a string")
    try:
        value = parse_json(raw)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise SpecializeParseError(f"failed to parse {label} JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SpecializeParseError(f"{label} JSON must be an object")
    return value


def parse_error_signature_response(raw: object, *, expected_sample_id: str) -> ErrorSignature:
    fields = {"sample_id", "task_pattern", "visual_focus", "candidate_difference",
              "parent_failure", "suggested_subdomain"}
    value = _exact(_parse(raw, "error signature"), fields, "error signature")
    try:
        result = ErrorSignature(**{field: value[field] for field in fields})
    except (TypeError, ValueError) as exc:
        raise SpecializeParseError(str(exc)) from exc
    if result.sample_id != expected_sample_id:
        raise SpecializeParseError("error signature sample_id does not match request")
    return result


def parse_split_failure_attribution_response(raw: object) -> dict[str, Any]:
    """Parse the Manager's evidence-grounded natural-language Split diagnosis."""

    fields = {"summary", "failure_categories", "details", "avoid_next_time"}
    value = _exact(_parse(raw, "split failure attribution"), fields,
                   "split failure attribution")
    summary = value["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise SpecializeParseError("split failure attribution summary must be non-empty")
    result: dict[str, Any] = {"summary": summary.strip()}
    for field in ("failure_categories", "details", "avoid_next_time"):
        items = value[field]
        if (not isinstance(items, list) or not items
                or any(not isinstance(item, str) or not item.strip() for item in items)):
            raise SpecializeParseError(f"{field} must be a non-empty string list")
        cleaned = [item.strip() for item in items]
        if len(set(cleaned)) != len(cleaned):
            raise SpecializeParseError(f"{field} must not contain duplicates")
        result[field] = cleaned
    if any(re.fullmatch(r"[a-z][a-z0-9_]*", item) is None
           for item in result["failure_categories"]):
        raise SpecializeParseError("failure_categories must use lower_snake_case")
    return result

def parse_cluster_proposal_response(
    raw: object,
    *,
    expected_sample_ids: Sequence[str],
    min_cluster_size: int,
    max_clusters: int,
    attempt_count: int,
    metrics: ModelCallMetrics,
    request_spec: SpecializeManagerRequestSpec,
) -> ClusterProposal:
    stored_raw = raw.strip() if isinstance(raw, str) else raw
    value = _exact(_parse(raw, "cluster proposal"),
                   {"clusters", "unclustered_sample_ids"}, "cluster proposal")
    if (not isinstance(value["clusters"], list)
            or not isinstance(value["unclustered_sample_ids"], list)):
        raise SpecializeParseError("cluster proposal fields must be lists")
    try:
        result = ClusterProposal(
            tuple(SemanticCluster.from_dict(item) for item in value["clusters"]),
            tuple(value["unclustered_sample_ids"]), stored_raw, attempt_count,
            metrics, request_spec)
    except (TypeError, ValueError) as exc:
        raise SpecializeParseError(str(exc)) from exc
    expected = set(expected_sample_ids)
    assigned = [sample_id for cluster in result.clusters for sample_id in cluster.sample_ids]
    assigned_set = set(assigned)
    unclustered_set = set(result.unclustered_sample_ids)
    observed = assigned_set | unclustered_set
    issues = []
    if len(result.clusters) > max_clusters:
        issues.append(
            f"cluster count {len(result.clusters)} exceeds maximum {max_clusters}")
    undersized = [
        f"{cluster.cluster_id}={len(cluster.sample_ids)}"
        for cluster in result.clusters if len(cluster.sample_ids) < min_cluster_size
    ]
    if undersized:
        issues.append(f"clusters smaller than minimum {min_cluster_size}: {undersized}")
    duplicates = sorted(sample_id for sample_id in assigned_set
                        if assigned.count(sample_id) > 1)
    if duplicates:
        issues.append(f"duplicate sample IDs: {duplicates}")
    overlap = sorted(assigned_set & unclustered_set)
    if overlap:
        issues.append(f"clustered/unclustered overlap: {overlap}")
    missing = sorted(expected - observed)
    if missing:
        issues.append(f"missing sample IDs: {missing}")
    unknown = sorted(observed - expected)
    if unknown:
        issues.append(f"unknown sample IDs: {unknown}")
    if issues:
        raise SpecializeParseError("; ".join(issues))
    return result


def parse_child_proposal_response(
    raw: object,
    *,
    cluster: SemanticCluster,
    attempt_count: int,
    metrics: ModelCallMetrics,
    request_spec: SpecializeManagerRequestSpec,
) -> ChildCriterionProposal:
    fields = {"criterion_name", "description", "rationale", "representative_sample_ids"}
    stored_raw = raw.strip() if isinstance(raw, str) else raw
    value = _exact(_parse(raw, "child proposal"), fields, "child proposal")
    if not isinstance(value["representative_sample_ids"], list):
        raise SpecializeParseError("representative_sample_ids must be a list")
    try:
        result = ChildCriterionProposal(
            cluster.cluster_id, value["criterion_name"], value["description"], value["rationale"],
            tuple(value["representative_sample_ids"]), stored_raw, attempt_count, metrics, request_spec,
        )
    except (TypeError, ValueError) as exc:
        raise SpecializeParseError(str(exc)) from exc
    if not set(result.representative_sample_ids).issubset(cluster.sample_ids):
        raise SpecializeParseError("representative samples must belong to the cluster")
    return result


def detect_specialize_trigger(
    context: EvolutionContext,
    parent_node_id: str,
    thresholds: Mapping[str, Any],
) -> SpecializeTriggerDecision:
    parent = context.rubric.get_node(parent_node_id)
    feedback = context.feedback.nodes[parent_node_id]
    if feedback.criterion_name != parent.criterion.name:
        raise ValueError("parent feedback criterion mismatch")
    required = {"tau_split", "tau_cov_high", "N_min_support", "N_min_wrong",
                "N_min_cluster", "max_children"}
    if not required.issubset(thresholds):
        raise ValueError("Specialize thresholds are incomplete")
    current = len(context.rubric.child_edges(parent_node_id))
    remaining = max(0, thresholds["max_children"] - current)
    reasons: list[str] = []
    if feedback.accuracy >= thresholds["tau_split"]:
        reasons.append("accuracy_not_below_tau_split")
    if feedback.coverage <= thresholds["tau_cov_high"]:
        reasons.append("coverage_below_tau_cov_high")
    if feedback.support < thresholds["N_min_support"]:
        reasons.append("support_below_minimum")
    if feedback.wrong < thresholds["N_min_wrong"]:
        reasons.append("wrong_below_minimum")
    if remaining < 2:
        reasons.append("remaining_child_capacity_below_two")
    wrong_ids = tuple(item.sample_id for item in feedback.errors if item.outcome == "wrong")
    if len(wrong_ids) != feedback.wrong:
        raise ValueError("feedback decisive-wrong references are inconsistent")
    return SpecializeTriggerDecision(parent_node_id, not reasons, tuple(reasons), wrong_ids,
                                     current, remaining)


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result or "child"


def deterministic_child_node_id(parent_node_id: str, cluster_id: str,
                                criterion_name: str) -> str:
    identity = f"{parent_node_id}\0{cluster_id}\0{criterion_name}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{parent_node_id}__{_slug(cluster_id)}__{suffix}"


def _image_sha256(row: Mapping[str, Any], image_field: str = "image_path") -> str:
    value = row.get(image_field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("representative sample lacks image_path")
    return hashlib.sha256(Path(value).read_bytes()).hexdigest()


def build_specialize_candidate(
    context: EvolutionContext,
    parent_node_id: str,
    cluster_proposal: ClusterProposal,
    children: Sequence[ChildCriterionProposal],
    dataset: Sequence[Mapping[str, Any]],
    signatures: Mapping[str, ErrorSignature],
) -> SpecializeCandidate:
    parent = context.rubric.get_node(parent_node_id)
    clusters = {item.cluster_id: item for item in cluster_proposal.clusters}
    children = tuple(children)
    if len(clusters) < 2 or len(children) != len(clusters):
        raise ValueError("each of at least two clusters must have exactly one child")
    if {item.cluster_id for item in children} != set(clusters):
        raise ValueError("children do not cover cluster proposal")
    names = [item.criterion_name for item in children]
    descriptions = [item.description for item in children]
    existing_names = {node.criterion.name for node in context.rubric.nodes.values()}
    if len(set(names)) != len(names) or existing_names & set(names):
        raise ValueError("child criterion names must be globally unique")
    if len(set(descriptions)) != len(descriptions):
        raise ValueError("child descriptions must be unique")
    rows = {str(row["sample_id"]): row for row in dataset}
    upserts: list[RubricNode] = []
    edges: list[RubricEdge] = []
    mapping: dict[str, str] = {}
    for child in children:
        cluster = clusters[child.cluster_id]
        node_id = deterministic_child_node_id(parent_node_id, cluster.cluster_id,
                                              child.criterion_name)
        mapping[cluster.cluster_id] = node_id
        examples = []
        for sample_id in child.representative_sample_ids:
            row = rows[sample_id]
            signature = signatures[sample_id]
            examples.append({
                "sample_id": sample_id,
                "question": row["question"], "A": row["A"], "B": row["B"],
                "gold": row["answer"], "signature_summary": signature.parent_failure,
                "image_sha256": _image_sha256(row),
            })
        upserts.append(RubricNode(
            node_id=node_id,
            criterion=RubricCriterionSnapshot(child.criterion_name, child.description,
                                               parent.criterion.score),
            examples=tuple(examples),
            lineage={"operator": "split", "parent_node_id": parent_node_id,
                     "cluster": cluster.to_dict(), "rationale": child.rationale,
                     "split_version": "1.0.0"},
        ))
        edges.append(RubricEdge(parent_node_id, node_id, EdgeCondition.ALWAYS))
    patch = RubricPatch(context.rubric.rubric_sha256, upsert_nodes=tuple(upserts),
                        add_edges=tuple(edges), root_ids=context.rubric.root_ids)
    payload = json.dumps(patch.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    edit = EditCandidate(f"split_{hashlib.sha256(payload).hexdigest()[:12]}",
                         OperatorKind.SPLIT,
                         f"Split {parent.criterion.name} into {len(children)} error subdomains",
                         patch)
    candidate = SpecializeCandidate(parent_node_id, edit, cluster_proposal, children, mapping)
    after = apply_rubric_patch(context.rubric, patch)
    diff = diff_rubrics(context.rubric, after)
    if diff.modified_node_ids or diff.removed_node_ids or diff.removed_edges or diff.root_ids_changed:
        raise ValueError("Split patch may only add children and edges")
    if any(edge.condition is not EdgeCondition.ALWAYS for edge in diff.added_edges):
        raise ValueError("Split v1 edges must be ALWAYS")
    refresh = plan_artifact_refresh(context.rubric, after)
    if refresh.refresh_gate_node_ids or refresh.remove_gate_criterion_names or refresh.router_stale:
        raise ValueError("Split v1 must not refresh Gate or Router")
    return candidate


def assemble_specialized_pairwise_prediction(
    base_prediction: PairwisePredictionOutput,
    child_prediction: PairwisePredictionOutput,
    candidate_rubric: StructuredRubric,
) -> PairwisePredictionOutput:
    if base_prediction.sample_ids != child_prediction.sample_ids:
        raise ValueError("base and child sample IDs differ")
    if base_prediction.sample_fingerprints != child_prediction.sample_fingerprints:
        raise ValueError("base and child sample fingerprints differ")
    if base_prediction.request_spec != child_prediction.request_spec:
        raise ValueError("base and child Pairwise request specs differ")
    base_criteria = {item.name: item.description for item in base_prediction.criteria}
    child_criteria = {item.name: item.description for item in child_prediction.criteria}
    if set(base_criteria) & set(child_criteria):
        raise ValueError("base and child criteria overlap")
    rubric_criteria = {node.criterion.name: node.criterion.description
                       for node in candidate_rubric.nodes.values()}
    if base_criteria | child_criteria != rubric_criteria:
        raise ValueError("combined criteria do not exactly cover candidate rubric")
    rows = []
    for base_row, child_row in zip(base_prediction.node_outputs, child_prediction.node_outputs):
        if set(base_row) != set(base_criteria) or set(child_row) != set(child_criteria):
            raise ValueError("Pairwise output row has missing or extra criteria")
        rows.append({**base_row, **child_row})
    ordered_nodes = tuple(candidate_rubric.get_node(node_id)
                          for node_id in candidate_rubric.preorder_node_ids())
    criteria = tuple(StructuredCriterionSnapshot(node.criterion.name, node.criterion.description)
                     for node in ordered_nodes)
    answers = tuple(aggregate_flat_votes(output.vote for output in row.values()) for row in rows)
    return PairwisePredictionOutput(base_prediction.sample_ids,
                                    base_prediction.sample_fingerprints,
                                    criteria, tuple(rows), answers,
                                    base_prediction.request_spec)


def project_pairwise_prediction(prediction: PairwisePredictionOutput,
                                rubric: StructuredRubric) -> PairwisePredictionOutput:
    expected = tuple(rubric.get_node(node_id).criterion.name
                     for node_id in rubric.preorder_node_ids())
    descriptions = {node.criterion.name: node.criterion.description
                    for node in rubric.nodes.values()}
    available = {item.name: item.description for item in prediction.criteria}
    if any(available.get(name) != descriptions[name] for name in expected):
        raise ValueError("prediction cannot be projected to rubric")
    rows = tuple({name: row[name] for name in expected} for row in prediction.node_outputs)
    answers = tuple(aggregate_flat_votes(output.vote for output in row.values()) for row in rows)
    return PairwisePredictionOutput(prediction.sample_ids, prediction.sample_fingerprints,
                                    tuple(StructuredCriterionSnapshot(name, descriptions[name])
                                          for name in expected), rows, answers,
                                    prediction.request_spec)


def execute_offline_m1(rubric: StructuredRubric, prediction: PairwisePredictionOutput,
                       dataset: Sequence[PairData]):
    executor = DualCascadeExecutor(rubric, OfflinePairwiseVoteBackend(prediction, rubric))
    execution = executor.execute_batch(dataset,
        ExecutionConfig(StructuredSystemVariant.M1_ALL_ROOTS_CASCADE))
    answers = tuple(trace.final_preference for trace in execution.traces)
    for trace in execution.traces:
        if replay_dual_execution_trace(trace, rubric) is not trace.final_preference:
            raise ValueError("M1 trace replay mismatch")
    return execution, answers


def _accuracy(values: Sequence[FinalPreference | Vote], gold: Sequence[str]) -> float:
    return sum(value.value == target for value, target in zip(values, gold)) / len(gold)


def _coverage(values: Sequence[FinalPreference | Vote]) -> float:
    return sum(value.value in {"A", "B"} for value in values) / len(values)


def evaluate_specialize_candidate(
    *,
    before_rubric: StructuredRubric,
    after_rubric: StructuredRubric,
    combined_prediction: PairwisePredictionOutput,
    child_prediction: PairwisePredictionOutput,
    dataset: Sequence[PairData],
    parent_node_id: str,
    cluster_proposal: ClusterProposal,
    candidate: SpecializeCandidate,
    policy: CandidateAcceptancePolicy,
) -> tuple[SpecializeEvaluation, Any, Any]:
    before_prediction = project_pairwise_prediction(combined_prediction, before_rubric)
    before_execution, before_answers = execute_offline_m1(before_rubric, before_prediction, dataset)
    after_execution, after_answers = execute_offline_m1(after_rubric, combined_prediction, dataset)
    gold = tuple(str(row["answer"]) for row in dataset)
    child_outputs = [output for row in child_prediction.node_outputs for output in row.values()]
    valid_rate = sum(output.parse_ok and output.answer_valid for output in child_outputs) / len(child_outputs)
    base_valid = sum(output.parse_ok and output.answer_valid
                     for row in before_prediction.node_outputs for output in row.values()) / (
                         len(before_prediction.node_outputs) * len(before_prediction.criteria))
    m1_diagnostic = evaluate_candidate(
        before_answers, after_answers, gold, before_valid_rate=base_valid,
        final_valid_rate=valid_rate, policy=policy)

    parent = before_rubric.get_node(parent_node_id)
    parent_name = parent.criterion.name
    parent_outputs = tuple(row[parent_name] for row in combined_prediction.node_outputs)
    parent_scope = tuple(
        output.parse_ok and output.answer_valid and output.vote in {Vote.A, Vote.B}
        for output in parent_outputs)
    parent_support = sum(parent_scope)
    if parent_support == 0:
        raise ValueError("Split parent has zero support and no applicability domain")
    parent_correct = tuple(
        parent_scope[index] and output.vote.value == gold[index]
        for index, output in enumerate(parent_outputs))
    parent_accuracy = sum(parent_correct) / parent_support

    cluster_by_name = {}
    for child in candidate.children:
        cluster_by_name[child.criterion_name] = next(
            cluster for cluster in cluster_proposal.clusters if cluster.cluster_id == child.cluster_id)
    child_diagnostics: list[ChildDiagnostic] = []
    row_index = {str(row["sample_id"]): index for index, row in enumerate(dataset)}
    for child in candidate.children:
        outputs = tuple(row[child.criterion_name] for row in combined_prediction.node_outputs)
        decisive = [
            parent_scope[index] and output.parse_ok and output.answer_valid
            and output.vote in {Vote.A, Vote.B}
            for index, output in enumerate(outputs)]
        correct = [
            decisive[index] and output.vote.value == gold[index]
            for index, output in enumerate(outputs)]
        support = sum(decisive)
        accuracy = sum(correct) / support if support else 0.0
        coverage = support / parent_support
        child_valid = sum(output.parse_ok and output.answer_valid for output in outputs) / len(outputs)
        cluster = cluster_by_name[child.criterion_name]
        cluster_indices = [row_index[sample_id] for sample_id in cluster.sample_ids
                           if parent_scope[row_index[sample_id]]]
        cluster_support = sum(decisive[index] for index in cluster_indices)
        cluster_correct = sum(correct[index] for index in cluster_indices)
        cluster_index_set = set(cluster_indices)
        non_target = [index for index, applicable in enumerate(parent_scope)
                      if applicable and index not in cluster_index_set]
        non_target_decisive = sum(decisive[index] for index in non_target)
        non_target_wrong = sum(decisive[index] and not correct[index] for index in non_target)
        non_target_abstain = sum(
            outputs[index].parse_ok and outputs[index].answer_valid
            and outputs[index].vote is Vote.ABSTAIN for index in non_target)
        node_id = candidate.node_id_by_cluster[child.cluster_id]
        child_rubric = apply_rubric_patch(after_rubric, RubricPatch(
            after_rubric.rubric_sha256, remove_node_ids=(node_id,), root_ids=after_rubric.root_ids))
        child_projected = project_pairwise_prediction(combined_prediction, child_rubric)
        _, leave_answers = execute_offline_m1(child_rubric, child_projected, dataset)
        after_correct = [value.value == target for value, target in zip(after_answers, gold)]
        leave_correct = [value.value == target for value, target in zip(leave_answers, gold)]
        # Per-child accuracy/coverage remain diagnostics only; Split acceptance is collective.
        fitness = accuracy
        child_diagnostics.append(ChildDiagnostic(
            node_id=node_id, criterion_name=child.criterion_name, support=support,
            accuracy=accuracy, coverage=coverage, valid_rate=child_valid, fitness=fitness,
            cluster_support=cluster_support,
            cluster_accuracy=cluster_correct / cluster_support if cluster_support else 0.0,
            non_target_decisive=non_target_decisive, non_target_wrong=non_target_wrong,
            non_target_abstain=non_target_abstain,
            corrected_count=sum(new and not old for new, old in zip(after_correct, leave_correct)),
            harmed_count=sum(not new and old for new, old in zip(after_correct, leave_correct)),
            leave_one_out_m1_delta=_accuracy(after_answers, gold) - _accuracy(leave_answers, gold),
        ))

    parent_votes = tuple(output.vote for output in parent_outputs)
    after_trace_by_sample = {trace.sample_id: trace for trace in after_execution.traces}
    subtree_votes = []
    subtree_conflicts = 0
    for index, row in enumerate(dataset):
        trace = after_trace_by_sample[str(row["sample_id"])]
        node_trace = next(node for node in trace.nodes if node.node_id == parent_node_id)
        subtree_votes.append(node_trace.subtree_vote)
        if (parent_scope[index]
                and node_trace.local_vote in {Vote.A, Vote.B}
                and node_trace.subtree_vote in {Vote.A, Vote.B}
                and node_trace.local_vote is not node_trace.subtree_vote):
            subtree_conflicts += 1

    specialized_correct = tuple(
        parent_scope[index] and vote.value == gold[index]
        for index, vote in enumerate(subtree_votes))
    specialized_accuracy = sum(specialized_correct) / parent_support
    accuracy_delta = specialized_accuracy - parent_accuracy
    corrected = tuple(
        str(row["sample_id"])
        for index, row in enumerate(dataset)
        if parent_scope[index] and not parent_correct[index] and specialized_correct[index])
    harmed = tuple(
        str(row["sample_id"])
        for index, row in enumerate(dataset)
        if parent_scope[index] and parent_correct[index] and not specialized_correct[index])

    # Split is a local operator: accept the complete child set exactly when the
    # parent+children subtree does not regress on the parent's frozen domain.
    reasons: list[str] = []
    if specialized_accuracy < parent_accuracy:
        reasons.append("specialized_accuracy_below_parent")
    paired = CandidateEvaluation(
        m1_diagnostic.before_accuracy, m1_diagnostic.after_accuracy,
        m1_diagnostic.accuracy_delta, m1_diagnostic.before_coverage,
        m1_diagnostic.after_coverage, m1_diagnostic.corrected_count,
        m1_diagnostic.harmed_count,
        m1_diagnostic.before_valid_rate, m1_diagnostic.final_valid_rate,
        EvolutionDecision.REJECT if reasons else EvolutionDecision.ACCEPT,
        tuple(reasons))

    sibling_pairs = list(combinations(candidate.children, 2))
    agreements = []
    conflict_samples: set[int] = set()
    for left, right in sibling_pairs:
        for index, row in enumerate(combined_prediction.node_outputs):
            if not parent_scope[index]:
                continue
            a, b = row[left.criterion_name], row[right.criterion_name]
            if (a.parse_ok and a.answer_valid and a.vote in {Vote.A, Vote.B}
                    and b.parse_ok and b.answer_valid and b.vote in {Vote.A, Vote.B}):
                agreements.append(a.vote is b.vote)
                if a.vote is not b.vote:
                    conflict_samples.add(index)
    risks = []
    if any(item.accuracy < parent_accuracy for item in child_diagnostics):
        risks.append("child_accuracy_below_parent")
    if specialized_accuracy < parent_accuracy:
        risks.append("subtree_local_regression")
    if m1_diagnostic.decision is EvolutionDecision.REJECT:
        risks.append("full_m1_guardrail_failed")
    subtree = SubtreeDiagnostic(
        parent_accuracy, _coverage(parent_votes), specialized_accuracy,
        _coverage(subtree_votes), corrected, harmed,
        sum(agreements) / len(agreements) if agreements else 0.0,
        len(conflict_samples), subtree_conflicts, tuple(risks),
    )
    # Split artifacts expose ACC-only metrics; length and coverage remain diagnostics.
    return SpecializeEvaluation(
        paired, tuple(child_diagnostics), subtree, valid_rate, parent_accuracy,
        specialized_accuracy, accuracy_delta), before_execution, after_execution

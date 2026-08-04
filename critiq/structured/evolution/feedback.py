"""All-node Pairwise feedback extraction for Phase 5 evolution."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Mapping, Sequence

from ..dual_worker import DualWorkerRequestSpec, PairwisePredictionOutput
from ..judgement import FinalPreference, Vote
from ..schema import StructuredRubric
from ..worker_output import structured_input_fingerprint
from .types import ErrorSampleRef, NodeFeedback, RubricFeedback


def criterion_fitness(
    accuracy: float,
    coverage: float,
    description_length: int,
    *,
    alpha: float = 2.0,
    lambda_: float = 0.1,
    len_max: int = 2000,
) -> float:
    """Length-regularized diagnostic fitness from the idea document."""

    if not 0.0 <= accuracy <= 1.0 or not 0.0 <= coverage <= 1.0:
        raise ValueError("accuracy and coverage must be in [0, 1]")
    if isinstance(description_length, bool) or description_length < 0:
        raise ValueError("description_length must be non-negative")
    if isinstance(len_max, bool) or not isinstance(len_max, int) or len_max < 1:
        raise ValueError("len_max must be a positive integer")
    if not math.isfinite(alpha) or not math.isfinite(lambda_) or alpha < 0 or lambda_ < 0:
        raise ValueError("fitness parameters must be finite and non-negative")
    normalized = description_length / len_max
    return accuracy - lambda_ * (math.exp(alpha * normalized) - 1.0) * coverage


def _validate_inputs(
    rubric: StructuredRubric,
    prediction: PairwisePredictionOutput,
    dataset: Sequence[Mapping[str, object]],
    m1_answers: Sequence[FinalPreference],
    expected_request_spec: DualWorkerRequestSpec,
) -> None:
    if len(dataset) != len(prediction.sample_ids) or len(m1_answers) != len(dataset):
        raise ValueError("dataset, prediction and M1 lengths differ")
    sample_ids = tuple(row.get("sample_id") for row in dataset)
    if sample_ids != prediction.sample_ids:
        raise ValueError("dataset and prediction sample order differ")
    if not isinstance(expected_request_spec, DualWorkerRequestSpec):
        raise TypeError("expected_request_spec must be DualWorkerRequestSpec")
    if prediction.request_spec != expected_request_spec:
        raise ValueError("Pairwise artifact request identity mismatch")
    actual_fingerprints = tuple(
        structured_input_fingerprint(
            row,
            image_field=expected_request_spec.image_field,
            question_field=expected_request_spec.question_field,
            sample_id_field=expected_request_spec.sample_id_field,
            max_data_chars=expected_request_spec.max_data_chars,
            encode_local_image=expected_request_spec.encode_local_image,
        )
        for row in dataset
    )
    if actual_fingerprints != prediction.sample_fingerprints:
        raise ValueError("dataset and Pairwise artifact sample fingerprints differ")
    if any(row.get("answer") not in {"A", "B"} for row in dataset):
        raise ValueError("dataset gold answers must be A or B")
    expected = {
        node.criterion.name: node.criterion.description
        for node in rubric.nodes.values()
    }
    actual = {item.name: item.description for item in prediction.criteria}
    if actual != expected:
        raise ValueError("Pairwise artifact does not exactly cover rubric criteria")
    if any(not isinstance(value, FinalPreference) for value in m1_answers):
        raise TypeError("M1 answers must be FinalPreference values")


def extract_rubric_feedback(
    rubric: StructuredRubric,
    prediction: PairwisePredictionOutput,
    dataset: Sequence[Mapping[str, object]],
    m1_answers: Sequence[FinalPreference],
    *,
    expected_request_spec: DualWorkerRequestSpec,
    alpha: float = 2.0,
    lambda_: float = 0.1,
    len_max: int = 2000,
) -> RubricFeedback:
    """Compute feedback without hiding nodes skipped by conditional traversal."""

    _validate_inputs(rubric, prediction, dataset, m1_answers, expected_request_spec)
    sample_count = len(dataset)
    node_feedback: dict[str, NodeFeedback] = {}

    for node_id in rubric.preorder_node_ids():
        node = rubric.get_node(node_id)
        name = node.criterion.name
        correct = wrong = abstain = answer_invalid = parse_invalid = 0
        errors: list[ErrorSampleRef] = []
        for index, row in enumerate(dataset):
            output = prediction.node_outputs[index][name]
            gold = str(row["answer"])
            if not output.parse_ok:
                parse_invalid += 1
                outcome = "parse_invalid"
            elif not output.answer_valid:
                answer_invalid += 1
                outcome = "answer_invalid"
            elif output.vote is Vote.ABSTAIN:
                abstain += 1
                outcome = "abstain"
            elif output.vote.value == gold:
                correct += 1
                continue
            else:
                wrong += 1
                outcome = "wrong"
            errors.append(
                ErrorSampleRef(
                    sample_id=str(row["sample_id"]),
                    criterion_name=name,
                    vote=output.vote,
                    gold=gold,
                    outcome=outcome,
                    thought=output.thought,
                )
            )
        support = correct + wrong
        accuracy = correct / support if support else 0.0
        coverage = support / sample_count
        valid_rate = (support + abstain) / sample_count
        length = len(node.criterion.description)
        node_feedback[node_id] = NodeFeedback(
            node_id=node_id,
            criterion_name=name,
            sample_count=sample_count,
            support=support,
            correct=correct,
            wrong=wrong,
            abstain=abstain,
            answer_invalid=answer_invalid,
            parse_invalid=parse_invalid,
            accuracy=accuracy,
            coverage=coverage,
            valid_rate=valid_rate,
            description_length=length,
            fitness=criterion_fitness(
                accuracy, coverage, length,
                alpha=alpha, lambda_=lambda_, len_max=len_max,
            ),
            errors=tuple(errors),
        )

    agreements: dict[str, float] = {}
    criterion_names = tuple(
        rubric.get_node(node_id).criterion.name
        for node_id in rubric.preorder_node_ids()
    )
    for left, right in combinations(criterion_names, 2):
        common = []
        for row in prediction.node_outputs:
            a, b = row[left], row[right]
            if (
                a.parse_ok and a.answer_valid and a.vote in {Vote.A, Vote.B}
                and b.parse_ok and b.answer_valid and b.vote in {Vote.A, Vote.B}
            ):
                common.append(a.vote is b.vote)
        agreements[f"{left}::{right}"] = sum(common) / len(common) if common else 0.0

    gaps: list[str] = []
    failures: list[str] = []
    conflicts: list[str] = []
    for index, (row, m1_answer) in enumerate(zip(dataset, m1_answers)):
        gold = str(row["answer"])
        has_correct_node = any(
            output.parse_ok
            and output.answer_valid
            and output.vote in {Vote.A, Vote.B}
            and output.vote.value == gold
            for output in prediction.node_outputs[index].values()
        )
        m1_correct = m1_answer in {FinalPreference.A, FinalPreference.B} and m1_answer.value == gold
        sample_id = str(row["sample_id"])
        if not has_correct_node:
            gaps.append(sample_id)
        if not m1_correct:
            failures.append(sample_id)
            if has_correct_node:
                conflicts.append(sample_id)

    return RubricFeedback(
        rubric_sha256=rubric.rubric_sha256,
        sample_ids=prediction.sample_ids,
        nodes=node_feedback,
        agreement_by_pair=agreements,
        rubric_gap_sample_ids=tuple(gaps),
        cascade_failure_sample_ids=tuple(failures),
        aggregation_conflict_sample_ids=tuple(conflicts),
    )

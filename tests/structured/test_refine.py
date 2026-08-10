"""Refine v1 trigger, proposal, patch and self-competition tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from critiq.agent import AgentCallMetrics
from critiq.structured import (
    DualWorkerRequestSpec,
    ErrorSampleRef,
    EvolutionContext,
    EvolutionDecision,
    ModelCallMetrics,
    NodeFeedback,
    OperatorKind,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    RefineManager,
    RefineParseError,
    RefineCandidate,
    RefineEvaluation,
    RubricCriterionSnapshot,
    RubricFeedback,
    RubricNode,
    SpecializeManagerRequestSpec,
    StructuredCriterionSnapshot,
    StructuredRubric,
    Vote,
    apply_rubric_patch,
    assemble_refined_pairwise_prediction,
    build_refine_candidate,
    detect_refine_trigger,
    evaluate_refine_candidate,
    parse_refine_proposal_response,
    structured_input_fingerprint,
)
from critiq.structured.aggregation import aggregate_flat_votes
from experiments.evolving_structured_rubrics.run_rubric_evolution import (
    _config,
    _phase5_config_view,
)
from experiments.evolving_structured_rubrics.refine_evolution import (
    _manifest_path,
    _same_pairwise_scientific_identity,
    _smoke_go_no_go,
)


def _rubric(description: str = "Old description") -> StructuredRubric:
    node = RubricNode(
        "root",
        RubricCriterionSnapshot("criterion", description, 1.0),
        examples=({"sample_id": "example"},),
        lineage={"source": "test"},
    )
    return StructuredRubric({"root": node}, (), ("root",))


def _feedback(rubric: StructuredRubric, *, accuracy: float = .7,
              coverage: float = .8, support: int = 20) -> RubricFeedback:
    correct = round(accuracy * support)
    wrong = support - correct
    total = round(support / coverage)
    actual_accuracy = correct / support
    actual_coverage = support / total
    node = rubric.get_node("root")
    errors = tuple(
        ErrorSampleRef(f"s{i}", node.criterion.name, Vote.B, "A", "wrong", "thought")
        for i in range(wrong)
    ) + tuple(
        ErrorSampleRef(f"a{i}", node.criterion.name, Vote.ABSTAIN, "A",
                       "abstain", "thought")
        for i in range(total - support)
    )
    item = NodeFeedback(
        "root", node.criterion.name, total, support, correct, wrong,
        total - support, 0, 0, actual_accuracy, actual_coverage, 1.0, 16, .2, errors,
    )
    return RubricFeedback(
        rubric.rubric_sha256, tuple(f"s{i}" for i in range(total)),
        {"root": item}, {}, (), (), (),
    )


def _manager_spec() -> SpecializeManagerRequestSpec:
    return SpecializeManagerRequestSpec.from_prompt(
        model="manager", backend_id="pool", prompt="prompt",
        decoding_config={"temperature": 0}, prompt_version="1.0.0",
        parser_version="1.0.0",
    )


def _description() -> str:
    return (
        "Criterion focus: Judge only the named visual relation.\n\n"
        "Applicable only when: the relation is visible and differs.\n\n"
        "Not applicable when: visual evidence is insufficient or answers tie.\n\n"
        "Decision rule: Output A/B only from this criterion; return None when "
        "not applicable or visual evidence is insufficient. Do not use overall answer "
        "quality and do not vote using other criteria."
    )


def _proposal(rubric: StructuredRubric):
    response = json.dumps({
        "criterion_name": "criterion",
        "description": _description(),
        "failure_analysis": ["The old boundary was broad."],
        "rationale": "Narrow the applicability boundary.",
        "representative_sample_ids": ["s0"],
    })
    return parse_refine_proposal_response(
        response, node=rubric.get_node("root"),
        allowed_representative_ids=("s0",), max_description_chars=1800,
        attempt_count=1, metrics=ModelCallMetrics(logical_evaluations=1),
        request_spec=_manager_spec(),
    )


def _vote(vote: Vote) -> PairwiseVoteOutput:
    answer = vote.value if vote in {Vote.A, Vote.B} else "None"
    return PairwiseVoteOutput(
        vote, True, json.dumps({"thought": "x", "answer": answer}),
        None, 1, "x", True,
    )


def _worker_spec() -> DualWorkerRequestSpec:
    return DualWorkerRequestSpec(
        "worker", "vllm-8001", "a" * 64, None, False,
        "image_path", "question", "sample_id", {"temperature": .5},
    )


def _rows(answers: tuple[str, ...]):
    return tuple({"sample_id": f"s{i}", "image_path": f"image-{i}.jpg",
                  "question": "Q", "A": "candidate A", "B": "candidate B",
                  "answer": answer} for i, answer in enumerate(answers))


def _prediction(description: str, votes: tuple[Vote, ...], dataset) -> PairwisePredictionOutput:
    criterion = StructuredCriterionSnapshot("criterion", description)
    rows = tuple({"criterion": _vote(vote)} for vote in votes)
    spec = _worker_spec()
    return PairwisePredictionOutput(
        tuple(f"s{i}" for i in range(len(votes))),
        tuple(structured_input_fingerprint(
            item, image_field=spec.image_field, question_field=spec.question_field,
            sample_id_field=spec.sample_id_field,
            max_data_chars=spec.max_data_chars,
            encode_local_image=spec.encode_local_image) for item in dataset),
        (criterion,), rows,
        tuple(aggregate_flat_votes((item["criterion"].vote,)) for item in rows),
        spec,
    )


class RefineTests(unittest.TestCase):
    def test_manager_retry_uses_parser_feedback_to_compress_description(self):
        rubric = _rubric()
        too_long = json.dumps({
            "criterion_name": "criterion",
            "description": _description() + "x" * 1800,
            "failure_analysis": ["broad"], "rationale": "repair",
            "representative_sample_ids": ["s0"],
        })
        valid = json.dumps({
            "criterion_name": "criterion", "description": _description(),
            "failure_analysis": ["broad"], "rationale": "repair",
            "representative_sample_ids": ["s0"],
        })

        class FakePool:
            backend_id = "pool"

            def __init__(self):
                self.responses = [too_long, valid]
                self.contents = []

            def call(self, content, **kwargs):
                self.contents.append(content)
                return self.responses.pop(0), AgentCallMetrics(api_attempts=1)

        pool = FakePool()
        manager = RefineManager(
            model="manager", backend_pool=pool, structured_max_retries=1)
        proposal = manager.generate(
            node=rubric.get_node("root"),
            evidence={"representative_sample_ids": ["s0"]},
            representative_rows=(), prior_failures=(),
            rubric_memory={"nodes": []}, max_description_chars=1800)
        self.assertEqual(proposal.attempt_count, 2)
        repair = pool.contents[1][0]["text"]
        self.assertIn("Structured retry correction", repair)
        self.assertIn("description exceeds 1800 characters", repair)
        self.assertIn("target at most 1400 characters", repair)
        self.assertIn('this exact allowlist', repair)
        self.assertIn('["s0"]', repair)

    def test_manager_retry_restricts_representatives_to_multimodal_allowlist(self):
        rubric = _rubric()
        invalid = json.dumps({
            "criterion_name": "criterion", "description": _description(),
            "failure_analysis": ["broad"], "rationale": "repair",
            "representative_sample_ids": ["text-only-case"],
        })
        valid = json.dumps({
            "criterion_name": "criterion", "description": _description(),
            "failure_analysis": ["broad"], "rationale": "repair",
            "representative_sample_ids": ["image-case"],
        })

        class FakePool:
            backend_id = "pool"

            def __init__(self):
                self.responses = [invalid, valid]
                self.contents = []

            def call(self, content, **kwargs):
                self.contents.append(content)
                return self.responses.pop(0), AgentCallMetrics(api_attempts=1)

        pool = FakePool()
        manager = RefineManager(
            model="manager", backend_pool=pool, structured_max_retries=1)
        proposal = manager.generate(
            node=rubric.get_node("root"),
            evidence={"representative_sample_ids": ["image-case"]},
            representative_rows=(), prior_failures=(),
            rubric_memory={"nodes": []}, max_description_chars=1800)
        self.assertEqual(proposal.representative_sample_ids, ("image-case",))
        repair = pool.contents[1][0]["text"]
        self.assertIn("representative_sample_ids must come from supplied evidence", repair)
        self.assertIn('["image-case"]', repair)
        self.assertIn("Do not select other sample IDs", repair)

    def test_manifest_paths_round_trip_from_json_strings(self):
        value = _manifest_path({"artifact": "output/example.json"}, "artifact")
        self.assertIsInstance(value, Path)
        self.assertEqual(value, Path("output/example.json"))
        with self.assertRaisesRegex(ValueError, "non-empty path"):
            _manifest_path({"artifact": ""}, "artifact")

    def test_pairwise_execution_route_may_change_but_science_may_not(self):
        frozen = _worker_spec()
        value = frozen.to_dict(); value["backend_id"] = "pool:new-8001-route"
        runtime = DualWorkerRequestSpec.from_dict(value)
        self.assertTrue(_same_pairwise_scientific_identity(frozen, runtime))
        value = runtime.to_dict(); value["decoding_config"] = {"temperature": .6}
        self.assertFalse(_same_pairwise_scientific_identity(
            frozen, DualWorkerRequestSpec.from_dict(value)))

    def test_smoke_report_waits_for_heldout_before_go(self):
        discovery = {"decision": "accept"}
        self.assertEqual(_smoke_go_no_go(discovery, None), "pending_heldout")
        self.assertEqual(_smoke_go_no_go(discovery, {
            "heldout_node_accuracy_delta": 0.0,
            "heldout_subtree_delta": 0.01,
        }), "go")
        self.assertEqual(_smoke_go_no_go(discovery, {
            "heldout_node_accuracy_delta": 0.01,
            "heldout_subtree_delta": -0.01,
        }), "stop_and_diagnose")
        self.assertEqual(_smoke_go_no_go({"decision": "reject"}, None),
                         "stop_and_diagnose")

    def test_phase5_config_accepts_refine_blocks_without_identity_drift(self):
        path = Path(
            "experiments/evolving_structured_rubrics/configs/"
            "rubric_evolution_phase5.example.json")
        config = _config(path.resolve())
        self.assertIn("refine_manager", config)
        self.assertIn("refine_operator", config)
        phase5 = _phase5_config_view(config)
        self.assertNotIn("refine_manager", phase5)
        self.assertNotIn("refine_operator", phase5)

    def test_trigger_uses_strict_accuracy_and_inclusive_coverage_support(self):
        thresholds = {"tau_acc": .55, "tau_refine": .8,
                      "tau_cov_high": .8, "N_min_support": 15}
        rubric = _rubric()
        context = EvolutionContext(rubric, _feedback(rubric))
        self.assertTrue(detect_refine_trigger(context, "root", thresholds).triggered)
        for accuracy in (.55, .8):
            context = EvolutionContext(
                rubric, _feedback(rubric, accuracy=accuracy, coverage=.8, support=20))
            self.assertFalse(detect_refine_trigger(context, "root", thresholds).triggered)
        context = EvolutionContext(
            rubric, _feedback(rubric, accuracy=.7, coverage=20 / 24, support=20))
        self.assertFalse(detect_refine_trigger(context, "root", thresholds).triggered)
        context = EvolutionContext(
            rubric, _feedback(rubric, accuracy=.5, coverage=.9, support=10))
        forced = detect_refine_trigger(context, "root", thresholds, forced=True)
        self.assertTrue(forced.triggered)
        self.assertTrue(forced.forced)
        self.assertGreater(len(forced.reasons), 0)

    def test_proposal_contract_and_patch_preserve_identity_and_topology(self):
        rubric = _rubric()
        proposal = _proposal(rubric)
        candidate = build_refine_candidate(
            EvolutionContext(rubric, _feedback(rubric)), "root", proposal)
        after = apply_rubric_patch(rubric, candidate.edit_candidate.patch)
        self.assertIs(candidate.edit_candidate.operator, OperatorKind.REFINE)
        self.assertEqual(after.root_ids, rubric.root_ids)
        self.assertEqual(after.edges, rubric.edges)
        self.assertEqual(after.get_node("root").criterion.name, "criterion")
        self.assertEqual(after.get_node("root").criterion.score, 1.0)
        self.assertEqual(after.get_node("root").examples,
                         rubric.get_node("root").examples)
        self.assertNotEqual(after.get_node("root").criterion.description,
                            rubric.get_node("root").criterion.description)
        self.assertEqual(RefineCandidate.from_dict(candidate.to_dict()), candidate)

    def test_parser_rejects_noop_name_change_missing_contract_and_length(self):
        rubric = _rubric()
        base = {
            "criterion_name": "criterion", "description": _description(),
            "failure_analysis": ["x"], "rationale": "y",
            "representative_sample_ids": ["s0"],
        }
        cases = []
        value = dict(base); value["criterion_name"] = "renamed"; cases.append(value)
        value = dict(base); value["description"] = "Old description"; cases.append(value)
        value = dict(base); value["description"] = "No required sections"; cases.append(value)
        value = dict(base); value["description"] = _description() + "x" * 1800; cases.append(value)
        for value in cases:
            with self.subTest(value=value["criterion_name"], length=len(value["description"])):
                with self.assertRaises(RefineParseError):
                    parse_refine_proposal_response(
                        json.dumps(value), node=rubric.get_node("root"),
                        allowed_representative_ids=("s0",), max_description_chars=1800,
                        attempt_count=1, metrics=ModelCallMetrics(logical_evaluations=1),
                        request_spec=_manager_spec())

    def test_self_competition_uses_each_criterion_own_support(self):
        rubric = _rubric()
        candidate = build_refine_candidate(
            EvolutionContext(rubric, _feedback(rubric)), "root", _proposal(rubric))
        after = apply_rubric_patch(rubric, candidate.edit_candidate.patch)
        rows = _rows(("A", "A", "A", "B"))
        old = _prediction("Old description", (Vote.A, Vote.B, Vote.ABSTAIN, Vote.B), rows)
        new_only = _prediction(_description(), (Vote.A, Vote.A, Vote.A, Vote.ABSTAIN), rows)
        combined = assemble_refined_pairwise_prediction(old, new_only, after, "root")
        evaluation, _, _ = evaluate_refine_candidate(
            before_rubric=rubric, after_rubric=after,
            before_prediction=old, combined_prediction=combined,
            candidate_prediction=new_only, dataset=rows, node_id="root", min_support=3)
        self.assertEqual(evaluation.old_node.support, 3)
        self.assertAlmostEqual(evaluation.old_node.accuracy, 2 / 3)
        self.assertEqual(evaluation.new_node.support, 3)
        self.assertEqual(evaluation.new_node.accuracy, 1.0)
        self.assertIs(evaluation.decision, EvolutionDecision.ACCEPT)
        self.assertEqual(RefineEvaluation.from_dict(evaluation.to_dict()), evaluation)

    def test_tie_and_support_collapse_reject_regardless_of_other_diagnostics(self):
        rubric = _rubric()
        candidate = build_refine_candidate(
            EvolutionContext(rubric, _feedback(rubric)), "root", _proposal(rubric))
        after = apply_rubric_patch(rubric, candidate.edit_candidate.patch)
        rows = _rows(("A", "B", "A"))
        old = _prediction("Old description", (Vote.A, Vote.B, Vote.B), rows)
        for votes, min_support, reason in (
                ((Vote.A, Vote.B, Vote.B), 1, "node_accuracy_not_strictly_improved"),
                ((Vote.A, Vote.ABSTAIN, Vote.ABSTAIN), 2, "support_below_minimum")):
            new_only = _prediction(_description(), votes, rows)
            combined = assemble_refined_pairwise_prediction(old, new_only, after, "root")
            result, _, _ = evaluate_refine_candidate(
                before_rubric=rubric, after_rubric=after,
                before_prediction=old, combined_prediction=combined,
                candidate_prediction=new_only, dataset=rows,
                node_id="root", min_support=min_support)
            self.assertIs(result.decision, EvolutionDecision.REJECT)
            self.assertIn(reason, result.reasons)

    def test_global_memory_is_required_by_manager(self):
        class FakePool:
            backend_id = "pool"

        manager = RefineManager(model="manager", backend_pool=FakePool())
        with self.assertRaisesRegex(ValueError, "requires rubric_memory"):
            manager.generate(
                node=_rubric().get_node("root"),
                evidence={"representative_sample_ids": ["s0"]},
                representative_rows=(), prior_failures=(), rubric_memory=None,
                max_description_chars=1800)


if __name__ == "__main__":
    unittest.main()

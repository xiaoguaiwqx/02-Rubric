"""Phase 6B Specialize operator semantics and artifact tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from critiq.specialize_prompts import (
    CHILD_GENERATION_PROMPT,
    CHILD_GENERATION_PROMPT_V1,
    CHILD_GENERATION_PROMPT_V2,
)

from critiq.agent import AgentCallMetrics
from experiments.evolving_structured_rubrics.run_rubric_evolution import (
    _activation_summary,
    _heldout_vote_metrics,
    _paired_heldout_comparison,
    _prior_split_failures,
    _visual_variant_rubric,
    _multimodal_child_config,
    _qwen35_signature_config,
    _split_set_summary,
)
from critiq.structured import (
    CandidateAcceptancePolicy,
    ClusterProposal,
    DualWorkerRequestSpec,
    EdgeCondition,
    EditCandidate,
    ErrorSampleRef,
    ErrorSignature,
    EvolutionContext,
    FinalPreference,
    ModelCallMetrics,
    NodeFeedback,
    OperatorKind,
    PairwisePredictionOutput,
    PairwiseVoteOutput,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricFeedback,
    RubricNode,
    RubricPatch,
    SemanticCluster,
    SpecializeCandidate,
    SpecializeManagerRequestSpec,
    SpecializeManager,
    SpecializeEvaluation,
    SpecializeParseError,
    StructuredCriterionSnapshot,
    StructuredRubric,
    Vote,
    apply_rubric_patch,
    assemble_specialized_pairwise_prediction,
    build_specialize_candidate,
    detect_specialize_trigger,
    deterministic_child_node_id,
    parse_child_proposal_response,
    parse_cluster_proposal_response,
    parse_error_signature_response,
    parse_split_failure_attribution_response,
    plan_artifact_refresh,
    evaluate_specialize_candidate,
    structured_input_fingerprint,
)
from critiq.structured.aggregation import aggregate_flat_votes
from critiq.structured.evolution.specialize_types import ChildCriterionProposal


def _manager_spec() -> SpecializeManagerRequestSpec:
    return SpecializeManagerRequestSpec.from_prompt(
        model="model", backend_id="pool", prompt="prompt", decoding_config={"temperature": 0},
        prompt_version="1.0.0", parser_version="1.0.0")


def _rubric() -> StructuredRubric:
    node = RubricNode("parent", RubricCriterionSnapshot("parent", "Parent criterion", 1.0))
    return StructuredRubric({"parent": node}, (), ("parent",))


def _context(*, accuracy: float = .25, coverage: float = 1.0,
             support: int = 20, wrong: int = 15) -> EvolutionContext:
    rubric = _rubric()
    correct = support - wrong
    errors = tuple(ErrorSampleRef(f"s{i}", "parent", Vote.B, "A", "wrong", "thought")
                   for i in range(wrong))
    feedback = NodeFeedback("parent", "parent", 20, support, correct, wrong,
                            20 - support, 0, 0, accuracy, coverage, 1.0, 16, .2, errors)
    overall = RubricFeedback(rubric.rubric_sha256, tuple(f"s{i}" for i in range(20)),
                             {"parent": feedback}, {}, (), (), ())
    return EvolutionContext(rubric, overall)


def _thresholds(**updates):
    value = {"tau_split": .7, "tau_cov_high": .8, "N_min_support": 15,
             "N_min_wrong": 15, "N_min_cluster": 5, "max_children": 5}
    value.update(updates)
    return value


def _vote(vote: Vote) -> PairwiseVoteOutput:
    return PairwiseVoteOutput(vote, True, '{"thought":"x","answer":"A"}', None, 1,
                              "x", True)


def _request_spec() -> DualWorkerRequestSpec:
    return DualWorkerRequestSpec("model", "pool", "a" * 64, None, False,
                                 "image_path", "question", "sample_id", {"temperature": .5})


def _prediction(criteria, rows):
    names = tuple(item.name for item in criteria)
    outputs = tuple({name: _vote(row[index]) for index, name in enumerate(names)} for row in rows)
    return PairwisePredictionOutput(
        tuple(f"s{i}" for i in range(len(rows))), tuple("f" * 64 for _ in rows), tuple(criteria),
        outputs, tuple(aggregate_flat_votes(item.vote for item in row.values()) for row in outputs),
        _request_spec())


class SpecializeTests(unittest.TestCase):
    def test_child_generation_prompt_v2_only_changes_description_contract(self):
        self.assertIs(CHILD_GENERATION_PROMPT, CHILD_GENERATION_PROMPT_V2)
        self.assertIn('"description": "a precise standalone pairwise judging criterion"',
                      CHILD_GENERATION_PROMPT_V1)
        for section in ("Criterion focus:", "Applicable only when:",
                        "Not applicable when:", "Decision rule:"):
            self.assertIn(section, CHILD_GENERATION_PROMPT_V2)

    def test_manager_cluster_retries_and_request_spec_is_deeply_immutable(self):
        ids = tuple(f"s{i}" for i in range(10))
        valid = json.dumps({"clusters": [
            {"cluster_id": "a", "label": "A", "shared_failure": "fa",
             "distinction": "da", "sample_ids": list(ids[:5])},
            {"cluster_id": "b", "label": "B", "shared_failure": "fb",
             "distinction": "db", "sample_ids": list(ids[5:])}],
            "unclustered_sample_ids": []})

        class FakePool:
            backend_id = "pool"

            def __init__(self):
                self.responses = ["not json", valid]
                self.calls = []

            def call(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return self.responses.pop(0), AgentCallMetrics(api_attempts=1)

        pool = FakePool()
        manager = SpecializeManager(model="model", backend_pool=pool,
            structured_max_retries=1,
            analysis_request_kwargs={"temperature": 0, "seed": 42, "nested": {"x": [1]}},
            clustering_request_kwargs={"temperature": 0, "seed": 42,
                                       "nested": {"x": [1]}},
            generation_request_kwargs={"temperature": .7, "seed": 42})
        signatures = tuple(ErrorSignature(sample_id, "task", "focus", "diff", "failure", "sub")
                           for sample_id in ids)
        result = manager.cluster(signatures, criterion_name="parent",
                                 min_cluster_size=5, max_clusters=5)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(result.metrics.api_attempts, 2)
        self.assertTrue(all(sample_id in pool.calls[0][0][0] for sample_id in ids))
        self.assertEqual(pool.calls[0][1]["agent_args"]["request_kwargs"]["temperature"], 0)
        self.assertEqual(pool.calls[0][0][0], pool.calls[1][0][0])
        spec = manager.request_specs()["semantic_cluster"]
        with self.assertRaises(TypeError):
            spec.decoding_config["nested"]["x"] = ()
        self.assertEqual(spec.decoding_config["temperature"], 0)
        self.assertEqual(SpecializeManagerRequestSpec.from_dict(spec.to_dict()), spec)

    def test_text_only_child_manager_uses_reasoning_request_without_images(self):
        response = json.dumps({
            "criterion_name": "spatial_accuracy",
            "description": "Judge spatial relations precisely.",
            "rationale": "Targets spatial relation errors.",
            "representative_sample_ids": ["s1"],
        })

        class FakePool:
            backend_id = "siliconflow"

            def __init__(self):
                self.call_args = None

            def call(self, *args, **kwargs):
                self.call_args = (args, kwargs)
                return response, AgentCallMetrics(api_attempts=1)

        pool = FakePool()
        request = {"temperature": .7, "seed": 42,
                   "extra_body": {"enable_thinking": True, "thinking_budget": 8192}}
        manager = SpecializeManager(
            model="deepseek-ai/DeepSeek-V4-Flash", backend_pool=pool,
            api_keys="secret-placeholder", child_input_mode="text",
            generation_request_kwargs=request)
        parent = _rubric().get_node("parent")
        cluster = SemanticCluster("spatial", "Spatial", "failure", "distinct", ("s1",))
        signature = ErrorSignature("s1", "task", "space", "difference", "failure", "spatial")
        row = {"sample_id": "s1", "question": "Where?", "A": "left", "B": "right",
               "answer": "A", "image_path": "this-file-must-not-be-opened"}
        child = manager.generate_child(
            parent=parent, cluster=cluster, signatures=(signature,),
            representative_rows=(row,), siblings=())
        content = pool.call_args[0][0]
        agent_args = pool.call_args[1]["agent_args"]
        self.assertIsInstance(content, str)
        self.assertIn('"question": "Where?"', content)
        self.assertNotIn("this-file-must-not-be-opened", content)
        self.assertEqual(agent_args["api_keys"], "secret-placeholder")
        self.assertTrue(agent_args["request_kwargs"]["extra_body"]["enable_thinking"])
        self.assertEqual(child.criterion_name, "spatial_accuracy")

    def test_manager_attributes_rejected_split_with_clustering_identity(self):
        response = json.dumps({
            "summary": "Broad children harmed correct parent decisions.",
            "failure_categories": ["child_too_broad"],
            "details": ["A spatial child activated on an object-identity sample."],
            "avoid_next_time": ["Use observable and disjoint applicability boundaries."],
        })

        class FakePool:
            backend_id = "siliconflow"

            def __init__(self):
                self.calls = []

            def call(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return response, AgentCallMetrics(api_attempts=1)

        pool = FakePool()
        manager = SpecializeManager(
            model="Qwen/Qwen3.5-397B-A17B", backend_pool=pool,
            clustering_request_kwargs={"temperature": .2, "seed": 42})
        artifact = manager.attribute_split_failure(
            parent=_rubric().get_node("parent"),
            signatures=(ErrorSignature(
                "s1", "task", "space", "difference", "failure", "spatial"),),
            cluster_proposal={"clusters": [{"cluster_id": "spatial"}]},
            children=(ChildCriterionProposal(
                "spatial", "spatial_accuracy", "Judge spatial relations.",
                "Targets spatial errors.", ("s1",), "raw", 1,
                ModelCallMetrics(), _manager_spec()),),
            local_metrics={"parent_accuracy": .7, "specialized_accuracy": .6},
            changed_predictions=({"sample_id": "s1", "outcome": "harmed"},))
        self.assertEqual(artifact["attribution"]["failure_categories"], ["child_too_broad"])
        self.assertEqual(artifact["request_spec"]["model"], "Qwen/Qwen3.5-397B-A17B")
        self.assertEqual(pool.calls[0][1]["request_type"], "split_failure_attribution")
        self.assertEqual(
            pool.calls[0][1]["agent_args"]["request_kwargs"]["temperature"], .2)

    def test_prior_split_failures_exposes_structured_and_natural_history(self):
        entry = {
            "parent_node_id": "parent", "decision": "reject",
            "parent_criterion": {"name": "parent", "description": "Parent criterion"},
            "structured_failure": {
                "reasons": ["specialized_accuracy_below_parent"],
                "mechanism_risks": ["child_accuracy_below_parent"]},
            "failure_attribution": {"attribution": {
                "summary": "Children were too broad.",
                "failure_categories": ["child_too_broad"],
                "details": ["They harmed one parent-correct sample."],
                "avoid_next_time": ["Tighten applicability."]}},
            "clusters": {"clusters": [{"cluster_id": "spatial"}]},
            "children": [{"criterion_name": "spatial_accuracy"}],
            "child_diagnostics": [], "parent_accuracy": .7,
            "specialized_accuracy": .6, "accuracy_delta": -.1,
            "corrected_sample_ids": [], "harmed_sample_ids": ["s1"],
        }
        history = {"schema_version": "2.0.0", "entries": [entry]}
        load_target = ("experiments.evolving_structured_rubrics."
                       "run_rubric_evolution.load_json")
        with patch.object(Path, "exists", return_value=True), \
                patch(load_target, return_value=history):
            prior = _prior_split_failures(Path("ignored"), "parent")
        self.assertEqual(prior[0]["failure_attribution"]["summary"],
                         "Children were too broad.")
        self.assertEqual(prior[0]["structured_failure"]["reasons"],
                         ["specialized_accuracy_below_parent"])
        self.assertEqual(prior[0]["harmed_sample_ids"], ["s1"])

    def test_example_config_unifies_all_split_manager_models(self):
        path = (Path(__file__).resolve().parents[2] / "experiments" /
                "evolving_structured_rubrics" / "configs" /
                "rubric_evolution_phase5.example.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        models = {profile["model"]
                  for profile in config["specialize_managers"].values()}
        self.assertEqual(models, {"Qwen/Qwen3.5-397B-A17B"})
    def test_trigger_selects_only_decisive_wrong(self):
        decision = detect_specialize_trigger(_context(), "parent", _thresholds())
        self.assertTrue(decision.triggered)
        self.assertEqual(len(decision.decisive_wrong_sample_ids), 15)
        self.assertEqual(decision.remaining_capacity, 5)

    def test_trigger_boundaries_and_capacity(self):
        decision = detect_specialize_trigger(_context(accuracy=.7, wrong=6), "parent", _thresholds())
        self.assertFalse(decision.triggered)
        self.assertIn("accuracy_not_below_tau_split", decision.reasons)
        children = {
            f"c{i}": RubricNode(f"c{i}", RubricCriterionSnapshot(f"c{i}", "child", 1.0))
            for i in range(4)
        }
        base = _context()
        rubric = StructuredRubric({"parent": base.rubric.get_node("parent"), **children},
            tuple(RubricEdge("parent", node_id, EdgeCondition.ALWAYS) for node_id in children),
            ("parent",))
        context = EvolutionContext(rubric, RubricFeedback(
            rubric.rubric_sha256, base.feedback.sample_ids, base.feedback.nodes,
            {}, (), (), ()))
        decision = detect_specialize_trigger(context, "parent", _thresholds())
        self.assertFalse(decision.triggered)
        self.assertEqual(decision.remaining_capacity, 1)

    def test_error_signature_strict_parser(self):
        payload = {"sample_id": "s1", "task_pattern": "qa", "visual_focus": "objects",
                   "candidate_difference": "identity", "parent_failure": "too broad",
                   "suggested_subdomain": "object identity"}
        result = parse_error_signature_response(f"```json\n{json.dumps(payload)}\n```",
                                                expected_sample_id="s1")
        self.assertEqual(result.sample_id, "s1")
        with self.assertRaises(SpecializeParseError):
            parse_error_signature_response(json.dumps({**payload, "extra": "x"}),
                                           expected_sample_id="s1")
        with self.assertRaises(SpecializeParseError):
            parse_error_signature_response(json.dumps(payload), expected_sample_id="s2")

    def test_split_failure_attribution_strict_parser(self):
        payload = {
            "summary": "The children overrode correct parent votes outside their clusters.",
            "failure_categories": ["child_too_broad", "unrelated_scene_activation"],
            "details": ["Two broad children harmed three parent-correct samples."],
            "avoid_next_time": ["Make applicability boundaries observable and disjoint."],
        }
        parsed = parse_split_failure_attribution_response(json.dumps(payload))
        self.assertEqual(parsed, payload)
        with self.assertRaises(SpecializeParseError):
            parse_split_failure_attribution_response(json.dumps({**payload, "extra": []}))
        with self.assertRaises(SpecializeParseError):
            parse_split_failure_attribution_response(json.dumps({
                **payload, "failure_categories": ["Not Snake Case"]}))
    def test_cluster_parser_validates_partition_and_capacity(self):
        ids = tuple(f"s{i}" for i in range(12))
        value = {"clusters": [
            {"cluster_id": "a", "label": "A", "shared_failure": "one",
             "distinction": "d1", "sample_ids": list(ids[:5])},
            {"cluster_id": "b", "label": "B", "shared_failure": "two",
             "distinction": "d2", "sample_ids": list(ids[5:10])}],
            "unclustered_sample_ids": list(ids[10:])}
        proposal = parse_cluster_proposal_response(
            "\n\n" + json.dumps(value) + "\n", expected_sample_ids=ids,
            min_cluster_size=5, max_clusters=5,
            attempt_count=1, metrics=ModelCallMetrics(), request_spec=_manager_spec())
        self.assertEqual(len(proposal.clusters), 2)
        self.assertEqual(proposal.raw_response, json.dumps(value))
        self.assertEqual(ClusterProposal.from_dict(proposal.to_dict()), proposal)
        value["clusters"][1]["sample_ids"][0] = ids[0]
        with self.assertRaisesRegex(
                SpecializeParseError,
                "duplicate sample IDs.*missing sample IDs"):
            parse_cluster_proposal_response(
                json.dumps(value), expected_sample_ids=ids, min_cluster_size=5, max_clusters=5,
                attempt_count=1, metrics=ModelCallMetrics(), request_spec=_manager_spec())

    def test_cluster_parser_allows_not_triggered_partition(self):
        ids = tuple(f"s{i}" for i in range(8))
        proposal = parse_cluster_proposal_response(
            json.dumps({"clusters": [], "unclustered_sample_ids": list(ids)}),
            expected_sample_ids=ids, min_cluster_size=5, max_clusters=5,
            attempt_count=1, metrics=ModelCallMetrics(), request_spec=_manager_spec())
        self.assertEqual(proposal.clusters, ())

    def test_child_parser_requires_cluster_representatives(self):
        cluster = SemanticCluster("a", "A", "failure", "distinct", ("s1", "s2", "s3", "s4", "s5"))
        raw = json.dumps({"criterion_name": "object_identity", "description": "Judge identity.",
                          "rationale": "Narrow identity check.",
                          "representative_sample_ids": ["s1", "s2"]})
        child = parse_child_proposal_response("\n\n" + raw + "\n", cluster=cluster, attempt_count=1,
                                              metrics=ModelCallMetrics(), request_spec=_manager_spec())
        self.assertEqual(child.criterion_name, "object_identity")
        self.assertEqual(child.raw_response, raw)
        with self.assertRaises(SpecializeParseError):
            parse_child_proposal_response(raw.replace('"s2"', '"other"'), cluster=cluster,
                                          attempt_count=1, metrics=ModelCallMetrics(),
                                          request_spec=_manager_spec())

    @patch("critiq.structured.evolution.specialize._image_sha256", return_value="e" * 64)
    def test_build_candidate_retains_parent_and_adds_always_children(self, _):
        context = _context()
        clusters = (
            SemanticCluster("a", "A", "failure a", "distinct a", tuple(f"s{i}" for i in range(5))),
            SemanticCluster("b", "B", "failure b", "distinct b", tuple(f"s{i}" for i in range(5, 10))),
        )
        cluster_proposal = ClusterProposal(clusters, tuple(f"s{i}" for i in range(10, 15)),
                                           "raw", 1, ModelCallMetrics(), _manager_spec())
        children = tuple(ChildCriterionProposal(
            cluster.cluster_id, f"child_{cluster.cluster_id}", f"Description {cluster.cluster_id}",
            "narrow", (cluster.sample_ids[0],), "raw", 1, ModelCallMetrics(), _manager_spec())
            for cluster in clusters)
        dataset = tuple({"sample_id": f"s{i}", "question": "q", "A": "a", "B": "b",
                         "answer": "A", "image_path": "unused"} for i in range(20))
        signatures = {f"s{i}": ErrorSignature(f"s{i}", "task", "focus", "diff", "failure", "sub")
                      for i in range(15)}
        candidate = build_specialize_candidate(context, "parent", cluster_proposal,
                                               children, dataset, signatures)
        after = apply_rubric_patch(context.rubric, candidate.edit_candidate.patch)
        self.assertIn("parent", after.nodes)
        self.assertEqual(len(after.nodes), 3)
        self.assertTrue(all(edge.condition is EdgeCondition.ALWAYS for edge in after.edges))
        refresh = plan_artifact_refresh(context.rubric, after)
        self.assertEqual(set(refresh.refresh_pairwise_node_ids), set(candidate.node_id_by_cluster.values()))
        self.assertEqual(refresh.refresh_gate_node_ids, ())
        self.assertFalse(refresh.router_stale)
        self.assertEqual(SpecializeCandidate.from_dict(candidate.to_dict()), candidate)

    def test_deterministic_child_node_id(self):
        first = deterministic_child_node_id("parent", "cluster", "criterion")
        self.assertEqual(first, deterministic_child_node_id("parent", "cluster", "criterion"))
        self.assertNotEqual(first, deterministic_child_node_id("parent", "other", "criterion"))

    def test_assemble_artifact_preserves_base_outputs(self):
        base = _prediction((StructuredCriterionSnapshot("parent", "Parent"),),
                           ((Vote.A,), (Vote.B,)))
        child = _prediction((StructuredCriterionSnapshot("child", "Child"),),
                            ((Vote.B,), (Vote.B,)))
        parent_node = RubricNode("parent", RubricCriterionSnapshot("parent", "Parent", 1.0))
        child_node = RubricNode("child", RubricCriterionSnapshot("child", "Child", 1.0))
        rubric = StructuredRubric({"parent": parent_node, "child": child_node},
                                  (RubricEdge("parent", "child", EdgeCondition.ALWAYS),),
                                  ("parent",))
        combined = assemble_specialized_pairwise_prediction(base, child, rubric)
        self.assertIs(combined.node_outputs[0]["parent"], base.node_outputs[0]["parent"])
        self.assertEqual([item.name for item in combined.criteria], ["parent", "child"])
        with self.assertRaises(ValueError):
            assemble_specialized_pairwise_prediction(base, base, rubric)

    def test_multimodal_child_ablation_changes_only_input_mode(self):
        control = {
            "specialize_managers": {
                "child_generation": {
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "backend_pool": {"pool_id": "same-pool"},
                    "api_key_env": "GUIJI_API_KEY",
                    "input_mode": "text",
                    "request_kwargs": {"temperature": 0.7, "seed": 42},
                    "vllm_identity": None,
                }
            },
            "sentinel": ["unchanged"],
        }
        treatment = _multimodal_child_config(control)
        self.assertEqual(control["specialize_managers"]["child_generation"]["input_mode"],
                         "text")
        self.assertEqual(treatment["specialize_managers"]["child_generation"]["input_mode"],
                         "multimodal")
        left = dict(control["specialize_managers"]["child_generation"])
        right = dict(treatment["specialize_managers"]["child_generation"])
        left.pop("input_mode"); right.pop("input_mode")
        self.assertEqual(left, right)
        self.assertEqual(control["sentinel"], treatment["sentinel"])
        with self.assertRaises(ValueError):
            _multimodal_child_config({})
        with self.assertRaises(ValueError):
            _multimodal_child_config(treatment)
    def test_qwen35_signature_ablation_changes_only_model_identity(self):
        error_profile = {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "backend_pool": {"pool_id": "local-8b"},
            "api_key_env": None,
            "input_mode": "multimodal",
            "request_kwargs": {"temperature": 0.2, "seed": 42},
            "vllm_identity": {"version": "0.11.0", "max_model_len": 25800},
        }
        child_profile = {
            "model": "Qwen/Qwen3.5-397B-A17B",
            "backend_pool": {"pool_id": "remote-397b"},
            "api_key_env": "GUIJI_API_KEY",
            "input_mode": "text",
            "request_kwargs": {"temperature": 0.7, "seed": 42},
            "vllm_identity": None,
        }
        control = {"specialize_managers": {
            "error_signature": error_profile,
            "semantic_cluster": {"sentinel": "unchanged"},
            "child_generation": child_profile,
        }, "sentinel": ["unchanged"]}
        treatment = _qwen35_signature_config(control)
        result = treatment["specialize_managers"]["error_signature"]
        self.assertEqual(result["model"], child_profile["model"])
        self.assertEqual(result["backend_pool"], child_profile["backend_pool"])
        self.assertEqual(result["api_key_env"], child_profile["api_key_env"])
        self.assertEqual(result["vllm_identity"], child_profile["vllm_identity"])
        self.assertEqual(result["input_mode"], error_profile["input_mode"])
        self.assertEqual(result["request_kwargs"], error_profile["request_kwargs"])
        self.assertEqual(control["specialize_managers"]["error_signature"], error_profile)
        self.assertEqual(treatment["specialize_managers"]["semantic_cluster"],
                         {"sentinel": "unchanged"})
        self.assertEqual(treatment["sentinel"], control["sentinel"])
        with self.assertRaises(ValueError):
            _qwen35_signature_config({})
    def test_split_set_summary_uses_weighted_target_accuracy(self):
        diagnostic_a = type("Diagnostic", (), {
            "cluster_support": 2, "cluster_accuracy": 0.5,
            "accuracy": 0.6, "coverage": 0.7, "fitness": 0.4,
            "non_target_decisive": 3, "non_target_wrong": 1})()
        diagnostic_b = type("Diagnostic", (), {
            "cluster_support": 3, "cluster_accuracy": 2 / 3,
            "accuracy": 0.8, "coverage": 0.5, "fitness": 0.6,
            "non_target_decisive": 4, "non_target_wrong": 2})()
        candidate = type("CandidateEvaluation", (), {
            "before_accuracy": 0.5, "after_accuracy": 0.6, "accuracy_delta": 0.1,
            "corrected_count": 2, "harmed_count": 1})()
        subtree = type("SubtreeDiagnostic", (), {
            "specialized_accuracy": 0.7, "specialized_coverage": 0.8})()
        evaluation = type("Evaluation", (), {
            "child_diagnostics": (diagnostic_a, diagnostic_b),
            "collective_fitness": 0.5, "parent_fitness": 0.7,
            "fitness_delta": -0.2, "candidate_evaluation": candidate,
            "subtree_diagnostic": subtree})()
        summary = _split_set_summary(evaluation)
        self.assertEqual(summary["target_cluster_support"], 5)
        self.assertEqual(summary["target_cluster_correct"], 3)
        self.assertAlmostEqual(summary["target_cluster_accuracy"], 0.6)
        self.assertEqual(summary["non_target_wrong"], 3)
        self.assertAlmostEqual(summary["mean_fitness"], 0.5)

    def test_activation_summary_is_restricted_to_parent_domain(self):
        base = {"samples": [
            {"sample_id": "s1", "node_outputs": {"parent": {"vote": "A"}}},
            {"sample_id": "s2", "node_outputs": {"parent": {"vote": "abstain"}}},
            {"sample_id": "s3", "node_outputs": {"parent": {"vote": "B"}}},
        ]}
        child = {"criteria": [{"name": "c1"}, {"name": "c2"}], "samples": [
            {"sample_id": "s1", "node_outputs": {
                "c1": {"vote": "A"}, "c2": {"vote": "B"}}},
            {"sample_id": "s2", "node_outputs": {
                "c1": {"vote": "A"}, "c2": {"vote": "A"}}},
            {"sample_id": "s3", "node_outputs": {
                "c1": {"vote": "abstain"}, "c2": {"vote": "B"}}},
        ]}
        with patch(
                "experiments.evolving_structured_rubrics.run_rubric_evolution.load_json",
                side_effect=(base, child)):
            summary = _activation_summary("base.json", "child.json", "parent")
        self.assertEqual(summary["parent_domain"], 2)
        self.assertEqual(summary["activation_distribution"], {"1": 1, "2": 1})
        self.assertAlmostEqual(summary["mean_active_children"], 1.5)
        self.assertEqual(summary["samples_with_all_children_active"], 1)
        self.assertEqual(summary["sibling_conflict_samples"], 1)
    def test_local_specialized_accuracy_controls_split_acceptance(self):
        dataset = (
            {"sample_id": "s0", "image_path": "http://image/0", "question": "q",
             "A": "a", "B": "b", "answer": "A"},
            {"sample_id": "s1", "image_path": "http://image/1", "question": "q",
             "A": "a", "B": "b", "answer": "A"},
        )
        spec = _request_spec()
        fingerprints = tuple(structured_input_fingerprint(
            row, image_field="image_path", question_field="question",
            sample_id_field="sample_id", max_data_chars=None, encode_local_image=False)
            for row in dataset)
        before = _rubric()
        children_nodes = {
            "child_a": RubricNode("child_a", RubricCriterionSnapshot("child_a", "A child", 1.0)),
            "child_b": RubricNode("child_b", RubricCriterionSnapshot("child_b", "B child", 1.0)),
        }
        after = StructuredRubric({"parent": before.get_node("parent"), **children_nodes}, (
            RubricEdge("parent", "child_a", EdgeCondition.ALWAYS),
            RubricEdge("parent", "child_b", EdgeCondition.ALWAYS)), ("parent",))
        base_rows = ({"parent": _vote(Vote.B)}, {"parent": _vote(Vote.A)})
        base = PairwisePredictionOutput(("s0", "s1"), fingerprints,
            (StructuredCriterionSnapshot("parent", "Parent criterion"),), base_rows,
            tuple(aggregate_flat_votes(item.vote for item in row.values()) for row in base_rows), spec)
        clusters = ClusterProposal((
            SemanticCluster("a", "A", "fa", "da", ("s0",)),
            SemanticCluster("b", "B", "fb", "db", ("s1",))), (), "raw", 1,
            ModelCallMetrics(), _manager_spec())
        child_proposals = (
            ChildCriterionProposal("a", "child_a", "A child", "r", ("s0",), "raw", 1,
                                   ModelCallMetrics(), _manager_spec()),
            ChildCriterionProposal("b", "child_b", "B child", "r", ("s1",), "raw", 1,
                                   ModelCallMetrics(), _manager_spec()),
        )
        patch = RubricPatch(before.rubric_sha256, upsert_nodes=tuple(children_nodes.values()),
            add_edges=after.edges, root_ids=before.root_ids)
        candidate = SpecializeCandidate("parent",
            EditCandidate("candidate", OperatorKind.SPLIT, "split", patch),
            clusters, child_proposals, {"a": "child_a", "b": "child_b"})

        def evaluate_rows(child_rows):
            child_prediction = PairwisePredictionOutput(("s0", "s1"), fingerprints, (
                StructuredCriterionSnapshot("child_a", "A child"),
                StructuredCriterionSnapshot("child_b", "B child")), child_rows,
                tuple(aggregate_flat_votes(item.vote for item in row.values())
                      for row in child_rows), spec)
            combined = assemble_specialized_pairwise_prediction(base, child_prediction, after)
            return evaluate_specialize_candidate(
                before_rubric=before, after_rubric=after, combined_prediction=combined,
                child_prediction=child_prediction, dataset=dataset, parent_node_id="parent",
                cluster_proposal=clusters, candidate=candidate,
                policy=CandidateAcceptancePolicy(.6, .95))[0]

        accepted = evaluate_rows((
            {"child_a": _vote(Vote.A), "child_b": _vote(Vote.A)},
            {"child_a": _vote(Vote.A), "child_b": _vote(Vote.A)},
        ))
        self.assertEqual(accepted.candidate_evaluation.decision.value, "accept")
        self.assertGreater(accepted.specialized_accuracy, accepted.parent_accuracy)
        self.assertIn("full_m1_guardrail_failed", accepted.subtree_diagnostic.mechanism_risks)
        self.assertEqual(SpecializeEvaluation.from_dict(accepted.to_dict()), accepted)
        artifact = accepted.to_dict()
        legacy = dict(artifact)
        legacy["parent_fitness"] = legacy.pop("parent_accuracy")
        legacy["collective_fitness"] = legacy.pop("specialized_accuracy")
        legacy["fitness_delta"] = legacy.pop("accuracy_delta")
        self.assertEqual(SpecializeEvaluation.from_dict(legacy), accepted)

        rejected = evaluate_rows((
            {"child_a": _vote(Vote.B), "child_b": _vote(Vote.B)},
            {"child_a": _vote(Vote.B), "child_b": _vote(Vote.B)},
        ))
        self.assertEqual(rejected.candidate_evaluation.decision.value, "reject")
        self.assertIn("specialized_accuracy_below_parent",
                      rejected.candidate_evaluation.reasons)

        unsupported = evaluate_rows((
            {"child_a": _vote(Vote.ABSTAIN), "child_b": _vote(Vote.ABSTAIN)},
            {"child_a": _vote(Vote.ABSTAIN), "child_b": _vote(Vote.ABSTAIN)},
        ))
        self.assertEqual(unsupported.candidate_evaluation.decision.value, "accept")
        self.assertEqual(unsupported.specialized_accuracy, unsupported.parent_accuracy)
        self.assertEqual(unsupported.candidate_evaluation.reasons, ())

class HeldoutVisualStageTest(unittest.TestCase):
    def test_vote_metrics_and_paired_comparison_use_gold(self):
        rows = (
            {"sample_id": "s0", "answer": "A"},
            {"sample_id": "s1", "answer": "B"},
            {"sample_id": "s2", "answer": "A"},
        )
        baseline = (FinalPreference.B, FinalPreference.B, FinalPreference.TIE)
        treatment = (FinalPreference.A, FinalPreference.A, FinalPreference.TIE)
        metrics = _heldout_vote_metrics(treatment, rows)
        self.assertEqual(metrics["correct_count"], 1)
        self.assertEqual(metrics["coverage_count"], 2)
        paired = _paired_heldout_comparison(baseline, treatment, rows)
        self.assertEqual(paired["corrected_sample_ids"], ["s0"])
        self.assertEqual(paired["harmed_sample_ids"], ["s1"])
        self.assertEqual(paired["mcnemar_exact_two_sided_p"], 1.0)

    def test_visual_variant_excludes_other_roots_and_unselected_children(self):
        nodes = {
            name: RubricNode(name, RubricCriterionSnapshot(name, name, 1.0))
            for name in ("parent", "child_a", "child_b", "other_root")
        }
        rubric = StructuredRubric(nodes, (
            RubricEdge("parent", "child_a", EdgeCondition.ALWAYS),
            RubricEdge("parent", "child_b", EdgeCondition.ALWAYS),
        ), ("parent", "other_root"))
        selected = _visual_variant_rubric(rubric, "parent", ("child_a",))
        self.assertEqual(set(selected.nodes), {"parent", "child_a"})
        self.assertEqual(selected.root_ids, ("parent",))
        self.assertEqual(selected.edges, (
            RubricEdge("parent", "child_a", EdgeCondition.ALWAYS),))
if __name__ == "__main__":
    unittest.main()

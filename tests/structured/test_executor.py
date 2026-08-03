import threading
import time
import unittest

from critiq.structured import (
    Applicability,
    BackendSource,
    CriterionStatus,
    EdgeCondition,
    ExecutionConfig,
    ExecutionTrace,
    FinalPreference,
    ModelCallMetrics,
    NodeBackendResult,
    NodeJudgement,
    OfflineNodeBackend,
    OfflineRouterBackend,
    PairPreference,
    RootRouterRequestSpec,
    RootRouterOutput,
    RootRoutingPredictionOutput,
    RootRoutingDecision,
    RouterBackendResult,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    StructuredCascadeExecutor,
    StructuredCriterionSnapshot,
    StructuredExecutionOutput,
    StructuredNodeOutput,
    StructuredPredictionOutput,
    StructuredRubric,
    StructuredSystemVariant,
    StructuredWorkerRequestSpec,
    aggregate_selected_roots,
    replay_execution_trace,
    root_set_sha256,
    root_snapshots,
    structured_input_fingerprint,
)


def judgement(applicable=Applicability.YES, a=CriterionStatus.PASS, b=CriterionStatus.PASS, preference=PairPreference.TIE):
    return StructuredNodeOutput(
        judgement=NodeJudgement(applicable, a, b, preference),
        raw_response="{}", parse_error=None, attempt_count=1,
    )


def make_forest():
    nodes = {
        node_id: RubricNode(node_id, RubricCriterionSnapshot(node_id + "_criterion", node_id + " description", 0.5))
        for node_id in ("r1", "c1", "g1", "c2", "r2")
    }
    edges = (
        RubricEdge("r1", "c1", EdgeCondition.PARENT_NONDECISIVE),
        RubricEdge("r1", "c2", EdgeCondition.PARENT_BOTH_PASS),
        RubricEdge("c1", "g1", EdgeCondition.ALWAYS),
    )
    return StructuredRubric(nodes=nodes, edges=edges, root_ids=("r1", "r2"))


OUTPUTS = {
    "r1": judgement(),
    "c1": judgement(Applicability.NO, CriterionStatus.UNCERTAIN, CriterionStatus.UNCERTAIN, PairPreference.UNCERTAIN),
    "g1": judgement(Applicability.YES, CriterionStatus.PASS, CriterionStatus.FAIL, PairPreference.A),
    "c2": judgement(Applicability.YES, CriterionStatus.FAIL, CriterionStatus.PASS, PairPreference.B),
    "r2": judgement(Applicability.YES, CriterionStatus.PASS, CriterionStatus.FAIL, PairPreference.A),
}


class FakeNodeBackend:
    def __init__(self):
        self.calls = []

    def sample_id(self, data):
        return data["sample_id"]

    def sample_fingerprint(self, data):
        return "a" * 64

    def evaluate(self, data, node):
        self.calls.append(node.node_id)
        zero = ModelCallMetrics()
        return NodeBackendResult(OUTPUTS[node.node_id], BackendSource.OFFLINE_ARTIFACT, zero, zero)


class FakeRouterBackend:
    def __init__(self, decision=None):
        self.calls = 0
        self.decision = decision or RootRoutingDecision(("r1",), {"r1": "relevant"}, True)

    def sample_id(self, data):
        return data["sample_id"]

    def sample_fingerprint(self, data):
        return "b" * 64

    def route(self, data, rubric):
        self.calls += 1
        output = RootRouterOutput(self.decision, "{}", None, 1)
        metrics = ModelCallMetrics(logical_evaluations=1, api_attempts=1, input_tokens=5, output_tokens=2, total_tokens=7)
        return RouterBackendResult(output, BackendSource.MODEL, metrics, metrics)


DATA = {"sample_id": "s1", "answer": "A"}


class StructuredExecutorVariantTest(unittest.TestCase):
    def execute(self, variant, concurrent=1, router=None):
        backend = FakeNodeBackend()
        executor = StructuredCascadeExecutor(make_forest(), backend, router)
        trace = executor.execute_one(DATA, ExecutionConfig(variant, concurrent))
        return trace, backend

    def test_five_variants_have_frozen_behavior(self):
        expected = {
            StructuredSystemVariant.B1_FLAT: (FinalPreference.A, {"r1", "c1", "g1", "c2", "r2"}),
            StructuredSystemVariant.H1_HIERARCHICAL_ONLY: (FinalPreference.A, {"r1", "c1", "g1", "c2", "r2"}),
            StructuredSystemVariant.G1_GATING_ONLY: (FinalPreference.TIE, {"r1", "c1", "c2", "r2"}),
            StructuredSystemVariant.M1_ALL_ROOTS_CASCADE: (FinalPreference.TIE, {"r1", "c1", "c2", "r2"}),
        }
        for variant, (answer, calls) in expected.items():
            with self.subTest(variant=variant):
                trace, backend = self.execute(variant)
                self.assertEqual(answer, trace.final_preference)
                self.assertEqual(calls, set(backend.calls))
                self.assertEqual(len(backend.calls), len(set(backend.calls)))
                self.assertEqual(answer, replay_execution_trace(make_forest(), trace))

    def test_m2_only_visits_selected_root_and_router_once(self):
        router = FakeRouterBackend()
        trace, backend = self.execute(StructuredSystemVariant.M2_ROUTED_ROOTS_CASCADE, router=router)
        self.assertEqual(FinalPreference.B, trace.final_preference)
        self.assertEqual({"r1", "c1", "c2"}, set(backend.calls))
        self.assertNotIn("r2", backend.calls)
        self.assertEqual(1, router.calls)
        self.assertEqual(2, trace.counterfactual_metrics.avoided_nodes)
        self.assertEqual(FinalPreference.B, replay_execution_trace(make_forest(), trace))

    def test_invalid_router_falls_back_to_all_roots(self):
        router = FakeRouterBackend(RootRoutingDecision((), {}, True))
        trace, backend = self.execute(StructuredSystemVariant.M2_ROUTED_ROOTS_CASCADE, router=router)
        self.assertTrue(trace.router.resolved.fallback_to_all_roots)
        self.assertIn("r2", backend.calls)
        self.assertEqual(FinalPreference.TIE, trace.final_preference)

    def test_disabled_variants_never_call_router(self):
        router = FakeRouterBackend()
        self.execute(StructuredSystemVariant.M1_ALL_ROOTS_CASCADE, router=router)
        self.assertEqual(0, router.calls)

    def test_bounded_concurrency_preserves_canonical_trace(self):
        sequential, _ = self.execute(StructuredSystemVariant.H1_HIERARCHICAL_ONLY, concurrent=1)
        concurrent, _ = self.execute(StructuredSystemVariant.H1_HIERARCHICAL_ONLY, concurrent=4)
        self.assertEqual(tuple(node.node_id for node in sequential.nodes), tuple(node.node_id for node in concurrent.nodes))
        self.assertEqual(sequential.final_preference, concurrent.final_preference)
        self.assertEqual(sequential.aggregation_input_votes, concurrent.aggregation_input_votes)

    def test_max_concurrent_globally_bounds_backend_calls(self):
        class CountingBackend(FakeNodeBackend):
            def __init__(self):
                super().__init__()
                self.active = 0
                self.peak = 0
                self.lock = threading.Lock()

            def evaluate(self, data, node):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                try:
                    time.sleep(0.01)
                    return super().evaluate(data, node)
                finally:
                    with self.lock:
                        self.active -= 1

        backend = CountingBackend()
        executor = StructuredCascadeExecutor(make_forest(), backend)
        executor.execute_one(
            DATA,
            ExecutionConfig(StructuredSystemVariant.H1_HIERARCHICAL_ONLY, max_concurrent=2),
        )
        self.assertLessEqual(backend.peak, 2)
        self.assertGreaterEqual(backend.peak, 2)

    def test_cross_sample_batch_shares_one_global_request_limit(self):
        class TrackingBackend(FakeNodeBackend):
            def __init__(self):
                super().__init__()
                self.active = 0
                self.peak = 0
                self.completed = set()
                self.lock = threading.Lock()

            def evaluate(self, data, node):
                parent = {"c1": "r1", "c2": "r1", "g1": "c1"}.get(
                    node.node_id)
                with self.lock:
                    if parent is not None:
                        self.assert_parent_completed(
                            data["sample_id"], parent)
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                try:
                    time.sleep(0.01)
                    zero = ModelCallMetrics()
                    return NodeBackendResult(
                        OUTPUTS[node.node_id],
                        BackendSource.OFFLINE_ARTIFACT,
                        zero,
                        zero,
                    )
                finally:
                    with self.lock:
                        self.active -= 1
                        self.completed.add(
                            (data["sample_id"], node.node_id))

            def assert_parent_completed(self, sample_id, parent):
                if (sample_id, parent) not in self.completed:
                    raise AssertionError(
                        f"{sample_id}:{parent} must finish before child")

        backend = TrackingBackend()
        executor = StructuredCascadeExecutor(make_forest(), backend)
        dataset = tuple(
            {"sample_id": f"s{index}", "answer": "A"}
            for index in range(10)
        )
        output = executor.execute_batch(
            dataset,
            ExecutionConfig(
                StructuredSystemVariant.H1_HIERARCHICAL_ONLY,
                max_concurrent=20,
                sample_concurrent=10,
                global_request_concurrent=3,
            ),
        )
        self.assertEqual(
            tuple(row["sample_id"] for row in dataset),
            tuple(trace.sample_id for trace in output.traces),
        )
        self.assertLessEqual(backend.peak, 3)
        self.assertGreaterEqual(backend.peak, 3)

    def test_router_and_workers_share_global_request_limit(self):
        class SharedTracker:
            def __init__(self):
                self.active = 0
                self.peak = 0
                self.lock = threading.Lock()

            def enter(self):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)

            def exit(self):
                with self.lock:
                    self.active -= 1

        tracker = SharedTracker()

        class TrackingNodeBackend(FakeNodeBackend):
            def evaluate(self, data, node):
                tracker.enter()
                try:
                    time.sleep(0.01)
                    return super().evaluate(data, node)
                finally:
                    tracker.exit()

        class TrackingRouterBackend(FakeRouterBackend):
            def route(self, data, rubric):
                tracker.enter()
                try:
                    time.sleep(0.01)
                    return super().route(data, rubric)
                finally:
                    tracker.exit()

        dataset = tuple(
            {"sample_id": f"s{index}", "answer": "A"}
            for index in range(10)
        )
        executor = StructuredCascadeExecutor(
            make_forest(),
            TrackingNodeBackend(),
            TrackingRouterBackend(),
        )
        output = executor.execute_batch(
            dataset,
            ExecutionConfig(
                StructuredSystemVariant.M2_ROUTED_ROOTS_CASCADE,
                max_concurrent=20,
                sample_concurrent=10,
                global_request_concurrent=4,
            ),
        )
        self.assertEqual(10, len(output.traces))
        self.assertLessEqual(tracker.peak, 4)
        self.assertGreaterEqual(tracker.peak, 4)

    def test_cross_sample_execution_config_validation(self):
        for kwargs in (
            {"sample_concurrent": 0},
            {"sample_concurrent": True},
            {"global_request_concurrent": 0},
            {"global_request_concurrent": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExecutionConfig(
                    StructuredSystemVariant.B1_FLAT,
                    **kwargs,
                )
    def test_trace_round_trip_and_tamper_detection(self):
        trace, _ = self.execute(StructuredSystemVariant.M1_ALL_ROOTS_CASCADE)
        restored = ExecutionTrace.from_dict(trace.to_dict())
        self.assertEqual(trace.final_preference, replay_execution_trace(make_forest(), restored))
        artifact = StructuredExecutionOutput(
            variant=trace.variant,
            rubric_sha256=trace.rubric_sha256,
            traces=(trace,),
        )
        restored_artifact = StructuredExecutionOutput.from_dict(artifact.to_dict())
        self.assertEqual(
            trace.final_preference,
            replay_execution_trace(make_forest(), restored_artifact.traces[0]),
        )
        tampered = trace.to_dict()
        tampered["final_preference"] = "A"
        with self.assertRaises(ValueError):
            replay_execution_trace(make_forest(), ExecutionTrace.from_dict(tampered))

        version_tampered = artifact.to_dict()
        version_tampered["trace_version"] = "999.0.0"
        with self.assertRaises(ValueError):
            StructuredExecutionOutput.from_dict(version_tampered)

        node_metrics_tampered = artifact.to_dict()
        node_metrics_tampered["traces"][0]["node_metrics"]["api_attempts"] += 1
        with self.assertRaisesRegex(ValueError, "node_metrics"):
            StructuredExecutionOutput.from_dict(node_metrics_tampered)

        disabled_router_tampered = artifact.to_dict()
        disabled_router_tampered["traces"][0]["router"]["metrics"]["api_attempts"] = 1
        with self.assertRaisesRegex(ValueError, "zero metrics"):
            StructuredExecutionOutput.from_dict(disabled_router_tampered)

        m2_trace, _ = self.execute(
            StructuredSystemVariant.M2_ROUTED_ROOTS_CASCADE,
            router=FakeRouterBackend(),
        )
        m2_artifact = StructuredExecutionOutput(
            variant=m2_trace.variant,
            rubric_sha256=m2_trace.rubric_sha256,
            traces=(m2_trace,),
        ).to_dict()
        m2_artifact["traces"][0]["router"]["metrics"]["api_attempts"] = 999
        with self.assertRaisesRegex(ValueError, "generation provenance"):
            StructuredExecutionOutput.from_dict(m2_artifact)

    def test_evaluation_tie_counts_as_error_and_zero_coverage(self):
        backend = FakeNodeBackend()
        executor = StructuredCascadeExecutor(make_forest(), backend)
        result = executor.evaluate_batch([DATA], ExecutionConfig(StructuredSystemVariant.M1_ALL_ROOTS_CASCADE))
        self.assertEqual(0.0, result.accuracy)
        self.assertEqual(0.0, result.coverage)
        self.assertEqual(1.0, result.tie_rate)

    def test_all_variants_share_one_offline_worker_artifact(self):
        rubric = make_forest()
        data = {
            "sample_id": "offline-s1",
            "image_path": "https://example.invalid/image.jpg",
            "question": "question",
            "A": "answer A",
            "B": "answer B",
            "answer": "A",
        }
        fingerprint = structured_input_fingerprint(
            data,
            image_field="image_path",
            question_field="question",
            sample_id_field="sample_id",
            encode_local_image=False,
        )
        criteria = tuple(
            StructuredCriterionSnapshot(
                rubric.get_node(node_id).criterion.name,
                rubric.get_node(node_id).criterion.description,
            )
            for node_id in rubric.preorder_node_ids()
        )
        outputs = {
            rubric.get_node(node_id).criterion.name: OUTPUTS[node_id]
            for node_id in rubric.preorder_node_ids()
        }
        names = tuple(criterion.name for criterion in criteria)
        worker_spec = StructuredWorkerRequestSpec(
            model="offline-model",
            worker_backend_id="offline-checkpoint",
            prompt_sha256="a" * 64,
            max_data_chars=None,
            encode_local_image=False,
            image_field="image_path",
            question_field="question",
            sample_id_field="sample_id",
            decoding_config={"temperature": 0},
        )
        worker_artifact = StructuredPredictionOutput(
            sample_ids=(data["sample_id"],),
            sample_fingerprints=(fingerprint,),
            criteria=criteria,
            node_outputs=(outputs,),
            flat_answers=(
                aggregate_selected_roots(
                    {name: outputs[name].local_decision.vote for name in names},
                    names,
                ),
            ),
            request_spec=worker_spec,
        )
        router_spec = RootRouterRequestSpec(
            model="offline-model",
            router_backend_id="offline-checkpoint",
            prompt_sha256="b" * 64,
            rubric_sha256=rubric.rubric_sha256,
            roots_sha256=root_set_sha256(rubric),
            max_data_chars=None,
            encode_local_image=False,
            image_field="image_path",
            question_field="question",
            sample_id_field="sample_id",
            decoding_config={"temperature": 0},
        )
        router_output = RootRouterOutput(
            RootRoutingDecision(("r1",), {"r1": "relevant"}, True),
            "{}",
            None,
            1,
        )
        router_artifact = RootRoutingPredictionOutput(
            sample_ids=(data["sample_id"],),
            sample_fingerprints=(fingerprint,),
            roots=root_snapshots(rubric),
            outputs=(router_output,),
            generation_metrics=(ModelCallMetrics(),),
            request_spec=router_spec,
            rubric_sha256=rubric.rubric_sha256,
        )
        node_backend = OfflineNodeBackend(worker_artifact, rubric)
        router_backend = OfflineRouterBackend(router_artifact, rubric)
        executor = StructuredCascadeExecutor(rubric, node_backend, router_backend)

        changed = dict(data)
        changed["A"] = "different answer A"
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            router_backend.route(changed, rubric)

        for variant in StructuredSystemVariant:
            with self.subTest(variant=variant):
                trace = executor.execute_one(data, ExecutionConfig(variant))
                self.assertEqual(0, trace.total_metrics.api_attempts)
                self.assertEqual(0, trace.total_metrics.total_tokens)
                self.assertEqual(trace.final_preference, replay_execution_trace(rubric, trace))


if __name__ == "__main__":
    unittest.main()

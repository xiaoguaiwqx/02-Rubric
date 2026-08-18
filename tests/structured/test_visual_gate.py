"""Focused offline tests for the Visual Grounding sibling Gate experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from critiq.agent import AgentCallMetrics
from critiq.structured import (
    BackendEndpointSpec,
    BackendPoolSpec,
    EdgeCondition,
    RubricCriterionSnapshot,
    RubricEdge,
    RubricNode,
    StructuredRubric,
)
from experiments.evolving_structured_rubrics import visual_gate as gate


def _description(label: str) -> str:
    return (
        f"Criterion focus: Focus {label}.\n\n"
        f"Applicable only when: Apply {label}.\n\n"
        f"Not applicable when: Exclude {label}.\n\n"
        f"Decision rule: Prefer the matching candidate for {label}."
    )


def _node(node_id: str, name: str, description: str) -> RubricNode:
    return RubricNode(
        node_id=node_id,
        criterion=RubricCriterionSnapshot(name, description, 1.0),
    )


def _rubric() -> StructuredRubric:
    parent = _node(gate.PARENT_NODE_ID, "visual_grounding_and_details", "Parent focus.")
    children = [
        _node(f"child_{index}", name, _description(name))
        for index, name in enumerate(gate.EXPECTED_CHILD_NAMES)
    ]
    nodes = {parent.node_id: parent, **{item.node_id: item for item in children}}
    return StructuredRubric(
        nodes=nodes,
        edges=tuple(RubricEdge(parent.node_id, item.node_id, EdgeCondition.ALWAYS)
                    for item in children),
        root_ids=(parent.node_id,),
    )


class VisualSiblingGateContractTest(unittest.TestCase):
    def test_contract_extracts_routing_sections_but_not_decision_rule(self):
        contract = gate.extract_routing_contract(_description("small objects"))
        self.assertEqual(
            {"focus", "applicable", "not_applicable"}, set(contract))
        self.assertNotIn("Prefer", json.dumps(contract))

    def test_contract_order_is_rubric_edge_order_and_prompt_is_compact(self):
        contract = gate.build_routing_contract(_rubric())
        self.assertEqual(
            gate.EXPECTED_CHILD_NAMES,
            tuple(item["criterion_name"] for item in contract["children"]),
        )
        prompt = gate.compact_system_prompt(contract)
        self.assertIn("Never decide whether A or B is better", prompt)
        self.assertNotIn("Decision rule:", prompt)
        for item in contract["children"]:
            self.assertIn(item["node_id"], prompt)

    def test_missing_section_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Not applicable"):
            gate.extract_routing_contract(
                "Criterion focus: x\nApplicable only when: y\nDecision rule: z")

    def test_retry_user_prompt_ends_with_nested_json_reminder(self):
        row = {"question": "q", "A": "a", "B": "b"}
        ordinary = gate._user_prompt(row)
        retry = gate._user_prompt(row, format_reminder=True)
        self.assertNotIn("never output a bare status string", ordinary)
        self.assertTrue(retry.endswith(
            'both "status" and "reason"; never output a bare status string.'))


class VisualSiblingGateParserTest(unittest.TestCase):
    def setUp(self):
        self.child_ids = ("c1", "c2")
        self.payload = {
            "decisions": {
                "c1": {"status": "applicable", "reason": "The disagreement concerns existence."},
                "c2": {"status": "not_applicable", "reason": "No spatial composition difference is present."},
            },
            "outside_local_scope": [],
        }

    def test_valid_output_and_uncertain_activation(self):
        parsed = gate.parse_sibling_gate_response(json.dumps(self.payload), self.child_ids)
        self.assertEqual(("c1",), gate.active_child_ids(parsed, self.child_ids, parse_ok=True))
        parsed["decisions"]["c2"]["status"] = "uncertain"
        self.assertEqual(
            ("c1", "c2"), gate.active_child_ids(parsed, self.child_ids, parse_ok=True))

    def test_cross_root_or_missing_child_is_rejected(self):
        self.payload["decisions"]["external"] = {
            "status": "applicable", "reason": "External criterion."}
        with self.assertRaisesRegex(ValueError, "exactly the direct children"):
            gate.parse_sibling_gate_response(json.dumps(self.payload), self.child_ids)

    def test_reason_content_and_length_are_not_validated(self):
        self.payload["decisions"]["c1"]["reason"] = "word " * 500
        self.payload["decisions"]["c2"]["reason"] = None
        self.payload["outside_local_scope"] = [
            {"criterion_name": "another_root", "reason": {"any": "JSON value"}}]
        parsed = gate.parse_sibling_gate_response(json.dumps(self.payload), self.child_ids)
        self.assertEqual("applicable", parsed["decisions"]["c1"]["status"])
        self.assertIsNone(parsed["decisions"]["c2"]["reason"])
        self.assertEqual(
            {"any": "JSON value"}, parsed["outside_local_scope"][0]["reason"])

    def test_invalid_output_falls_back_to_all_children(self):
        self.assertEqual(
            self.child_ids,
            gate.active_child_ids({}, self.child_ids, parse_ok=False),
        )

class _FakePool:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def call(self, *_args, **_kwargs):
        value = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return value, AgentCallMetrics(
            api_attempts=1, input_tokens=10, output_tokens=5,
            total_tokens=15, usage_complete=True)


class VisualSiblingGateRetryAndCacheTest(unittest.TestCase):
    def _inputs(self, image_path: Path):
        contract = gate.build_routing_contract(_rubric())
        child_ids = [item["node_id"] for item in contract["children"]]
        request_spec = {
            "child_ids": child_ids,
            "decoding_config": {"temperature": 0.2, "max_tokens": 2048},
            "identity": "unit-test",
        }
        row = {
            "sample_id": "s1", "image_path": str(image_path),
            "question": "What differs?", "A": "one", "B": "two", "answer": "A",
        }
        payload = {
            "decisions": {
                child_id: {"status": "not_applicable", "reason": "No matching disagreement."}
                for child_id in child_ids
            },
            "outside_local_scope": [],
        }
        return contract, child_ids, request_spec, row, json.dumps(payload)

    def test_valid_result_is_cached_without_a_second_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"fake-image")
            contract, _ids, spec, row, raw = self._inputs(image)
            pool = _FakePool([raw])
            first, _ = gate._route_one(
                {"model": "test", "api_retry_attempts": 0},
                {"gate_max_retries": 3}, contract, spec, row, pool, root / "cache")
            second, metrics = gate._route_one(
                {"model": "test", "api_retry_attempts": 0},
                {"gate_max_retries": 3}, contract, spec, row, pool, root / "cache")
            self.assertTrue(first["parse_ok"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(1, pool.calls)
            self.assertEqual(1, metrics.cache_hits)

    def test_final_parse_failure_retries_five_times_then_activates_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"fake-image")
            contract, child_ids, spec, row, _raw = self._inputs(image)
            pool = _FakePool(["not-json"])
            result, metrics = gate._route_one(
                {"model": "test", "api_retry_attempts": 0},
                {"gate_max_retries": 5}, contract, spec, row, pool, root / "cache")
            self.assertFalse(result["parse_ok"])
            self.assertEqual(child_ids, result["active_child_ids"])
            self.assertEqual(6, result["attempt_count"])
            self.assertEqual(5, metrics.parse_retries)

    def test_failed_cache_is_retried_while_valid_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"fake-image")
            contract, _child_ids, spec, row, raw = self._inputs(image)
            failed_pool = _FakePool(["not-json"])
            failed, _ = gate._route_one(
                {"model": "test", "api_retry_attempts": 0},
                {"gate_max_retries": 0}, contract, spec, row, failed_pool, root / "cache")
            self.assertFalse(failed["parse_ok"])

            recovery_pool = _FakePool([raw])
            recovered, _ = gate._route_one(
                {"model": "test", "api_retry_attempts": 0},
                {"gate_max_retries": 5}, contract, spec, row, recovery_pool, root / "cache")
            self.assertTrue(recovered["parse_ok"])
            self.assertTrue(recovered["recovered_from_failed_cache"])
            self.assertTrue(recovered["format_reminder_used"])
            self.assertFalse(recovered["cache_hit"])
            self.assertEqual(1, recovery_pool.calls)


class VisualSiblingGatePoolTest(unittest.TestCase):
    def test_gate_pool_uses_both_frozen_endpoints_and_sums_capacity(self):
        config = {
            "backend_pool": {
                "pool_id": "source",
                "common_checkpoint_id": "Qwen/Qwen3-VL-8B-Instruct",
                "global_request_concurrency": 99,
                "endpoints": [
                    {"endpoint_id": "vllm-8000", "base_url": "http://localhost:8000/v1",
                     "checkpoint_root": "model", "max_concurrency": 20},
                    {"endpoint_id": "vllm-8001", "base_url": "http://localhost:8001/v1",
                     "checkpoint_root": "model", "max_concurrency": 20},
                ],
            }
        }
        spec = gate._gate_pool_spec(config, ["vllm-8000", "vllm-8001"])
        self.assertEqual(("vllm-8000", "vllm-8001"),
                         tuple(item.endpoint_id for item in spec.endpoints))
        self.assertEqual(40, spec.global_request_concurrency)
        self.assertEqual("visual-gate-dual-v2", spec.pool_id)

    def test_gate_pool_rejects_single_endpoint_protocol(self):
        config = {
            "backend_pool": {
                "pool_id": "source", "common_checkpoint_id": "model",
                "global_request_concurrency": 20,
                "endpoints": [
                    {"endpoint_id": "vllm-8000", "base_url": "http://localhost:8000/v1",
                     "checkpoint_root": "model", "max_concurrency": 20},
                ],
            }
        }
        with self.assertRaisesRegex(RuntimeError, "requires endpoints"):
            gate._gate_pool_spec(config, ["vllm-8000"])

    def test_request_uses_prompt_json_without_server_constrained_decoding(self):
        config = {"model": "Qwen/Qwen3-VL-8B-Instruct"}
        protocol = {
            "gate_endpoints": ["vllm-8000", "vllm-8001"],
            "gate_temperature": 0.2,
            "gate_max_tokens": 2048,
            "gate_seed": 42,
            "gate_max_retries": 5,
        }
        contract = gate.build_routing_contract(_rubric())
        spec = BackendPoolSpec(
            "visual-gate-dual-v2", "Qwen/Qwen3-VL-8B-Instruct", 2,
            (BackendEndpointSpec("vllm-8000", "http://localhost:8000/v1", "model", 1),
             BackendEndpointSpec("vllm-8001", "http://localhost:8001/v1", "model", 1)),
        )
        request = gate._request_spec(config, protocol, contract, spec)
        self.assertNotIn("response_format", request["decoding_config"])
        self.assertEqual(0.2, request["decoding_config"]["temperature"])
        self.assertEqual(2048, request["decoding_config"]["max_tokens"])


class VisualSiblingGateHeldoutAuthorizationTest(unittest.TestCase):
    def _report(self, go: bool = False):
        return {
            "go_heldout": go,
            "position_audit": {
                "exact_status_match_rate": 0.70,
                "per_child_status_agreement": {
                    "c1": 0.85, "c2": 0.95, "c3": 0.90, "c4": 0.95,
                },
            },
            "systems": {
                "all_children": {"visual_subtree": {"coverage_count": 87}},
                "dynamic_gate": {"visual_subtree": {"coverage_count": 86}},
            },
        }

    def test_failed_frozen_gate_still_blocks_standard_heldout(self):
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            gate._heldout_authorization(
                self._report(), exploratory_override=False)

    def test_exploratory_override_preserves_original_failures(self):
        value = gate._heldout_authorization(
            self._report(), exploratory_override=True)
        self.assertTrue(value["posthoc_override"])
        self.assertFalse(value["discovery_go_heldout"])
        failures = value["frozen_failures_preserved"]
        self.assertAlmostEqual(0.9125, failures["mean_per_child_status_agreement"])
        self.assertEqual(-1, failures["visual_coverage_count_delta"])

    def test_protocol_pass_needs_no_override(self):
        value = gate._heldout_authorization(
            self._report(go=True), exploratory_override=False)
        self.assertEqual("frozen_protocol_pass", value["mode"])
        self.assertFalse(value["posthoc_override"])


if __name__ == "__main__":
    unittest.main()

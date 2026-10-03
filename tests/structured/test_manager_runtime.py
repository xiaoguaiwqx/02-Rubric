"""Offline contracts for frozen Manager prompts, retry limits, and caches."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from experiments.evolving_structured_rubrics import manager_runtime as runtime
from experiments.evolving_structured_rubrics import subtree_local_reflection_manager as local


def response(text, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                 finish_reason=finish_reason)],
        usage=None,
    )


class TestManagerRuntime(unittest.TestCase):
    def setUp(self):
        self.config = dict(model="offline-manager", base_url="http://offline/v1",
                           concurrency=4, timeout=300,
                           request_kwargs=dict(temperature=0.2))

    def test_current_stage_prompts_match_frozen_five_root_protocol(self):
        expected = {
            "signature": "d5e828cf47fb8652734227cbf0e5f6493a1bcd48a79ad6cf8bde580c289ab1cb",
            "cluster": "27b2b75935e6832a34f3d9586f46600339e11e0531bc2eef4bfce1d3b437ed5e",
            "children": "0d8f548a17c61793447c95d25926349be58427102f3bd699650f070b9f83379e",
            "case_reflection": "eb4ed7467f9fb3c24f154df3cab7552241aa8f8ec17b1670b82b7e04a202daf0",
            "subtree_split": "494cbde262d048cac9ea06299c43691cd99e92cb9693380e02761ec1dbf2be99",
        }
        self.assertEqual(set(runtime.PROMPTS), {"signature", "cluster", "children"})
        prompts = local.local_prompts_for_root_count(5)
        self.assertEqual(set(prompts), set(expected))
        for stage, digest in expected.items():
            with self.subTest(stage=stage):
                self.assertEqual(hashlib.sha256(prompts[stage].encode()).hexdigest(), digest)

    def test_truncation_retry_parser_and_cached_request_are_preserved(self):
        valid = r'{"applicable":true,"signature":"Check \sqrt{5}","basis":"Visible formula"}'
        create = Mock(side_effect=[response(valid, "length"), response(valid)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        manager = runtime.Manager(self.config, client=client)
        payload = dict(root=dict(root_id="r1"))
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            path = Path(temporary) / "signature.json"
            parsed = manager.call("signature", path, payload)
            saved = path.read_bytes()
            cached = manager.call("signature", path, payload)
            self.assertEqual(path.read_bytes(), saved)
        record = json.loads(saved)
        self.assertEqual(parsed, dict(applicable=True, signature=r"Check \sqrt{5}",
                                      basis="Visible formula"))
        self.assertEqual(cached, parsed)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(record["request"], dict(
            stage="signature", model=self.config["model"], base_url=self.config["base_url"],
            prompt=runtime.PROMPTS["signature"], request_kwargs=self.config["request_kwargs"],
            payload=payload, images=[]))
        self.assertEqual(len(record["attempts"]), 2)
        self.assertIn("response truncated", record["attempts"][0]["error"])
        request = create.call_args_list[1].kwargs
        self.assertEqual(request["messages"][0]["content"], runtime.PROMPTS["signature"])
        self.assertIn("Previous output validation failed: ValueError: response truncated",
                      request["messages"][1]["content"][-1]["text"])

    def test_default_retry_limit_stops_after_ten_attempts(self):
        create = Mock(side_effect=ValueError("offline failure"))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        manager = runtime.Manager(self.config, client=client)
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            path = Path(temporary) / "signature.json"
            with self.assertRaisesRegex(RuntimeError, "failed after 10 attempts"):
                manager.call("signature", path, {})
            self.assertEqual(len(runtime.load_json(path)["attempts"]), 10)
        self.assertEqual(create.call_count, 10)


if __name__ == "__main__":
    unittest.main()

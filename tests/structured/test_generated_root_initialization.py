"""Offline checks for configurable warmup, shared histories, and resume."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from structured_rubrics.agent import AgentCallMetrics
from experiments.evolving_structured_rubrics import generated_root_initialization as generated
from tests.structured.core_fixtures import rows


class OfflineAgent:
    def __init__(self, *, fail_warmup_at=None, calls=None, forks=None):
        self.history = []
        self.client = Mock()
        self.client.with_options.return_value = self.client
        self.api_retry_attempts = 0
        self.last_call_metrics = AgentCallMetrics()
        self.fail_warmup_at = fail_warmup_at
        self.calls = [] if calls is None else calls
        self.forks = [] if forks is None else forks

    def __call__(self, content, stream=False):
        if self.api_retry_attempts != 0:
            raise AssertionError("Root generation must use the explicit retry loop")
        kind = "warmup" if isinstance(content, list) else "generation"
        self.calls.append(kind)
        if kind == "warmup":
            index = len(self.history) // 2 + 1
            if index == self.fail_warmup_at:
                return None
            response = f"Evidence from comparison {index}"
        else:
            count = 5 if "exactly five roots" in content else 3
            response = json.dumps(dict(count_reason="Distinct responsibilities", roots=[
                dict(name=f"criterion_{index}", description=f"Check reusable evidence {index}")
                for index in range(count)]))
        self.history.extend((dict(role="user", content=content),
                             dict(role="assistant", content=response)))
        return response

    def fork(self):
        self.forks.append(deepcopy(self.history))
        branch = OfflineAgent(calls=self.calls, forks=self.forks)
        branch.history = deepcopy(self.history)
        branch.api_retry_attempts = 50
        return branch


class TestGeneratedRootInitialization(unittest.TestCase):
    def fixture(self, target):
        image = target / "image.png"
        image.write_bytes(b"offline-image")
        data = rows(100)
        for row in data:
            row["image_path"] = str(image.resolve())
        config = dict(model="offline-manager", base_url="http://offline/v1",
                      request_kwargs=dict(temperature=0.2))
        return data, config

    def run_pair(self, data, config, target, agent, attempts=10, **kwargs):
        with patch.object(generated, "Manager") as manager, \
                patch.object(generated, "_agent", return_value=agent), redirect_stdout(StringIO()):
            manager.return_value._read_keys.return_value = ["offline-key"]
            return generated.generate_r0_pair(data, seed=11, manager_config=config,
                output_dir=target, attempt_limit=attempts,
                count_instructions=generated.LEGACY_COUNT_INSTRUCTIONS, **kwargs)

    def test_five_sample_selection_preserves_order_mapping_and_global_random_state(self):
        with TemporaryDirectory() as temporary:
            data, _ = self.fixture(Path(temporary))
            frozen = deepcopy(data)
            state = random.getstate()
            selected = generated._warmup_rows(data, 11)
            self.assertEqual(selected, generated._warmup_rows(data, 11, warmup_count=5))
            self.assertEqual(random.getstate(), state)
            self.assertEqual(data, frozen)
            self.assertEqual([(row["sample_id"], row["flipped"]) for row in selected], [
                ("sample-57", True), ("sample-65", True), ("sample-71", True),
                ("sample-99", False), ("sample-59", True)])
            for row in selected:
                self.assertEqual(row["original_answer"], "A")
                self.assertEqual((row["A"], row["B"], row["answer"]),
                                 ("candidate-b", "candidate-a", "B") if row["flipped"] else
                                 ("candidate-a", "candidate-b", "A"))

    def test_g5_and_gn_share_five_replies_and_resume_without_model_calls(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            data, config = self.fixture(target)
            first = OfflineAgent()
            rubrics = self.run_pair(data, config, target, first)
            self.assertEqual(first.calls, ["warmup"] * 5 + ["generation"] * 2)
            self.assertEqual(len(rubrics["g5"].root_ids), 5)
            self.assertEqual(len(rubrics["gn"].root_ids), 3)
            self.assertEqual(first.forks[0], first.forks[1])
            self.assertEqual(len(first.forks[0]), 10)
            g5 = generated.load_json(target / "g5/r0/generation.json")["request"]
            gn = generated.load_json(target / "gn/r0/generation.json")["request"]
            self.assertEqual(g5["warmup_history_sha256"], gn["warmup_history_sha256"])
            self.assertEqual(g5["warmup_request_sha256"], gn["warmup_request_sha256"])
            self.assertIn("From the five comparisons", g5["prompt"])
            self.assertIn("five is allowed", gn["prompt"])
            files = {path: path.read_bytes() for path in target.rglob("*.json")}
            resumed = OfflineAgent()
            restored = self.run_pair(data, config, target, resumed)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(resumed.forks, first.forks)
            self.assertEqual({key: value.to_dict() for key, value in restored.items()},
                             {key: value.to_dict() for key, value in rubrics.items()})
            self.assertEqual({path: path.read_bytes() for path in files}, files)

    def test_partial_warmup_resume_keeps_successful_prefix(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            data, config = self.fixture(target)
            first = OfflineAgent(fail_warmup_at=4)
            with self.assertRaisesRegex(RuntimeError, "warmup sample.*failed after 1 attempts"):
                self.run_pair(data, config, target, first, attempts=1)
            transcript = target / "warmup/transcript.json"
            prefix = deepcopy(generated.load_json(transcript)["turns"][:3])
            resumed = OfflineAgent()
            rubrics = self.run_pair(data, config, target, resumed, attempts=2)
            self.assertEqual(resumed.calls, ["warmup"] * 2 + ["generation"] * 2)
            saved = generated.load_json(transcript)
            self.assertEqual(saved["turns"][:3], prefix)
            self.assertEqual(len(saved["turns"]), 5)
            self.assertEqual(len(saved["turns"][3]["attempts"]), 2)
            self.assertEqual(set(rubrics), {"g5", "gn"})
            self.assertEqual(resumed.forks[0], resumed.forks[1])

    def test_ten_sample_warmup_shares_history_and_resumes_without_model_calls(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            data, config = self.fixture(target)
            first = OfflineAgent()
            rubrics = self.run_pair(data, config, target, first, warmup_count=10)
            self.assertEqual(first.calls, ["warmup"] * 10 + ["generation"] * 2)
            self.assertEqual(len(rubrics["g5"].root_ids), 5)
            self.assertEqual(first.forks[0], first.forks[1])
            self.assertEqual(len(first.forks[0]), 20)
            transcript = generated.load_json(target / "warmup/transcript.json")
            self.assertEqual(transcript["request"]["sample_count"], 10)
            self.assertEqual(len({turn["sample_id"] for turn in transcript["turns"]}), 10)
            g5 = generated.load_json(target / "g5/r0/generation.json")["request"]
            gn = generated.load_json(target / "gn/r0/generation.json")["request"]
            self.assertIn("From the ten comparisons", g5["prompt"])
            self.assertEqual(g5["warmup_history_sha256"], gn["warmup_history_sha256"])
            resumed = OfflineAgent()
            restored = self.run_pair(data, config, target, resumed, warmup_count=10)
            self.assertEqual(resumed.calls, [])
            self.assertEqual({key: value.to_dict() for key, value in restored.items()},
                             {key: value.to_dict() for key, value in rubrics.items()})

    def test_changed_warmup_count_rejects_existing_cache_without_model_calls(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            data, config = self.fixture(target)
            self.run_pair(data, config, target, OfflineAgent())
            files = {path: path.read_bytes() for path in target.rglob("*.json")}
            changed = OfflineAgent()
            with self.assertRaisesRegex(ValueError, "warmup input changed"):
                self.run_pair(data, config, target, changed, warmup_count=10)
            self.assertEqual(changed.calls, [])
            self.assertEqual({path: path.read_bytes() for path in files}, files)

    def test_warmup_count_must_fit_training_rows(self):
        data = rows(100)
        for count in (0, -1, 101):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "warmup_count"):
                generated._warmup_rows(data, 11, warmup_count=count)


if __name__ == "__main__":
    unittest.main()

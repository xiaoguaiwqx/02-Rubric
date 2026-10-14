"""Offline regression checks for backend changes with frozen run snapshots."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.evolving_structured_rubrics import rubric_pipeline as base
from experiments.evolving_structured_rubrics import subtree_local_reflection as local


def configuration(root):
    return dict(
        protocol=local.PROTOCOL, max_epochs=5, data_root=str(root),
        datasets=dict(discovery="discovery.jsonl", dev="dev.jsonl", vlrb="vlrb.parquet"),
        worker=dict(
            model="offline-worker", temperature=0.5, max_tokens=2048,
            api_retry_attempts=0,
            backend_pool=dict(
                pool_id="offline-pool", common_checkpoint_id="offline-checkpoint",
                global_request_concurrency=50,
                endpoints=[dict(
                    endpoint_id="worker-one", base_url="http://offline-one/v1",
                    checkpoint_root="offline-checkpoint", max_concurrency=50)])),
        manager=dict(
            model="offline-manager", base_url="http://offline-manager/v1",
            api_key_env="OFFLINE_API_KEY", concurrency=4, timeout=300,
            stage_concurrency=dict(signature=12),
            request_kwargs=dict(extra_body=dict(enable_thinking=False))),
        acceptance_metric="strict_acc", preservation_case_count=5)


def with_two_worker_endpoints(config):
    current = deepcopy(config)
    pool = current["worker"]["backend_pool"]
    pool["global_request_concurrency"] = 75
    pool["endpoints"][0]["max_concurrency"] = 25
    pool["endpoints"].append(dict(
        endpoint_id="worker-two", base_url="http://offline-two/v1",
        checkpoint_root="offline-checkpoint", max_concurrency=50))
    return current


class TestRuntimeConfiguration(unittest.TestCase):
    def test_local_check_accepts_75_slots_and_keeps_both_frozen_snapshots(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            target = Path(temp)
            frozen = configuration(target)
            legacy = deepcopy(frozen)
            legacy["protocol"] = base.PROTOCOL
            base.write(target / "run_config.json", frozen)
            base.write(target / "checks/run_config.json", legacy)
            snapshots = {
                path: path.read_bytes()
                for path in (target / "run_config.json", target / "checks/run_config.json")}
            image = target / "image.png"
            image.write_bytes(b"offline-image")
            (target / "vlrb.parquet").write_bytes(b"offline-parquet")
            data = {
                split: [dict(sample_id=f"{split}-{index}", image_path=str(image))
                        for index in range(count)]
                for split, count in (("discovery", 100), ("dev", 150))}
            for split, items in data.items():
                base.write(target / "checks" / f"{split}_samples.json", items)
            current = with_two_worker_endpoints(frozen)
            with patch.object(base, "load_rows", side_effect=lambda config, split: data[split]), \
                    patch.object(local, "make_manager") as manager, \
                    patch.object(local.system, "evaluate") as evaluate:
                local.check(current, target)
                manager.assert_not_called()
                evaluate.assert_not_called()
            for path, content in snapshots.items():
                self.assertEqual(path.read_bytes(), content)
            settings = base.load_json(target / "runtime_settings.json")
            self.assertEqual(settings["worker_backend_pool"], current["worker"]["backend_pool"])
            self.assertEqual(settings["manager_timeout"], current["manager"]["timeout"])
            self.assertEqual(settings["manager_stage_concurrency"],
                             current["manager"]["stage_concurrency"])

    def test_backend_override_still_rejects_worker_and_checkpoint_changes(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            target = Path(temp)
            frozen = configuration(target)
            snapshot = target / "run_config.json"
            base.write(snapshot, frozen)
            saved = snapshot.read_bytes()
            changes = (
                (("model",), "different-worker"),
                (("temperature",), 0.7),
                (("max_tokens",), 4096),
                (("backend_pool", "common_checkpoint_id"), "different-checkpoint"),
                (("backend_pool", "pool_id"), "different-pool"))
            for keys, value in changes:
                with self.subTest(field=".".join(keys)):
                    current = with_two_worker_endpoints(frozen)
                    destination = current["worker"]
                    for key in keys[:-1]:
                        destination = destination[key]
                    destination[keys[-1]] = value
                    with self.assertRaisesRegex(ValueError, "run configuration changed"):
                        base.freeze_config(current, target, allow_worker_backend_change=True)
                    self.assertEqual(snapshot.read_bytes(), saved)

    def test_default_freeze_rejects_worker_backend_changes(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            target = Path(temp)
            frozen = configuration(target)
            snapshot = target / "run_config.json"
            base.write(snapshot, frozen)
            saved = snapshot.read_bytes()
            with self.assertRaisesRegex(ValueError, "run configuration changed"):
                base.freeze_config(with_two_worker_endpoints(frozen), target)
            self.assertEqual(snapshot.read_bytes(), saved)

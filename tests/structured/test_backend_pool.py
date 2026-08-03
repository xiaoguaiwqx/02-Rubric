from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from critiq.agent import AgentCallMetrics
from critiq.structured.backend_pool import (
    AvailableSlotBackendPool,
    BackendEndpointSpec,
    BackendPoolSpec,
)


class BackendPoolTest(unittest.TestCase):
    @staticmethod
    def _spec() -> BackendPoolSpec:
        return BackendPoolSpec(
            "pool",
            "checkpoint",
            2,
            (
                BackendEndpointSpec("a", "http://a/v1", "/a", 1),
                BackendEndpointSpec("b", "http://b/v1", "/b", 1),
            ),
        )

    def test_pool_identity_is_stable_and_validates_duplicates(self):
        self.assertEqual(self._spec().backend_id, self._spec().backend_id)
        with self.assertRaises(ValueError):
            BackendPoolSpec(
                "pool",
                "checkpoint",
                2,
                (
                    BackendEndpointSpec("a", "http://a/v1", "/a", 1),
                    BackendEndpointSpec("a", "http://b/v1", "/b", 1),
                ),
            )

    def test_first_available_slots_enforce_endpoint_and_global_limits(self):
        active = {"http://a/v1": 0, "http://b/v1": 0}
        maximum = dict(active)
        lock = threading.Lock()

        class FakeAgent:
            def __init__(self, **kwargs):
                self.base_url = kwargs["base_url"]
                self.last_call_metrics = AgentCallMetrics()

            def __call__(self, prompt, stream=False):
                del prompt, stream
                with lock:
                    active[self.base_url] += 1
                    maximum[self.base_url] = max(
                        maximum[self.base_url], active[self.base_url])
                time.sleep(0.02)
                with lock:
                    active[self.base_url] -= 1
                self.last_call_metrics = AgentCallMetrics(
                    api_attempts=1,
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                    usage_complete=True,
                    latency_seconds=0.02,
                )
                return "ok"

        pool = AvailableSlotBackendPool(self._spec())
        with patch("critiq.structured.backend_pool.Agent", FakeAgent):
            threads = [
                threading.Thread(
                    target=pool.call,
                    args=("x",),
                    kwargs={
                        "request_type": "pairwise",
                        "request_key": str(index),
                        "structured_attempt": 1,
                        "agent_args": {"model": "m"},
                    },
                )
                for index in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(8, len(pool.records))
        self.assertLessEqual(maximum["http://a/v1"], 1)
        self.assertLessEqual(maximum["http://b/v1"], 1)
        self.assertEqual(8, sum(pool.records_by_endpoint().values()))
        self.assertTrue(all(value > 0 for value in pool.records_by_endpoint().values()))
        self.assertTrue(all(record.usage_complete for record in pool.records))


if __name__ == "__main__":
    unittest.main()

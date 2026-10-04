"""Offline regression tests for the frozen S0 initialization procedure."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.evolving_structured_rubrics import rubric_pipeline as pipeline
from experiments.evolving_structured_rubrics import init_split as initialization
from tests.structured.core_fixtures import artifact, report, rows


class TestRubricInitialization(unittest.TestCase):
    def test_primary_errors_expand_with_complete_roots_and_local_evidence(self):
        data = rows(6)
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        for primary_count in (3, 4):
            with self.subTest(primary_count=primary_count), TemporaryDirectory() as temporary:
                target = Path(temporary)
                baseline = artifact(data, r0, 6)
                for index, sample in enumerate(baseline["samples"]):
                    for call in sample["replicates"]["0"]["subtrees"].values():
                        call["parsed"] = report("B" if index < primary_count else "A")
                calls = []

                class Manager:
                    config = dict(concurrency=1)

                    def call(self, stage, path, payload, images=(), *, user_text=None):
                        calls.append((stage, path.relative_to(target), deepcopy(payload),
                                      [item["sample_id"] for item in images], user_text))
                        if stage == "signature":
                            return dict(applicable=True, signature="Verify image evidence",
                                        basis="Supported by the case")
                        ids = [item["signature_id"] for item in payload["signatures"]]
                        if stage == "cluster":
                            # Four errors allow clustering, but one cluster still
                            # requires expansion before S0 can be created.
                            clusters = [dict(pattern="first", signature_ids=ids[:2])]
                            if len(ids) >= 6:
                                clusters.append(dict(pattern="second", signature_ids=ids[2:4]))
                            return dict(clusters=clusters)
                        return dict(children=[dict(name="evidence", description="Check visible evidence"),
                                              dict(name="scope", description="Check question scope")],
                                    change_summary="Initial split")

                evaluations = []

                def evaluate(config, output, name, items, rubric, **kwargs):
                    evaluations.append(name)
                    return baseline if name == "r0/system" else artifact(items, rubric, 6)

                with patch.object(pipeline, "evaluate", side_effect=evaluate):
                    s0, _ = initialization.initialize({}, target, data, Manager(), 10, r0=r0)
                self.assertEqual(evaluations, ["r0/system", "init/system"])
                self.assertEqual(s0.root_ids, r0.root_ids)
                for root in s0.root_ids:
                    self.assertEqual(len(s0.children(root)), 2)
                counts = Counter(stage for stage, _, _, _, _ in calls)
                self.assertEqual(counts["signature"], 30)
                self.assertEqual(counts["cluster"], 5 if primary_count == 3 else 10)
                self.assertEqual(counts["children"], 5)
                for stage, path, payload, images, user_text in calls:
                    self.assertEqual([r["root_id"] for r in payload["roots"]], list(r0.root_ids))
                    self.assertTrue(all(r in user_text for r in r0.root_ids))
                    root = payload["root"]["root_id"]
                    if stage == "signature":
                        self.assertEqual(set(payload), {"roots", "root", "case"})
                        case = payload["case"]
                        self.assertEqual(set(case), {"sample_id", "question", "A", "B", "gold", "worker"})
                        expected = baseline["samples"][int(case["sample_id"].split("-")[-1])]["replicates"]["0"]["subtrees"][root]["parsed"]
                        self.assertEqual(case["worker"], expected)
                        self.assertNotIn("arbiter", str(payload))
                        self.assertEqual(images, [case["sample_id"]])
                    else:
                        self.assertNotIn("revision", payload)
                        self.assertEqual(images, [])
                        self.assertEqual(len(payload["signatures"]), primary_count if "primary" in path.parts else 6)
                        if stage == "children":
                            self.assertIn("expanded", path.parts)
                            self.assertEqual(len(payload["clusters"]["clusters"]), 2)
                state = pipeline.load_json(target / "state.json")
                self.assertEqual((state["epoch"], state["baseline"], state["completed"]),
                                 (0, "init/system", False))

    def test_insufficient_supported_signatures_do_not_create_s0(self):
        data = rows(6)
        r0 = pipeline.build_multicrit_open_ended_init_rubric()
        baseline = artifact(data, r0, 6)

        class Manager:
            config = dict(concurrency=1)

            def call(self, stage, path, payload, images=(), *, user_text=None):
                if stage != "signature":
                    raise AssertionError("Unsupported signatures must not trigger clustering")
                return dict(applicable=False, basis="No supported local failure")

        with TemporaryDirectory() as temporary, patch.object(pipeline, "evaluate", return_value=baseline):
            target = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "insufficient supported patterns"):
                initialization.initialize({}, target, data, Manager(), 10, r0=r0)
            self.assertFalse((target / "init/rubric.json").exists())
            self.assertFalse((target / "state.json").exists())


if __name__ == "__main__":
    unittest.main()

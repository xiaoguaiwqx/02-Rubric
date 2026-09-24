"""Offline checks for the Hallucination100 transfer protocol."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.evolving_structured_rubrics import vlrb_hallucination_transfer as transfer


def record(source: str, index: int) -> dict:
    prefix = {"povid": "hallucination_pair", "rlaif-v": "RLAIF-V",
              "rlhf-v": "RLHF-V"}[source]
    return dict(sample_id=f"{prefix}-{index}", benchmark_id=f"{prefix}-{index}",
                group="hallucination")


class TestHallucinationTransfer(unittest.TestCase):
    def test_prepare_uses_first_frozen_vlrb_order_and_relabels_gold(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            records = [dict(sample_id=f"hallucination_pair-{i}",
                            benchmark_id=f"hallucination_pair-{i}",
                            image_path=f"image-{i}.png", image_sha256=f"hash-{i}",
                            question=f"question-{i}", responses=[f"preferred-{i}", f"other-{i}"],
                            preferred_original_index=0)
                       for i in range(100)]
            split = dict(train_ids=[item["sample_id"] for item in records],
                         heldout_ids=[], excluded_discovery_overlap=[],
                         excluded_train_overlap=[])
            schedule = {item["sample_id"]: (i % 2, 1 - i % 2, i % 2)
                        for i, item in enumerate(records)}
            config = dict(data_root=str(root), datasets=dict(vlrb="source.parquet",
                                                             discovery="old.jsonl"))

            class Rubric:
                rubric_sha256 = "init-hash"

            with (patch.object(transfer, "load_json", return_value=config),
                  patch.object(transfer, "_records", return_value=records),
                  patch.object(transfer.base, "load_rows", return_value=[]),
                  patch.object(transfer, "_features", return_value={}),
                  patch.object(transfer, "select_split", return_value=split),
                  patch.object(transfer, "file_sha256", return_value="source-hash"),
                  patch.object(transfer.StructuredRubric, "load_json", return_value=Rubric()),
                  patch.object(transfer.vlrb, "_order_schedule", return_value=schedule)):
                transfer.prepare(root / "history", root / "output", 11)

            path = root / "output/seed11/discovery_100.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], split["train_ids"])
            self.assertEqual((rows[0]["A"], rows[0]["B"], rows[0]["answer"]),
                             ("preferred-0", "other-0", "A"))
            self.assertEqual((rows[1]["A"], rows[1]["B"], rows[1]["answer"]),
                             ("other-1", "preferred-1", "B"))
            self.assertEqual(Counter(row["answer"] for row in rows), {"A": 50, "B": 50})

    def test_seeded_split_has_quotas_and_excludes_discovery_overlap(self):
        records = ([record("povid", i) for i in range(70)]
                   + [record("rlaif-v", i) for i in range(36)]
                   + [record("rlhf-v", i) for i in range(12)])
        features = {
            item["sample_id"]: (
                int.from_bytes(sha256(item["sample_id"].encode()).digest()[:8], "big"),
                (sha256(item["sample_id"].encode()).hexdigest(),
                 ("answer a", "answer b")))
            for item in records}
        discovery_features = {"old": features[records[0]["sample_id"]]}
        first = transfer.select_split(records, [{"sample_id": "old"}], 11,
                                      features, discovery_features)
        second = transfer.select_split(records, [{"sample_id": "old"}], 11,
                                       features, discovery_features)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train_ids"]), 100)
        self.assertTrue(set(first["train_ids"]).isdisjoint(first["heldout_ids"]))
        self.assertNotIn(records[0]["sample_id"], first["train_ids"])
        self.assertNotIn(records[0]["sample_id"], first["heldout_ids"])
        self.assertEqual(Counter(transfer.vlrb._official_dataset(sid)
                                 for sid in first["train_ids"]), transfer.QUOTAS)

    def test_subset_strict_accuracy_counts_abstention_as_wrong(self):
        records = tuple(dict(sample_id=str(i), preferred_original_index=0)
                        for i in range(3))
        result = transfer._subset(records, [0, 1, None], {"0", "1", "2"})
        self.assertEqual((result["correct"], result["covered"], result["total"]),
                         (1, 2, 3))
        self.assertAlmostEqual(result["strict_acc"], 1 / 3)
        self.assertAlmostEqual(result["covered_acc"], 1 / 2)
        paired = transfer._paired(records, [1, 0, None], [0, 1, None],
                                  {"0", "1", "2"})
        self.assertEqual((paired["corrected"], paired["harmed"]), (1, 1))

    def test_run_seeds_frozen_init_without_initial_split(self):
        class Rubric:
            def to_dict(self):
                return {"frozen": True}

        with TemporaryDirectory() as temp:
            target = Path(temp) / "seed11"
            rows = [dict(sample_id=str(i)) for i in range(100)]
            split = {"train_ids": [str(i) for i in range(100)]}
            with (patch.object(transfer, "_load_prepared",
                               return_value=(target, {"manager": {}}, split)),
                  patch.object(transfer, "check"),
                  patch.object(transfer, "load_dotenv"),
                  patch.object(transfer.base, "load_rows", return_value=rows),
                  patch.object(transfer.StructuredRubric, "load_json", return_value=Rubric()),
                  patch.object(transfer.base, "evaluate") as evaluate,
                  patch.object(transfer.base, "initialize") as initialize,
                  patch.object(transfer.local, "make_manager", return_value="manager") as make_manager,
                  patch.object(transfer.local, "run") as evolve):
                transfer.run(Path(temp) / "old", Path(temp), 11, 4)
            initialize.assert_not_called()
            self.assertEqual(evaluate.call_args.args[2], "init/system")
            self.assertEqual(evolve.call_count, 1)
            self.assertEqual(make_manager.call_args.args[1], 10)
            self.assertEqual(evolve.call_args.kwargs["attempts"], 4)
            self.assertEqual(evolve.call_args.kwargs["manager"], "manager")
            state = transfer.load_json(target / "state.json")
            self.assertEqual(state["rubric"], {"frozen": True})
            self.assertEqual(state["baseline"], "init/system")

    def test_vlrb_evaluates_only_new_final_with_old_init_baseline(self):
        class Rubric:
            root_ids = ("root",)

            def __init__(self, signature):
                self.signature = signature
                self.rubric_sha256 = signature

        with TemporaryDirectory() as temp:
            target = Path(temp) / "seed11"
            transfer.write(target / "state.json", {"completed": True, "rubric": {}})
            records = tuple(dict(sample_id=str(i)) for i in range(1247))
            old = dict(rubric_sha256="init", k=3, samples=[
                dict(sample_id=str(i), replicates={str(j): dict(order=(0, 1, 0)[j])
                                                    for j in range(3)})
                for i in range(1247)])
            orders = {str(i): (0, 1, 0) for i in range(1247)}
            with (patch.object(transfer, "_load_prepared", return_value=(target, {}, {})),
                  patch.object(transfer, "_records", return_value=records),
                  patch.object(transfer.system, "load", return_value=old),
                  patch.object(transfer.StructuredRubric, "load_json", return_value=Rubric("init")),
                  patch.object(transfer.StructuredRubric, "from_dict", return_value=Rubric("final")),
                  patch.object(transfer.vlrb, "_order_schedule", return_value=orders),
                  patch.object(transfer.system, "_root_subtree_sha256",
                               side_effect=lambda rubric, root: rubric.signature),
                  patch.object(transfer.system.support, "vlrb_rows", return_value=[]),
                  patch.object(transfer.base, "evaluate", return_value={}) as evaluate,
                  patch.object(transfer.aligned, "_votes", return_value=[]),
                  patch.object(transfer.official, "_system_metrics",
                               return_value={"strict_accuracy": .7})):
                transfer.vlrb_final(Path(temp) / "old", Path(temp), 11, 4)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(evaluate.call_args.args[2], "vlrb/final")
            self.assertIs(evaluate.call_args.kwargs["baseline"], old)
            self.assertEqual(evaluate.call_args.kwargs["changed"], ["root"])

    def test_summary_averages_seed_deltas_without_pooling_samples(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for seed, init, discovery, final in (
                    (11, .7, .69, .72), (29, .71, .70, .73),
                    (47, .68, .67, .66)):
                metrics = {
                    name: dict(full={"strict_acc": score},
                               heldout={"strict_acc": score})
                    for name, score in (("common_init", init),
                                        ("discovery_final", discovery),
                                        ("hallucination_final", final))}
                transfer.write(root / f"seed{seed}/transfer_report.json",
                               dict(seed=seed, metrics=metrics))
            transfer.summary(root)
            value = transfer.load_json(root / "summary.json")
            self.assertEqual(len(value["rows"]), 3)
            self.assertEqual(value["deltas"]["heldout_delta_vs_init"]["positive_seeds"], 2)
            self.assertAlmostEqual(value["deltas"]["heldout_delta_vs_init"]["mean"],
                                   (.02 + .02 - .02) / 3)


if __name__ == "__main__":
    unittest.main()

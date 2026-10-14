"""Offline preparation and entry checks for frozen VLRB training splits."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.evolving_structured_rubrics import run_subtree_experiment as entry
from experiments.evolving_structured_rubrics import subtree_local_reflection as local
from experiments.evolving_structured_rubrics.experiment_utils import atomic_write_json, load_json
from tests.structured.core_fixtures import rows


class TestVLRBTrainingSplits(unittest.TestCase):
    def fixture(self, root, *, category="reasoning", train_count=70, heldout_count=247):
        parquet = root / "test.parquet"
        parquet.write_bytes(b"offline benchmark")
        prefix = "mathverse" if category == "reasoning" else "hallucination"
        records = [dict(sample_id=f"{prefix}_{i}", benchmark_id=f"{prefix}_{i}",
                        group=category, image_path="shared/image.png", image_sha256="image",
                        question=f"Question {i}", responses=[f"First {i}", f"Second {i}"],
                        preferred_original_index=i % 2)
                   for i in range(train_count + heldout_count)]
        split = dict(seed=11, train_ids=[r["sample_id"] for r in records[:train_count]],
                     heldout_ids=[r["sample_id"] for r in records[train_count:]],
                     parquet_sha256=entry.file_sha256(parquet))
        config = dict(data_root=str(root), datasets={"vlrb": "test.parquet"})
        if category == "reasoning":
            split.update(category=category, train_count=train_count, heldout_count=heldout_count,
                         train_jsonl="discovery_70.jsonl", train_jsonl_sha256="not-created")
            config.update(split_manifest="split.json", discovery_sample_count=train_count)
        return config, split, records

    def test_prepare_reuses_frozen_reasoning_jsonl_without_rewriting_or_copying(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, split, records = self.fixture(root)
            source = root / "discovery_70.jsonl"
            data = rows(70)
            for row, sample_id in zip(data, split["train_ids"]):
                row["sample_id"] = sample_id
            source.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
            split["train_jsonl_sha256"] = entry.file_sha256(source)
            atomic_write_json(root / "split.json", split)
            before = source.read_bytes()
            output = root / "run"
            with patch.object(entry.vlrb_official, "_read_records", return_value=records):
                entry.prepare(config, output, 11)
                first = (output / "config.json").read_bytes()
                entry.prepare(config, output, 11)
            prepared = load_json(output / "config.json")
            self.assertEqual(prepared["discovery_sample_count"], 70)
            self.assertEqual(Path(prepared["datasets"]["discovery"]), source)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual((output / "config.json").read_bytes(), first)
            self.assertFalse((output / "discovery_70.jsonl").exists())

    def test_prepare_rebuilds_missing_reasoning_rows_with_frozen_legacy_ab_mapping(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, split, records = self.fixture(root)
            atomic_write_json(root / "split.json", split)
            with patch.object(entry.vlrb_official, "_read_records", return_value=records), \
                    patch.object(entry.vlrb_official, "_order_schedule",
                                 wraps=entry.vlrb_official._order_schedule) as schedule:
                entry.prepare(config, root / "run", 11)
            schedule.assert_called_once_with(records, protocol=entry.vlrb_official.LEGACY_ORDER_PROTOCOL)
            path = root / "run/discovery_70.jsonl"
            prepared_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_id"] for row in prepared_rows], split["train_ids"])
            orders = entry.vlrb_official._order_schedule(
                records, protocol=entry.vlrb_official.LEGACY_ORDER_PROTOCOL)
            for row, record in zip(prepared_rows, records):
                order = orders[row["sample_id"]][0]
                self.assertEqual((row["A"], row["B"]),
                                 (record["responses"][order], record["responses"][1 - order]))
                self.assertEqual(row["answer"],
                                 "A" if record["preferred_original_index"] == order else "B")
                self.assertEqual(row["source"], "reasoning_tasks")

    def test_prepare_keeps_default_hallucination100_behavior(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, split, records = self.fixture(root, category="hallucination",
                                                   train_count=100, heldout_count=648)
            manifest = root / "hallucination.json"
            atomic_write_json(manifest, split)
            with patch.object(entry, "SPLIT_MANIFEST", manifest), \
                    patch.object(entry.vlrb_official, "_read_records", return_value=records):
                entry.prepare(config, root / "run", 11)
            prepared = load_json(root / "run/config.json")
            self.assertNotIn("discovery_sample_count", prepared)
            path = Path(prepared["datasets"]["discovery"])
            self.assertEqual(path.name, "discovery_100.jsonl")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 100)

    def test_roots_entry_passes_70_rows_and_ten_warmup_examples(self):
        args = SimpleNamespace(stage="roots", variant="g5", seed=11,
                               config=Path("config.json"), output_root=Path("run"),
                               root_attempt_limit=10, warmup_count=10)
        config = dict(discovery_sample_count=70, manager={})
        data = rows(70)
        with redirect_stdout(StringIO()), \
                patch.object(entry, "load_json", return_value=config), \
                patch.object(entry, "load_dotenv"), \
                patch.object(entry.rubric_pipeline, "load_rows", return_value=data), \
                patch.object(entry.generated_root_initialization, "generate_r0_pair") as generate:
            entry.run_stage(args)
        self.assertEqual(generate.call_args.args, (data,))
        self.assertEqual(generate.call_args.kwargs["expected_count"], 70)
        self.assertEqual(generate.call_args.kwargs["warmup_count"], 10)
        self.assertEqual(generate.call_args.kwargs["variants"], ("g5",))

    def test_local_check_accepts_70_discovery_rows_without_model_calls(self):
        path = Path(__file__).resolve().parents[2] / (
            "experiments/evolving_structured_rubrics/configs/reasoning70.example.json")
        config = load_json(path)
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            image = root / "image.png"
            image.write_bytes(b"offline image")
            (root / "test.parquet").write_bytes(b"offline parquet")
            config["data_root"] = str(root)
            config["datasets"] = dict(discovery="train.jsonl", dev="dev.jsonl", vlrb="test.parquet")
            for name, count in (("train", 70), ("dev", 150)):
                data = [dict(sample_id=f"{name}-{i}", image_path=str(image), answer="A")
                        for i in range(count)]
                (root / f"{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
            with patch.object(local, "make_manager") as manager, \
                    patch.object(local.system, "evaluate") as evaluate:
                local.check(config, root / "g5")
                manager.assert_not_called()
                evaluate.assert_not_called()
            self.assertEqual(len(load_json(root / "g5/checks/discovery_samples.json")), 70)


if __name__ == "__main__":
    unittest.main()

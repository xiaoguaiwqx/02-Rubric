"""Offline R0/S0/Final summaries for the frozen Hallucination100 comparison."""

from __future__ import annotations

from collections import Counter
from math import comb
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from structured_rubrics.structured.schema import StructuredRubric

from . import aligned_system_runtime as system
from . import rubric_pipeline, vlrb_official
from .experiment_utils import atomic_write_json, load_json


SPLIT_MANIFEST = (Path(__file__).resolve().parents[2] / "docs/experiments"
                  / "vlrb-hallucination100-generated-roots/seed11_split.json")
VARIANTS = ("f5", "g5", "gn")


def _r0_artifact(target: Path, relative: str) -> Path:
    """Prefer local R0 artifacts, then follow the recorded reuse source."""
    target = target.resolve()
    path = target / relative
    while not path.is_file() and (target / "source.json").is_file():
        target = Path(load_json(target / "source.json")["source_run"])
        path = target / relative
    return path.resolve()


def subsets(records: Sequence[Mapping[str, Any]], split: Mapping[str, Any]) -> dict[str, set[str]]:
    """Use the original 100/648/1146 split, including its near-duplicate exclusion."""
    all_ids = {row["sample_id"] for row in records}
    train = set(split["train_ids"])
    heldout = set(split["heldout_ids"])
    groups = {name: {row["sample_id"] for row in records if row["group"] == name}
              for name in ("general", "hallucination", "reasoning")}
    clean = heldout | groups["general"] | groups["reasoning"]
    if (len(all_ids), len(train), len(heldout), len(clean)) != (1247, 100, 648, 1146):
        raise RuntimeError("frozen VLRB subset sizes changed")
    if train & clean or clean - all_ids:
        raise RuntimeError("clean VLRB subset overlaps training or is absent")
    return dict(full=all_ids, train=train, nontrain_1147=all_ids - train,
                clean_1146=clean, heldout_hallucination=heldout, **groups)


def subset_metrics(records: Sequence[Mapping[str, Any]], predictions: Sequence[int | None],
                   ids: set[str]) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("VLRB prediction count differs from records")
    chosen = [(item, prediction) for item, prediction in zip(records, predictions)
              if item["sample_id"] in ids]
    if len(chosen) != len(ids):
        raise RuntimeError("subset includes IDs missing from VLRB predictions")
    correct = sum(prediction == item["preferred_original_index"]
                  for item, prediction in chosen)
    covered = sum(prediction is not None for _, prediction in chosen)
    count = len(chosen)
    return dict(correct=correct, total=count, strict_acc=correct / count if count else None,
                covered=covered, coverage=covered / count if count else None,
                covered_acc=correct / covered if covered else None)


def paired_metrics(records: Sequence[Mapping[str, Any]], before: Sequence[int | None],
                   after: Sequence[int | None], ids: set[str]) -> dict[str, Any]:
    if len(records) != len(before) or len(records) != len(after):
        raise ValueError("paired VLRB prediction count differs from records")
    corrected, harmed = [], []
    for item, first, second in zip(records, before, after):
        sid = item["sample_id"]
        if sid not in ids:
            continue
        good_first = first == item["preferred_original_index"]
        good_second = second == item["preferred_original_index"]
        if good_second and not good_first:
            corrected.append(sid)
        elif good_first and not good_second:
            harmed.append(sid)
    discordant = len(corrected) + len(harmed)
    tail = sum(comb(discordant, k) for k in range(min(len(corrected), len(harmed)) + 1))
    return dict(corrected=len(corrected), harmed=len(harmed),
                mcnemar_exact_two_sided_p=min(1.0, 2 * tail / (2 ** discordant)),
                corrected_ids=corrected, harmed_ids=harmed)


def paired_bootstrap_ci(records: Sequence[Mapping[str, Any]],
                        before: Sequence[int | None], after: Sequence[int | None],
                        ids: set[str], *, iterations: int = 10_000,
                        seed: int = 20260925) -> dict[str, Any]:
    deltas = [int(after[index] == row["preferred_original_index"])
              - int(before[index] == row["preferred_original_index"])
              for index, row in enumerate(records) if row["sample_id"] in ids]
    if len(deltas) != len(ids):
        raise RuntimeError("paired bootstrap subset differs from frozen IDs")
    generator = random.Random(seed)
    sample_count = len(deltas)
    values = [sum(deltas[generator.randrange(sample_count)]
                  for _ in range(sample_count)) / sample_count
              for _ in range(iterations)]
    values.sort()
    return dict(metric="strict_accuracy_delta", estimate=sum(deltas) / sample_count,
                ci95=[values[int(0.025 * iterations)],
                      values[min(iterations - 1, int(0.975 * iterations))]],
                iterations=iterations, seed=seed)


def _report_orders(records: Sequence[Mapping[str, Any]], path: Path):
    """Validate old reports against their frozen protocol, and new reports against random swaps."""
    protocol = load_json(path).get("order_protocol", vlrb_official.LEGACY_ORDER_PROTOCOL)
    return protocol, vlrb_official._order_schedule(records, protocol=protocol)


def _validated_system(path: Path, rubric: StructuredRubric,
                      records: Sequence[Mapping[str, Any]],
                      orders: Mapping[str, Sequence[int]]) -> tuple[dict, list[int | None]]:
    value = system.load(path)
    ids = [row["sample_id"] for row in records]
    if value["rubric_sha256"] != rubric.rubric_sha256 or value["k"] != 3:
        raise RuntimeError(f"{path}: Rubric or K differs")
    if [row["sample_id"] for row in value["samples"]] != ids:
        raise RuntimeError(f"{path}: VLRB sample order differs")
    if value["metrics"]["technical_failure_count"]:
        raise RuntimeError(f"{path}: unresolved technical failure")
    for sample in value["samples"]:
        if tuple(sample["orders"]) != tuple(orders[sample["sample_id"]]):
            raise RuntimeError(f"{path}: A/B schedule differs for {sample['sample_id']}")
    if value["metrics"] != system.metrics(value, system.support.vlrb_rows(records)):
        raise RuntimeError(f"{path}: saved metrics differ from sample-level predictions")
    predictions = vlrb_official.official_system_metrics(
        records, vlrb_official._votes(value))["original_index_predictions"]
    return value, predictions


def _root_predictions(value: Mapping[str, Any], root_id: str) -> list[int | None]:
    predictions = []
    for sample in value["samples"]:
        votes = []
        for replicate in range(value["k"]):
            item = sample["replicates"][str(replicate)]
            answer = system._answer(item["subtrees"][root_id], int(item["order"]))
            votes.append(0 if answer == "A" else 1 if answer == "B" else None)
        predictions.append(vlrb_official._majority(votes))
    return predictions


def _training_root_metrics(value: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                           rubric: StructuredRubric) -> dict[str, dict]:
    result = {}
    for root in rubric.root_ids:
        predictions = []
        for sample in value["samples"]:
            replicate = sample["replicates"]["0"]
            answer = system._answer(replicate["subtrees"][root], int(replicate["order"]))
            if answer not in {"A", "B", "None"}:
                raise RuntimeError(f"training root {root}: unresolved technical failure")
            predictions.append(answer)
        total = len(rows)
        covered = sum(answer in {"A", "B"} for answer in predictions)
        correct = sum(answer == row["answer"] for answer, row in zip(predictions, rows))
        result[root] = dict(correct=correct, covered=covered, total=total,
                            covered_acc=correct / covered if covered else None,
                            coverage=covered / total, strict_acc=correct / total)
    return result


def _worker_cost(directory: Path) -> dict[str, dict]:
    totals = {kind: dict(logical_calls=0, model_generations=0, api_attempts=0,
                         input_tokens=0, output_tokens=0, latency_seconds=0.0,
                         missing_usage_calls=0, missing_usage_attempts=0)
              for kind in ("subtree", "arbiter")}
    if not directory.is_dir():
        return totals
    for path in directory.rglob("*.json"):
        section = path.relative_to(directory).parts[0]
        if section not in {"r0", "init", "vlrb"} and not (
                len(section) == 3 and section.startswith("e")
                and section[1:].isdigit()):
            continue
        item = load_json(path)
        if "cache_key" not in item:
            continue
        kind = item.get("request", {}).get("request_key", {}).get("kind")
        kind = ("subtree" if kind == "aligned_unified_subtree"
                else "arbiter" if kind == "aligned_global_arbiter" else None)
        if kind is None:
            raise RuntimeError(f"unrecognized Worker cache request: {path}")
        metrics = item.get("metrics", {})
        cost = totals[kind]
        cost["logical_calls"] += 1
        cost["model_generations"] += item.get("model_generation_count", 0)
        cost["api_attempts"] += metrics.get("api_attempts", 0) or 0
        for field in ("input_tokens", "output_tokens", "latency_seconds"):
            cost[field] += metrics.get(field, 0) or 0
        missing = sum(attempt.get("usage_complete") is False
                      or attempt.get("input_tokens") is None
                      or attempt.get("output_tokens") is None
                      for attempt in item.get("attempt_metrics", ()))
        cost["missing_usage_attempts"] += missing
        if missing:
            cost["missing_usage_calls"] += 1
    return totals


def _generation_cost(root: Path, variant: str) -> dict | None:
    if variant == "f5":
        return None
    result = {}
    target = root / variant
    for label, path in (
            ("shared_warmup", _r0_artifact(target, "../warmup/transcript.json")),
            ("root_generation", _r0_artifact(target, "r0/generation.json"))):
        value = load_json(path)
        entries = value["turns"] if label == "shared_warmup" else [value]
        attempts = [attempt for entry in entries for attempt in entry["attempts"]]
        result[label] = dict(
            successful_turns=(sum("response" in entry for entry in entries)
                              if label == "shared_warmup" else int("parsed" in value)),
            attempts=len(attempts),
            api_attempts=sum(item.get("metrics", {}).get("api_attempts", 0)
                             for item in attempts),
            input_tokens=sum(item.get("metrics", {}).get("input_tokens") or 0
                             for item in attempts),
            output_tokens=sum(item.get("metrics", {}).get("output_tokens") or 0
                              for item in attempts),
            latency_seconds=sum(item.get("metrics", {}).get("latency_seconds", 0)
                                for item in attempts),
            missing_usage_attempts=sum(item.get("metrics", {}).get("input_tokens") is None
                                       or item.get("metrics", {}).get("output_tokens") is None
                                       for item in attempts),
        )
    return result


def _rubrics(target: Path) -> dict[str, StructuredRubric]:
    return dict(r0=StructuredRubric.load_json(target / "r0/rubric.json"),
                s0=StructuredRubric.load_json(target / "init/rubric.json"),
                final=StructuredRubric.load_json(target / "final.json"))


def variant_report(config: dict, root: Path, variant: str,
                   records: Sequence[Mapping[str, Any]],
                   slices: Mapping[str, set[str]]) -> None:
    """Append the old experiment's training, slice, root, pair, and cost metrics."""
    target = root / variant
    result = load_json(target / "report.json")
    rows = rubric_pipeline.load_rows(config, "discovery")
    rubrics = _rubrics(target)
    order_protocol, orders = _report_orders(records, target / "vlrb/initial.json")
    systems, roots, predictions = {}, {}, {}
    wall_seconds = {}
    relatives = ["vlrb/r0.json"]
    if variant != "f5":
        relatives += ["r0/generation.json", "../warmup/transcript.json"]
    r0_artifacts = {relative: _r0_artifact(target, relative) for relative in relatives}
    reused_artifacts = {relative: str(path) for relative, path in r0_artifacts.items()
                        if path != (target / relative).resolve()}
    for stage, name in (("r0", "r0"), ("s0", "initial"), ("final", "final")):
        path = r0_artifacts["vlrb/r0.json"] if stage == "r0" else target / f"vlrb/{name}.json"
        value, votes = _validated_system(path, rubrics[stage],
                                         records, orders)
        predictions[stage] = votes
        systems[stage] = {label: subset_metrics(records, votes, ids)
                          for label, ids in slices.items()}
        root_votes = {root_id: _root_predictions(value, root_id)
                      for root_id in rubrics[stage].root_ids}
        roots[stage] = {root_id: {
            label: subset_metrics(records, votes, ids)
            for label, ids in slices.items()}
            for root_id, votes in root_votes.items()}
        wall_seconds[stage + "_vlrb"] = value.get("wall_seconds")
    state = load_json(target / "state.json")
    for stage, relative in (("r0", "r0/system.json"), ("s0", "init/system.json"),
                            ("final", state["baseline"] + ".json")):
        value = system.load(target / relative)
        if ([sample["sample_id"] for sample in value["samples"]]
                != [row["sample_id"] for row in rows]
                or value["k"] != 1
                or value["rubric_sha256"] != rubrics[stage].rubric_sha256):
            raise RuntimeError(f"{target / relative}: training artifact differs")
        if value["metrics"] != system.metrics(value, rows):
            raise RuntimeError(f"{target / relative}: saved training metrics differ")
        systems[stage + "_train"] = value["metrics"]
        roots[stage + "_train"] = _training_root_metrics(value, rows, rubrics[stage])
        wall_seconds[stage + "_train"] = value.get("wall_seconds")
    paired = {}
    for comparison, before, after in (("r0_to_s0", "r0", "s0"),
                                      ("r0_to_final", "r0", "final"),
                                      ("final_vs_s0", "s0", "final")):
        paired[comparison] = {}
        for label, ids in slices.items():
            item = paired_metrics(records, predictions[before], predictions[after], ids)
            if label == "heldout_hallucination":
                item["bootstrap"] = paired_bootstrap_ci(
                    records, predictions[before], predictions[after], ids)
            paired[comparison][label] = item
    epochs = [load_json(path) for path in sorted(target.glob("e[0-9][0-9]/summary.json"))]
    costs = dict(
        nodes={stage: len(rubric.nodes) for stage, rubric in rubrics.items()},
        epochs_completed=state["epoch"],
        evolution_wall_seconds=sum(item.get("wall_seconds", 0) for item in epochs),
        manager=rubric_pipeline.manager_cost([target / "init", *sorted(target.glob("e[0-9][0-9]"))]),
        worker=_worker_cost(target / "cache"),
        root_generation=_generation_cost(root, variant),
        wall_seconds_by_artifact=wall_seconds,
        source_output=str(target),
        historical_cost_reused=bool(reused_artifacts),
        reused_artifacts=reused_artifacts,
    )
    generation = None
    if variant != "f5":
        item = load_json(r0_artifacts["r0/generation.json"])
        generation = dict(count_reason=item["parsed"]["count_reason"],
                          warmup_history_sha256=item["request"]["warmup_history_sha256"],
                          attempt_count=len(item["attempts"]))
    result["generated_roots"] = dict(
        order_protocol=order_protocol, order_seed=vlrb_official.SEED,
        root_count=len(rubrics["r0"].root_ids),
        root_catalog=[dict(root_id=root_id,
                           name=rubrics["r0"].get_node(root_id).criterion.name,
                           description=rubrics["r0"].get_node(root_id).criterion.description)
                      for root_id in rubrics["r0"].root_ids],
        root_generation=generation, systems=systems, roots=roots,
        paired=paired, costs=costs)
    atomic_write_json(target / "report.json", result)
    heldout = systems["final"]["heldout_hallucination"]
    print(f"{variant} Final heldout: {heldout['correct']}/{heldout['total']} "
          f"({heldout['strict_acc']:.2%})", flush=True)


def artifact_worker_cost(values: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    """Sum saved compact telemetry without traversing thousands of cache files."""
    totals = {kind: dict(logical_calls=0, model_generations=0, api_attempts=0,
                         input_tokens=0, output_tokens=0, latency_seconds=0.0,
                         missing_usage_calls=0) for kind in ("subtree", "arbiter")}
    seen = set()
    for value in values:
        for sample in value["samples"]:
            for replicate in sample["replicates"].values():
                groups = (("subtree", replicate["subtrees"].values()),
                          ("arbiter", [replicate["arbiter"]]))
                for kind, calls in groups:
                    for call in calls:
                        key = call.get("cache_key")
                        if key is not None:
                            if key in seen:
                                continue
                            seen.add(key)
                        metrics = call.get("metrics", {})
                        cost = totals[kind]
                        cost["logical_calls"] += 1
                        cost["model_generations"] += call.get("model_generation_count", 0)
                        for field in ("api_attempts", "input_tokens", "output_tokens",
                                      "latency_seconds"):
                            cost[field] += metrics.get(field, 0) or 0
                        if (metrics.get("usage_complete") is False
                                or metrics.get("input_tokens") is None
                                or metrics.get("output_tokens") is None):
                            cost["missing_usage_calls"] += 1
    return totals


def init_report(config: dict, target: Path, source: Path) -> dict:
    """Compare a frozen new S0 with the saved R0 and old S0, without Final."""
    from .model_call_support import file_sha256

    source = Path(source).resolve()
    if (target / "source.json").is_file():
        provenance = load_json(target / "source.json")
        if Path(provenance["source_run"]).resolve() != source:
            raise ValueError("comparison source differs from initialization source")
        if any(file_sha256(source / name) != digest
               for name, digest in provenance["source_files"].items()):
            raise ValueError("saved comparison source changed")
    r0 = StructuredRubric.load_json(source / "r0/rubric.json")
    reused = StructuredRubric.load_json(target / "r0/rubric.json")
    if r0.rubric_sha256 != reused.rubric_sha256:
        raise ValueError("new initialization did not use the comparison R0")
    rubrics = dict(r0=r0, old_s0=StructuredRubric.load_json(source / "init/rubric.json"),
                   new_s0=StructuredRubric.load_json(target / "init/rubric.json"))
    for rubric in rubrics.values():
        if (rubric.root_ids != r0.root_ids
                or any(rubric.get_node(root).to_dict() != r0.get_node(root).to_dict()
                       for root in r0.root_ids)):
            raise ValueError("initialization changed the frozen roots")
    parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
    records = vlrb_official._read_records(target / "vlrb", parquet_path=parquet)
    slices = subsets(records, load_json(SPLIT_MANIFEST))
    order_protocol, orders = _report_orders(records, target / "vlrb/initial.json")
    systems, roots, official, predictions, values = {}, {}, {}, {}, {}
    r0_predictions = _r0_artifact(source, "vlrb/r0.json")
    for label, path in (("r0", r0_predictions),
                        ("old_s0", source / "vlrb/initial.json"),
                        ("new_s0", target / "vlrb/initial.json")):
        value, votes = _validated_system(path, rubrics[label], records, orders)
        values[label], predictions[label] = value, votes
        systems[label] = {name: subset_metrics(records, votes, ids)
                          for name, ids in slices.items()}
        roots[label] = {root: {name: subset_metrics(
            records, _root_predictions(value, root), ids) for name, ids in slices.items()}
            for root in r0.root_ids}
        metrics = vlrb_official.official_system_metrics(records, vlrb_official._votes(value))
        official[label] = {key: item for key, item in metrics.items()
                           if key not in {"original_index_predictions", "predictions_by_order"}}
    paired = {}
    for label, before, after in (("r0_to_old_s0", "r0", "old_s0"),
                                 ("r0_to_new_s0", "r0", "new_s0"),
                                 ("old_s0_to_new_s0", "old_s0", "new_s0")):
        paired[label] = {}
        for name, ids in slices.items():
            item = paired_metrics(records, predictions[before], predictions[after], ids)
            item["net_corrected"] = item["corrected"] - item["harmed"]
            if name in {"heldout_hallucination", "clean_1146", "nontrain_1147"}:
                item["bootstrap"] = paired_bootstrap_ci(
                    records, predictions[before], predictions[after], ids)
            paired[label][name] = item
    training = {label: load_json(directory / "init/system.json")
                for label, directory in (("old_s0", source), ("new_s0", target))}
    rows = rubric_pipeline.load_rows(config, "discovery")
    if {r["sample_id"] for r in rows} != slices["train"]:
        raise ValueError("initialization discovery set differs from the frozen split")
    for label, value in training.items():
        if (value["k"] != 1 or value["rubric_sha256"] != rubrics[label].rubric_sha256
                or [s["sample_id"] for s in value["samples"]] != [r["sample_id"] for r in rows]
                or value["metrics"]["technical_failure_count"]
                or value["metrics"] != system.metrics(value, rows)):
            raise ValueError(f"{label}: discovery report differs")
    result = dict(
        protocol="init-split-comparison-v1", k=3, source_run=str(source),
        order_protocol=order_protocol, order_seed=vlrb_official.SEED,
        rubric_sha256={label: rubric.rubric_sha256 for label, rubric in rubrics.items()},
        systems=systems, roots=roots, official=official, paired=paired,
        discovery_k1={label: value["metrics"] for label, value in training.items()},
        initialization=load_json(target / "init/summary.json"),
        costs=dict(
            old_s0=dict(manager=rubric_pipeline.manager_cost([source / "init"]),
                        worker=artifact_worker_cost([training["old_s0"], values["old_s0"]])),
            new_s0=dict(manager=rubric_pipeline.manager_cost([target / "init"]),
                        worker=artifact_worker_cost([training["new_s0"], values["new_s0"]])),
            reused_r0_new_calls=0,
            vlrb_wall_seconds={label: value.get("wall_seconds") for label, value in values.items()},
        ),
    )
    atomic_write_json(target / "init_comparison.json", result)
    for label in ("r0", "old_s0", "new_s0"):
        metrics = systems[label]["heldout_hallucination"]
        print(f"{label} heldout: {metrics['correct']}/{metrics['total']} "
              f"({metrics['strict_acc']:.2%}, K=3)", flush=True)
    return result


def combined_report(root: Path, records: Sequence[Mapping[str, Any]],
                    slices: Mapping[str, set[str]]) -> None:
    """Write F5/G5/GN comparisons after all three variant reports exist."""
    paths = [root / variant / "report.json" for variant in VARIANTS]
    if not all(path.is_file() for path in paths):
        return
    reports = {variant: load_json(root / variant / "report.json").get("generated_roots")
               for variant in VARIANTS}
    if not all(reports.values()):
        return
    warmup = load_json(_r0_artifact(root / "g5", "../warmup/transcript.json"))
    samples = warmup["request"]["samples"]
    result = dict(
        seed=load_json(SPLIT_MANIFEST)["seed"],
        warmup=dict(sample_ids=[sample["sample_id"] for sample in samples],
                    source_counts=dict(Counter(sample["source"] for sample in samples)),
                    flipped_sample_ids=[sample["sample_id"] for sample in samples
                                        if sample["flipped"]]),
        root_counts={v: reports[v]["root_count"] for v in VARIANTS},
        root_catalog={v: reports[v]["root_catalog"] for v in VARIANTS},
        root_generation={v: reports[v]["root_generation"] for v in ("g5", "gn")},
        systems={v: reports[v]["systems"] for v in VARIANTS},
        roots={v: reports[v]["roots"] for v in VARIANTS},
        costs={v: reports[v]["costs"] for v in VARIANTS}, paired={})
    order_protocol, orders = _report_orders(records, root / "f5/vlrb/initial.json")
    result["order_protocol"] = order_protocol
    result["order_seed"] = vlrb_official.SEED
    predictions = {}
    for variant in VARIANTS:
        rubrics = _rubrics(root / variant)
        for stage, name in (("r0", "r0"), ("s0", "initial"), ("final", "final")):
            path = (_r0_artifact(root / variant, "vlrb/r0.json") if stage == "r0"
                    else root / variant / f"vlrb/{name}.json")
            _, predictions[variant, stage] = _validated_system(
                path, rubrics[stage], records, orders)
    comparisons = (
        ("gn_final_vs_f5_final", ("f5", "final"), ("gn", "final")),
        ("g5_final_vs_f5_final", ("f5", "final"), ("g5", "final")),
        ("gn_final_vs_g5_final", ("g5", "final"), ("gn", "final")),
        *((f"{variant}_final_vs_s0", (variant, "s0"), (variant, "final"))
          for variant in VARIANTS),
        *((f"{variant}_s0_vs_r0", (variant, "r0"), (variant, "s0"))
          for variant in VARIANTS),
        *((f"{variant}_final_vs_r0", (variant, "r0"), (variant, "final"))
          for variant in VARIANTS),
    )
    for label, before_key, after_key in comparisons:
        before, after = predictions[before_key], predictions[after_key]
        result["paired"][label] = {}
        for name, ids in slices.items():
            item = paired_metrics(records, before, after, ids)
            if name == "heldout_hallucination":
                item["bootstrap"] = paired_bootstrap_ci(records, before, after, ids)
            result["paired"][label][name] = item
    atomic_write_json(root / "report.json", result)


def report(config: dict, root: Path, variant: str) -> None:
    rows = rubric_pipeline.load_rows(config, "discovery")
    split = load_json(SPLIT_MANIFEST)
    if [row["sample_id"] for row in rows] != split["train_ids"]:
        return  # The original Discovery100 protocol uses the ordinary subtree report.
    target = root / variant
    required = [_r0_artifact(target, "vlrb/r0.json"),
                target / "vlrb/initial.json", target / "vlrb/final.json"]
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise RuntimeError(f"complete R0/S0/Final VLRB evaluation before report: {missing}")
    parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
    records = vlrb_official._read_records(target / "vlrb", parquet_path=parquet)
    slices = subsets(records, split)
    variant_report(config, root, variant, records, slices)
    combined_report(root, records, slices)

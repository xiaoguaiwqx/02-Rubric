"""Three-seed VLRB Hallucination100 evolution with a shared frozen Init.

Only each new Final receives VLRB inference. Historical Init and Discovery
Final predictions are reused for all full-set and held-out comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import json
from math import comb
from pathlib import Path
import random
import re

from dotenv import load_dotenv
from PIL import Image, ImageOps
from critiq.structured.schema import StructuredRubric

from . import aligned_system_runtime as system
from . import framework_v6 as base
from . import subtree_local_reflection as local
from . import vl_rewardbench as vlrb
from . import vl_rewardbench_aligned_evolution as aligned
from . import vl_rewardbench_phase10 as official
from .experiment_utils import atomic_write_json as write, load_json


SEEDS = (11, 29, 47)
QUOTAS = {"povid": 60, "rlaif-v": 31, "rlhf-v": 9}
DEFAULT_HISTORY = base.ROOT / "output/subtree_local_reflection/discovery100_27b_strict_preserve5"
DEFAULT_OUTPUT = base.ROOT / "output/vlrb_hallucination100_transfer"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def _text_key(question: str, responses: list[str] | tuple[str, str]) -> tuple[str, tuple[str, str]]:
    return _normal(question), tuple(sorted(_normal(item) for item in responses))


def _near_text(left: tuple[str, tuple[str, str]], right: tuple[str, tuple[str, str]]) -> bool:
    if left == right:
        return True
    question_a, answers_a = left
    question_b, answers_b = right
    if min(len(question_a), len(question_b)) < .85 * max(len(question_a), len(question_b)):
        return False
    if SequenceMatcher(None, question_a, question_b).ratio() < .92:
        return False
    return all(SequenceMatcher(None, x, y).ratio() >= .92
               for x, y in zip(answers_a, answers_b))


def _image_hash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8))
        pixels = [image.getpixel((col, row)) for row in range(8) for col in range(9)]
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return result


def _features(records: list[dict], *, old: bool = False) -> dict[str, tuple[int, tuple]]:
    result = {}
    for item in records:
        sid = str(item["sample_id"])
        responses = [item["A"], item["B"]] if old else item["responses"]
        result[sid] = (_image_hash(Path(item["image_path"])),
                       _text_key(item["question"], responses))
    return result


def _overlap(feature: tuple[int, tuple], others: list[tuple[int, tuple]]) -> str | None:
    image_hash, text_key = feature
    for other_hash, other_text in others:
        if (image_hash ^ other_hash).bit_count() <= 4:
            return "same_or_near_image"
        if _near_text(text_key, other_text):
            return "same_or_near_pair"
    return None


def select_split(records: list[dict], discovery: list[dict], seed: int,
                 features: dict[str, tuple[int, tuple]],
                 discovery_features: dict[str, tuple[int, tuple]]) -> dict:
    """Freeze source-stratified training IDs before any model predictions."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    old_features = list(discovery_features.values())
    eligible, cross_excluded = [], []
    for item in records:
        if item["group"] != "hallucination":
            continue
        reason = _overlap(features[item["sample_id"]], old_features)
        if reason:
            cross_excluded.append(dict(sample_id=item["sample_id"], reason=reason))
        else:
            eligible.append(item)
    selected = []
    selected_features = []
    for source, quota in QUOTAS.items():
        candidates = sorted((item for item in eligible
                             if vlrb._official_dataset(item["benchmark_id"]) == source),
                            key=lambda item: item["sample_id"])
        random.Random(f"{seed}:{source}").shuffle(candidates)
        for item in candidates:
            if _overlap(features[item["sample_id"]], selected_features):
                continue
            selected.append(item)
            selected_features.append(features[item["sample_id"]])
            if sum(vlrb._official_dataset(x["benchmark_id"]) == source
                   for x in selected) == quota:
                break
        else:
            raise RuntimeError(f"{source}: cannot select {quota} unique pairs")
    chosen = {item["sample_id"] for item in selected}
    heldout, train_excluded = [], []
    for item in eligible:
        if item["sample_id"] in chosen:
            continue
        reason = _overlap(features[item["sample_id"]], selected_features)
        if reason:
            train_excluded.append(dict(sample_id=item["sample_id"], reason=reason))
        else:
            heldout.append(item["sample_id"])
    if len(selected) != 100 or len(chosen) != 100:
        raise RuntimeError("training split is not exactly 100 unique pairs")
    return dict(seed=seed, source_quotas=QUOTAS,
                train_ids=[item["sample_id"] for item in selected],
                heldout_ids=heldout, excluded_discovery_overlap=cross_excluded,
                excluded_train_overlap=train_excluded,
                eligible_source_counts=dict(Counter(
                    vlrb._official_dataset(item["benchmark_id"]) for item in eligible)))


def _records(config: dict, output_root: Path) -> tuple[dict, ...]:
    parquet = Path(config["data_root"]) / config["datasets"]["vlrb"]
    return vlrb._read_records(output_root / "dataset", parquet_path=parquet)


def _path(output_root: Path, seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    return output_root / f"seed{seed}"


def prepare(history: Path, output_root: Path, seed: int) -> None:
    target = _path(output_root, seed)
    prior = load_json(history / "run_config.json")
    records = list(_records(prior, output_root))
    discovery = base.load_rows(prior, "discovery")
    features = _features(records)
    old_features = _features(discovery, old=True)
    split = select_split(records, discovery, seed, features, old_features)
    split.update(parquet_sha256=file_sha256(
        Path(prior["data_root"]) / prior["datasets"]["vlrb"]),
        discovery_sha256=file_sha256(
            Path(prior["data_root"]) / prior["datasets"]["discovery"]),
        historical_init_sha256=StructuredRubric.load_json(
            history / "init/rubric.json").rubric_sha256)
    selected = set(split["train_ids"])
    by_id = {item["sample_id"]: item for item in records}
    first_order = {sid: order[0] for sid, order in vlrb._order_schedule(records).items()}
    rows = []
    for sid in split["train_ids"]:
        item = by_id[sid]
        order = first_order[sid]
        rows.append(dict(sample_id=sid, image_path=item["image_path"],
                         image_sha256=item["image_sha256"], question=item["question"],
                         A=item["responses"][order], B=item["responses"][1 - order],
                         answer="A" if item["preferred_original_index"] == order else "B",
                         source=vlrb._official_dataset(item["benchmark_id"]),
                         domain="visual"))
    if len(selected) != 100 or len(rows) != 100:
        raise RuntimeError("prepared training set is not exactly 100 pairs")
    config = deepcopy(prior)
    config["datasets"]["discovery"] = str((target / "discovery_100.jsonl").resolve())
    payloads = {
        target / "split.json": json.dumps(split, ensure_ascii=False, indent=2) + "\n",
        target / "discovery_100.jsonl": "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        target / "config.json": json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    }
    for path, content in payloads.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"frozen preparation differs: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    print(f"seed={seed}: train=100 heldout={len(split['heldout_ids'])} "
          f"discovery_overlap={len(split['excluded_discovery_overlap'])} "
          f"train_overlap={len(split['excluded_train_overlap'])}", flush=True)


def _load_prepared(history: Path, output_root: Path, seed: int) -> tuple[Path, dict, dict]:
    target = _path(output_root, seed)
    config = load_json(target / "config.json")
    split = load_json(target / "split.json")
    source = load_json(history / "run_config.json")
    expected = deepcopy(source)
    expected["datasets"]["discovery"] = str((target / "discovery_100.jsonl").resolve())
    if config != expected or split["seed"] != seed:
        raise RuntimeError("prepared config or seed differs from historical protocol")
    if split["historical_init_sha256"] != StructuredRubric.load_json(
            history / "init/rubric.json").rubric_sha256:
        raise RuntimeError("historical Init changed")
    if split["parquet_sha256"] != file_sha256(
            Path(config["data_root"]) / config["datasets"]["vlrb"]):
        raise RuntimeError("VLRB parquet changed after sampling")
    if split["discovery_sha256"] != file_sha256(
            Path(source["data_root"]) / source["datasets"]["discovery"]):
        raise RuntimeError("historical Discovery100 changed after sampling")
    return target, config, split


def check(history: Path, output_root: Path, seed: int) -> None:
    target, config, split = _load_prepared(history, output_root, seed)
    local.check(config, target)
    rows = base.load_rows(config, "discovery")
    if [row["sample_id"] for row in rows] != split["train_ids"]:
        raise RuntimeError("training sample order changed")
    print(f"seed={seed}: prepared split and frozen config checked", flush=True)


def run(history: Path, output_root: Path, seed: int, attempts: int,
        manager_attempts: int = 10) -> None:
    target, config, split = _load_prepared(history, output_root, seed)
    load_dotenv(config.get("env_file"), override=False)
    check(history, output_root, seed)
    rows = base.load_rows(config, "discovery")
    if [row["sample_id"] for row in rows] != split["train_ids"]:
        raise RuntimeError("training sample order changed")
    init = StructuredRubric.load_json(history / "init/rubric.json")
    if not (target / "state.json").exists():
        write(target / "init/rubric.json", init.to_dict())
        base.evaluate(config, target, "init/system", rows, init, attempts=attempts)
        write(target / "state.json", dict(epoch=0, rubric=init.to_dict(),
              baseline="init/system", completed=False, stop_reason=None))
    manager = local.make_manager(
        dict(config["manager"], env_file=config.get("env_file", ".env")),
        manager_attempts)
    local.run(config, target, attempts=attempts, manager=manager)


def vlrb_final(history: Path, output_root: Path, seed: int, attempts: int) -> None:
    target, config, split = _load_prepared(history, output_root, seed)
    state = load_json(target / "state.json")
    if not state["completed"]:
        raise RuntimeError("evolution must finish before VLRB evaluation")
    records = _records(config, output_root)
    if len(records) != 1247:
        raise RuntimeError("VLRB row count changed")
    old = system.load(history / "vlrb/initial.json")
    init = StructuredRubric.load_json(history / "init/rubric.json")
    final = StructuredRubric.from_dict(state["rubric"])
    orders = vlrb._order_schedule(records)
    if old["rubric_sha256"] != init.rubric_sha256 or old["k"] != 3:
        raise RuntimeError("historical Init VLRB result is incompatible")
    if [item["sample_id"] for item in old["samples"]] != [item["sample_id"] for item in records]:
        raise RuntimeError("historical Init VLRB sample order differs")
    if any(tuple(item["replicates"][str(index)]["order"] for index in range(3))
           != orders[item["sample_id"]] for item in old["samples"]):
        raise RuntimeError("historical Init VLRB K=3 order differs")
    changed = [root for root in final.root_ids
               if system._root_subtree_sha256(init, root)
               != system._root_subtree_sha256(final, root)]
    if changed:
        result = base.evaluate(config, target, "vlrb/final",
                               system.support.vlrb_rows(records), final,
                               baseline=old, changed=changed, orders=orders,
                               attempts=attempts)
    else:
        result = old
        write(target / "vlrb/final.json", old)
    metrics = official._system_metrics(records, aligned._votes(result))
    write(target / "vlrb/official_final.json", metrics)
    print(f"seed={seed}: VLRB official Strict ACC={metrics['strict_accuracy']:.2%}", flush=True)


def _subset(records: tuple[dict, ...], predictions: list[int | None],
            ids: set[str]) -> dict:
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


def _paired(records: tuple[dict, ...], before: list[int | None],
            after: list[int | None], ids: set[str]) -> dict:
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
    p_value = min(1.0, 2 * tail / (2 ** discordant))
    return dict(corrected=len(corrected), harmed=len(harmed),
                mcnemar_exact_two_sided_p=p_value,
                corrected_ids=corrected, harmed_ids=harmed)


def _historical_metrics(history: Path, records: tuple[dict, ...]) -> dict:
    stored = load_json(history / "vlrb/report.json")["official"]
    ids = [item["sample_id"] for item in records]
    schedule = vlrb._order_schedule(records)
    result = {}
    for label, artifact, rubric_file in (
            ("initial", "initial", "init/rubric.json"),
            ("final", "final", "final.json")):
        value = system.load(history / f"vlrb/{artifact}.json")
        rubric = StructuredRubric.load_json(history / rubric_file)
        if value["rubric_sha256"] != rubric.rubric_sha256 or value["k"] != 3:
            raise RuntimeError(f"historical {label} rubric or K differs")
        if [item["sample_id"] for item in value["samples"]] != ids:
            raise RuntimeError(f"historical {label} sample IDs differ")
        if any(tuple(item["replicates"][str(index)]["order"] for index in range(3))
               != schedule[item["sample_id"]] for item in value["samples"]):
            raise RuntimeError(f"historical {label} A/B schedule differs")
        metrics = official._system_metrics(records, aligned._votes(value))
        if metrics["original_index_predictions"] != stored[label]["original_index_predictions"]:
            raise RuntimeError(f"historical {label} official predictions differ")
        if (metrics["correct_count"] != stored[label]["correct_count"]
                or metrics["groups"] != stored[label]["groups"]
                or metrics["source_groups"] != stored[label]["source_groups"]):
            raise RuntimeError(f"historical {label} Gold or group metrics differ")
        result[label] = metrics
    return result


def report(history: Path, output_root: Path, seed: int) -> None:
    target, config, split = _load_prepared(history, output_root, seed)
    records = _records(config, output_root)
    old = _historical_metrics(history, records)
    stored_new = load_json(target / "vlrb/official_final.json")
    new_artifact = system.load(target / "vlrb/final.json")
    final = StructuredRubric.from_dict(load_json(target / "state.json")["rubric"])
    if new_artifact["rubric_sha256"] != final.rubric_sha256:
        raise RuntimeError("new Final VLRB artifact differs from frozen rubric")
    if [item["sample_id"] for item in new_artifact["samples"]] != [
            item["sample_id"] for item in records]:
        raise RuntimeError("new Final VLRB sample order differs")
    new = official._system_metrics(records, aligned._votes(new_artifact))
    if new["original_index_predictions"] != stored_new["original_index_predictions"]:
        raise RuntimeError("new Final official predictions differ from saved results")
    predictions = {
        "common_init": old["initial"]["original_index_predictions"],
        "discovery_final": old["final"]["original_index_predictions"],
        "hallucination_final": new["original_index_predictions"],
    }
    if any(len(value) != len(records) for value in predictions.values()):
        raise RuntimeError("official prediction array length differs from VLRB")
    heldout = set(split["heldout_ids"])
    full = {item["sample_id"] for item in records}
    groups = {name: {item["sample_id"] for item in records if item["group"] == name}
              for name in ("general", "hallucination", "reasoning")}
    sources = {name: {item["sample_id"] for item in records
                      if vlrb._official_dataset(item["benchmark_id"]) == name}
               for name in QUOTAS}
    metrics = {}
    for label, values in predictions.items():
        metrics[label] = dict(full=_subset(records, values, full),
                              heldout=_subset(records, values, heldout),
                              groups={name: _subset(records, values, ids)
                                      for name, ids in groups.items()},
                              hallucination_sources={name: _subset(records, values, ids)
                                                     for name, ids in sources.items()},
                              heldout_hallucination_sources={
                                  name: _subset(records, values, ids & heldout)
                                  for name, ids in sources.items()})
    paired = {label: _paired(records, predictions["common_init"], values, heldout)
              for label, values in predictions.items() if label != "common_init"}
    paired["new_vs_discovery"] = _paired(
        records, predictions["discovery_final"], predictions["hallucination_final"], heldout)
    result = dict(seed=seed, train_ids=split["train_ids"],
                  heldout_ids=split["heldout_ids"], metrics=metrics, paired=paired,
                  historical_output=str(history), new_output=str(target))
    write(target / "transfer_report.json", result)
    print(json.dumps({name: value["heldout"] for name, value in metrics.items()},
                     ensure_ascii=False), flush=True)


def summary(output_root: Path) -> None:
    reports = [load_json(_path(output_root, seed) / "transfer_report.json")
               for seed in SEEDS]
    rows = []
    for seed, item in zip(SEEDS, reports):
        if item["seed"] != seed:
            raise RuntimeError(f"seed{seed} transfer report has wrong seed")
        metrics = item["metrics"]
        rows.append(dict(seed=seed,
            full={name: value["full"] for name, value in metrics.items()},
            heldout={name: value["heldout"] for name, value in metrics.items()},
            heldout_delta_vs_init=(metrics["hallucination_final"]["heldout"]["strict_acc"]
                                   - metrics["common_init"]["heldout"]["strict_acc"]),
            heldout_delta_vs_discovery=(metrics["hallucination_final"]["heldout"]["strict_acc"]
                                        - metrics["discovery_final"]["heldout"]["strict_acc"])))
    deltas = {}
    for key in ("heldout_delta_vs_init", "heldout_delta_vs_discovery"):
        values = [row[key] for row in rows]
        deltas[key] = dict(mean=sum(values) / len(values), minimum=min(values),
                           maximum=max(values), positive_seeds=sum(v > 0 for v in values))
    result = dict(seeds=list(SEEDS), rows=rows, deltas=deltas,
                  note="Held-out sets overlap across seeds; do not pool rows as independent samples.")
    write(output_root / "summary.json", result)
    print(json.dumps(deltas, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "run", "vlrb", "report", "summary"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--historical-output", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt-limit", type=int, default=4)
    parser.add_argument("--manager-attempt-limit", type=int, default=10)
    args = parser.parse_args()
    if args.attempt_limit < 1:
        parser.error("attempt-limit must be positive")
    if args.manager_attempt_limit < 1:
        parser.error("manager-attempt-limit must be positive")
    if args.stage != "summary" and args.seed is None:
        parser.error("--seed is required except for summary")
    history = args.historical_output.resolve()
    output_root = args.output_root.resolve()
    if args.stage == "summary":
        summary(output_root)
    elif args.stage == "prepare":
        prepare(history, output_root, args.seed)
    elif args.stage == "check":
        check(history, output_root, args.seed)
    elif args.stage == "run":
        run(history, output_root, args.seed, args.attempt_limit,
            args.manager_attempt_limit)
    elif args.stage == "vlrb":
        vlrb_final(history, output_root, args.seed, args.attempt_limit)
    else:
        report(history, output_root, args.seed)


if __name__ == "__main__":
    main()

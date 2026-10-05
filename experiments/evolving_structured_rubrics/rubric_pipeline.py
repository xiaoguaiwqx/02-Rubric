"""Shared Rubric construction, data, evaluation, configuration, and costs."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from structured_rubrics.structured.schema import StructuredRubric, RubricNode, RubricCriterionSnapshot, RubricEdge
from structured_rubrics.structured.semantics import EdgeCondition
from . import aligned_system_runtime as system
from .experiment_utils import atomic_write_json as write, load_json
from .manager_runtime import nonempty

PROTOCOL = "framework-v6-hierarchical-v1"


MULTICRIT_OPEN_ENDED_CRITERIA = (
    (
        "completeness_and_coverage",
        "Completeness and Coverage",
        "Address the full scope of the task in the user’s query, covering all major elements specified in the prompt as well as relevant visual aspects and contextual cues.",
    ),
    (
        "visual_grounding_and_details",
        "Visual Grounding and Details",
        "Reference observable elements in the image such as objects, spatial relationships, colors, or text, and bases its description or analysis on these details.",
    ),
    (
        "factuality_no_hallucination",
        "Factuality / No Hallucination",
        "Avoid visual or factual errors, ensuring all details and claims are presented in the image or reasonably supported by the prompt.",
    ),
    (
        "creativity_and_expressiveness",
        "Creativity and Expressiveness",
        "Demonstrates imagination and originality when appropriate, or precise and knowledgeable articulation for analytical tasks, while remaining contextually appropriate.",
    ),
    (
        "clarity_and_coherence",
        "Clarity and Coherence",
        "Communicates ideas clearly and logically, with fluent language, well-organized structure, and smooth flow of information.",
    ),
)


def build_multicrit_open_ended_init_rubric() -> StructuredRubric:
    """Build the deterministic five-root Phase 5 initialization rubric."""

    nodes: dict[str, RubricNode] = {}
    root_ids: list[str] = []
    for index, (name, display_name, description) in enumerate(
        MULTICRIT_OPEN_ENDED_CRITERIA, 1
    ):
        node_id = f"init_{index:02d}_{name}"
        root_ids.append(node_id)
        nodes[node_id] = RubricNode(
            node_id=node_id,
            criterion=RubricCriterionSnapshot(name, description, 1.0),
            examples=(),
            lineage={
                "initialization": "human_defined_seed",
                "source": "Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following",
                "source_url": "https://arxiv.org/abs/2511.21662",
                "task_family": "open_ended_generation",
                "source_display_name": display_name,
                "source_order": index,
                "init_version": "multicrit_open_ended_v1",
            },
        )
    return StructuredRubric(nodes=nodes, edges=(), root_ids=tuple(root_ids))


def project_rubric(rubric, root_ids=None):
    """Allowlist projection: examples/lineage never enter Manager prompts."""
    def node(n):
        return dict(node_id=n.node_id, name=n.criterion.name,
                    description=n.criterion.description)
    return [dict(root_id=r, **node(rubric.get_node(r)),
                 children=[node(c) for c in rubric.children(r)])
            for r in (rubric.root_ids if root_ids is None else root_ids)]


def replace_groups(current, groups):
    if not set(groups) <= set(current.root_ids):
        raise ValueError("replacement contains an unknown root")
    nodes = {k: v for k, v in current.nodes.items()
             if k in current.root_ids or current.root_id_for(k) not in groups}
    edges = [e for e in current.edges if current.root_id_for(e.parent_id) not in groups]
    for root, children in groups.items():
        if not 2 <= len(children) <= 5:
            raise ValueError(f"{root}: require 2-5 children")
        for i, child in enumerate(children, 1):
            node_id = f"{root}_c{i:02d}"
            # Scope criterion names by root to avoid cross-root name collisions.
            name = child['name']
            if not name.startswith(f"{root}__"):
                name = f"{root}__{name}"
            nodes[node_id] = RubricNode(node_id, RubricCriterionSnapshot(
                name, child["description"]))
            edges.append(RubricEdge(root, node_id, EdgeCondition.ALWAYS))
    return StructuredRubric(nodes, tuple(edges), current.root_ids)


def validate_children(result):
    """Shared complete-child-group schema for initialization and evolution."""
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    children = result.get("children")
    if not isinstance(children, list) or not 2 <= len(children) <= 5:
        raise ValueError("need 2-5 children")
    names = []
    for child in children:
        child["name"] = nonempty(child.get("name"), "child name")
        child["description"] = nonempty(child.get("description"), "child description")
        names.append(child["name"])
    if len(names) != len(set(names)):
        raise ValueError("child names must be distinct within the root")
    nonempty(result.get("change_summary"), "change_summary")
    return result


def load_rows(config, split):
    data_root = Path(config["data_root"])
    path = data_root / config["datasets"][split]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    for row in rows:
        row["image_path"] = str((data_root / row["image_path"]).resolve())
        if row.get("answer") not in {"A", "B"}:
            raise ValueError(f"invalid gold: {row.get('sample_id')}")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError(f"duplicate sample IDs in {path}")
    return rows


def system_records(value):
    return {s["sample_id"]: s["replicates"]["0"] for s in value["samples"]}


def _log_system_metrics(name, value, vlrb_records=None):
    metrics = value["metrics"]
    label = ""
    if vlrb_records is not None:
        from . import vlrb_official as vlrb
        metrics = vlrb.official_system_metrics(vlrb_records, vlrb._votes(value))
        label = " (official K=3)"
    print(f"{name}: Strict ACC={metrics['strict_accuracy']:.2%}{label}", flush=True)


def evaluate(config, target, name, rows, rubric, *, baseline=None, changed=None,
             orders=None, attempts=10, vlrb_records=None):
    path = target / f"{name}.json"
    if path.exists():
        value = system.load(path)
        if value["rubric_sha256"] != rubric.rubric_sha256:
            raise ValueError(f"{path}: rubric changed")
        if [s["sample_id"] for s in value["samples"]] != [r["sample_id"] for r in rows]:
            raise ValueError(f"{path}: sample order changed")
        if not value["metrics"]["technical_failure_count"]:
            if vlrb_records is not None:
                _log_system_metrics(name, value, vlrb_records)
            return value
    worker = config["worker"]
    value = system.evaluate(
        worker, output_path=path, cache_dir=target / "cache" / name,
        split_name=name, rows=rows, rubric=rubric,
        settings=system.RuntimeSettings(worker["temperature"], worker["max_tokens"],
                                        attempts - 1, retain_arbiter_reason=True),
        baseline=baseline, changed_root_ids=changed, orders_by_id=orders,
        endpoint_ids=[e["endpoint_id"] for e in worker["backend_pool"]["endpoints"]],
        total_attempt_limit=attempts)
    if value["metrics"]["technical_failure_count"]:
        raise RuntimeError(f"{path}: technical failures; inspect cache and resume with a larger --attempt-limit")
    _log_system_metrics(name, value, vlrb_records)
    return value


def parallel(items, fn, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))


def evaluate_external(config, target, dataset, name, rubric, attempts=10,
                      *, baseline=None, changed=None, records=None):
    """Evaluate a specified frozen Rubric, independently of evolution state."""
    if dataset == "vlrb":
        from . import vlrb_official as vlrb
        if records is None:
            records = vlrb._read_records(target / "vlrb", parquet_path=Path(config["data_root"]) / config["datasets"]["vlrb"])
        rows = system.support.vlrb_rows(records)
        orders = vlrb._order_schedule(records)
    else:
        rows, orders = load_rows(config, dataset), None
    value = evaluate(config, target, f"{dataset}/{name}", rows, rubric,
                     orders=orders, attempts=attempts, baseline=baseline,
                     changed=changed, vlrb_records=records)
    return value, records, rows


def external(config, target, dataset, attempts):
    state = load_json(target / "state.json")
    if not state["completed"]:
        raise RuntimeError("Freeze Final before external evaluation")
    initial = StructuredRubric.load_json(target / "init/rubric.json")
    final = StructuredRubric.from_dict(state["rubric"])
    before, records, rows = evaluate_external(
        config, target, dataset, "initial", initial, attempts)
    changed = [r for r in final.root_ids
               if system._root_subtree_sha256(initial, r) != system._root_subtree_sha256(final, r)]
    after = before if not changed else evaluate_external(
        config, target, dataset, "final", final, attempts, baseline=before,
        changed=changed, records=records)[0]
    if not changed:
        write(target / f"{dataset}/final.json", after)
        if records is not None:
            _log_system_metrics(f"{dataset}/final", after, records)
    report = dict(initial=before["metrics"], final=after["metrics"],
                  paired=system.paired(before["metrics"], after["metrics"], rows),
                  k=before["k"], identical_rubric_reuse=not changed)
    if records is not None:
        from . import vlrb_official as vlrb
        a = vlrb.official_system_metrics(records, vlrb._votes(before))
        b = vlrb.official_system_metrics(records, vlrb._votes(after))
        report["official"] = dict(initial=a, final=b,
            paired=vlrb._paired(records, a["original_index_predictions"], b["original_index_predictions"]))
    write(target / f"{dataset}/report.json", report)


def manager_cost(directories):
    cost = dict(calls=0, attempts=0, input_tokens=0, output_tokens=0,
                        usage_missing_attempts=0, reasoning_tokens=0, latency_seconds=0.0)
    # Manager paths are short, known subdirectories; do not load huge Worker caches here.
    for top in directories:
        for path in top.rglob("*.json"):
            if path.name in {"candidate_system.json", "system.json"}:
                continue
            value = load_json(path)
            if not isinstance(value, dict) or "attempts" not in value or "request" not in value:
                continue
            cost["calls"] += 1
            for attempt in value["attempts"]:
                cost["attempts"] += 1
                cost["latency_seconds"] += attempt["wall_seconds"]
                usage = attempt.get("usage")
                if usage is None:
                    cost["usage_missing_attempts"] += 1
                    continue
                cost["input_tokens"] += usage.get("prompt_tokens", 0) or 0
                cost["output_tokens"] += usage.get("completion_tokens", 0) or 0
                cost["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return cost


def freeze_config(config, target, *, allow_stage_concurrency_change=False,
                  allow_worker_backend_change=False):
    """Keep the initial snapshot while allowing selected runtime settings to change."""
    frozen = target / "run_config.json"
    if frozen.exists():
        previous = load_json(frozen)
        previous["manager"]["timeout"] = config["manager"]["timeout"]
        if allow_stage_concurrency_change:
            previous["manager"]["stage_concurrency"] = config["manager"].get("stage_concurrency", {})
        if allow_worker_backend_change:
            previous_pool = previous["worker"]["backend_pool"]
            current_pool = config["worker"]["backend_pool"]
            for key in ("global_request_concurrency", "endpoints"):
                previous_pool[key] = current_pool[key]
        if previous != config:
            raise ValueError("run configuration changed; use a new output directory")
    else:
        write(frozen, config)
    print(f"Manager request timeout={config['manager']['timeout']}s", flush=True)


def check(config, target, *, require_manager_thinking=True, allow_stage_concurrency_change=False,
          discovery_count=100, allow_worker_backend_change=False):
    if config["protocol"] != PROTOCOL or not 1 <= config["max_epochs"] <= 5:
        raise ValueError("wrong protocol or max_epochs outside 1-5")
    if require_manager_thinking and config["manager"]["request_kwargs"].get("extra_body", {}).get("enable_thinking") is not True:
        raise ValueError("This experiment requires Manager thinking enabled")
    for split, count in (("discovery", discovery_count), ("dev", 150)):
        rows = load_rows(config, split)
        if len(rows) != count:
            raise ValueError(f"{split}: expected {count} frozen samples")
        for row in rows:
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError(row["image_path"])
        manifest = target / f"{split}_samples.json"
        # Store the actual split, not predictions or a new fingerprint infrastructure.
        if manifest.exists() and load_json(manifest) != rows:
            raise ValueError(f"{split}: dataset changed; use a new run directory")
        write(manifest, rows)
    if not (Path(config["data_root"]) / config["datasets"]["vlrb"]).is_file():
        raise FileNotFoundError("configured VLRB parquet is missing")
    freeze_config(config, target, allow_stage_concurrency_change=allow_stage_concurrency_change,
                  allow_worker_backend_change=allow_worker_backend_change)
    print(f"check passed: Discovery{discovery_count}, Dev150, VLRB path; no API calls", flush=True)

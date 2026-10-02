"""S0 initialization and frozen external evaluation for local reflection."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from critiq.structured.schema import StructuredRubric, RubricNode, RubricCriterionSnapshot, RubricEdge
from critiq.structured.semantics import EdgeCondition
from . import aligned_system_runtime as system
from .experiment_utils import atomic_write_json as write, load_json

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


def case_payload(row, record, previous=None, root=None, previous_before=None):
    if record["order"] != 0:
        raise ValueError("reflection expects the frozen Discovery K=1 original order")
    reports = {r: c["parsed"] for r, c in record["subtrees"].items()
               if root is None or r == root}
    arbiter = record["arbiter"]["parsed"]
    if any(not isinstance(arbiter.get(k), str) or not arbiter[k].strip()
           for k in ("analysis_a", "analysis_b", "thought")):
        raise ValueError(f"{row['sample_id']}: missing full Arbiter reason")
    case = {k: row[k] for k in ("sample_id", "question", "A", "B")}
    case.update(gold=row["answer"], current=dict(reports=reports, arbiter=arbiter))
    if previous is not None:
        case["previous_candidate"] = dict(
            reports={r: c["parsed"] for r, c in previous["subtrees"].items()
                     if root is None or r == root}, arbiter=previous["arbiter"]["parsed"])
    if previous_before is not None:
        case["previous_baseline"] = dict(
            reports={r: c["parsed"] for r, c in previous_before["subtrees"].items()
                     if root is None or r == root}, arbiter=previous_before["arbiter"]["parsed"])
    return case


def evaluate(config, target, name, rows, rubric, *, baseline=None, changed=None,
             orders=None, attempts=4):
    path = target / f"{name}.json"
    if path.exists():
        value = system.load(path)
        if value["rubric_sha256"] != rubric.rubric_sha256:
            raise ValueError(f"{path}: rubric changed")
        if [s["sample_id"] for s in value["samples"]] != [r["sample_id"] for r in rows]:
            raise ValueError(f"{path}: sample order changed")
        if not value["metrics"]["technical_failure_count"]:
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
    print(f"{name}: Strict ACC={value['metrics']['strict_accuracy']:.2%}", flush=True)
    return value


def parallel(items, fn, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))


def signatures(manager, directory, root, rubric, rows, current, goal="", previous=None):
    records = system_records(current)
    old = system_records(previous) if previous is not None else {}
    def one(row):
        sid = row["sample_id"]
        payload = dict(root=project_rubric(rubric, [root])[0], revision_goal=goal,
                       case=case_payload(row, records[sid], old.get(sid), root))
        result = manager.call("signature", directory / f"{row['_signature_id']}.json",
                              payload, [row])
        return dict(signature_id=row["_signature_id"], sample_id=sid, **result)
    return parallel(rows, one, manager.config.get("stage_concurrency", {}).get(
        "signature", manager.config["concurrency"]))


def generate_group(manager, directory, root, rubric, library, reflection, *, initial):
    selected = [s for s in library if s["applicable"]]
    if len(selected) < (4 if initial else 2):
        return None
    payload = dict(root=project_rubric(rubric, [root])[0], initial=initial,
                   revision=reflection, signatures=selected)
    clusters = manager.call("cluster", directory / "clusters.json", payload)
    if len(clusters["clusters"]) < (2 if initial else 1):
        return None
    proposal = manager.call("children", directory / "children.json",
                            dict(**payload, clusters=clusters))
    # Validate the actual schema before treating a model proposal as complete.
    replace_groups(rubric, {root: proposal["children"]})
    return proposal


def initialize(config, target, rows, manager, attempts, r0=None):
    r0 = build_multicrit_open_ended_init_rubric() if r0 is None else r0
    write(target / "r0/rubric.json", r0.to_dict())
    baseline = evaluate(config, target, "r0/system", rows, r0, attempts=attempts)
    records = system_records(baseline)
    groups = {}
    for i, root in enumerate(r0.root_ids, 1):
        directory = target / f"init/r{i:02d}"
        primary = [r for r in rows if records[r["sample_id"]]["subtrees"][root]["parsed"]["answer"] != r["answer"]]
        library = signatures(manager, directory / "signatures", root, r0, primary, baseline)
        reflection = dict(action="revise", revision_goal="Initial split: improve reusable local judging instructions.")
        proposal = generate_group(manager, directory / "primary", root, r0, library,
                                  reflection, initial=True)
        if proposal is None:
            seen = {r["sample_id"] for r in primary}
            extra = [r for r in rows if r["sample_id"] not in seen]
            library += signatures(manager, directory / "signatures", root, r0, extra, baseline)
            proposal = generate_group(manager, directory / "expanded", root, r0, library,
                                      reflection, initial=True)
        write(directory / "library.json", library)
        if proposal is None:
            raise RuntimeError(f"{root}: insufficient supported patterns for two initial clusters; S0 not created")
        groups[root] = proposal["children"]
    s0 = replace_groups(r0, groups)
    write(target / "init/rubric.json", s0.to_dict())
    initial_system = evaluate(config, target, "init/system", rows, s0, attempts=attempts)
    if not (target / "state.json").exists():
        write(target / "state.json", dict(epoch=0, rubric=s0.to_dict(),
              baseline="init/system", completed=False, stop_reason=None))
    return s0, initial_system


def external(config, target, dataset, attempts):
    state = load_json(target / "state.json")
    if not state["completed"]:
        raise RuntimeError("Freeze Final before external evaluation")
    records = None
    if dataset == "vlrb":
        from . import current_vlrb as vlrb
        records = vlrb._read_records(target / "vlrb", parquet_path=Path(config["data_root"]) / config["datasets"]["vlrb"])
        rows = system.support.vlrb_rows(records)
        orders = vlrb._order_schedule(records)
    else:
        rows, orders = load_rows(config, dataset), None
    initial = StructuredRubric.load_json(target / "init/rubric.json")
    final = StructuredRubric.from_dict(state["rubric"])
    before = evaluate(config, target, f"{dataset}/initial", rows, initial, orders=orders, attempts=attempts)
    changed = [r for r in final.root_ids
               if system._root_subtree_sha256(initial, r) != system._root_subtree_sha256(final, r)]
    after = before if not changed else evaluate(
        config, target, f"{dataset}/final", rows, final, orders=orders, attempts=attempts,
        baseline=before, changed=changed)
    if not changed:
        write(target / f"{dataset}/final.json", after)
    report = dict(initial=before["metrics"], final=after["metrics"],
                  paired=system.paired(before["metrics"], after["metrics"], rows),
                  k=before["k"], identical_rubric_reuse=not changed)
    if records is not None:
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


def freeze_config(config, target, *, allow_stage_concurrency_change=False):
    """Keep the initial snapshot; allow only Manager timeout to change on resume."""
    frozen = target / "run_config.json"
    if frozen.exists():
        previous = load_json(frozen)
        previous["manager"]["timeout"] = config["manager"]["timeout"]
        if allow_stage_concurrency_change:
            previous["manager"]["stage_concurrency"] = config["manager"].get("stage_concurrency", {})
        if previous != config:
            raise ValueError("run configuration changed; use a new output directory")
    else:
        write(frozen, config)
    print(f"Manager request timeout={config['manager']['timeout']}s", flush=True)


def check(config, target, *, require_manager_thinking=True, allow_stage_concurrency_change=False,
          discovery_count=100):
    if config["protocol"] != PROTOCOL or not 1 <= config["max_epochs"] <= 5:
        raise ValueError("wrong protocol or max_epochs outside 1-5")
    if require_manager_thinking and config["manager"]["request_kwargs"].get("extra_body", {}).get("enable_thinking") is not True:
        raise ValueError("This experiment requires Manager thinking enabled")
    pool = config["worker"]["backend_pool"]
    if pool["global_request_concurrency"] != 50 or sum(e["max_concurrency"] for e in pool["endpoints"]) != 50:
        raise ValueError("Worker shared concurrency must be 50")
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
    freeze_config(config, target, allow_stage_concurrency_change=allow_stage_concurrency_change)
    print(f"check passed: Discovery{discovery_count}, Dev150, VLRB path; no API calls", flush=True)

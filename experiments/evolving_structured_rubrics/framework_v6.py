"""Framework v6: five-root initialization, hierarchical reflection, joint acceptance.

Run `all` to check, smoke, initialize, evolve, and evaluate frozen S0/Final.
All intermediate outputs live under one run directory and resume in place.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time

from dotenv import load_dotenv
from critiq.structured.schema import (
    RubricCriterionSnapshot, RubricEdge, RubricNode, StructuredRubric,
)
from critiq.structured.semantics import EdgeCondition
from . import aligned_system_runtime as system
from . import joint_split_evolution as joint
from .experiment_utils import atomic_write_json as write, load_json
from .framework_v6_manager import Manager
from .rubric_factory import build_multicrit_open_ended_init_rubric

PROTOCOL = "framework-v6-hierarchical-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / ".local/framework_v6/config.json"
DEFAULT_OUTPUT = ROOT / "output/fw6"


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


def round_robin(rows, key, seed=42):
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        buckets[str(key(row))].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    output = []
    while any(buckets.values()):
        for name in sorted(buckets):
            if buckets[name]:
                output.append(buckets[name].pop())
    return output


def select_cases(rows, current, previous_summary=None):
    records = system_records(current)
    prediction = dict(zip((r["sample_id"] for r in rows), current["metrics"]["predictions"]))
    def key(row):
        local = records[row["sample_id"]]["subtrees"]
        pattern = tuple((r, c["parsed"]["answer"]) for r, c in sorted(local.items()))
        return (row.get("source", "unknown"), row.get("domain", "unknown"), pattern)
    ordered = round_robin(rows, key)
    wrong = [r for r in ordered if prediction[r["sample_id"]] != r["answer"]]
    correct = [r for r in ordered if prediction[r["sample_id"]] == r["answer"]]
    chosen, reasons = {}, {}
    def add(items, limit, reason):
        n = 0
        for row in items:
            sid = row["sample_id"]
            if n == limit:
                break
            if sid not in chosen:
                chosen[sid] = row
                reasons[sid] = reason
                n += 1
    # Reserve flip and correct slots before filling the larger error stratum.
    comp = (previous_summary or {}).get("comparison", {})
    for label in ("corrected", "harmed"):
        ids = set(comp.get(f"{label}_sample_ids", []))
        add([r for r in ordered if r["sample_id"] in ids], 4, f"previous_{label}")
    add(correct, max(0, min(8, len(correct)) - sum(
        prediction[sid] == row["answer"] for sid, row in chosen.items())), "correct_boundary")
    add(wrong, 24, "current_error")
    add(wrong + correct, 40 - len(chosen), "fill")
    selected = list(chosen.values())
    return selected, dict(sample_ids=list(chosen), reasons=reasons,
                         source_counts=dict(Counter(r.get("source", "unknown") for r in selected)),
                         current_correct=sum(prediction[r["sample_id"]] == r["answer"] for r in selected),
                         current_wrong=sum(prediction[r["sample_id"]] != r["answer"] for r in selected))


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
    return parallel(rows, one, manager.config["concurrency"])


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


def initialize(config, target, rows, manager, attempts):
    r0 = build_multicrit_open_ended_init_rubric()
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


def reflect(config, manager, directory, rows, current_rubric, current,
            libraries, previous=None, previous_summary=None, previous_rubric=None,
            previous_before=None):
    selected, coverage = select_cases(rows, current, previous_summary)
    records, old = system_records(current), system_records(previous) if previous else {}
    old_before = system_records(previous_before) if previous_before else {}
    cases = {r["sample_id"]: case_payload(r, records[r["sample_id"]], old.get(r["sample_id"]),
                                         previous_before=old_before.get(r["sample_id"]))
             for r in selected}
    history = None if not previous_summary else {
        k: previous_summary[k] for k in ("epoch", "accepted", "comparison", "changes")}
    batches = [selected[i:i + 10] for i in range(0, len(selected), 10)]
    def system_batch(item):
        i, batch = item
        batch_history = history
        if history is not None:
            allowed = {r["sample_id"] for r in batch}
            batch_history = dict(history, comparison={
                k: [sid for sid in v if sid in allowed] if k.endswith("sample_ids") else v
                for k, v in history["comparison"].items()})
        payload = dict(rubric=project_rubric(current_rubric),
                       cases=[cases[r["sample_id"]] for r in batch], previous_outcome=batch_history)
        if previous_rubric is not None:
            payload["previous_candidate_rubric"] = project_rubric(previous_rubric)
        return manager.call("system", directory / f"system/b{i:02d}.json", payload, batch)
    feedback = {root: [] for root in current_rubric.root_ids}
    observations = []
    for result in parallel(list(enumerate(batches, 1)), system_batch, manager.config["concurrency"]):
        for root, issues in result["root_feedback"].items():
            feedback[root].extend(issues)
        observations.extend(result.get("system_observations", []))
    write(directory / "system_feedback.json", dict(root_feedback=feedback, system_observations=observations))
    selected_by_id = {r["sample_id"]: r for r in selected}
    coverage["roots"] = {}
    packets = {}
    for i, root in enumerate(current_rubric.root_ids, 1):
        # Round robin across issues; do not allow the first issue to consume the packet.
        issues = feedback[root]
        issue_lists = [list(dict.fromkeys(issue["sample_ids"])) for issue in issues]
        ids = []
        while any(issue_lists) and len(ids) < 12:
            for pending in issue_lists:
                if pending and len(ids) < 12:
                    sid = pending.pop(0)
                    if sid not in ids:
                        ids.append(sid)
        correct = [r for r in selected if records[r["sample_id"]]["arbiter"]["parsed"]["answer"] == r["answer"]]
        correct.sort(key=lambda r: records[r["sample_id"]]["subtrees"][root]["parsed"]["answer"] == "None")
        boundary = [r["sample_id"] for r in correct if r["sample_id"] not in ids][:8]
        ids += boundary
        packet_rows = [selected_by_id[sid] for sid in ids]
        root_history = None if history is None else dict(history,
            changes={root: history["changes"][root]} if root in history["changes"] else {},
            comparison={k: [sid for sid in v if sid in ids] if k.endswith("sample_ids") else v
                        for k, v in history["comparison"].items()})
        payload = dict(root=project_rubric(current_rubric, [root])[0], feedback=issues,
                       cases=[case_payload(r, records[r["sample_id"]], old.get(r["sample_id"]), root,
                                           old_before.get(r["sample_id"]))
                              for r in packet_rows],
                       signature_library=[s for s in libraries[root] if s["applicable"]],
                       previous_outcome=root_history)
        if previous_rubric is not None:
            payload["previous_candidate_subtree"] = project_rubric(previous_rubric, [root])[0]
        packets[root] = (i, payload, packet_rows)
        coverage["roots"][root] = dict(feedback_count=len(issues),
            cited_case_count=len({sid for item in issues for sid in item["sample_ids"]}),
            input_case_count=len(ids), sample_ids=ids, boundary_sample_ids=boundary)
    write(directory / "coverage.json", coverage)
    def root_call(root):
        i, payload, packet_rows = packets[root]
        return root, manager.call("root", directory / f"r{i:02d}/reflection.json", payload, packet_rows)
    reflections = dict(parallel(list(current_rubric.root_ids), root_call, manager.config["concurrency"]))
    return reflections, packets, feedback


def run(config, target, attempts=4, manager=None):
    rows = load_rows(config, "discovery")
    for i, row in enumerate(rows, 1):
        row["_signature_id"] = f"S{i:03d}"
    manager = manager or Manager(config["manager"], attempts)
    if not (target / "state.json").exists():
        initialize(config, target, rows, manager, attempts)
    state = load_json(target / "state.json")
    if state["completed"]:
        write(target / "final.json", state["rubric"])
        return
    roots = StructuredRubric.from_dict(state["rubric"]).root_ids
    libraries = {r: load_json(target / f"init/r{i:02d}/library.json") for i, r in enumerate(roots, 1)}
    rows_by_id = {r["sample_id"]: r for r in rows}
    for epoch in range(state["epoch"] + 1, config["max_epochs"] + 1):
        started = time.perf_counter()
        directory = target / f"e{epoch:02d}"
        current = StructuredRubric.from_dict(state["rubric"])
        baseline = system.load(target / f"{state['baseline']}.json")
        previous_summary = load_json(target / f"e{epoch-1:02d}/summary.json") if epoch > 1 else None
        previous = system.load(target / f"e{epoch-1:02d}/candidate_system.json") if epoch > 1 else None
        previous_rubric = StructuredRubric.load_json(target / f"e{epoch-1:02d}/candidate.json") if epoch > 1 else None
        previous_before = system.load(target / f"{previous_summary['baseline_system']}.json") if epoch > 1 else None
        reflections, packets, feedback = reflect(config, manager, directory, rows, current,
                                                 baseline, libraries, previous, previous_summary,
                                                 previous_rubric, previous_before)
        groups, changes = {}, {}
        for i, root in enumerate(roots, 1):
            reflection = reflections[root]
            if reflection["action"] == "preserve":
                continue
            d = directory / f"r{i:02d}"
            selected_old = {s["signature_id"]: s for s in libraries[root]
                            if s["signature_id"] in reflection["use_signature_ids"]}
            # Refresh exactly the currently cited local cases, never all old errors.
            ids = sorted({sid for issue in feedback[root] for sid in issue["sample_ids"]})
            local_rows = [rows_by_id[sid] for sid in ids]
            fresh = signatures(manager, d / "signatures", root, current, local_rows,
                               baseline, reflection["revision_goal"], previous)
            selected_old.update({s["signature_id"]: s for s in fresh})
            library = list(selected_old.values())
            write(d / "generation_signatures.json", library)
            proposal = generate_group(manager, d, root, current, library, reflection, initial=False)
            if proposal is None:
                raise RuntimeError(f"epoch={epoch} root={root}: revise has insufficient supported clustered evidence")
            groups[root] = proposal["children"]
            changes[root] = dict(before=project_rubric(current, [root])[0]["children"],
                                 after=proposal["children"], summary=proposal["change_summary"])
        if not groups:
            state.update(epoch=epoch, completed=True, stop_reason="all_roots_preserved")
            write(directory / "summary.json", dict(epoch=epoch, accepted=False,
                  no_candidate=True, root_actions=reflections, changes={},
                  current_strict=baseline["metrics"]["strict_accuracy"], manager_cost=manager_cost([directory]),
                  wall_seconds=time.perf_counter()-started))
            write(target / "state.json", state)
            break
        candidate = replace_groups(current, groups)
        write(directory / "candidate.json", candidate.to_dict())
        changed = [r for r in groups if system._root_subtree_sha256(current, r)
                   != system._root_subtree_sha256(candidate, r)]
        if changed:
            after = evaluate(config, target, f"e{epoch:02d}/candidate_system", rows, candidate,
                             baseline=baseline, changed=changed, attempts=attempts)
        else:
            after = baseline
            write(directory / "candidate_system.json", after)
        accepted, comparison = joint.joint_decision(baseline, after, rows)
        summary = dict(epoch=epoch, accepted=accepted, root_actions=reflections, changes=changes,
                       baseline_system=state["baseline"], manager_cost=manager_cost([directory]),
                       current_strict=baseline["metrics"]["strict_accuracy"],
                       candidate_strict=after["metrics"]["strict_accuracy"], comparison=comparison,
                       wall_seconds=time.perf_counter()-started)
        write(directory / "summary.json", summary)
        if accepted:
            state.update(rubric=candidate.to_dict(), baseline=f"e{epoch:02d}/candidate_system")
        state["epoch"] = epoch
        write(target / "state.json", state)
        print(json.dumps({k: summary[k] for k in ("epoch", "accepted", "current_strict", "candidate_strict")}), flush=True)
    state.update(completed=True, stop_reason=state["stop_reason"] or "max_epochs")
    write(target / "state.json", state)
    write(target / "final.json", state["rubric"])


def external(config, target, dataset, attempts):
    state = load_json(target / "state.json")
    if not state["completed"]:
        raise RuntimeError("Freeze Final before external evaluation")
    records = None
    if dataset == "vlrb":
        from . import vl_rewardbench as vlrb
        from . import vl_rewardbench_phase10 as official
        from . import vl_rewardbench_aligned_evolution as aligned
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
        a = official._system_metrics(records, aligned._votes(before))
        b = official._system_metrics(records, aligned._votes(after))
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


def report(config, target):
    state = load_json(target / "state.json")
    rows = load_rows(config, "discovery")
    initial = system.load(target / "init/system.json")
    final = system.load(target / f"{state['baseline']}.json")
    epochs = [load_json(p) for p in sorted(target.glob("e*/summary.json"))]
    cost = manager_cost([target / "smoke", target / "init", *sorted(target.glob("e[0-9][0-9]"))])
    worker_cost = dict(logical_calls=0, model_generations=0, api_attempts=0,
                       input_tokens=0, output_tokens=0, latency_seconds=0.0)
    for path in (target / "cache").rglob("*.json"):
        value = load_json(path)
        if "cache_key" not in value:
            continue
        worker_cost["logical_calls"] += 1
        worker_cost["model_generations"] += value.get("model_generation_count", 0)
        for key in ("api_attempts", "input_tokens", "output_tokens", "latency_seconds"):
            worker_cost[key] += value.get("metrics", {}).get(key, 0) or 0
    worker_cost["scope"] = "Unique persisted Worker/Arbiter calls including smoke/retries; reused reports not counted twice."
    examples = []
    rows_by_id = {r["sample_id"]: r for r in rows}
    for epoch in epochs:
        if epoch.get("no_candidate"):
            continue
        d = target / f"e{epoch['epoch']:02d}"
        before = system_records(system.load(target / f"{epoch['baseline_system']}.json"))
        after = system_records(system.load(d / "candidate_system.json"))
        feedback = load_json(d / "system_feedback.json")["root_feedback"]
        for label in ("corrected", "harmed"):
            for sid in epoch["comparison"][f"{label}_sample_ids"][:3]:
                examples.append(dict(epoch=epoch["epoch"], outcome=label, sample_id=sid,
                    image_path=rows_by_id[sid]["image_path"],
                    case=case_payload(rows_by_id[sid], before[sid], after[sid]),
                    changes=epoch["changes"],
                    feedback={r: [x for x in issues if sid in x["sample_ids"]] for r, issues in feedback.items()}))
    write(target / "mechanism_examples.json", examples)
    result = dict(protocol=PROTOCOL, completed=state["completed"], stop_reason=state["stop_reason"],
                  r0=system.load(target / "r0/system.json")["metrics"],
                  initial=initial["metrics"], final=final["metrics"],
                  paired=system.paired(initial["metrics"], final["metrics"], rows),
                  epochs=epochs, manager_cost=cost, worker_cost=worker_cost,
                  coverage=[load_json(p) for p in sorted(target.glob("e*/coverage.json"))])
    for dataset in ("dev", "vlrb"):
        path = target / f"{dataset}/report.json"
        if path.exists():
            result[dataset] = load_json(path)
    write(target / "report.json", result)
    print(json.dumps(dict(initial_strict=result["initial"]["strict_accuracy"],
                          final_strict=result["final"]["strict_accuracy"],
                          completed=state["completed"], manager_cost=cost), ensure_ascii=False), flush=True)


def freeze_config(config, target):
    """Keep the initial snapshot; allow only Manager timeout to change on resume."""
    frozen = target / "run_config.json"
    if frozen.exists():
        previous = load_json(frozen)
        previous["manager"]["timeout"] = config["manager"]["timeout"]
        if previous != config:
            raise ValueError("run configuration changed; use a new output directory")
    else:
        write(frozen, config)
    print(f"Manager request timeout={config['manager']['timeout']}s", flush=True)


def check(config, target):
    if config["protocol"] != PROTOCOL or not 1 <= config["max_epochs"] <= 5:
        raise ValueError("wrong protocol or max_epochs outside 1-5")
    if config["manager"]["request_kwargs"].get("extra_body", {}).get("enable_thinking") is not True:
        raise ValueError("This experiment requires Manager thinking enabled")
    pool = config["worker"]["backend_pool"]
    if pool["global_request_concurrency"] != 50 or sum(e["max_concurrency"] for e in pool["endpoints"]) != 50:
        raise ValueError("Worker shared concurrency must be 50")
    for split, count in (("discovery", 100), ("dev", 150)):
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
    freeze_config(config, target)
    print("check passed: Discovery100, Dev150, VLRB path; no API calls", flush=True)


def smoke(config, target, attempts):
    rows = load_rows(config, "discovery")[:10]
    for i, row in enumerate(rows, 1):
        row["_signature_id"] = f"S{i:03d}"
    rubric = build_multicrit_open_ended_init_rubric()
    manager = Manager(config["manager"], attempts)
    # Model identity is checked on the actual server; no legacy /version pin.
    from urllib.request import urlopen
    endpoint = config["worker"]["backend_pool"]["endpoints"][0]
    with urlopen(endpoint["base_url"].rstrip("/") + "/models", timeout=30) as response:
        models = json.load(response)
    if config["worker"]["model"] not in {item["id"] for item in models["data"]}:
        raise ValueError("Worker endpoint does not expose the configured model")
    value = evaluate(config, target, "smoke/system", rows, rubric, attempts=attempts)
    reflect(config, manager, target / "smoke", rows, rubric, value,
            {r: [] for r in rubric.root_ids})
    write(target / "smoke/completed.json", dict(samples=len(rows), reason_preserved=True,
          manager_request_kwargs=config["manager"]["request_kwargs"]))
    print("smoke passed: image/reason/reflection chain; not a gain or clustering test", flush=True)


def configure(args):
    data_root = args.data_root.resolve()
    template = ROOT / "experiments/evolving_structured_rubrics/configs/framework_v6.example.json"
    value = load_json(template)
    value.update(data_root=str(data_root), env_file=str((args.env_file or data_root / ".env").resolve()))
    endpoint = value["worker"]["backend_pool"]["endpoints"][0]
    endpoint["base_url"] = args.worker_url.rstrip("/")
    if not endpoint["base_url"].endswith("/v1"):
        endpoint["base_url"] += "/v1"
    if args.config.exists():
        if load_json(args.config) != value:
            raise ValueError(f"{args.config} exists with different settings; edit intentionally or choose another path")
    else:
        write(args.config, value)
    print(f"Configuration ready: {args.config}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("configure", "check", "smoke", "run", "dev", "vlrb", "report", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--worker-url", default="http://localhost:8000/v1")
    parser.add_argument("--attempt-limit", type=int, default=4,
                        help="cumulative attempts per logical call, including cached failed attempts")
    args = parser.parse_args()
    if args.attempt_limit < 1:
        parser.error("attempt-limit must be positive")
    if args.stage == "configure":
        configure(args)
        return
    config = load_json(args.config)
    if config.get("env_file"):
        load_dotenv(config["env_file"], override=False)
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    check(config, target)
    stages = ("smoke", "run", "dev", "vlrb", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "check":
            continue
        print(f"=== framework-v6 stage={stage} ===", flush=True)
        if stage == "smoke":
            smoke(config, target, args.attempt_limit)
        elif stage == "run":
            run(config, target, args.attempt_limit)
        elif stage in {"dev", "vlrb"}:
            external(config, target, stage, args.attempt_limit)
        else:
            report(config, target)
    print("Requested stages completed.", flush=True)


if __name__ == "__main__":
    main()

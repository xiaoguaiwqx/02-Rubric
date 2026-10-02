"""Initialize by clustered signatures, then evolve roots with isolated critiques.

No system feedback, no evolution-time clustering, no global acceptance gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
from pathlib import Path
import time

from dotenv import load_dotenv

from . import rubric_pipeline as base
from . import aligned_system_runtime as system
from .experiment_utils import atomic_write_json as write, load_json
from .subtree_local_reflection_manager import make_manager

PROTOCOL = "subtree-local-reflection-v1"
ROOT = Path(__file__).resolve().parents[2]
StructuredRubric = base.StructuredRubric


def predictions(value, rows):
    samples = {s["sample_id"]: s for s in value["samples"]}
    if len(samples) != len(value["samples"]) or set(samples) != {r["sample_id"] for r in rows}:
        raise ValueError("root report sample IDs differ from evaluation rows")
    result = {}
    for row in rows:
        sample = samples[row["sample_id"]]
        answer = system._answer(sample["call"], sample["order"])
        if answer not in {"A", "B", "None"}:
            raise RuntimeError(f"unresolved root report: {row['sample_id']}")
        result[row["sample_id"]] = answer
    return result


def local_metrics(value, rows):
    pred = predictions(value, rows)
    n = len(rows)
    covered = sum(v in {"A", "B"} for v in pred.values())
    correct = sum(pred[r["sample_id"]] == r["answer"] for r in rows)
    return dict(correct=correct, covered=covered, total=n,
                covered_acc=correct / covered if covered else None,
                coverage=covered / n if n else 0,
                strict_acc=correct / n if n else 0)


def compare(before, after, rows, acceptance_metric="covered_acc"):
    if before["root_id"] != after["root_id"]:
        raise ValueError("root identity changed")
    a, b = local_metrics(before, rows), local_metrics(after, rows)
    if acceptance_metric == "strict_acc":
        accepted = b["correct"] > a["correct"]
    elif acceptance_metric == "covered_acc":
        accepted = bool(b["covered"] and (
            b["correct"] > 0 if not a["covered"] else
            b["correct"] * a["covered"] > a["correct"] * b["covered"]))
    else:
        raise ValueError(f"unknown acceptance metric: {acceptance_metric}")
    old, new = predictions(before, rows), predictions(after, rows)
    transitions = {k: [] for k in ("corrected", "harmed", "ab_to_none", "none_to_ab")}
    ledger = []
    common = []
    for row in rows:
        sid, gold = row["sample_id"], row["answer"]
        x, y = old[sid], new[sid]
        if x != gold and y == gold:
            transitions["corrected"].append(sid)
        if x == gold and y != gold:
            transitions["harmed"].append(sid)
        if x != "None" and y == "None":
            transitions["ab_to_none"].append(sid)
        if x == "None" and y != "None":
            transitions["none_to_ab"].append(sid)
        if x != "None" and y != "None":
            common.append((x == gold, y == gold))
        ledger.append(dict(sample_id=sid, gold=gold, before=x, after=y))
    return dict(accepted=accepted, acceptance_metric=acceptance_metric,
                before=a, after=b, transitions=transitions,
                zero_coverage_baseline=a["covered"] == 0,
                common_coverage=dict(total=len(common),
                    before_correct=sum(x for x, _ in common),
                    after_correct=sum(y for _, y in common)), ledger=ledger)


def local_case(row, root_report):
    if root_report["order"] != 0 or not root_report["call"].get("parse_ok"):
        raise ValueError("reflection requires a valid original-order report")
    report = root_report["call"]["parsed"]
    if any(not isinstance(report.get(k), str) for k in
           ("analysis_a", "analysis_b", "thought", "answer")):
        raise ValueError("missing complete local report")
    return dict(sample_id=row["sample_id"], question=row["question"],
                A=row["A"], B=row["B"], gold=row["answer"], report=report)


def root_evaluate(config, target, name, rows, rubric, root, attempts):
    path = target / f"{name}.json"
    if path.exists():
        value = load_json(path)
        if value["root_subtree_sha256"] != system._root_subtree_sha256(rubric, root):
            raise ValueError(f"candidate changed: {path}")
        if not value["metrics"]["technical_failure_count"]:
            predictions(value, rows)
            return value
    worker = config["worker"]
    value = system.evaluate_root(worker, output_path=path,
        cache_dir=target / "cache" / name, split_name=name, rows=rows,
        rubric=rubric, root_id=root, scope_sample_ids=[r["sample_id"] for r in rows],
        settings=system.RuntimeSettings(worker["temperature"], worker["max_tokens"], attempts-1),
        total_attempt_limit=attempts,
        endpoint_ids=[e["endpoint_id"] for e in worker["backend_pool"]["endpoints"]])
    predictions(value, rows)
    return value


def preservation_rows(rows, reports, epoch, root, count, seed):
    pred = predictions(reports, rows)
    eligible = [r for r in rows if pred[r["sample_id"]] == r["answer"]]
    rng = random.Random(f"{seed}:{epoch}:{root}")
    return rng.sample(eligible, min(count, len(eligible)))


def root_signature(rubric, root):
    """Ignore generated node IDs when deciding whether descriptions changed."""
    return [(n.criterion.name, n.criterion.description) for n in rubric.children(root)]


def evolve_epoch(config, target, epoch, rows, current, baseline, manager, attempts):
    started = time.perf_counter()
    directory = target / f"e{epoch:02d}"
    ids = [r["sample_id"] for r in rows]
    full = base.project_rubric(current)
    before = {root: system.root_from_system(baseline, rows, root, ids)
              for root in current.root_ids}
    jobs = []
    for i, root in enumerate(current.root_ids, 1):
        pred = predictions(before[root], rows)
        selected = [r for r in rows if pred[r["sample_id"]] != r["answer"]]
        write(directory / f"r{i:02d}/selected_cases.json", [r["sample_id"] for r in selected])
        samples = {v["sample_id"]: v for v in before[root]["samples"]}
        jobs.extend((i, root, row, samples[row["sample_id"]]) for row in selected)

    def reflect(job):
        i, root, row, record = job
        # The file index avoids using arbitrary dataset IDs as filesystem paths.
        index = ids.index(row["sample_id"])
        payload = dict(rubric=full, root=full[i-1], case=local_case(row, record))
        result = manager.call("case_reflection",
            directory / f"r{i:02d}/reflections/s{index:04d}.json", payload, [row])
        return root, dict(sample_id=row["sample_id"], **result)

    feedback = {root: [] for root in current.root_ids}
    concurrency = manager.config.get("stage_concurrency", {}).get(
        "case_reflection", manager.config["concurrency"])
    for root, result in base.parallel(jobs, reflect, concurrency):
        if result["critique"].strip():
            feedback[root].append(dict(sample_id=result["sample_id"], critique=result["critique"]))
    groups, accepted_reports, summaries = {}, {}, {}
    # Sequential roots keep the runtime's single global Worker pool at 50.
    for i, root in enumerate(current.root_ids, 1):
        d = directory / f"r{i:02d}"
        write(d / "critiques.json", feedback[root])
        if not feedback[root]:
            summaries[root] = dict(action="preserve", reason="no_actionable_critique",
                                   before=local_metrics(before[root], rows), critique_count=0)
            continue
        payload = dict(rubric=full, root=full[i-1], critiques=feedback[root])
        count = config.get("preservation_case_count", 0)
        if count:
            keep = preservation_rows(rows, before[root], epoch, root, count,
                                     config.get("preservation_seed", 42))
            records = {v["sample_id"]: v for v in before[root]["samples"]}
            payload["preservation_cases"] = [local_case(r, records[r["sample_id"]]) for r in keep]
            payload["preservation_instruction"] = (
                "Critiques are proposed repairs, not mandatory rules. The preservation_cases "
                "are current judgments matching global gold, not guaranteed correct reasoning. "
                "Use their images and reports to preserve evidence-supported capabilities within "
                "this root's scope while resolving critiques. Do not force global gold agreement "
                "outside that scope. In change_summary explain repairs and preserved capabilities. "
                "Do not put examples or sample IDs in the generated criteria.")
            write(d / "preservation_cases.json", payload["preservation_cases"])
            proposal = manager.call("subtree_split", d / "split.json", payload, keep)
        else:
            proposal = manager.call("subtree_split", d / "split.json", payload)
        candidate = base.replace_groups(current, {root: proposal["children"]})
        write(d / "candidate.json", candidate.to_dict())
        if root_signature(candidate, root) == root_signature(current, root):
            candidate_reports = before[root]
            write(d / "candidate_reports.json", candidate_reports)
        else:
            candidate_reports = root_evaluate(config, target,
                f"e{epoch:02d}/r{i:02d}/candidate_reports", rows, candidate, root, attempts)
        decision = compare(before[root], candidate_reports, rows,
                           config.get("acceptance_metric", "covered_acc"))
        decision.update(critique_count=len(feedback[root]), change_summary=proposal["change_summary"],
                        action="accepted" if decision["accepted"] else "rejected",
                        before_children=full[i-1]["children"],
                        after_children=base.project_rubric(candidate, [root])[0]["children"])
        summaries[root] = decision
        write(d / "comparison.json", decision)
        if decision["accepted"]:
            groups[root] = proposal["children"]
            accepted_reports[root] = candidate_reports
        print(f"e{epoch:02d} root={root}: {decision['action']} "
              f"{decision['acceptance_metric']}="
              f"{decision['before'][decision['acceptance_metric']]} -> "
              f"{decision['after'][decision['acceptance_metric']]}", flush=True)
    updated = base.replace_groups(current, groups)
    write(directory / "rubric.json", updated.to_dict())
    # Assemble proven reports for accepted roots; only Arbiter is regenerated.
    bundle = deepcopy(baseline)
    for root, value in accepted_reports.items():
        calls = {v["sample_id"]: v["call"] for v in value["samples"]}
        for sample in bundle["samples"]:
            sample["replicates"]["0"]["subtrees"][root] = calls[sample["sample_id"]]
    bundle["rubric_sha256"] = updated.rubric_sha256
    bundle["root_subtree_sha256"] = {r: system._root_subtree_sha256(updated, r) for r in updated.root_ids}
    if groups:
        after = base.evaluate(config, target, f"e{epoch:02d}/system", rows, updated,
                              baseline=bundle, changed=[], attempts=attempts)
    else:
        after = baseline
        write(directory / "system.json", after)
    summary = dict(epoch=epoch, roots=summaries, accepted_roots=list(groups),
                   no_actionable_feedback=not any(feedback.values()),
                   system_before=baseline["metrics"], system_after=after["metrics"],
                   wall_seconds=time.perf_counter()-started)
    write(directory / "summary.json", summary)
    return updated, after, summary


def run(config, target, attempts=10, manager=None, rows=None, r0=None):
    rows = base.load_rows(config, "discovery") if rows is None else rows
    for i, row in enumerate(rows, 1):
        row["_signature_id"] = f"S{i:03d}"
    manager = manager or make_manager(dict(config["manager"], env_file=config.get("env_file", ".env")), attempts)
    if not (target / "state.json").exists():
        base.initialize(config, target, rows, manager, attempts, r0=r0)
    state = load_json(target / "state.json")
    if state["completed"]:
        write(target / "final.json", state["rubric"])
        return
    for epoch in range(state["epoch"] + 1, config["max_epochs"] + 1):
        current = StructuredRubric.from_dict(state["rubric"])
        baseline = system.load(target / f"{state['baseline']}.json")
        updated, _, summary = evolve_epoch(config, target, epoch, rows, current,
                                           baseline, manager, attempts)
        state.update(epoch=epoch, rubric=updated.to_dict(), baseline=f"e{epoch:02d}/system")
        if summary["no_actionable_feedback"]:
            state.update(completed=True, stop_reason="no_actionable_feedback")
        write(target / "state.json", state)
        if state["completed"]:
            break
    state.update(completed=True, stop_reason=state.get("stop_reason") or "max_epochs")
    write(target / "state.json", state)
    write(target / "final.json", state["rubric"])


def smoke(config, target, attempts):
    """Ten-case wiring test with fixed child fixtures, not a clustered S0."""
    target = target / "smoke"
    rows = base.load_rows(config, "discovery")[:10]
    cfg = deepcopy(config)
    cfg["max_epochs"] = 1
    if not (target / "state.json").exists():
        bare = base.build_multicrit_open_ended_init_rubric()
        groups = {r: [dict(name="scope_check", description=f"Apply the scope of {bare.get_node(r).criterion.name}; disregard unrelated preferences."),
                      dict(name="evidence_check", description="Verify relevant claims against the image, question and candidate text before comparing.")]
                  for r in bare.root_ids}
        fixture = base.replace_groups(bare, groups)
        write(target / "init/rubric.json", fixture.to_dict())
        base.evaluate(cfg, target, "init/system", rows, fixture, attempts=attempts)
        write(target / "state.json", dict(epoch=0, completed=False, stop_reason=None,
              rubric=fixture.to_dict(), baseline="init/system"))
    run(cfg, target, attempts, rows=rows)
    print("smoke complete: fixture-based one-round test; not scientific initialization", flush=True)


def report(config, target):
    state = load_json(target / "state.json")
    result = dict(protocol=PROTOCOL, completed=state["completed"],
        initial=load_json(target / "init/system.json")["metrics"],
        final=load_json(target / f"{state['baseline']}.json")["metrics"],
        epochs=[load_json(p) for p in sorted(target.glob("e*/summary.json"))],
        manager_cost=base.manager_cost([target / "init", *sorted(target.glob("e[0-9][0-9]"))]))
    result["smoke_manager_cost"] = base.manager_cost([target / "smoke"])
    worker_cost = dict(logical_calls=0, model_generations=0, api_attempts=0,
                       input_tokens=0, output_tokens=0, latency_seconds=0.0)
    for cache in (target / "cache", target / "smoke/cache"):
        for path in cache.rglob("*.json"):
            value = load_json(path)
            if "cache_key" not in value:
                continue
            worker_cost["logical_calls"] += 1
            worker_cost["model_generations"] += value.get("model_generation_count", 0)
            for key in ("api_attempts", "input_tokens", "output_tokens", "latency_seconds"):
                worker_cost[key] += value.get("metrics", {}).get(key, 0) or 0
    result["worker_cost_including_smoke"] = worker_cost
    examples = []
    for epoch in result["epochs"]:
        for i, (root, decision) in enumerate(epoch["roots"].items(), 1):
            directory = target / f"e{epoch['epoch']:02d}/r{i:02d}"
            if "transitions" not in decision:
                continue
            critiques = {v["sample_id"]: v["critique"] for v in load_json(directory / "critiques.json")}
            for outcome in ("corrected", "harmed"):
                for sid in decision["transitions"][outcome][:3]:
                    examples.append(dict(epoch=epoch["epoch"], root=root, sample_id=sid,
                        outcome=outcome, accepted=decision["accepted"], critique=critiques.get(sid),
                        before_children=decision["before_children"], after_children=decision["after_children"],
                        prediction=next(v for v in decision["ledger"] if v["sample_id"] == sid),
                        artifact_directory=str(directory)))
    write(target / "mechanism_examples.json", examples)
    for dataset in ("dev", "vlrb"):
        path = target / dataset / "report.json"
        if path.exists():
            result[dataset] = load_json(path)
    result["external_complete"] = all(d in result for d in ("dev", "vlrb"))
    write(target / "report.json", result)
    print(json.dumps(dict(completed=result["completed"], external_complete=result["external_complete"],
          initial_strict=result["initial"]["strict_accuracy"], final_strict=result["final"]["strict_accuracy"])), flush=True)


def check(config, target):
    count = config.get("preservation_case_count", 0)
    if type(count) is not int or count < 0:
        raise ValueError("preservation_case_count must be a nonnegative integer")
    if config.get("acceptance_metric", "covered_acc") not in {"covered_acc", "strict_acc"}:
        raise ValueError("acceptance_metric must be covered_acc or strict_acc")
    if config["protocol"] != PROTOCOL:
        raise ValueError("wrong local-reflection protocol")
    # Existing frozen runs store this internal check snapshot under the v6 identity.
    legacy = deepcopy(config)
    legacy["protocol"] = base.PROTOCOL
    base.check(legacy, target / "checks", require_manager_thinking=False,
               allow_stage_concurrency_change=True,
               discovery_count=config.get("discovery_sample_count", 100))
    discovery_ids = {r["sample_id"] for r in base.load_rows(config, "discovery")}
    if discovery_ids.intersection(r["sample_id"] for r in base.load_rows(config, "dev")):
        raise ValueError("Discovery and Dev sample IDs overlap")
    base.freeze_config(config, target, allow_stage_concurrency_change=True)
    write(target / "runtime_settings.json", dict(manager_timeout=config["manager"]["timeout"],
          manager_stage_concurrency=config["manager"].get("stage_concurrency", {})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("configure", "check", "smoke", "run", "dev", "vlrb", "report", "all"))
    parser.add_argument("--config", type=Path, default=ROOT / ".local/subtree_local_reflection/config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/subtree_local_reflection/main")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--worker-url", default="http://10.102.137.255:8000/v1")
    parser.add_argument("--attempt-limit", type=int, default=10)
    args = parser.parse_args()
    if args.attempt_limit < 1:
        parser.error("attempt-limit must be positive")
    if args.stage == "configure":
        cfg = load_json(ROOT / "experiments/evolving_structured_rubrics/configs/subtree_local_reflection.example.json")
        cfg.update(data_root=str(args.data_root.resolve()),
                   env_file=str((args.env_file or args.data_root / ".env").resolve()))
        url = args.worker_url.rstrip("/")
        cfg["worker"]["backend_pool"]["endpoints"][0]["base_url"] = url if url.endswith("/v1") else url + "/v1"
        if args.config.exists() and load_json(args.config) != cfg:
            raise ValueError("configuration exists with different settings; edit intentionally")
        write(args.config, cfg)
        print(f"Configuration ready: {args.config}")
        return
    config = load_json(args.config)
    load_dotenv(config.get("env_file"), override=False)
    target = args.output_dir.resolve()
    check(config, target)
    for stage in (("smoke", "run", "dev", "vlrb", "report") if args.stage == "all" else (args.stage,)):
        print(f"=== subtree-local-reflection: {stage} ===", flush=True)
        if stage == "smoke":
            smoke(config, target, args.attempt_limit)
        elif stage == "run":
            run(config, target, args.attempt_limit)
        elif stage in {"dev", "vlrb"}:
            base.external(config, target, stage, args.attempt_limit)
        elif stage == "report":
            report(config, target)
    print("Requested stages completed.", flush=True)


if __name__ == "__main__":
    main()

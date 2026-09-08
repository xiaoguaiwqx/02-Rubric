"""Split-only: local root errors generate candidates; one joint system accepts them."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

from critiq.structured import (
    EvolutionContext, PairwisePredictionOutput, StructuredRubric,
    detect_specialize_trigger,
)
from . import aligned_system_runtime as system
from . import discovery_v2_prompt_v2_evolution as phase17
from . import split_evolution as split
from .experiment_utils import atomic_write_json as write, load_json
from .rubric_factory import build_multicrit_open_ended_init_rubric


def runtime_config(config):
    value = phase17._runtime_config(config)
    value['evolution_policy']['trigger_thresholds'].update(
        tau_split=.75, tau_cov_high=.80, N_min_support=15, N_min_wrong=15,
        N_min_cluster=2)
    value['_manager_compact_sample_ids'] = True
    return value


def protocol(config):
    return replace(phase17.PROTOCOL, experiment_dir='joint_split_v1',
                   manager_model=config['specialize_managers']['error_signature']['model'],
                   allow_legacy_signature_reuse=False)


def evaluate(config, output, name, rows, rubric, *, baseline=None, changed=None,
             orders=None, attempts=4):
    path = output / f'{name}.json'
    if path.exists():
        value = system.load(path)
        if value['rubric_sha256'] != rubric.rubric_sha256:
            raise ValueError(f'{name}: saved prediction belongs to another rubric')
        if [s['sample_id'] for s in value['samples']] != [r['sample_id'] for r in rows]:
            raise ValueError(f'{name}: sample order changed')
        if value['metrics']['technical_failure_count'] == 0:
            return value
    value = system.evaluate(
        config, output_path=path, cache_dir=output/'cache'/name,
        split_name=name, rows=rows, rubric=rubric,
        settings=system.RuntimeSettings(.5, 2048, 3),
        baseline=baseline, changed_root_ids=changed, orders_by_id=orders,
        total_attempt_limit=attempts,
        endpoint_ids=[e['endpoint_id'] for e in config['backend_pool']['endpoints']])
    if value['metrics']['technical_failure_count']:
        raise RuntimeError(f'{name}: unresolved technical failures; resume with --attempt-limit 11')
    return value


def joint_decision(before, after, rows):
    if before['metrics']['technical_failure_count'] or after['metrics']['technical_failure_count']:
        raise RuntimeError('Technical failures are not competition losses')
    comparison = system.paired(before['metrics'], after['metrics'], rows)
    return comparison['net_corrected'] > 0, comparison


def merge_candidates(initial, current, candidates):
    """Rebase additive patches for still-unsplit, unchanged initial roots only."""
    rebased = {}
    for root, candidate in candidates.items():
        if (current.get_node(root) != initial.get_node(root)
                or any(e.parent_id == root for e in current.edges)):
            raise ValueError('Only unchanged, unsplit initial roots may propose children')
        patch = candidate.edit_candidate.patch
        if patch.base_rubric_sha256 != initial.rubric_sha256:
            raise ValueError('Candidate must originate from frozen initial root evidence')
        rebased[root] = replace(candidate, edit_candidate=replace(
            candidate.edit_candidate, patch=replace(patch, base_rubric_sha256=current.rubric_sha256)))
    return split.merge_accepted_rubrics(current, rebased)


def rejection_feedback(root, before, after, rows, comparison, candidate):
    """Bounded observations, not causal blame; no extra model call."""
    buckets = [[], [], []]
    for index, row in enumerate(rows):
        old = before['samples'][index]['replicates']['0']
        new = after['samples'][index]['replicates']['0']
        a = before['metrics']['predictions'][index]
        b = after['metrics']['predictions'][index]
        local = system._answer(new['subtrees'][root], new['order'])
        gold = row['answer']
        bucket = (0 if a == gold and b != gold and local != gold else
                  1 if a != gold and b != gold and local != gold else
                  2 if a != gold and b == gold and local == gold else None)
        if bucket is None:
            continue
        buckets[bucket].append(dict(
            sample_id=row['sample_id'], gold=gold, system_before=a, system_after=b,
            root_before=system._answer(old['subtrees'][root], old['order']),
            root_after=local,
            root_reason=str(new['subtrees'][root]['parsed'].get('thought', ''))[:600],
            arbiter_reason=str(new['arbiter']['parsed'].get('thought', ''))[:400]))
    examples = buckets[0][:4] + buckets[1][:2] + buckets[2][:2]
    return dict(
        scope='joint_system_not_root_causal_attribution',
        instruction='The batch failed, not necessarily this root. Reconsider clusters and children; preserve useful boundaries. Do not treat system metrics as local accuracy.',
        baseline_strict=before['metrics']['strict_accuracy'],
        candidate_strict=after['metrics']['strict_accuracy'],
        corrected=comparison['corrected'], harmed=comparison['harmed'],
        net_corrected=comparison['net_corrected'], examples=examples,
        children=[dict(name=c.criterion_name, description=c.description)
                  for c in candidate.children])


def initialize(config, target, rows, attempts):
    initial = build_multicrit_open_ended_init_rubric()
    initial.save_json(target/'initial.json')
    pred_path = target/'initial_pairwise.json'
    prediction = (PairwisePredictionOutput.load_json(pred_path) if pred_path.exists()
                  else phase17._worker_prediction(config, target/'pairwise', initial,
                                                   rows, 'initial_roots'))
    if any(not o.parse_ok or not o.answer_valid for row in prediction.node_outputs for o in row.values()):
        raise RuntimeError('Initial Pairwise has technical failures; resolve before selecting root errors')
    prediction.save_json(pred_path)
    _, votes = split.execute_offline_m1(initial, prediction, rows)
    feedback = split.extract_rubric_feedback(
        initial, prediction, rows, votes, expected_request_spec=prediction.request_spec)
    baseline = evaluate(config, target, 'initial_system', rows, initial, attempts=attempts)
    state_path = target/'state.json'
    if not state_path.exists():
        thresholds = runtime_config(config)['evolution_policy']['trigger_thresholds']
        eligible = [root for root in initial.root_ids if detect_specialize_trigger(
            EvolutionContext(initial, feedback), root, thresholds).triggered]
        write(state_path, dict(epoch=0, pending=eligible, accepted=[], feedback={},
                               rubric=initial.to_dict(), baseline='initial_system', completed=False))
    return initial, prediction, feedback, baseline


def run(config, target, attempts=4, max_epochs=5):
    rows = phase17._rows(config, 'discovery')
    initial, prediction, feedback, _ = initialize(config, target, rows, attempts)
    state = load_json(target/'state.json')
    if state['completed']:
        print('Joint Split already complete; use report/evaluate.', flush=True)
        return
    runtime = runtime_config(config)
    managers, _, specs, _ = split._managers(runtime, protocol(config))
    for epoch in range(state['epoch'] + 1, max_epochs + 1):
        if not state['pending']:
            break
        started = time.perf_counter()
        current = StructuredRubric.from_dict(state['rubric'])
        baseline = system.load(target/f"{state['baseline']}.json")
        epoch_dir = target/f'e{epoch:02d}'
        memory, memory_hash = split.freeze_rubric_memory(epoch_dir, current)
        results, candidates = {}, {}
        for root in state['pending']:
            attempt_dir = split._attempt(epoch_dir, root, epoch)
            if (attempt_dir/'proposal_failure.json').exists():
                results[root] = 'proposal_invalid'
                continue
            print(f'joint split epoch={epoch} root={root} cluster_cached={(attempt_dir/"cluster_proposal.json").exists()}', flush=True)
            try:
                result = split._prepare(
                    runtime, target, epoch_dir, root, epoch, initial, prediction,
                    feedback, rows, {'attempts': []}, managers, specs,
                    protocol(config), memory, memory_hash,
                    prior_failures_override=state['feedback'].get(root, []),
                    root_scoped_names=True)
                results[root] = result['decision']
                if result.get('candidate') is not None:
                    candidates[root] = result['candidate']
            except split.ProposalInvalid as exc:
                results[root] = 'proposal_invalid'
                write(attempt_dir/'proposal_failure.json', split._failure_details(exc))
            except split.SpecializeManagerFailure as exc:
                write(attempt_dir/'manager_failure.json', exc.to_dict())
                if split._manager_failure_kind(exc) == split.TRANSPORT_FAILED:
                    raise
                results[root] = 'proposal_invalid'
                write(attempt_dir/'proposal_failure.json', exc.to_dict())
            # Transport/Manager failures stop the stage: do not silently drop a root.
        collisions = split.colliding_roots(candidates)
        existing_names = {n.criterion.name for n in current.nodes.values()}
        collided = {r for owners in collisions.values() for r in owners}
        collided.update(r for r,c in candidates.items()
                        if any(child.criterion_name in existing_names for child in c.children))
        for root in collided:
            candidates.pop(root)
            results[root] = 'cross_root_collision'
        if candidates:
            candidate = merge_candidates(initial, current, candidates)
            candidate.save_json(epoch_dir/'candidate.json')
            name = f'e{epoch:02d}/candidate_system'
            after = evaluate(config, target, name, rows, candidate, baseline=baseline,
                             changed=list(candidates), attempts=attempts)
            accepted, comparison = joint_decision(baseline, after, rows)
            for root, proposal in candidates.items():
                results[root] = 'committed' if accepted else 'rejected_with_batch'
                if not accepted:
                    state['feedback'][root] = [rejection_feedback(
                        root, baseline, after, rows, comparison, proposal)]
            if accepted:
                state['rubric'] = candidate.to_dict()
                state['baseline'] = name
                state['accepted'].extend(candidates)
        else:
            accepted, comparison = False, dict(corrected=0, harmed=0, net_corrected=0)
            after = baseline
        state['pending'] = [r for r in state['pending']
                            if results[r] not in ('committed', 'not_eligible')]
        summary = dict(epoch=epoch, batch='accepted' if accepted else 'rejected' if candidates else 'no_candidate',
                       participating_roots=list(candidates), root_status=results,
                       baseline_strict=baseline['metrics']['strict_accuracy'],
                       candidate_strict=after['metrics']['strict_accuracy'],
                       comparison=comparison, wall_seconds=time.perf_counter()-started)
        write(epoch_dir/'summary.json', summary)
        StructuredRubric.from_dict(state['rubric']).save_json(epoch_dir/'committed.json')
        state['epoch'] = epoch
        state['completed'] = not state['pending'] or epoch >= max_epochs
        write(target/'state.json', state)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    state['completed'] = True
    write(target/'state.json', state)
    StructuredRubric.from_dict(state['rubric']).save_json(target/'final.json')


def report(config, target):
    state = load_json(target/'state.json')
    if not state['completed']:
        raise RuntimeError('Evolution is not complete')
    final = StructuredRubric.from_dict(state['rubric'])
    final.save_json(target/'final.json')
    rows = phase17._rows(config, 'discovery')
    before = system.load(target/'initial_system.json')
    after = system.load(target/f"{state['baseline']}.json")
    calls = [load_json(p) for p in (target/'cache').rglob('*.json')]
    calls = [c for c in calls if 'cache_key' in c and 'metrics' in c]
    cost = {key: sum(c['metrics'].get(key, 0) or 0 for c in calls)
            for key in ('api_attempts', 'input_tokens', 'output_tokens', 'latency_seconds')}
    cost['scope'] = 'Persisted system calls including smoke/retries; excludes Pairwise and Manager telemetry stored in their own artifacts. Latency sums are not wall time.'
    value = dict(initial=before['metrics'], final=after['metrics'], system_cost=cost,
                 paired=system.paired(before['metrics'], after['metrics'], rows),
                 epochs=[load_json(p) for p in sorted(target.glob('e*/summary.json'))],
                 final_rubric_sha256=final.rubric_sha256, exploratory=True)
    write(target/'report.json', value)
    print(json.dumps(value['paired'], ensure_ascii=False))


def external(config, target, dataset, attempts):
    state = load_json(target/'state.json')
    if not state['completed']:
        raise RuntimeError('Freeze the final structure before external evaluation')
    from . import vl_rewardbench as vlrb
    from . import vl_rewardbench_prompt_v2 as control
    from . import vl_rewardbench_phase10 as metrics
    from . import vl_rewardbench_aligned_evolution as aligned
    if dataset == 'vlrb':
        records = control._records()
        rows = system.support.vlrb_rows(records)
        orders = vlrb._order_schedule(records)
    else:
        rows = phase17._rows(config, dataset)
        orders = None  # Existing aligned evolution heldout/dev protocol: K=1.
    initial = build_multicrit_open_ended_init_rubric()
    final = StructuredRubric.from_dict(state['rubric'])
    before = evaluate(config, target, f'{dataset}/initial', rows, initial,
                      orders=orders, attempts=attempts)
    after = before if initial.rubric_sha256 == final.rubric_sha256 else evaluate(
        config, target, f'{dataset}/final', rows, final, orders=orders, attempts=attempts)
    if dataset == 'vlrb':
        a = metrics._system_metrics(records, aligned._votes(before))
        b = metrics._system_metrics(records, aligned._votes(after))
        paired = vlrb._paired(records, a['original_index_predictions'], b['original_index_predictions'])
    else:
        a, b = before['metrics'], after['metrics']
        paired = system.paired(a, b, rows)
    write(target/f'{dataset}/report.json', dict(initial=a, final=b, paired=paired,
          exploratory=True, k=before['k'], identical_rubric_reuse=before is after,
          heldout_overlap_policy='known source overlap; exploratory, not a clean test' if dataset=='heldout' else None))
    print(json.dumps({k:v for k,v in paired.items() if not k.endswith('sample_ids')}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('configure', 'check', 'smoke', 'run', 'report', 'dev', 'heldout', 'vlrb'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-config', type=Path)
    parser.add_argument('--manager-url')
    parser.add_argument('--worker-url')
    parser.add_argument('--output-dir', type=Path, default=Path('output/js27b'))
    parser.add_argument('--attempt-limit', type=int, default=4,
                        help='Total cached system attempts; 11 resumes only unresolved calls')
    args = parser.parse_args()
    if args.stage == 'configure':
        if not args.source_config or not args.manager_url or not args.worker_url:
            parser.error('configure requires --source-config, --manager-url, --worker-url')
        if args.config.exists():
            raise ValueError('Local config already exists; inspect it instead of overwriting')
        config = deepcopy(load_json(args.source_config))
        config['model'] = 'Qwen/Qwen3-VL-8B-Instruct'
        pool = config['backend_pool']
        pool['endpoints'] = [pool['endpoints'][0]]
        pool['global_request_concurrency'] = 100
        pool['endpoints'][0].update(base_url=args.worker_url, max_concurrency=100)
        config['discovery_v2_prompt_v2_evolution'].update(
            worker_endpoints=[pool['endpoints'][0]['endpoint_id']],
            compact_manager_sample_ids=True, split_min_cluster_size=2)
        config['structured_max_retries'] = 3
        config['worker_request_kwargs'] = {'temperature': .5, 'max_tokens': 2048}
        for stage, settings in config['specialize_managers'].items():
            if stage not in ('error_signature', 'semantic_cluster', 'child_generation'):
                continue
            settings['model'] = 'Qwen/Qwen3.5-27B'
            capacity = 30 if stage == 'error_signature' else 1
            settings['backend_pool']['endpoints'] = [settings['backend_pool']['endpoints'][0]]
            settings['backend_pool']['global_request_concurrency'] = capacity
            settings['backend_pool']['endpoints'][0].update(base_url=args.manager_url, max_concurrency=capacity)
            request = settings['request_kwargs']
            request.pop('thinking_token_budget', None)
            request['max_completion_tokens'] = 16384
            request.setdefault('extra_body', {})['chat_template_kwargs'] = {'enable_thinking': False}
        args.config.parent.mkdir(parents=True, exist_ok=True)
        write(args.config, config)
        print(f'Created {args.config}; no model requests')
        return
    config = load_json(args.config)
    pool = split.base.BackendPoolSpec.from_dict(config['backend_pool'])
    if len(pool.endpoints) != 1 or pool.global_request_concurrency != 100 or pool.endpoints[0].max_concurrency != 100:
        raise ValueError('Expected one 8B endpoint with total concurrency 100')
    if config['model'] != 'Qwen/Qwen3-VL-8B-Instruct':
        raise ValueError('Pairwise and system inference must use Qwen3-VL-8B-Instruct')
    for stage in ('error_signature', 'semantic_cluster', 'child_generation'):
        manager = config['specialize_managers'][stage]
        if (manager['model'] != 'Qwen/Qwen3.5-27B'
                or manager['request_kwargs'].get('max_completion_tokens') != 16384
                or manager['request_kwargs'].get('extra_body', {}).get(
                    'chat_template_kwargs', {}).get('enable_thinking') is not False):
            raise ValueError('Expected 27B Manager, thinking=false, max_completion_tokens=16384')
    if args.attempt_limit not in (4, 11):
        raise ValueError('Use 4 initial attempts or 11 total technical rescue attempts')
    target = args.output_dir
    target.mkdir(parents=True, exist_ok=True)
    # Snapshot is local run metadata, not a new fingerprint/migration system.
    frozen = target/'config.json'
    if frozen.exists() and load_json(frozen) != config:
        raise ValueError('Config changed: use a separate output directory')
    write(frozen, config)
    if args.stage == 'check':
        for name in ('discovery', 'dev'):
            rows = phase17._rows(config, name)
            print(f'{name}: {len(rows)} samples')
        print('Split-only, K=1, Clean S5, single Worker endpoint, total concurrency=100')
    elif args.stage == 'smoke':
        from urllib.request import urlopen
        endpoints = [(pool.endpoints[0].base_url, config['model'])]
        manager = config['specialize_managers']['error_signature']
        endpoints.append((manager['backend_pool']['endpoints'][0]['base_url'], manager['model']))
        for url, model in endpoints:
            with urlopen(url.rstrip('/')+'/models', timeout=15) as response:
                served = {entry['id'] for entry in json.load(response)['data']}
            if model not in served:
                raise ValueError(f'Expected model {model} not served by {url}')
        rows = phase17._rows(config, 'discovery')[:5]
        rubric = build_multicrit_open_ended_init_rubric()
        phase17._worker_prediction(config, target/'smoke/pairwise', rubric, rows, 'smoke')
        evaluate(config, target, 'smoke/system', rows, rubric, attempts=args.attempt_limit)
    elif args.stage == 'run':
        run(config, target, args.attempt_limit)
    elif args.stage == 'report':
        report(config, target)
    else:
        external(config, target, args.stage, args.attempt_limit)


if __name__ == '__main__':
    main()

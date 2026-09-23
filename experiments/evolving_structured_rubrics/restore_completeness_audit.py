"""Restore Init completeness reports; rerun only the unchanged Arbiter."""
import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from critiq.structured.schema import StructuredRubric
from experiments.evolving_structured_rubrics import framework_v6 as base
from experiments.evolving_structured_rubrics import aligned_system_runtime as system
from experiments.evolving_structured_rubrics import vl_rewardbench as vlrb
from experiments.evolving_structured_rubrics import vl_rewardbench_phase10 as official
from experiments.evolving_structured_rubrics import vl_rewardbench_aligned_evolution as aligned
from experiments.evolving_structured_rubrics.experiment_utils import load_json, atomic_write_json as write


def run(source, target, attempts):
    config = load_json(source / 'run_config.json')
    initial = StructuredRubric.load_json(source / 'init/rubric.json')
    final = StructuredRubric.load_json(source / 'final.json')
    root = final.root_ids[0]
    children = [dict(name=n.criterion.name, description=n.criterion.description)
                for n in initial.children(root)]
    hybrid = base.replace_groups(final, {root: children})
    write(target / 'rubric.json', hybrid.to_dict())
    write(target / 'protocol.json', dict(source=str(source.resolve()), restored_root=root,
        datasets=['discovery', 'dev', 'vlrb'], only_arbiter=True,
        note='Post-hoc diagnosis; no evolution or model selection. Original orders preserved.'))
    summaries = {}
    for dataset in ['discovery', 'dev', 'vlrb']:
        a = load_json(source / ('init/system.json' if dataset == 'discovery' else f'{dataset}/initial.json'))
        b = load_json(source / ('e05/system.json' if dataset == 'discovery' else f'{dataset}/final.json'))
        records = None
        if dataset == 'vlrb':
            records = vlrb._read_records(source / 'vlrb', parquet_path=Path(config['data_root']) / config['datasets']['vlrb'])
            rows = system.support.vlrb_rows(records)
        else:
            rows = base.load_rows(config, dataset)
        ai = {s['sample_id']: s for s in a['samples']}
        mixed = deepcopy(b)
        for s in mixed['samples']:
            original = ai[s['sample_id']]
            assert s['orders'] == original['orders']
            for key, rep in s['replicates'].items():
                assert rep['order'] == original['replicates'][key]['order']
                replacement = original['replicates'][key]['subtrees'][root]
                assert replacement['parse_ok']
                rep['subtrees'][root] = deepcopy(replacement)
        orders = {s['sample_id']: s['orders'] for s in mixed['samples']}
        # Guard the scientific intervention: any attempted Worker call is an error.
        with patch.object(system, '_call_subtree', side_effect=AssertionError('Unexpected Worker call')):
            result = base.evaluate(config, target, dataset, rows, hybrid,
                baseline=mixed, changed=[], orders=orders, attempts=attempts)
        for before, after in zip(mixed['samples'], result['samples']):
            assert before['sample_id'] == after['sample_id']
            for key, rep in after['replicates'].items():
                assert not rep['regenerated_root_ids']
                for r in final.root_ids:
                    assert rep['subtrees'][r]['parsed'] == before['replicates'][key]['subtrees'][r]['parsed']
        summary = dict(original_final=b['metrics'], restored=result['metrics'],
            paired_vs_final=system.paired(b['metrics'], result['metrics'], rows))
        if records is not None:
            old = official._system_metrics(records, aligned._votes(b))
            new = official._system_metrics(records, aligned._votes(result))
            summary['official'] = dict(original_final=old, restored=new,
                paired_vs_final=vlrb._paired(records, old['original_index_predictions'], new['original_index_predictions']))
        summaries[dataset] = summary
        write(target / f'{dataset}_comparison.json', summary)
        print(f'{dataset}: comparison saved', flush=True)
    write(target / 'report.json', dict(completed=True, datasets=summaries))
    print('Restore completeness audit complete', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--attempt-limit', type=int, default=10)
    args = parser.parse_args()
    run(args.source, args.output_dir, args.attempt_limit)

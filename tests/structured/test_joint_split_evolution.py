"""Offline tests: no Manager or model endpoint is contacted."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from experiments.evolving_structured_rubrics import joint_split_evolution as j
from tests.structured.test_specialize import _context, _manager_spec, _thresholds
from critiq.structured import (
    SemanticCluster, ClusterProposal, ChildCriterionProposal, ErrorSignature,
    ModelCallMetrics, build_specialize_candidate, RubricNode, RubricCriterionSnapshot,
    StructuredRubric,
    Vote,
)


ROWS = [dict(sample_id='s1', answer='A'), dict(sample_id='s2', answer='B')]


def result(answers):
    return {'metrics': dict(predictions=answers, technical_failure_count=0,
                            strict_accuracy=sum(a==r['answer'] for a,r in zip(answers, ROWS))/2)}


class TestJointSplit(unittest.TestCase):
    def test_root_scoped_child_name_only_and_idempotent(self):
        child = ChildCriterionProposal('c', 'accuracy', 'Description', 'Rationale',
                    ('s1',), 'original raw response', 1, ModelCallMetrics(), _manager_spec())
        normalized = j.split._root_scoped_child(child, 'init_01_completeness')
        self.assertEqual(normalized.criterion_name, 'init_01_completeness__accuracy')
        self.assertIs(j.split._root_scoped_child(normalized, 'init_01_completeness'), normalized)
        expected = child.to_dict()
        expected['criterion_name'] = normalized.criterion_name
        self.assertEqual(normalized.to_dict(), expected)
        self.assertNotEqual(normalized.criterion_name,
                            j.split._root_scoped_child(child, 'init_04_creativity').criterion_name)
        # Identical names within a root remain identical for the existing validator.
        self.assertEqual(normalized.criterion_name,
                         j.split._root_scoped_child(child, 'init_01_completeness').criterion_name)

    @patch('critiq.structured.evolution.specialize._image_sha256', return_value='e'*64)
    def test_real_additive_candidate_rebases_without_mutating_parent_evidence(self, _):
        context = _context()
        clusters = tuple(SemanticCluster(k,k,'failure','distinction',tuple(f's{i}' for i in indices))
                         for k,indices in [('a',range(5)),('b',range(5,10))])
        proposal = ClusterProposal(clusters,tuple(f's{i}' for i in range(10,15)),
                                   'raw',1,ModelCallMetrics(),_manager_spec())
        children = tuple(ChildCriterionProposal(c.cluster_id,'child_'+c.cluster_id,
                         'Description '+c.cluster_id,'narrow',(c.sample_ids[0],),'raw',1,
                         ModelCallMetrics(),_manager_spec()) for c in clusters)
        rows = [dict(sample_id=f's{i}',question='q',A='a',B='b',answer='A',image_path='unused') for i in range(20)]
        signatures = {f's{i}':ErrorSignature(f's{i}','task','focus','diff','failure','sub') for i in range(15)}
        candidate = build_specialize_candidate(context,'parent',proposal,children,rows,signatures)
        # Exercise the actual shared candidate path with fake Manager outputs.
        outputs = {key: SimpleNamespace(signature=sig, to_dict=lambda: {}) for key,sig in signatures.items()}
        cluster_manager = Mock()
        cluster_manager.cluster.return_value = proposal
        child_manager = Mock()
        child_manager.generate_child.side_effect = children
        managers = {'error_signature':Mock(),'semantic_cluster':cluster_manager,'child_generation':child_manager}
        specs = {key:_manager_spec().to_dict() for key in managers}
        pred = SimpleNamespace(node_outputs=[{'parent':SimpleNamespace(vote=Vote.B)} for _ in rows])
        packet = [{'scope':'joint_system_not_root_causal_attribution'}]
        with tempfile.TemporaryDirectory() as folder, patch.object(j.split,'_signatures',return_value=(outputs,'unused',{'v1_identity_match':0})):
            prepared = j.split._prepare({'evolution_policy':{'trigger_thresholds':_thresholds()}},
                Path(folder),Path(folder)/'e01','parent',1,context.rubric,pred,context.feedback,
                rows,{'attempts':[]},managers,specs,prior_failures_override=packet,
                root_scoped_names=True)
            self.assertEqual(prepared['decision'],'prepared')
            self.assertEqual([c.criterion_name for c in prepared['candidate'].children],
                             ['parent__child_a', 'parent__child_b'])
            for artifact in (prepared['attempt_dir']/'children').glob('*.json'):
                saved = j.split.load_json(artifact)
                self.assertTrue(saved['criterion_name'].startswith('parent__'))
                self.assertEqual(saved['raw_response'], 'raw')
            self.assertEqual(cluster_manager.cluster.call_args.kwargs['prior_failures'],packet)
            self.assertEqual(child_manager.generate_child.call_args.kwargs['prior_failures'],packet)
        current = StructuredRubric(nodes={**context.rubric.nodes,'other':RubricNode(
                    'other',RubricCriterionSnapshot('other','Other criterion',1.0))},
                    edges=(),root_ids=('parent','other'))
        merged = j.merge_candidates(context.rubric,current,{'parent':candidate})
        self.assertEqual(len(merged.nodes),4)
        self.assertEqual(merged.get_node('parent'),context.rubric.get_node('parent'))
        self.assertEqual(candidate.edit_candidate.patch.base_rubric_sha256,context.rubric.rubric_sha256)
        with self.assertRaisesRegex(ValueError,'unsplit'):
            j.merge_candidates(context.rubric,merged,{'parent':candidate})

    def test_positive_zero_negative_and_technical_failure(self):
        before = result(['B', 'B'])
        for answers, accepted in [(['A', 'B'], True), (['B', 'B'], False), (['B', 'A'], False)]:
            self.assertEqual(j.joint_decision(before, result(answers), ROWS)[0], accepted)
        broken = result(['A', 'B'])
        broken['metrics']['technical_failure_count'] = 1
        with self.assertRaises(RuntimeError):
            j.joint_decision(before, broken, ROWS)

    def test_single_endpoint_runtime_and_historical_default(self):
        config = {'backend_pool': dict(pool_id='one', common_checkpoint_id='m',
                    global_request_concurrency=1, endpoints=[dict(endpoint_id='one',
                    base_url='http://unused/v1', checkpoint_root='m', max_concurrency=1)])}
        rubric = j.build_multicrit_open_ended_init_rubric()
        call = dict(parse_ok=True, parsed={'answer': 'A'})
        fake = dict(sample_id='s1', orders=[0], replicates={'0': dict(order=0,
                    subtrees={r:call for r in rubric.root_ids}, arbiter=call)})
        with tempfile.TemporaryDirectory() as folder, patch.object(j.system, '_one_sample', return_value=fake):
            kwargs = dict(output_path=Path(folder)/'out.json', cache_dir=Path(folder)/'cache',
                          split_name='test', rows=ROWS[:1], rubric=rubric,
                          settings=j.system.RuntimeSettings(.5,2048,3))
            with self.assertRaises(RuntimeError):
                j.system.evaluate(config, **kwargs)
            value = j.system.evaluate(config, endpoint_ids=['one'], **kwargs)
            self.assertEqual(value['metrics']['strict_accuracy'], 1)

    def test_cached_complete_system_never_calls_model(self):
        rubric = j.build_multicrit_open_ended_init_rubric()
        value = dict(rubric_sha256=rubric.rubric_sha256,
                     samples=[{'sample_id':r['sample_id']} for r in ROWS], **result(['A','B']))
        with tempfile.TemporaryDirectory() as folder, patch.object(j.system, 'load', return_value=value), \
                patch.object(j.system, 'evaluate', side_effect=AssertionError('must reuse')):
            path = Path(folder)
            j.write(path/'baseline.json', value)
            self.assertEqual(j.evaluate({}, path, 'baseline', ROWS, rubric), value)

    def test_batch_accepts_only_valid_roots_and_resume_skips_generation(self):
        self._run_batch(True)

    def test_rejected_batch_preserves_structure_and_feedback(self):
        self._run_batch(False)

    def _run_batch(self, positive):
        rubric = j.build_multicrit_open_ended_init_rubric()
        roots = list(rubric.root_ids[:2])
        before, after = result(['B','B']), result(['A','B'] if positive else ['B','B'])
        fake_candidate = SimpleNamespace(children=[])
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            target = Path(folder)
            j.write(target/'state.json', dict(epoch=0,pending=roots,accepted=[],feedback={},
                    rubric=rubric.to_dict(),baseline='initial_system',completed=False))
            stack.enter_context(patch.object(j.phase17, '_rows', return_value=ROWS))
            stack.enter_context(patch.object(j, 'initialize', return_value=(rubric,None,None,before)))
            stack.enter_context(patch.object(j, 'runtime_config', return_value={}))
            stack.enter_context(patch.object(j, 'protocol', return_value=None))
            stack.enter_context(patch.object(j.split, '_managers', return_value=({},None,{},None)))
            stack.enter_context(patch.object(j.system, 'load', return_value=before))
            stack.enter_context(patch.object(j, 'evaluate', return_value=after))
            stack.enter_context(patch.object(j, 'merge_candidates', return_value=rubric))
            stack.enter_context(patch.object(j, 'rejection_feedback', return_value={'scope':'joint'}))
            prepare = stack.enter_context(patch.object(j.split, '_prepare', side_effect=[
                dict(decision='prepared',candidate=fake_candidate), j.split.ProposalInvalid('cluster','bad')]))
            stack.enter_context(patch.object(j.split, '_evaluate', side_effect=AssertionError('no local competition')))
            j.run({}, target, max_epochs=1)
            state = j.load_json(target/'state.json')
            self.assertEqual(state['accepted'], roots[:1] if positive else [])
            self.assertEqual(state['feedback'], {} if positive else {roots[0]:[{'scope':'joint'}]})
            self.assertEqual(j.load_json(target/'e01/summary.json')['root_status'][roots[1]], 'proposal_invalid')
            self.assertIsNone(prepare.call_args.kwargs.get('decisive_sample_allowlist'))
            j.run({}, target, max_epochs=1)
            self.assertEqual(prepare.call_count, 2)

    def test_feedback_is_bounded_and_separates_system_and_root(self):
        rows = [dict(sample_id=f's{i}',answer='A') for i in range(30)]
        def make(answer):
            call = dict(parse_ok=True,parsed=dict(answer=answer,thought='x'*1000))
            return dict(metrics=dict(predictions=[answer]*30,strict_accuracy=answer=='A'),
                        samples=[dict(replicates={'0':dict(order=0,subtrees={'r':call},arbiter=call)}) for _ in rows])
        packet = j.rejection_feedback('r',make('A'),make('B'),rows,
                    dict(corrected=0,harmed=30,net_corrected=-30),SimpleNamespace(children=[]))
        self.assertLessEqual(len(packet['examples']),8)
        self.assertEqual(packet['scope'],'joint_system_not_root_causal_attribution')
        self.assertEqual(packet['examples'][0]['root_after'],'B')
        self.assertLessEqual(len(packet['examples'][0]['root_reason']),600)


if __name__ == '__main__':
    unittest.main()

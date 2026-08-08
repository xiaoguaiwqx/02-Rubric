import unittest
from unittest.mock import patch
import experiments.evolving_structured_rubrics.split_evolution as split_module
from experiments.evolving_structured_rubrics.run_rubric_evolution import _validate_phase6_config_against_manifest
from pathlib import Path
from types import SimpleNamespace
from experiments.evolving_structured_rubrics.split_evolution import (
 ACCEPTED, ATTRIBUTION_INVALID, COMPETITION_REJECTED, POLICY_V1, PROPOSAL_INVALID,
 SCIENTIFIC_ATTEMPT_OUTCOMES, TRANSPORT_FAILED, AttributionInvalid, TransportFailed,
 _abort_program, _history_projection, _manager_failure_kind, _pause_transport, colliding_roots, merge_accepted_rubrics,
 _required_failure_attribution, non_degenerate_acceptance_check, require_heldout_treatment, retryable_roots,
 runtime_acceptance_policy, should_stop_after_epoch, signature_identity, validate_phase5_lineage, validate_policy)
from critiq.structured import (EdgeCondition, RubricCriterionSnapshot,
 RubricEdge, RubricNode, RubricPatch, StructuredRubric)
from critiq.structured.evolution.specialize_manager import SpecializeManagerFailure
from critiq.structured.telemetry import ModelCallMetrics

class SplitEvolutionStateTests(unittest.TestCase):
 def history(self,**states):return {'root_states':{k:{'status':v} for k,v in states.items()}}
 def test_accepted_root_lock(self):
  self.assertEqual(retryable_roots(self.history(a='accepted_locked',b='retryable',c='not_eligible',d='exhausted')),('b',))
 def test_min_three_and_early_stop(self):
  self.assertFalse(should_stop_after_epoch(2,self.history(a='accepted_locked')));self.assertTrue(should_stop_after_epoch(3,self.history(a='accepted_locked')));self.assertFalse(should_stop_after_epoch(5,self.history(a='retryable')))
 def test_collision_rejects_all_owners(self):
  child=lambda n:SimpleNamespace(criterion_name=n);candidate=lambda *ns:SimpleNamespace(children=tuple(child(n) for n in ns))
  self.assertEqual(colliding_roots({'z':candidate('shared','z_only'),'a':candidate('shared','a_only')}),{'shared':['a','z']})
 def test_signature_cache_identity(self):
  trigger={'decisive_wrong_sample_ids':['s1','s2']};spec={'model':'397b'};first=signature_identity('p',trigger,['A','B'],spec)
  self.assertEqual(first,signature_identity('p',trigger,['A','B'],spec));self.assertNotEqual(first,signature_identity('p',trigger,['B','B'],spec));self.assertNotEqual(first,signature_identity('p',{'decisive_wrong_sample_ids':['s1']},['A','B'],spec))
 def test_synchronous_merge_is_root_order_invariant(self):
  pa=RubricNode('a',RubricCriterionSnapshot('a','A',1.0));pb=RubricNode('b',RubricCriterionSnapshot('b','B',1.0));rubric=StructuredRubric({'a':pa,'b':pb},(),('a','b'))
  def candidate(root,child):
   node=RubricNode(child,RubricCriterionSnapshot(child,child,1.0));patch=RubricPatch(rubric.rubric_sha256,upsert_nodes=(node,),add_edges=(RubricEdge(root,child,EdgeCondition.ALWAYS),));return SimpleNamespace(edit_candidate=SimpleNamespace(patch=patch))
  a=candidate('a','a_child');b=candidate('b','b_child');self.assertEqual(merge_accepted_rubrics(rubric,{'a':a,'b':b}).rubric_sha256,merge_accepted_rubrics(rubric,{'b':b,'a':a}).rubric_sha256)
 def test_exact_policy(self):
  config={'split_evolution':dict(POLICY_V1),'evolution_policy':{'trigger_thresholds':{'tau_split':.7,'tau_cov_high':.8}}};self.assertEqual(validate_policy(config),POLICY_V1);config['split_evolution']['max_epochs']=6
  with self.assertRaises(ValueError):validate_policy(config)
 def test_phase5_lineage_guard_separates_output_from_endpoint(self):
  validate_phase5_lineage(Path('output/evolving_structured_rubrics/rubric_evolution_phase5'))
  with self.assertRaisesRegex(ValueError,'frozen rubric_evolution_phase5'):
   validate_phase5_lineage(Path('output/evolving_structured_rubrics/rubric_evolution_phase5_prompt_8001'))
 def test_phase6_may_route_frozen_pairwise_identity_only_to_8001(self):
  policy={'policy_version':'phase5-v2'}
  manifest={'experiment_id':'phase5','evolution_policy':policy,'pairwise_request_spec':{'model':'worker','backend_id':'frozen-dual-pool','prompt_sha256':'0'*64,'max_data_chars':None,'encode_local_image':True,'image_field':'image_path','question_field':'question','sample_id_field':'sample_id','decoding_config':{'temperature':.5}}}
  config={'experiment_id':'phase5','model':'worker','worker_request_kwargs':{'temperature':.5},'evolution_policy':policy,'backend_pool':{'endpoints':[{'endpoint_id':'vllm-8001'}]}}
  _validate_phase6_config_against_manifest(config,manifest)
  config['evolution_policy']={'policy_version':'changed'}
  with self.assertRaisesRegex(RuntimeError,'evolution policy changed'):
   _validate_phase6_config_against_manifest(config,manifest)
 def test_runtime_policy_uses_plain_config_not_artifact_schema(self):
  policy=runtime_acceptance_policy({'evolution_policy':{'candidate_acceptance':{'min_accuracy_delta':0.0,'min_valid_rate':0.95}}})
  self.assertEqual(policy.min_accuracy_delta,0.0);self.assertEqual(policy.min_valid_rate,0.95)
 def test_manager_transport_and_parse_failures_are_distinct(self):
  transport=SpecializeManagerFailure('semantic_cluster',None,'no response',2,ModelCallMetrics(api_attempts=2,error_count=2))
  parse=SpecializeManagerFailure('semantic_cluster','{}','bad schema',2,ModelCallMetrics(api_attempts=2,error_count=0))
  self.assertEqual(_manager_failure_kind(transport),TRANSPORT_FAILED);self.assertEqual(_manager_failure_kind(parse),PROPOSAL_INVALID)
 def test_rejection_attribution_transport_pauses_before_history_commit(self):
  failure=SpecializeManagerFailure('split_failure_attribution',None,'offline',2,ModelCallMetrics(api_attempts=2,error_count=2))
  manager=SimpleNamespace(attribute_split_failure=lambda **kwargs:(_ for _ in ()).throw(failure))
  with patch.object(split_module,'_write') as write,self.assertRaises(TransportFailed):
   _required_failure_attribution(manager,Path('attempt'))
  self.assertTrue(any(call.args[0].name=='failure_attribution_failure.json' for call in write.call_args_list))
 def test_schema_invalid_attribution_is_not_scientific_history(self):
  failure=SpecializeManagerFailure('split_failure_attribution','{}','bad schema',2,ModelCallMetrics(api_attempts=2,error_count=0))
  manager=SimpleNamespace(attribute_split_failure=lambda **kwargs:(_ for _ in ()).throw(failure))
  with patch.object(split_module,'_write') as write,self.assertRaises(AttributionInvalid):
   _required_failure_attribution(manager,Path('attempt'))
  self.assertNotIn(ATTRIBUTION_INVALID,SCIENTIFIC_ATTEMPT_OUTCOMES)
  invalid=[call.args[1] for call in write.call_args_list if call.args[0].name=='attribution_invalid.json'][0]
  self.assertFalse(invalid['history_appended']);self.assertTrue(invalid['competition_decision_not_committed'])
 def test_successful_attribution_is_durable_before_rejection_history(self):
  value={'schema_version':'1.0.0','attribution':{'summary':'too broad','failure_categories':['child_too_broad'],'details':['x'],'avoid_next_time':['narrow scope']}}
  manager=SimpleNamespace(attribute_split_failure=lambda **kwargs:value)
  with patch.object(split_module,'_write') as write:
   self.assertIs(_required_failure_attribution(manager,Path('attempt')),value)
  self.assertEqual(write.call_args.args,(Path('attempt')/'failure_attribution.json',value))
 def test_history_projection_keeps_semantics_but_drops_raw_payload(self):
  history={'attempts':[{'root_id':'r','attempt':1,'decision':COMPETITION_REJECTED,'history_payload':{'structured_failure':{'code':'x','stage':'semantic_cluster','details':{'parse_error':'bad','attempt_count':2,'raw_response':'secret','metrics':{'input_tokens':999}}},'children_summary':[{'criterion_name':'c'}],'local_metrics':{'accuracy_delta':-0.1}}}]}
  projected=_history_projection(history,'r');self.assertEqual(projected[0]['failure']['details'],{'parse_error':'bad','attempt_count':2});self.assertNotIn('secret',str(projected));self.assertEqual(projected[0]['children_summary'][0]['criterion_name'],'c')
 def test_empty_acceptance_is_not_vacuous_success(self):
  self.assertIsNone(non_degenerate_acceptance_check({'attempts':[]}));self.assertTrue(non_degenerate_acceptance_check({'attempts':[{'decision':ACCEPTED,'accuracy_delta':0.0}]}))
 def test_heldout_requires_nonempty_treatment(self):
  node=RubricNode('r',RubricCriterionSnapshot('r','R',1.0));rubric=StructuredRubric({'r':node},(),('r',));history={'root_states':{'r':{'status':'not_eligible'}}}
  with self.assertRaisesRegex(RuntimeError,'no_treatment'):require_heldout_treatment(rubric,history)
 def test_program_error_aborts_without_recording_scientific_outcome(self):
  with patch.object(split_module,'_write') as write,patch.object(split_module,'_set_run_status') as status:
   _abort_program(Path('unused'),1,'root',1,'evaluation',ValueError('bug'))
  self.assertEqual(write.call_args.args[1]['outcome'],'program_error');self.assertEqual(status.call_args.args[1],'aborted')
 def test_transport_failure_pauses_without_recording_scientific_outcome(self):
  with patch.object(split_module,'_write') as write,patch.object(split_module,'_set_run_status') as status:
   _pause_transport(Path('unused'),1,'root',1,'semantic_cluster',{'api_attempts':22})
  self.assertEqual(write.call_args.args[1]['outcome'],TRANSPORT_FAILED);self.assertEqual(status.call_args.args[1],'paused')
if __name__=='__main__':unittest.main()

"""Five-root Split-only synchronous multi-epoch evolution experiment.

Discovery stages never resolve/read heldout data. Imported lazily by the legacy runner.
"""
from __future__ import annotations
import json, re, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from critiq.structured import (
 CandidateAcceptancePolicy, ChildCriterionProposal, ClusterProposal,
 DualWorkerRequestSpec, ErrorSignatureOutput, EvolutionContext,
 PairwisePredictionOutput, RubricFeedback, SpecializeCandidate,
 SpecializeEvaluation, SpecializeManagerFailure, StructuredCriterionSnapshot, StructuredRubric,
 apply_rubric_patch, assemble_specialized_pairwise_prediction,
 build_specialize_candidate, detect_specialize_trigger,
 evaluate_specialize_candidate, execute_offline_m1, extract_rubric_feedback,
 project_pairwise_prediction)
from critiq.structured.aggregation import aggregate_child_subtrees, aggregate_flat_votes
from critiq.structured.judgement import Vote
from .experiment_utils import atomic_write_json, canonical_sha256, load_json, load_jsonl_dataset
from .rubric_factory import file_sha256
from . import run_rubric_evolution as base

EXPERIMENT_DIR='phase6_split_only_evolution_v2'
MEMORY_EXPERIMENT_DIR='phase6_split_only_evolution_global_memory_v1'
VISUAL_RETRY_EXPERIMENT_DIR='phase8_visual_split_retry_feedback_v2'
LOCKED_RETRY_V2_EXPERIMENT_DIR='phase8_visual_split_retry_locked_v2'
LOCKED_RETRY_V2_HELDOUT_DIR='heldout500_diagnostic'
VISUAL_RETRY_SOURCE_DIR='phase7_split_refine_evolution_v1'
VISUAL_RETRY_ROOT='init_02_visual_grounding_and_details'
CONTROL_EXPERIMENT_DIR=EXPERIMENT_DIR
MANAGER_MODEL='Qwen/Qwen3.5-397B-A17B'
PAIRWISE_ENDPOINT='vllm-8001'
PHASE5_OUTPUT_DIR='rubric_evolution_phase5'
# Frozen Split v1 contract.  Any semantic change to the trigger, scheduling,
# acceptance, retry/history, or Manager-memory rules requires a new version and
# a new experiment directory; historical Control/Treatment stages remain replayable.
POLICY_V1={
 'candidate_scope':'initial_roots_only','lock_root_after_accept':True,
 'min_epochs':3,'max_epochs':5,'retry_rejected_roots':True,
 'synchronous_epoch_commit':True,'reuse_signatures_if_parent_unchanged':True,
 'heldout_access':'final_stage_only'}
HELDOUT_PROTOCOL={
 'decision':'reuse_original_heldout500',
 'rationale':'v1 exposed only an Init=Final null treatment and no child heldout predictions',
 'selected_before_v2_freeze':True}
MEMORY_HELDOUT_PROTOCOL={
 'decision':'reuse_control_heldout500_exploratory',
 'control_experiment':CONTROL_EXPERIMENT_DIR,
 'confirmatory':False,
 'selection_after_heldout_forbidden':True}
MEMORY_CONFIG_V1={
 'variant':'global_rubric_v1',
 'control_experiment_dir':CONTROL_EXPERIMENT_DIR,
 'memory_stages':['semantic_cluster','child_generation'],
 'refresh':'epoch_start_committed_rubric',
 'signature_source':'control_read_only_exact_identity',
 'heldout_protocol':'reuse_exploratory'}


@dataclass(frozen=True)
class EvolutionProtocol:
 experiment_dir: str
 stage_prefix: str
 rubric_memory_mode: str = 'none'
 control_experiment_dir: str | None = None
 read_only_control_signatures: bool = False
 pairwise_endpoint: str = PAIRWISE_ENDPOINT
 retry_feedback_mode: str = 'aggregate_v1'
 max_attempts: int | None = None
 fixed_root_id: str | None = None
 source_experiment_dir: str | None = None
 allow_configured_endpoint_pool: bool = False
 allow_legacy_signature_reuse: bool = True

 @property
 def is_memory_treatment(self):return self.rubric_memory_mode=='global_rubric_v1'

 @property
 def is_retry_treatment(self):return self.retry_feedback_mode=='child_diagnostic_v2'

 @property
 def is_locked_retry_v2(self):return self.retry_feedback_mode=='locked_sample_v3'


CONTROL_PROTOCOL=EvolutionProtocol(EXPERIMENT_DIR,'split-evolution')
MEMORY_PROTOCOL=EvolutionProtocol(
 MEMORY_EXPERIMENT_DIR,'split-memory','global_rubric_v1',CONTROL_EXPERIMENT_DIR,True)
VISUAL_RETRY_PROTOCOL=EvolutionProtocol(
 VISUAL_RETRY_EXPERIMENT_DIR,'split-retry-visual','global_rubric_v1',None,False,
 'vllm-8000','child_diagnostic_v2',3,VISUAL_RETRY_ROOT,VISUAL_RETRY_SOURCE_DIR)
LOCKED_RETRY_V2_PROTOCOL=EvolutionProtocol(
 LOCKED_RETRY_V2_EXPERIMENT_DIR,'split-retry-v2','global_rubric_v1',None,False,
 'vllm-8000','locked_sample_v3',2,VISUAL_RETRY_ROOT,VISUAL_RETRY_SOURCE_DIR)


ACCEPTED='accepted'
COMPETITION_REJECTED='competition_rejected'
PROPOSAL_INVALID='proposal_invalid'
TRANSPORT_FAILED='transport_failed'
ATTRIBUTION_INVALID='attribution_invalid'
PROGRAM_ERROR='program_error'
VALID_COMPETITION_OUTCOMES={ACCEPTED,COMPETITION_REJECTED}
SCIENTIFIC_ATTEMPT_OUTCOMES=VALID_COMPETITION_OUTCOMES|{PROPOSAL_INVALID,'cross_root_collision'}

class ProposalInvalid(RuntimeError):
 def __init__(self,stage,message,details=None):
  super().__init__(message);self.stage=stage;self.details=dict(details or {})

class TransportFailed(RuntimeError):
 def __init__(self,stage,message,details=None):
  super().__init__(message);self.stage=stage;self.details=dict(details or {})

class AttributionInvalid(RuntimeError):
 def __init__(self,message,details=None):
  super().__init__(message);self.stage='split_failure_attribution';self.details=dict(details or {})


def experiment_root(output:Path,protocol:EvolutionProtocol=CONTROL_PROTOCOL)->Path:
 return output/protocol.experiment_dir

def _heldout_protocol(protocol):
 return MEMORY_HELDOUT_PROTOCOL if protocol.is_memory_treatment else HELDOUT_PROTOCOL

def validate_phase5_lineage(output:Path):
 """Prevent a Split evolution run from silently using another Phase-5 lineage."""
 if output.resolve().name != PHASE5_OUTPUT_DIR:
  raise ValueError(
   f'Split evolution must use the frozen {PHASE5_OUTPUT_DIR} output directory; '
   f'got {output}. The Pairwise API endpoint remains {PAIRWISE_ENDPOINT}.')

def _write(path:Path,value:Any):
 path.parent.mkdir(parents=True,exist_ok=True);atomic_write_json(path,value)

def validate_policy(config,protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 if config.get('split_evolution')!=POLICY_V1:raise ValueError('split_evolution must equal frozen v1 policy')
 if protocol.read_only_control_signatures and config.get('split_manager_memory_ablation')!=MEMORY_CONFIG_V1:
  raise ValueError('split_manager_memory_ablation must equal frozen global-rubric v1 protocol')
 t=config['evolution_policy']['trigger_thresholds']
 if t['tau_split']!=.70 or t['tau_cov_high']!=.80:raise ValueError('trigger must be ACC < .70 and Coverage > .80')
 if protocol.is_retry_treatment:
  block=config.get('split_retry_experiment') or {}
  expected={
   'variant':'visual_grounding_child_diagnostic_v2',
   'root_id':VISUAL_RETRY_ROOT,
   'source_experiment_dir':VISUAL_RETRY_SOURCE_DIR,
   'max_attempts':3,
   'pairwise_endpoint':'vllm-8000',
   'split_retry_feedback_mode':'child_diagnostic_v2',
  }
  drift={key:(block.get(key),value) for key,value in expected.items()
         if block.get(key)!=value}
  if drift:raise ValueError(f'split_retry_experiment protocol drift: {drift}')
 if protocol.is_locked_retry_v2:
  block=config.get('split_retry_v2_experiment') or {}
  expected={
   'variant':'visual_grounding_locked_sample_v3',
   'root_id':VISUAL_RETRY_ROOT,
   'source_experiment_dir':VISUAL_RETRY_SOURCE_DIR,
   'max_repair_attempts':2,
   'pairwise_endpoint':'vllm-8000',
   'retry_feedback_mode':'locked_sample_v3',
   'strong_child_min_support':15,
   'strong_child_min_net_corrected':3,
  }
  drift={key:(block.get(key),value) for key,value in expected.items()
         if block.get(key)!=value}
  if drift:raise ValueError(f'split_retry_v2_experiment protocol drift: {drift}')
 return dict(POLICY_V1)

def retryable_roots(history):
 return tuple(sorted(k for k,v in history['root_states'].items() if v['status'] in {'eligible','retryable'}))

def should_stop_after_epoch(epoch,history):return epoch>=3 and not retryable_roots(history)


def runtime_acceptance_policy(config):
 """Build a runtime policy from config, not a versioned artifact."""
 return CandidateAcceptancePolicy(**config['evolution_policy']['candidate_acceptance'])

def _manager_failure_kind(exc):
 metrics=exc.metrics
 if exc.raw_response is None and metrics.api_attempts>0 and metrics.error_count>=metrics.api_attempts:
  return TRANSPORT_FAILED
 return PROPOSAL_INVALID

def _failure_details(exc):
 if isinstance(exc,SpecializeManagerFailure):return exc.to_dict()
 return {'type':type(exc).__name__,'message':str(exc),**getattr(exc,'details',{})}

def signature_identity(parent_id,trigger,parent_outputs,request_spec):
 return canonical_sha256({'parent_node_id':parent_id,'decisive_wrong_sample_ids':trigger['decisive_wrong_sample_ids'],'parent_predictions':list(parent_outputs),'request_spec':request_spec})

def colliding_roots(candidates):
 owners=defaultdict(set)
 for root,c in candidates.items():
  for child in c.children:owners[child.criterion_name].add(root)
 return {k:sorted(v) for k,v in owners.items() if len(v)>1}

def merge_accepted_rubrics(rubric,candidates):
 nodes=dict(rubric.nodes);edges=list(rubric.edges)
 for root in sorted(candidates):
  p=candidates[root].edit_candidate.patch
  if p.base_rubric_sha256!=rubric.rubric_sha256:raise ValueError('candidate base mismatch')
  if p.remove_node_ids or p.remove_edges:raise ValueError('Split patch must be additive')
  for n in p.upsert_nodes:
   if n.node_id in nodes:raise ValueError('duplicate node ID')
   nodes[n.node_id]=n
  edges.extend(p.add_edges)
 return StructuredRubric(nodes=nodes,edges=tuple(edges),root_ids=rubric.root_ids)

def merge_predictions(base_prediction,children,rubric):
 rows=[dict(x) for x in base_prediction.node_outputs];desc={x.name:x.description for x in base_prediction.criteria}
 for pred in children:
  if pred.sample_ids!=base_prediction.sample_ids or pred.sample_fingerprints!=base_prediction.sample_fingerprints or pred.request_spec!=base_prediction.request_spec:raise ValueError('prediction identity mismatch')
  for c in pred.criteria:
   if c.name in desc:raise ValueError('duplicate criterion prediction')
   desc[c.name]=c.description
  for row,shard in zip(rows,pred.node_outputs):row.update(shard)
 ordered=[rubric.get_node(i).criterion for i in rubric.preorder_node_ids()]
 if set(desc)!={x.name for x in ordered}:raise ValueError('predictions do not cover committed rubric')
 rows=tuple({c.name:r[c.name] for c in ordered} for r in rows)
 answers=tuple(aggregate_flat_votes(x.vote for x in r.values()) for r in rows)
 return PairwisePredictionOutput(
  base_prediction.sample_ids,base_prediction.sample_fingerprints,
  tuple(StructuredCriterionSnapshot(c.name,c.description) for c in ordered),
  rows,answers,base_prediction.request_spec,
  semantics_version=base_prediction.semantics_version,
  schema_version=base_prediction.schema_version,
  prompt_version=base_prediction.prompt_version,
  parser_version=base_prediction.parser_version)

def _epoch(target,n):return target/'epochs'/f'epoch_{n:02d}'
def root_shard(root):
 parts=root.split('_',2)
 return f'r{parts[1]}' if len(parts)>2 and parts[0]=='init' and parts[1].isdigit() else 'r'+canonical_sha256(root)[:8]
def _attempt(d,root,n):return d/'roots'/root_shard(root)/f'attempt_{n:02d}'
def _history(target):return load_json(target/'evolution_history.json')
def _save_history(target,h):_write(target/'evolution_history.json',h)

def _snapshot(d,rubric,pred,rows,manifest):
 d.mkdir(parents=True,exist_ok=True);execution,votes=execute_offline_m1(rubric,pred,rows)
 feedback=extract_rubric_feedback(rubric,pred,rows,votes,expected_request_spec=DualWorkerRequestSpec.from_dict(manifest['pairwise_request_spec']))
 rubric.save_json(d/'rubric_committed.json');pred.save_json(d/'discovery_pairwise.json');execution.save_json(d/'m1_execution.json');_write(d/'feedback.json',feedback.to_dict())
 return votes,feedback

def rubric_memory_snapshot(rubric):
 """Build the deterministic, metric-free Manager memory for one epoch."""
 return {
  'schema_version':'1.0.0',
  'rubric_sha256':rubric.rubric_sha256,
  'root_ids':list(rubric.root_ids),
  'nodes':[{
   'node_id':node_id,
   'criterion_name':rubric.get_node(node_id).criterion.name,
   'description':rubric.get_node(node_id).criterion.description,
  } for node_id in rubric.preorder_node_ids()],
  'edges':[edge.to_dict() for edge in sorted(
   rubric.edges,key=lambda x:(x.parent_id,x.child_id,x.condition.value))],
 }

def freeze_rubric_memory(epoch_dir,rubric):
 snapshot=rubric_memory_snapshot(rubric);path=epoch_dir/'rubric_memory.json'
 if path.exists() and load_json(path)!=snapshot:
  raise RuntimeError(f'epoch rubric memory drift: {path}')
 if not path.exists():_write(path,snapshot)
 return snapshot,canonical_sha256(snapshot)

def _managers(config,protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 managers={};profiles={};specs={};identities={}
 for stage in ('error_signature','semantic_cluster','child_generation'):
  memory_mode=(protocol.rubric_memory_mode
               if stage in {'semantic_cluster','child_generation'} else 'none')
  m,_,p,ids=base._manager_runtime(config,stage,rubric_memory_mode=memory_mode)
  if protocol.is_retry_treatment or protocol.is_locked_retry_v2:
   m.retry_feedback_mode=protocol.retry_feedback_mode
   p=dict(p);p['retry_feedback_mode']=protocol.retry_feedback_mode
  if protocol.is_locked_retry_v2 and stage=='child_generation':
   # v2's boundary packets are intentionally multimodal even though the
   # historic global Split configuration used text-only child generation.
   m.child_input_mode='multimodal';p=dict(p);p['input_mode']='multimodal'
  if p['model']!=MANAGER_MODEL:raise RuntimeError(f'{stage} must use {MANAGER_MODEL}')
  managers[stage]=m;profiles[stage]=p;specs[stage]=m.request_specs()[stage].to_dict();identities[stage]=ids
  if stage=='semantic_cluster':
   specs['split_failure_attribution']=m.request_specs()['split_failure_attribution'].to_dict()
 return managers,profiles,specs,identities

def _control_signature_manifest(output,rubric,pred,triggers):
 control=output/CONTROL_EXPERIMENT_DIR
 required=(control/'frozen_manifest.json',control/'evolution_history.json',
           control/'final'/'discovery_report.json',control/'heldout500'/'report.json',
           control/'final_report.json')
 missing=[str(path) for path in required if not path.exists()]
 if missing:raise RuntimeError(f'Control v2 is incomplete: missing {missing}')
 control_manifest=load_json(required[0]);control_history=load_json(required[1])
 if not control_history.get('completed'):raise RuntimeError('Control v2 evolution is not complete')
 status=load_json(control/'stage_status.json')
 if any(status.get(stage,{}).get('status')!='passed'
        for stage in ('freeze','run','report','heldout','final_report')):
  raise RuntimeError('Control v2 stages are not all passed')
 if control_manifest['initial_rubric_sha256']!=rubric.rubric_sha256:
  raise RuntimeError('Control initial Rubric does not match Treatment')
 if control_manifest['initial_root_ids']!=list(rubric.root_ids):
  raise RuntimeError('Control initial roots do not match Treatment')
 if control_manifest['initial_triggers']!=triggers:
  raise RuntimeError('Control triggers or decisive-wrong IDs do not match Treatment')
 if control_manifest['pairwise_request_spec']!=pred.request_spec.to_dict():
  raise RuntimeError('Control Pairwise request identity does not match Treatment')
 entries=[];root_sources={}
 for root in rubric.root_ids:
  trigger=triggers[root]
  if not trigger['triggered']:continue
  name=rubric.get_node(root).criterion.name
  parent_outputs=[row[name].vote.value for row in pred.node_outputs]
  attempt_paths=sorted(control.glob(f'epochs/epoch_*/roots/{root_shard(root)}/attempt_*/error_signatures.json'))
  if not attempt_paths:raise RuntimeError(f'Control signature attempt missing: {root}')
  attempt_artifact=load_json(attempt_paths[0]);outputs=attempt_artifact.get('outputs',{})
  if set(outputs)!=set(trigger['decisive_wrong_sample_ids']):
   raise RuntimeError(f'Control signature sample set drift: {root}')
  request_specs={canonical_sha256(value['request_spec']):value['request_spec'] for value in outputs.values()}
  if len(request_specs)!=1:raise RuntimeError(f'Control root used mixed signature identities: {root}')
  error_spec=next(iter(request_specs.values()))
  identity=signature_identity(root,trigger,parent_outputs,error_spec)
  if attempt_artifact.get('signature_identity')!=identity:
   raise RuntimeError(f'Control signature identity does not replay: {root}')
  cache=control/'signature_cache'/root_shard(root)/identity[:16]
  root_sources[root]={'signature_identity':identity,'request_spec':error_spec,
                      'attempt_artifact':str(attempt_paths[0].resolve()),
                      'attempt_artifact_sha256':file_sha256(attempt_paths[0])}
  for sample_id in trigger['decisive_wrong_sample_ids']:
   source=cache/base._safe_artifact_name(sample_id)
   if not source.exists():raise RuntimeError(f'Control signature missing: {root}/{sample_id}')
   value=ErrorSignatureOutput.from_dict(load_json(source))
   if value.signature is None or value.signature.sample_id!=sample_id:
    raise RuntimeError(f'Control signature invalid: {root}/{sample_id}')
   if value.request_spec.to_dict()!=error_spec:
    raise RuntimeError(f'Control signature identity drift: {root}/{sample_id}')
   entries.append({'root_id':root,'sample_id':sample_id,
                   'signature_identity':identity,'source_path':str(source.resolve()),
                   'source_sha256':file_sha256(source)})
 return {'schema_version':'1.0.0','control_experiment':CONTROL_EXPERIMENT_DIR,
         'generated':0,'reused':len(entries),'entries':entries,
         'root_sources':root_sources,
         'manifest_sha256':canonical_sha256(control_manifest),
         'history_sha256':canonical_sha256(control_history)}

def freeze(config,output,protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 validate_policy(config,protocol);validate_phase5_lineage(output);target=experiment_root(output,protocol)
 if (target/'frozen_manifest.json').exists():print(f'{protocol.stage_prefix}-freeze already passed');return
 phase5,rubric,rows,pred,feedback=base._phase6_inputs(config,output)
 pool=base.BackendPoolSpec.from_dict(config['backend_pool'])
 if protocol.allow_configured_endpoint_pool:
  if protocol.pairwise_endpoint not in {item.endpoint_id for item in pool.endpoints}:
   raise RuntimeError(f'Pairwise endpoint {protocol.pairwise_endpoint} is not configured')
 elif len(pool.endpoints)!=1 or pool.endpoints[0].endpoint_id!=PAIRWISE_ENDPOINT:
  raise RuntimeError('Pairwise must use only vllm-8001')
 _,profiles,specs,ids=_managers(config,protocol);roots=tuple(rubric.root_ids)
 if len(roots)!=5:raise RuntimeError('expected five initial roots')
 ctx=EvolutionContext(rubric,feedback);thresholds=config['evolution_policy']['trigger_thresholds']
 triggers={r:detect_specialize_trigger(ctx,r,thresholds).to_dict() for r in roots}
 eligible_roots=tuple(r for r in roots if triggers[r]['triggered'])
 signature_manifest=None
 if protocol.is_memory_treatment:
  control_manifest=load_json(output/CONTROL_EXPERIMENT_DIR/'frozen_manifest.json')
  for stage in ('semantic_cluster','child_generation'):
   expected=dict(control_manifest['manager_profiles'][stage]);actual=dict(profiles[stage]);actual.pop('rubric_memory_mode',None)
   if actual!=expected:raise RuntimeError(f'Control Manager profile drift: {stage}')
  current_error=profiles['error_signature'];control_error=control_manifest['manager_profiles']['error_signature']
  for key in ('model','api_key_env','input_mode','request_kwargs','vllm_identity'):
   if current_error[key]!=control_error[key]:raise RuntimeError(f'Control ErrorSignature profile drift: {key}')
  if current_error['backend_pool']['common_checkpoint_id']!=control_error['backend_pool']['common_checkpoint_id']:
   raise RuntimeError('Control ErrorSignature checkpoint identity drift')
  endpoint_identity=lambda profile:[(x['endpoint_id'],x['base_url'],x['checkpoint_root']) for x in profile['backend_pool']['endpoints']]
  if (current_error['backend_pool']['pool_id']!=control_error['backend_pool']['pool_id']
      or endpoint_identity(current_error)!=endpoint_identity(control_error)):
   raise RuntimeError('Control ErrorSignature endpoint identity drift')
  signature_manifest=_control_signature_manifest(output,rubric,pred,triggers)
  current_spec=specs['error_signature']
  scientific_keys=('model','prompt_sha256','decoding_config','prompt_version','parser_version')
  for root,source in signature_manifest['root_sources'].items():
   if any(current_spec[key]!=source['request_spec'][key] for key in scientific_keys):
    raise RuntimeError(f'Control ErrorSignature scientific request identity drift: {root}')
  _write(target/'control_signature_reuse_manifest.json',signature_manifest)
 manifest={'schema_version':'1.0.0','experiment':protocol.experiment_dir,'policy':POLICY_V1,'manager_seed':42,'phase5_manifest_sha256':canonical_sha256(phase5),'discovery_dataset_sha256':phase5['discovery_dataset_sha256'],'initial_rubric_sha256':rubric.rubric_sha256,'initial_root_ids':list(roots),'initial_triggers':triggers,'initial_eligible_root_ids':list(eligible_roots),'initial_eligible_root_count':len(eligible_roots),'pairwise_request_spec':pred.request_spec.to_dict(),'pairwise_endpoint':PAIRWISE_ENDPOINT,'manager_profiles':profiles,'manager_request_specs':specs,'manager_endpoint_identities':ids,'heldout_access':'forbidden_until_heldout_stage','heldout_protocol':_heldout_protocol(protocol)}
 if protocol.is_memory_treatment:manifest.update({'rubric_memory_mode':protocol.rubric_memory_mode,'control_experiment':protocol.control_experiment_dir,'control_signature_reuse_manifest_sha256':canonical_sha256(signature_manifest)})
 _write(target/'frozen_manifest.json',manifest);votes,_=_snapshot(_epoch(target,0),rubric,pred,rows,manifest)
 if protocol.is_memory_treatment:
  memory,memory_hash=freeze_rubric_memory(_epoch(target,0),rubric)
  manifest['epoch_00_rubric_memory_sha256']=memory_hash;_write(target/'frozen_manifest.json',manifest)
 _write(_epoch(target,0)/'summary.json',{'epoch':0,'m1':base._metrics(votes,rows),'triggers':triggers,'accepted_roots':[]})
 states={r:{'status':'eligible' if x['triggered'] else 'not_eligible','attempt_count':0,'accepted_epoch':None,'children':[]} for r,x in triggers.items()}
 _save_history(target,{'schema_version':'1.0.0','current_epoch':0,'completed':False,'stop_reason':None,'root_states':states,'attempts':[]})
 _write(target/'stage_status.json',{'freeze':{'status':'passed'}});print(json.dumps({'eligible':len(eligible_roots),'eligible_roots':list(eligible_roots),'target':str(target)},indent=2))

def _history_projection(history,root,include_retry_diagnostics=False):
 items=[]
 for record in history['attempts']:
  if record['root_id']!=root or record['decision']==ACCEPTED:continue
  payload=record.get('history_payload') or {};failure=payload.get('structured_failure') or {};details=failure.get('details') or {}
  compact_failure={'code':failure.get('code'),'stage':failure.get('stage'),'details':{k:details[k] for k in ('type','message','parse_error','attempt_count','collisions') if k in details}}
  item={'attempt':record['attempt'],'outcome':record['decision'],'failure':compact_failure,'cluster_summary':payload.get('cluster_summary'),'children_summary':payload.get('children_summary',[]),'local_metrics':payload.get('local_metrics'),'corrected_sample_ids':payload.get('corrected_sample_ids',[]),'harmed_sample_ids':payload.get('harmed_sample_ids',[]),'natural_language_attribution':payload.get('natural_language_attribution')}
  if include_retry_diagnostics:
   item['child_retry_diagnostics']=payload.get('child_retry_diagnostics')
   item['preservation_directives']=payload.get('preservation_directives',[])
  items.append(item)
 return items

def _prior(history,root,include_retry_diagnostics=False):
 source={'attempts':history.get('seed_attempts',[])}
 return (_history_projection(source,root,include_retry_diagnostics)
         +_history_projection(history,root,include_retry_diagnostics))

def _retry_feedback(prior):
 attempts=[{'attempt':x.get('attempt'),'outcome':x.get('outcome'),
            'child_retry_diagnostics':x.get('child_retry_diagnostics'),
            'preservation_directives':x.get('preservation_directives',[]),
            'natural_language_attribution':x.get('natural_language_attribution')}
           for x in prior if x.get('child_retry_diagnostics')]
 return {'schema_version':'2.0.0','attempts':attempts,
         'latest':attempts[-1] if attempts else None,
         'preservation_implementation':{
          'mode':'manager_directive_only','criterion_hash_identical_reuse':False,
          'pairwise_prediction_reused':False,
          'note':'v2 preserves strong-child semantics through explicit Manager directives; exact artifact reuse is not claimed.'}}

def _decisive(output):
 return output.parse_ok and output.answer_valid and output.vote in {Vote.A,Vote.B}

def build_child_retry_diagnostics(*,candidate,combined,rows,parent_node_id,parent_name,evaluation):
 """Build child-level evidence on the parent's frozen decisive scope.

 This artifact is intentionally richer than the frozen Split-v1 schema.  It is
 treatment-only evidence supplied to the next Manager retry; it never changes
 the ACC-only acceptance rule.
 """
 parent_outputs=[row[parent_name] for row in combined.node_outputs]
 parent_scope=[_decisive(x) for x in parent_outputs]
 parent_support=sum(parent_scope)
 if not parent_support:raise ValueError('retry diagnostics parent scope is empty')
 gold=[str(row['answer']) for row in rows]
 cluster_by_id={x.cluster_id:x for x in candidate.cluster_proposal.clusters}
 child_outputs={child.criterion_name:[row[child.criterion_name] for row in combined.node_outputs]
                for child in candidate.children}

 def subtree_vote(index,excluded=None):
  votes=[outputs[index].vote for name,outputs in child_outputs.items()
         if name!=excluded and _decisive(outputs[index])]
  return aggregate_child_subtrees(parent_outputs[index].vote,votes)
 full=[subtree_vote(i) for i in range(len(rows))]
 full_accuracy=sum(parent_scope[i] and full[i].value==gold[i]
                   for i in range(len(rows)))/parent_support
 result=[]
 for child in candidate.children:
  outputs=child_outputs[child.criterion_name]
  decisive=[parent_scope[i] and _decisive(x) for i,x in enumerate(outputs)]
  indices=[i for i,x in enumerate(decisive) if x]
  support=len(indices);correct=sum(outputs[i].vote.value==gold[i] for i in indices)
  parent_correct=sum(parent_outputs[i].vote.value==gold[i] for i in indices)
  child_acc=correct/support if support else 0.0
  parent_same=parent_correct/support if support else 0.0
  corrected=[str(rows[i]['sample_id']) for i in indices
             if outputs[i].vote.value==gold[i] and parent_outputs[i].vote.value!=gold[i]]
  harmed=[str(rows[i]['sample_id']) for i in indices
          if outputs[i].vote.value!=gold[i] and parent_outputs[i].vote.value==gold[i]]
  single=[outputs[i].vote if _decisive(outputs[i]) else parent_outputs[i].vote
          for i in range(len(rows))]
  single_acc=sum(parent_scope[i] and single[i].value==gold[i]
                 for i in range(len(rows)))/parent_support
  cluster=set(cluster_by_id[child.cluster_id].sample_ids)
  target=[i for i in indices if str(rows[i]['sample_id']) in cluster]
  non_target=[i for i in indices if str(rows[i]['sample_id']) not in cluster]
  target_corrected=[str(rows[i]['sample_id']) for i in target
                    if outputs[i].vote.value==gold[i] and parent_outputs[i].vote.value!=gold[i]]
  target_harmed=[str(rows[i]['sample_id']) for i in target
                 if outputs[i].vote.value!=gold[i] and parent_outputs[i].vote.value==gold[i]]
  omitted=[subtree_vote(i,child.criterion_name) for i in range(len(rows))]
  omitted_acc=sum(parent_scope[i] and omitted[i].value==gold[i]
                  for i in range(len(rows)))/parent_support
  local_delta=full_accuracy-omitted_acc
  strong=(support>=15 and child_acc>parent_same and local_delta>=0)
  action=('preserve_exact' if strong else 'refine'
          if support>=15 and child_acc>=parent_same-.03 else 'replace')
  result.append({
   'cluster_id':child.cluster_id,'criterion_name':child.criterion_name,
   'criterion_sha256':canonical_sha256({'criterion_name':child.criterion_name,
                                        'description':child.description}),
   'support':support,'coverage_on_parent_scope':support/parent_support,
   'accuracy':child_acc,'parent_accuracy_same_support':parent_same,
   'accuracy_delta_same_support':child_acc-parent_same,
   'single_child_specialized_accuracy':single_acc,
   'single_child_specialized_delta':single_acc-evaluation.parent_accuracy,
   'net_corrected':len(corrected)-len(harmed),
   'corrected_sample_ids':corrected,'harmed_sample_ids':harmed,
   'target':{'decisive':len(target),'correct':sum(outputs[i].vote.value==gold[i] for i in target),
             'wrong':sum(outputs[i].vote.value!=gold[i] for i in target),
             'corrected_sample_ids':target_corrected,'harmed_sample_ids':target_harmed,
             'net_corrected':len(target_corrected)-len(target_harmed)},
   'non_target':{'decisive':len(non_target),'correct':sum(outputs[i].vote.value==gold[i] for i in non_target),
                 'wrong':sum(outputs[i].vote.value!=gold[i] for i in non_target)},
   'local_leave_one_child_out_specialized_delta':local_delta,
   'recommended_action':action,'strong_child':strong,
  })
 pairs=[]
 for i,left in enumerate(candidate.children):
  for right in candidate.children[i+1:]:
   lo=child_outputs[left.criterion_name];ro=child_outputs[right.criterion_name]
   joint=[j for j in range(len(rows)) if parent_scope[j] and _decisive(lo[j]) and _decisive(ro[j])]
   conflicts=[str(rows[j]['sample_id']) for j in joint if lo[j].vote is not ro[j].vote]
   pairs.append({'left':left.criterion_name,'right':right.criterion_name,
                 'joint_decisive':len(joint),'conflict_count':len(conflicts),
                 'conflict_rate':len(conflicts)/len(joint) if joint else 0.0,
                 'conflict_sample_ids':conflicts})
 return {'schema_version':'2.0.0','parent_node_id':parent_node_id,
         'parent_scope_support':parent_support,'parent_accuracy':evaluation.parent_accuracy,
         'specialized_accuracy':evaluation.specialized_accuracy,
         'children':result,'sibling_pairs':pairs,
         'preservation_directives':[{'criterion_name':x['criterion_name'],
          'cluster_id':x['cluster_id'],'action':x['recommended_action'],
          'criterion_sha256':x['criterion_sha256']} for x in result]}

def _require_split_clusters(cluster):
 if len(cluster.clusters)<2:
  raise ProposalInvalid('semantic_cluster','Split requires at least two valid clusters',{
   'reason':'insufficient_cluster_count','valid_clusters':len(cluster.clusters),
   'required_minimum':2,'cluster_ids':[x.cluster_id for x in cluster.clusters],
   'unclustered_count':len(cluster.unclustered_sample_ids)})


def _phase5_output_from_experiment(target: Path) -> Path:
 """Resolve the Phase-5 output root for both top-level and nested experiments."""
 current=target.resolve()
 while current.name!=PHASE5_OUTPUT_DIR and current.parent!=current:
  current=current.parent
 if current.name!=PHASE5_OUTPUT_DIR:
  # Unit tests and reusable callers may place an experiment in an arbitrary
  # temporary root.  Preserve the historical top-level layout in that case.
  return target.parent
 return current

def _signatures(target,manager,parent,trigger,node_feedback,rows_by_id,identity,spec,
                 protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 phase5_output=_phase5_output_from_experiment(target)
 cache=target/'signature_cache'/root_shard(parent.node_id)/identity[:16];cache.mkdir(parents=True,exist_ok=True);v1_cache=phase5_output/'phase6_split_only_evolution_v1'/'signature_cache'/root_shard(parent.node_id)/identity[:16]
 control_cache=phase5_output/CONTROL_EXPERIMENT_DIR/'signature_cache'/root_shard(parent.node_id)/identity[:16]
 errors={x.sample_id:x for x in node_feedback.errors if x.outcome=='wrong'};values={};total=len(trigger.decisive_wrong_sample_ids);reuse={'same_run':0,'v1_identity_match':0,'generated':0};pending=[]
 if protocol.read_only_control_signatures:reuse['control_v2_exact_identity']=0
 if protocol.is_retry_treatment:
  reuse['phase7_visual_exact_identity']=0
  source_attempt=_latest_visual_source_attempt(target.parent/protocol.source_experiment_dir)
  frozen=load_json(target/'frozen_manifest.json') if (target/'frozen_manifest.json').exists() else {}
  source_attempt=Path(frozen.get('source_latest_attempt_dir',source_attempt))
  signatures_path=source_attempt/'error_signatures.json'
  expected_hash=frozen.get('source_error_signatures_sha256')
  if expected_hash is not None and file_sha256(signatures_path)!=expected_hash:
   raise RuntimeError('frozen Phase7 Visual ErrorSignature bundle drift')
  source_outputs=load_json(signatures_path)['outputs']
 def accept(sid,value,shard,source_kind):
  if value.signature is None:
   details={'sample_id':sid,'parse_error':value.parse_error,'metrics':value.metrics.to_dict()}
   if value.raw_response is None and value.metrics.api_attempts>0 and value.metrics.error_count>=value.metrics.api_attempts:raise TransportFailed('error_signature',f'transport failed for signature {sid}',details)
   raise ProposalInvalid('error_signature',f'invalid signature {sid}: {value.parse_error}',details)
  if value.request_spec.to_dict()!=spec:raise RuntimeError('signature request identity mismatch')
  if not shard.exists():
   source_payload={'source':source_kind,'source_run':None if source_kind=='generated' else 'phase6_split_only_evolution_v1'}
   if source_kind=='control_v2_exact_identity':source_payload={'source':source_kind,'source_run':CONTROL_EXPERIMENT_DIR,'source_sha256':file_sha256(control_cache/base._safe_artifact_name(sid))}
   if source_kind=='phase7_visual_exact_identity':source_payload={'source':source_kind,'source_run':protocol.source_experiment_dir,'source_attempt':str(source_attempt)}
   _write(shard,value.to_dict());_write(shard.with_suffix('.source.json'),source_payload)
  values[sid]=value
 for sid in trigger.decisive_wrong_sample_ids:
  shard=cache/base._safe_artifact_name(sid);v1_shard=v1_cache/base._safe_artifact_name(sid);control_shard=control_cache/base._safe_artifact_name(sid)
  if shard.exists():accept(sid,ErrorSignatureOutput.from_dict(load_json(shard)),shard,'same_run');reuse['same_run']+=1
  elif protocol.is_retry_treatment:
   if sid not in source_outputs:raise RuntimeError(f'Phase7 Visual signature missing: {sid}')
   accept(sid,ErrorSignatureOutput.from_dict(source_outputs[sid]),shard,'phase7_visual_exact_identity');reuse['phase7_visual_exact_identity']+=1
  elif protocol.read_only_control_signatures:
   if not control_shard.exists():raise RuntimeError(f'Control signature missing: {parent.node_id}/{sid}')
   accept(sid,ErrorSignatureOutput.from_dict(load_json(control_shard)),shard,'control_v2_exact_identity');reuse['control_v2_exact_identity']+=1
  elif protocol.allow_legacy_signature_reuse and v1_shard.exists():accept(sid,ErrorSignatureOutput.from_dict(load_json(v1_shard)),shard,'v1_identity_match');reuse['v1_identity_match']+=1
  else:pending.append((sid,shard))
 completed=len(values)
 if completed:print(f'split-evolution signatures cached={completed}/{total} root={parent.node_id}',flush=True)
 if protocol.read_only_control_signatures and pending:
  raise RuntimeError('Treatment may not generate ErrorSignatures online')
 if protocol.is_retry_treatment and pending:
  raise RuntimeError('Visual retry treatment may not generate ErrorSignatures online')
 max_workers=min(max(1,len(pending)),manager.backend_pool.spec.global_request_concurrency)
 if pending:
  print(f'split-evolution signatures submitted={len(pending)} concurrency={max_workers} root={parent.node_id}',flush=True)
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
   futures={executor.submit(manager.infer_signature,rows_by_id[sid],parent,errors[sid]):(sid,shard) for sid,shard in pending}
   for future in as_completed(futures):
    sid,shard=futures[future];accept(sid,future.result(),shard,'generated');reuse['generated']+=1;completed+=1
    print(f'split-evolution signatures completed={completed}/{total} root={parent.node_id} sample={sid}',flush=True)
 out={sid:values[sid] for sid in trigger.decisive_wrong_sample_ids}
 return out,cache,reuse
def _prepare(config,target,epoch_dir,root,attempt_no,rubric,pred,feedback,rows,history,managers,specs,
             protocol:EvolutionProtocol=CONTROL_PROTOCOL,rubric_memory=None,rubric_memory_sha256=None):
 d=_attempt(epoch_dir,root,attempt_no);d.mkdir(parents=True,exist_ok=True);ctx=EvolutionContext(rubric,feedback);t=detect_specialize_trigger(ctx,root,config['evolution_policy']['trigger_thresholds']);_write(d/'trigger.json',t.to_dict())
 if not t.triggered:return {'root_id':root,'decision':'not_eligible','attempt_dir':d}
 parent=rubric.get_node(root);pname=parent.criterion.name;parent_outputs=[x[pname].vote.value for x in pred.node_outputs]
 signature_spec=specs['error_signature'];ident=signature_identity(root,t.to_dict(),parent_outputs,signature_spec)
 if protocol.read_only_control_signatures:
  source_manifest=load_json(target/'control_signature_reuse_manifest.json');source=source_manifest['root_sources'].get(root)
  if source is None:raise RuntimeError(f'Control signature source is not frozen for {root}')
  signature_spec=source['request_spec'];ident=source['signature_identity']
  if signature_identity(root,t.to_dict(),parent_outputs,signature_spec)!=ident:raise RuntimeError(f'Control signature identity no longer matches parent scope: {root}')
 rows_by_id={str(x['sample_id']):x for x in rows}
 outputs,cache,reuse=_signatures(target,managers['error_signature'],parent,t,feedback.nodes[root],rows_by_id,ident,signature_spec,protocol);signatures={k:v.signature for k,v in outputs.items()}
 signature_artifact={'signature_identity':ident,'cache_dir':str(cache),'reuse':reuse,'source_v1_read_only':reuse['v1_identity_match']>0,'outputs':{k:v.to_dict() for k,v in outputs.items()}}
 if protocol.read_only_control_signatures:signature_artifact.update({'control_signature_source':CONTROL_EXPERIMENT_DIR,'generated':reuse['generated']})
 _write(d/'error_signatures.json',signature_artifact)
 thresholds=config['evolution_policy']['trigger_thresholds'];prior=_prior(history,root,protocol.is_retry_treatment);retry_feedback=_retry_feedback(prior) if protocol.is_retry_treatment else None
 _write(d/'history_projection.json',{'schema_version':'1.0.0','root_id':root,'attempt':attempt_no,'history':prior,'projection_sha256':canonical_sha256(prior)})
 if protocol.is_retry_treatment:_write(d/'retry_feedback.json',retry_feedback)
 cluster_path=d/'cluster_proposal.json'
 if protocol.is_memory_treatment:
  if rubric_memory is None or rubric_memory_sha256!=canonical_sha256(rubric_memory):raise RuntimeError('missing or invalid epoch rubric memory')
  signature_source=(protocol.source_experiment_dir if protocol.is_retry_treatment
                    else CONTROL_EXPERIMENT_DIR if protocol.read_only_control_signatures
                    else 'same_trajectory_exact_identity_or_generated')
  _write(d/'rubric_memory_ref.json',{'rubric_memory_mode':protocol.rubric_memory_mode,'rubric_memory_sha256':rubric_memory_sha256,'signature_source':signature_source,'control_signature_source':signature_source if protocol.read_only_control_signatures else None,'semantic_cluster_request_spec':specs['semantic_cluster'],'child_generation_request_spec':specs['child_generation']})
 cluster=ClusterProposal.from_dict(load_json(cluster_path)) if cluster_path.exists() else managers['semantic_cluster'].cluster(tuple(signatures.values()),criterion_name=pname,min_cluster_size=thresholds['N_min_cluster'],max_clusters=t.remaining_capacity,prior_failures=prior,rubric_memory=rubric_memory,retry_feedback=retry_feedback)
 if not cluster_path.exists():_write(cluster_path,cluster.to_dict())
 _require_split_clusters(cluster)
 children=[]
 for i,c in enumerate(cluster.clusters,1):
  reps=c.sample_ids[:3];child_path=d/'children'/f'{c.cluster_id}.json'
  child=ChildCriterionProposal.from_dict(load_json(child_path)) if child_path.exists() else managers['child_generation'].generate_child(parent=parent,cluster=c,signatures=[signatures[x] for x in c.sample_ids],representative_rows=[rows_by_id[x] for x in reps],siblings=children,prior_failures=prior,rubric_memory=rubric_memory,retry_feedback=retry_feedback)
  children.append(child)
  if not child_path.exists():_write(child_path,child.to_dict())
  print(f'split-evolution children {i}/{len(cluster.clusters)} root={root}',flush=True)
 candidate=build_specialize_candidate(ctx,root,cluster,children,rows,signatures);_write(d/'candidate.json',candidate.to_dict());after=apply_rubric_patch(rubric,candidate.edit_candidate.patch);after.save_json(d/'candidate_rubric.json')
 return {'root_id':root,'decision':'prepared','attempt_dir':d,'candidate':candidate,'after_rubric':after,'signatures':signatures,'retry_feedback':retry_feedback}

def _failure(result,code,stage='competition',attribution=None,details=None):
 c=result.get('candidate');e=result.get('evaluation')
 cluster_summary=None if c is None else [{'cluster_id':x.cluster_id,'label':x.label,'sample_ids':list(x.sample_ids)} for x in c.cluster_proposal.clusters]
 children_summary=[] if c is None else [{'cluster_id':x.cluster_id,'criterion_name':x.criterion_name,'description':x.description[:800]} for x in c.children]
 metrics=None if e is None else {'parent_accuracy':e.parent_accuracy,'specialized_accuracy':e.specialized_accuracy,'accuracy_delta':e.accuracy_delta}
 compact_attribution=None if attribution is None else attribution.get('attribution',attribution)
 retry=result.get('child_retry_diagnostics')
 payload={'structured_failure':{'code':code,'stage':stage,'details':dict(details or {})},'cluster_summary':cluster_summary,'children_summary':children_summary,'local_metrics':metrics,'corrected_sample_ids':[] if e is None else list(e.subtree_diagnostic.corrected_sample_ids),'harmed_sample_ids':[] if e is None else list(e.subtree_diagnostic.harmed_sample_ids),'natural_language_attribution':compact_attribution}
 if retry is not None:
  payload['child_retry_diagnostics']=retry
  payload['preservation_directives']=retry['preservation_directives']
 return payload

def _proposal_failure_payload(d,stage,details):
 payload=_failure({'candidate':None},'proposal_invalid',stage,details=details)
 cluster_path=d/'cluster_proposal.json'
 if cluster_path.exists():
  cluster=ClusterProposal.from_dict(load_json(cluster_path));payload['cluster_summary']=[{'cluster_id':x.cluster_id,'label':x.label,'sample_ids':list(x.sample_ids)} for x in cluster.clusters]
 children=[]
 for path in sorted((d/'children').glob('*.json')) if (d/'children').exists() else ():
  child=ChildCriterionProposal.from_dict(load_json(path));children.append({'cluster_id':child.cluster_id,'criterion_name':child.criterion_name,'description':child.description})
 payload['children_summary']=children;return payload

def _required_failure_attribution(manager,attempt_dir,**kwargs):
 """Finish a rejected competition only after durable, structured attribution."""
 try:
  attribution=manager.attribute_split_failure(**kwargs)
 except SpecializeManagerFailure as exc:
  _write(attempt_dir/'failure_attribution_failure.json',exc.to_dict())
  if _manager_failure_kind(exc)==TRANSPORT_FAILED:
   raise TransportFailed('split_failure_attribution','Split rejection attribution transport failed',exc.to_dict()) from exc
  invalid={'outcome':ATTRIBUTION_INVALID,'stage':'split_failure_attribution','details':exc.to_dict(),'competition_decision_not_committed':True,'history_appended':False}
  _write(attempt_dir/'attribution_invalid.json',invalid)
  raise AttributionInvalid('Split rejection attribution remained schema-invalid after structured retries',invalid) from exc
 if not isinstance(attribution,Mapping) or not isinstance(attribution.get('attribution'),Mapping):
  invalid={'outcome':ATTRIBUTION_INVALID,'stage':'split_failure_attribution','details':{'message':'missing structured attribution payload'},'competition_decision_not_committed':True,'history_appended':False}
  _write(attempt_dir/'attribution_invalid.json',invalid)
  raise AttributionInvalid('Split rejection attribution is missing',invalid)
 _write(attempt_dir/'failure_attribution.json',attribution)
 return attribution

def _evaluate(config,epoch_dir,rows,rubric,pred,result,pool,manager,feedback,
              protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 c=result['candidate'];after=result['after_rubric'];ids=tuple(c.node_id_by_cluster.values());child_rubric=StructuredRubric(nodes={i:after.get_node(i) for i in ids},edges=(),root_ids=ids)
 label='children'
 try:child,artifact,valid=base._generate_pairwise(config,result['attempt_dir'],child_rubric,rows,label,execution_backend_pool=pool,request_backend_id=pred.request_spec.backend_id,request_level_progress=True)
 except RuntimeError as exc:
  if 'final-valid rate below' in str(exc):raise TransportFailed('pairwise_worker',str(exc)) from exc
  raise
 combined=assemble_specialized_pairwise_prediction(pred,child,after);combined.save_json(result['attempt_dir']/'combined_pairwise.json')
 policy=runtime_acceptance_policy(config);evaluation,_,after_execution=evaluate_specialize_candidate(before_rubric=rubric,after_rubric=after,combined_prediction=combined,child_prediction=child,dataset=rows,parent_node_id=result['root_id'],cluster_proposal=c.cluster_proposal,candidate=c,policy=policy)
 _write(result['attempt_dir']/'evaluation.json',evaluation.to_dict());records=base._split_parent_scope_predictions(context=EvolutionContext(rubric,feedback),candidate=c,combined_prediction=combined,after_execution=after_execution,rows=rows);_write(result['attempt_dir']/'specialized_predictions.json',records)
 decision=ACCEPTED if evaluation.specialized_accuracy>=evaluation.parent_accuracy else COMPETITION_REJECTED
 result.update({'evaluation':evaluation,'child_prediction':child,'combined':combined,'pairwise_artifact':str(artifact),'child_valid_rate':valid,'changed_predictions':records,'decision':decision})
 if protocol.is_retry_treatment:
  diagnostics=build_child_retry_diagnostics(candidate=c,combined=combined,rows=rows,
   parent_node_id=result['root_id'],parent_name=rubric.get_node(result['root_id']).criterion.name,
   evaluation=evaluation)
  result['child_retry_diagnostics']=diagnostics
  _write(result['attempt_dir']/'child_retry_diagnostics.json',diagnostics)
 if decision==COMPETITION_REJECTED:
  attribution=_required_failure_attribution(manager,result['attempt_dir'],parent=rubric.get_node(result['root_id']),signatures=tuple(result['signatures'].values()),cluster_proposal=c.cluster_proposal.to_dict(),children=c.children,local_metrics=evaluation.to_dict(),changed_predictions=records,retry_feedback=result.get('child_retry_diagnostics'))
  result['history_payload']=_failure(result,'specialized_accuracy_below_parent',attribution=attribution)
 return result
def _set_run_status(target,status,details):
 value=load_json(target/'stage_status.json');value['run']={'status':status,'details':details};_write(target/'stage_status.json',value)

def _pause_transport(target,epoch_no,root,attempt_no,stage,details):
 d=_attempt(_epoch(target,epoch_no),root,attempt_no);payload={'outcome':TRANSPORT_FAILED,'stage':stage,'details':details};_write(d/'transport_failure.json',payload);_set_run_status(target,'paused',{'epoch':epoch_no,'root_id':root,'attempt':attempt_no,**payload});print(f'split-evolution paused transport root={root} stage={stage}',flush=True)

def _abort_program(target,epoch_no,root,attempt_no,stage,exc):
 d=_attempt(_epoch(target,epoch_no),root,attempt_no);payload={'outcome':PROGRAM_ERROR,'stage':stage,'type':type(exc).__name__,'message':str(exc)};_write(d/'program_error.json',payload);_set_run_status(target,'aborted',{'epoch':epoch_no,'root_id':root,'attempt':attempt_no,**payload})

def run(config,output,protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 validate_policy(config,protocol);validate_phase5_lineage(output);runtime_acceptance_policy(config);target=experiment_root(output,protocol)
 if not (target/'frozen_manifest.json').exists():raise RuntimeError(f'run {protocol.stage_prefix}-freeze first')
 manifest=load_json(target/'frozen_manifest.json');history=_history(target)
 if history['completed']:print('split-evolution-run already completed');return
 _,_,rows,_,_=base._phase6_inputs(config,output);managers,profiles,specs,_=_managers(config,protocol);pool=base._single_endpoint_execution_pool(config,protocol.pairwise_endpoint);max_epochs=protocol.max_attempts or POLICY_V1['max_epochs']
 if protocol.read_only_control_signatures:
  specs['error_signature']=manifest['manager_request_specs']['error_signature']
  for stage in ('semantic_cluster','child_generation'):
   if specs[stage]!=manifest['manager_request_specs'][stage]:raise RuntimeError(f'Treatment Manager request identity drift after freeze: {stage}')
 elif protocol.is_retry_treatment:
  for stage in ('error_signature','semantic_cluster','child_generation'):
   if specs[stage]!=manifest['manager_request_specs'][stage]:raise RuntimeError(f'Visual retry Manager request identity drift after freeze: {stage}')
 _set_run_status(target,'running',{'current_epoch':history['current_epoch']})
 for epoch_no in range(history['current_epoch']+1,max_epochs+1):
  epoch_started=time.monotonic();previous=_epoch(target,epoch_no-1);epoch_dir=_epoch(target,epoch_no);rubric=StructuredRubric.load_json(previous/'rubric_committed.json');pred=PairwisePredictionOutput.load_json(previous/'discovery_pairwise.json');feedback=RubricFeedback.from_dict(load_json(previous/'feedback.json'));scheduled=retryable_roots(history);results={};rubric_memory=rubric_memory_sha256=None
  if protocol.is_memory_treatment:rubric_memory,rubric_memory_sha256=freeze_rubric_memory(epoch_dir,rubric)
  print(f'split-evolution epoch={epoch_no} scheduled={list(scheduled)}',flush=True);started={}
  for root in scheduled:
   started[root]=time.monotonic();n=history['root_states'][root]['attempt_count']+1
   try:results[root]=_prepare(config,target,epoch_dir,root,n,rubric,pred,feedback,rows,history,managers,specs,protocol,rubric_memory,rubric_memory_sha256)
   except TransportFailed as exc:
    _pause_transport(target,epoch_no,root,n,exc.stage,_failure_details(exc));return
   except ProposalInvalid as exc:
    d=_attempt(epoch_dir,root,n);payload=_proposal_failure_payload(d,exc.stage,_failure_details(exc));_write(d/'proposal_failure.json',payload);results[root]={'root_id':root,'decision':PROPOSAL_INVALID,'attempt_dir':d,'history_payload':payload}
   except SpecializeManagerFailure as exc:
    d=_attempt(epoch_dir,root,n);_write(d/'manager_failure.json',exc.to_dict());kind=_manager_failure_kind(exc)
    if kind==TRANSPORT_FAILED:_pause_transport(target,epoch_no,root,n,exc.stage,exc.to_dict());return
    payload=_proposal_failure_payload(d,exc.stage,exc.to_dict());results[root]={'root_id':root,'decision':PROPOSAL_INVALID,'attempt_dir':d,'history_payload':payload}
   except Exception as exc:
    _abort_program(target,epoch_no,root,n,'candidate_construction',exc);raise
  prepared={r:x['candidate'] for r,x in results.items() if x['decision']=='prepared'};collisions=colliding_roots(prepared);collided={r for owners in collisions.values() for r in owners}
  for root in collided:
   results[root]['decision']='cross_root_collision';results[root]['history_payload']=_failure(results[root],'cross_root_child_name_collision','synchronous_commit',details={'collisions':collisions});_write(results[root]['attempt_dir']/'collision.json',results[root]['history_payload'])
  for root in sorted(set(prepared)-collided):
   n=history['root_states'][root]['attempt_count']+1
   try:results[root]=_evaluate(config,epoch_dir,rows,rubric,pred,results[root],pool,managers['semantic_cluster'],feedback,protocol)
   except TransportFailed as exc:
    _pause_transport(target,epoch_no,root,n,exc.stage,_failure_details(exc));return
   except AttributionInvalid as exc:
    payload={'outcome':ATTRIBUTION_INVALID,'stage':exc.stage,'details':exc.details,'competition_decision_not_committed':True,'history_appended':False}
    _write(results[root]['attempt_dir']/'attribution_invalid.json',payload);_set_run_status(target,'failed',{'epoch':epoch_no,'root_id':root,'attempt':n,**payload});print(f'split-evolution attribution invalid root={root}',flush=True);return
   except Exception as exc:
    _abort_program(target,epoch_no,root,n,'pairwise_or_evaluation',exc);raise
  accepted={r:x['candidate'] for r,x in results.items() if x['decision']==ACCEPTED}
  try:
   committed=merge_accepted_rubrics(rubric,accepted);committed_pred=merge_predictions(pred,[results[r]['child_prediction'] for r in sorted(accepted)],committed);votes,_=_snapshot(epoch_dir,committed,committed_pred,rows,manifest)
  except Exception as exc:
   _abort_program(target,epoch_no,'__epoch__',0,'synchronous_commit',exc);raise
  records=[]
  for root in scheduled:
   result=results[root];state=history['root_states'][root];decision=result['decision']
   if decision not in SCIENTIFIC_ATTEMPT_OUTCOMES:raise RuntimeError(f'non-scientific outcome reached commit: {decision}')
   if decision==COMPETITION_REJECTED and not isinstance((result.get('history_payload') or {}).get('natural_language_attribution'),Mapping):
    raise RuntimeError(f'rejected Split cannot commit without natural-language attribution: {root}')
   state['attempt_count']+=1
   if decision==ACCEPTED:state.update({'status':'accepted_locked','accepted_epoch':epoch_no,'children':[x.criterion_name for x in result['candidate'].children]})
   else:state['status']='exhausted' if state['attempt_count']>=max_epochs else 'retryable'
   e=result.get('evaluation');record={'epoch':epoch_no,'root_id':root,'attempt':state['attempt_count'],'decision':decision,'competition_completed':decision in VALID_COMPETITION_OUTCOMES,'attempt_dir':str(result['attempt_dir']),'parent_accuracy':None if e is None else e.parent_accuracy,'specialized_accuracy':None if e is None else e.specialized_accuracy,'accuracy_delta':None if e is None else e.accuracy_delta,'elapsed_seconds':time.monotonic()-started[root],'history_payload':result.get('history_payload')};history['attempts'].append(record);records.append(record)
  history['current_epoch']=epoch_no
  if ((protocol.is_retry_treatment and not retryable_roots(history))
      or should_stop_after_epoch(epoch_no,history) or epoch_no==max_epochs):
   history['completed']=True;history['stop_reason']='max_valid_attempts_reached' if any(x['status']=='exhausted' for x in history['root_states'].values()) else 'no_retryable_roots'
  metrics=base._metrics(votes,rows);_write(epoch_dir/'summary.json',{'epoch':epoch_no,'scheduled_roots':list(scheduled),'accepted_roots':sorted(accepted),'collisions':collisions,'m1':metrics,'attempts':records,'rubric_node_count':len(committed.nodes),'retryable_roots':list(retryable_roots(history)),'epoch_wall_seconds':time.monotonic()-epoch_started});_save_history(target,history);print(f"split-evolution epoch={epoch_no} accepted={sorted(accepted)} m1_acc={metrics['accuracy']:.4f}",flush=True)
  if history['completed']:break
 _set_run_status(target,'passed' if history['completed'] else 'partial',{'current_epoch':history['current_epoch'],'stop_reason':history['stop_reason']})
def smoke(config,output,protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 """Run one isolated root through a complete Split attempt without committing it."""
 validate_policy(config,protocol);validate_phase5_lineage(output);runtime_acceptance_policy(config);target=experiment_root(output,protocol);manifest_path=target/'frozen_manifest.json'
 if not manifest_path.exists():raise RuntimeError(f'run {protocol.stage_prefix}-freeze first')
 manifest=load_json(manifest_path)
 if manifest.get('heldout_protocol')!=_heldout_protocol(protocol):raise RuntimeError('heldout protocol was not frozen before smoke')
 smoke_dir=target/'smoke';report_path=smoke_dir/'report.json'
 if report_path.exists():print('split-evolution-smoke already passed; reusing report');print(json.dumps(load_json(report_path),indent=2));return
 _,_,rows,_,_=base._phase6_inputs(config,output);rubric=StructuredRubric.load_json(_epoch(target,0)/'rubric_committed.json');pred=PairwisePredictionOutput.load_json(_epoch(target,0)/'discovery_pairwise.json');feedback=RubricFeedback.from_dict(load_json(_epoch(target,0)/'feedback.json'));root='init_02_visual_grounding_and_details'
 if not manifest['initial_triggers'][root]['triggered']:raise RuntimeError('smoke root is not eligible')
 managers,_,specs,_=_managers(config,protocol);pool=base._single_endpoint_execution_pool(config,PAIRWISE_ENDPOINT);history={'root_states':{root:{'status':'eligible','attempt_count':0}},'attempts':[]};started=time.monotonic();rubric_memory=rubric_memory_sha256=None
 if protocol.is_memory_treatment:
  specs['error_signature']=manifest['manager_request_specs']['error_signature']
  for stage in ('semantic_cluster','child_generation'):
   if specs[stage]!=manifest['manager_request_specs'][stage]:raise RuntimeError(f'Treatment Manager request identity drift after freeze: {stage}')
 if protocol.is_memory_treatment:rubric_memory,rubric_memory_sha256=freeze_rubric_memory(smoke_dir,rubric)
 try:
  result=_prepare(config,target,smoke_dir,root,1,rubric,pred,feedback,rows,history,managers,specs,protocol,rubric_memory,rubric_memory_sha256);result=_evaluate(config,smoke_dir,rows,rubric,pred,result,pool,managers['semantic_cluster'],feedback,protocol)
 except TransportFailed as exc:
  payload={'schema_version':'1.0.0','status':'paused','outcome':TRANSPORT_FAILED,'stage':exc.stage,'details':_failure_details(exc),'formal_evolution_mutated':False,'heldout_accessed':False};_write(smoke_dir/'status.json',payload);print(json.dumps(payload,indent=2));return
 except AttributionInvalid as exc:
  payload={'schema_version':'1.0.0','status':'failed','outcome':ATTRIBUTION_INVALID,'stage':exc.stage,'details':exc.details,'competition_decision_not_committed':True,'history_appended':False,'formal_evolution_mutated':False,'heldout_accessed':False};_write(smoke_dir/'status.json',payload);print(json.dumps(payload,indent=2));return
 except SpecializeManagerFailure as exc:
  kind=_manager_failure_kind(exc);payload={'schema_version':'1.0.0','status':'paused' if kind==TRANSPORT_FAILED else 'failed','outcome':kind,'stage':exc.stage,'details':exc.to_dict(),'formal_evolution_mutated':False,'heldout_accessed':False};_write(smoke_dir/'status.json',payload);print(json.dumps({'status':payload['status'],'outcome':kind,'stage':exc.stage},indent=2));return
 except ProposalInvalid as exc:
  payload={'schema_version':'1.0.0','status':'failed','outcome':PROPOSAL_INVALID,'stage':exc.stage,'details':_failure_details(exc),'formal_evolution_mutated':False,'heldout_accessed':False};_write(smoke_dir/'status.json',payload);print(json.dumps(payload,indent=2));return
 except Exception as exc:
  _write(smoke_dir/'program_error.json',{'outcome':PROGRAM_ERROR,'type':type(exc).__name__,'message':str(exc)});raise
 evaluation=result['evaluation'];report_value={'schema_version':'1.0.0','status':'passed','root_id':root,'decision':result['decision'],'parent_accuracy':evaluation.parent_accuracy,'specialized_accuracy':evaluation.specialized_accuracy,'accuracy_delta':evaluation.accuracy_delta,'corrected_sample_ids':list(evaluation.subtree_diagnostic.corrected_sample_ids),'harmed_sample_ids':list(evaluation.subtree_diagnostic.harmed_sample_ids),'child_valid_rate':result['child_valid_rate'],'children':[{'cluster_id':x.cluster_id,'criterion_name':x.criterion_name,'description':x.description} for x in result['candidate'].children],'elapsed_seconds':time.monotonic()-started,'formal_evolution_mutated':False,'heldout_accessed':False,'pairwise_endpoint':PAIRWISE_ENDPOINT};_write(report_path,report_value);status=load_json(target/'stage_status.json');status['smoke']={'status':'passed','details':{'root_id':root,'decision':result['decision'],'report':str(report_path)}};_write(target/'stage_status.json',status);print(json.dumps(report_value,indent=2))
def _md_table(headers,rows):
 lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
 return '\n'.join(lines+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def _audit_discovery_state(config,target,history,rows):
 completed=[];accepted=[]
 for record in history['attempts']:
  if record['decision'] not in VALID_COMPETITION_OUTCOMES:continue
  if record['decision']==COMPETITION_REJECTED and not isinstance(((record.get('history_payload') or {}).get('natural_language_attribution')),Mapping):
   raise RuntimeError(f'rejected Split history lacks natural-language attribution: {record["root_id"]} attempt={record["attempt"]}')
  d=Path(record['attempt_dir'])
  required=('candidate.json','candidate_rubric.json','combined_pairwise.json','evaluation.json')
  missing=[name for name in required if not (d/name).exists()]
  if missing:raise RuntimeError(f'completed competition missing artifacts: {record["root_id"]} {missing}')
  candidate=SpecializeCandidate.from_dict(load_json(d/'candidate.json'));after=StructuredRubric.load_json(d/'candidate_rubric.json');combined=PairwisePredictionOutput.load_json(d/'combined_pairwise.json');before=StructuredRubric.load_json(_epoch(target,record['epoch']-1)/'rubric_committed.json');ids=tuple(candidate.node_id_by_cluster.values());child_rubric=StructuredRubric(nodes={i:after.get_node(i) for i in ids},edges=(),root_ids=ids);child=project_pairwise_prediction(combined,child_rubric)
  replay,_,_=evaluate_specialize_candidate(before_rubric=before,after_rubric=after,combined_prediction=combined,child_prediction=child,dataset=rows,parent_node_id=record['root_id'],cluster_proposal=candidate.cluster_proposal,candidate=candidate,policy=runtime_acceptance_policy(config));stored=load_json(d/'evaluation.json')
  if canonical_sha256(replay.to_dict())!=canonical_sha256(stored):raise RuntimeError(f'evaluation replay mismatch: {record["root_id"]} attempt={record["attempt"]}')
  expected=ACCEPTED if replay.specialized_accuracy>=replay.parent_accuracy else COMPETITION_REJECTED
  if record['decision']!=expected:raise RuntimeError(f'decision mismatch: {record["root_id"]} attempt={record["attempt"]}')
  completed.append(record);accepted.extend([record] if expected==ACCEPTED else [])
 for root,state in history['root_states'].items():
  records=[x for x in history['attempts'] if x['root_id']==root]
  if state['attempt_count']!=len(records):raise RuntimeError(f'attempt count mismatch: {root}')
  if state['status']=='accepted_locked' and len([x for x in records if x['decision']==ACCEPTED])!=1:raise RuntimeError(f'accepted state mismatch: {root}')
 initial=StructuredRubric.load_json(_epoch(target,0)/'rubric_committed.json');final=StructuredRubric.load_json(_epoch(target,history['current_epoch'])/'rubric_committed.json');accepted_node_ids=[]
 for record in accepted:
  candidate=SpecializeCandidate.from_dict(load_json(Path(record['attempt_dir'])/'candidate.json'));accepted_node_ids.extend(candidate.node_id_by_cluster.values());expected_names=[x.criterion_name for x in candidate.children]
  if history['root_states'][record['root_id']]['children']!=expected_names:raise RuntimeError(f'accepted child state mismatch: {record["root_id"]}')
 if len(final.nodes)!=len(initial.nodes)+len(accepted_node_ids) or any(node_id not in final.nodes for node_id in accepted_node_ids):raise RuntimeError('committed rubric growth does not match accepted candidates')
 return {'scheduled_attempts':len(history['attempts']),'competition_completed':len(completed),'accepted':len(accepted),'competition_rejected':sum(x['decision']==COMPETITION_REJECTED for x in completed),'proposal_invalid':sum(x['decision']==PROPOSAL_INVALID for x in history['attempts']),'treatment_nonempty':bool(accepted),'scientific_status':'valid' if completed else 'inconclusive_no_completed_competition'}

def _audit_memory_protocol(target,history):
 source_manifest=load_json(target/'control_signature_reuse_manifest.json');attempts=0
 for record in history['attempts']:
  d=Path(record['attempt_dir']);signature_path=d/'error_signatures.json'
  if not signature_path.exists():continue
  signatures=load_json(signature_path);attempts+=1
  if signatures.get('generated')!=0 or signatures['reuse'].get('generated')!=0:
   raise RuntimeError(f'Treatment generated ErrorSignatures online: {record["root_id"]}')
  source=source_manifest['root_sources'][record['root_id']]
  if signatures['signature_identity']!=source['signature_identity']:
   raise RuntimeError(f'Treatment signature source drift: {record["root_id"]}')
  ref=load_json(d/'rubric_memory_ref.json');memory_path=_epoch(target,record['epoch'])/'rubric_memory.json';memory=load_json(memory_path)
  if ref['rubric_memory_sha256']!=canonical_sha256(memory):
   raise RuntimeError(f'Treatment root used wrong epoch memory: {record["root_id"]}')
 for epoch_no in range(history['current_epoch']+1):
  memory=load_json(_epoch(target,epoch_no)/'rubric_memory.json')
  expected=StructuredRubric.load_json(_epoch(target,max(0,epoch_no-1))/'rubric_committed.json') if epoch_no else StructuredRubric.load_json(_epoch(target,0)/'rubric_committed.json')
  if memory!=rubric_memory_snapshot(expected):raise RuntimeError(f'Rubric memory does not match epoch-start commit: {epoch_no}')
 return {'attempts_checked':attempts,'error_signatures_generated':0,'control_signatures_reused':source_manifest['reused'],'epoch_memories_checked':history['current_epoch']+1}

def report(config,output,protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 validate_policy(config,protocol);validate_phase5_lineage(output);target=experiment_root(output,protocol);history=_history(target)
 if not history['completed']:raise RuntimeError('split-evolution-run must complete first')
 _,_,rows,init_pred,_=base._phase6_inputs(config,output);audit=_audit_discovery_state(config,target,history,rows)
 if protocol.is_memory_treatment:audit['manager_memory_protocol']=_audit_memory_protocol(target,history)
 manifest=load_json(target/'frozen_manifest.json');final_epoch=_epoch(target,history['current_epoch']);rubric=StructuredRubric.load_json(final_epoch/'rubric_committed.json');pred=PairwisePredictionOutput.load_json(final_epoch/'discovery_pairwise.json');init_rubric=StructuredRubric.load_json(_epoch(target,0)/'rubric_committed.json');_,init_votes=execute_offline_m1(init_rubric,init_pred,rows);_,final_votes=execute_offline_m1(rubric,pred,rows)
 by_epoch_root={(x['epoch'],x['root_id']):x for x in history['attempts']};trajectory=[]
 for epoch_no in range(history['current_epoch']+1):
  for root,state in history['root_states'].items():
   item=by_epoch_root.get((epoch_no,root))
   if item is not None:trajectory.append(item);continue
   if epoch_no==0:decision='eligible' if manifest['initial_triggers'][root]['triggered'] else 'not_eligible'
   elif state['accepted_epoch'] is not None and epoch_no>state['accepted_epoch']:decision='accepted_locked'
   elif state['status']=='not_eligible':decision='not_eligible'
   else:decision='not_scheduled'
   trajectory.append({'epoch':epoch_no,'root_id':root,'attempt':None,'decision':decision,'parent_accuracy':None,'specialized_accuracy':None,'accuracy_delta':None})
 roots=[]
 for root,state in history['root_states'].items():
  attempts=[x for x in history['attempts'] if x['root_id']==root];last=attempts[-1] if attempts else {}
  roots.append({'root_id':root,'criterion_name':rubric.get_node(root).criterion.name,'status':state['status'],'attempts':state['attempt_count'],'children':state['children'],'parent_accuracy':last.get('parent_accuracy'),'specialized_accuracy':last.get('specialized_accuracy'),'delta':last.get('accuracy_delta')})
 retries=[]
 for root in history['root_states']:
  items=[x for x in history['attempts'] if x['root_id']==root]
  if len(items)>1:
   retries.append({'root_id':root,'attempts':len(items),'decisions':[x['decision'] for x in items],'failure_reasons':[None if not x.get('history_payload') else x['history_payload'].get('structured_failure') for x in items],'metric_deltas':[x.get('accuracy_delta') for x in items]})
 natural=any(any(x['decision']==COMPETITION_REJECTED for x in items[:-1]) and any(x['decision'] in VALID_COMPETITION_OUTCOMES for x in items[1:]) for r in history['root_states'] for items in [[x for x in history['attempts'] if x['root_id']==r]])
 discovery={'init':base._metrics(init_votes,rows),'final':base._metrics(final_votes,rows),'paired':base._paired_heldout_comparison(init_votes,final_votes,rows)}
 value={'schema_version':'1.0.0','validity_audit':audit,'trajectory':trajectory,'final_roots':roots,'discovery_m1':discovery,'retry_history':retries,'history_evidence':{'natural_reject_retry_observed':natural,'statement':'Natural reject→retry observed; inspect retry table.' if natural else 'History mechanism only verified by tests; this trajectory produced no natural evidence.'},'cost':_cost_summary(target,history),'final_rubric_sha256':rubric.rubric_sha256}
 if protocol.is_memory_treatment:
  value['protocol']=protocol.rubric_memory_mode;value['rubric_memory_epochs']=[]
  for path in sorted(target.glob('epochs/epoch_*/rubric_memory.json')):
   memory=load_json(path);value['rubric_memory_epochs'].append({'epoch':int(path.parent.name.split('_')[1]),'rubric_memory_sha256':canonical_sha256(memory),'rubric_sha256':memory['rubric_sha256'],'node_count':len(memory['nodes'])})
 (target/'final').mkdir(exist_ok=True);rubric.save_json(target/'final'/'rubric.json');_write(target/'final'/'discovery_report.json',value);status=load_json(target/'stage_status.json');status['report']={'status':'passed'};_write(target/'stage_status.json',status);print(json.dumps(discovery,indent=2))

def _cost_summary(target,history):
 provenance=list(target.glob('epochs/epoch_*/roots/*/attempt_*/provenance/*.json'));pair_api=pair_in=pair_out=0;pair_latency=0.0;pair_usage_incomplete=0
 for path in provenance:
  calls=load_json(path).get('calls',[]);pair_api+=sum(int(x.get('api_attempts',1)) for x in calls);pair_in+=sum(int(x.get('prompt_tokens') or 0) for x in calls);pair_out+=sum(int(x.get('completion_tokens') or 0) for x in calls);pair_latency+=sum(float(x.get('latency_seconds') or 0) for x in calls);pair_usage_incomplete+=sum(not bool(x.get('usage_complete',True)) for x in calls)
 manager_api=manager_in=manager_out=0;manager_latency=0.0;manager_failed_api=0;manager_usage_incomplete=0
 metric_files=list(target.glob('epochs/epoch_*/roots/*/attempt_*/cluster_proposal.json'))+list(target.glob('epochs/epoch_*/roots/*/attempt_*/children/*.json'))+list(target.glob('epochs/epoch_*/roots/*/attempt_*/failure_attribution.json'))
 for path in metric_files:
  m=load_json(path).get('metrics',{});manager_api+=int(m.get('api_attempts') or 0);manager_in+=int(m.get('input_tokens') or 0);manager_out+=int(m.get('output_tokens') or 0);manager_latency+=float(m.get('call_latency_seconds') or 0);manager_usage_incomplete+=not bool(m.get('usage_complete',True))
 failure_files=list(target.glob('epochs/epoch_*/roots/*/attempt_*/manager_failure.json'))+list(target.glob('epochs/epoch_*/roots/*/attempt_*/failure_attribution_failure.json'))
 for path in failure_files:
  m=load_json(path).get('metrics',{});attempts=int(m.get('api_attempts') or 0);manager_failed_api+=attempts;manager_api+=attempts;manager_in+=int(m.get('input_tokens') or 0);manager_out+=int(m.get('output_tokens') or 0);manager_latency+=float(m.get('call_latency_seconds') or 0);manager_usage_incomplete+=not bool(m.get('usage_complete',True))
 reused_signature_artifacts=0;reused_control_signature_artifacts=0
 for path in target.glob('signature_cache/*/*/*.json'):
  if path.name.endswith('.source.json'):continue
  source_path=path.with_suffix('.source.json');source=load_json(source_path).get('source') if source_path.exists() else 'generated'
  if source=='v1_identity_match':reused_signature_artifacts+=1;continue
  if source=='control_v2_exact_identity':reused_control_signature_artifacts+=1;continue
  m=load_json(path).get('metrics',{});manager_api+=int(m.get('api_attempts') or 0);manager_in+=int(m.get('input_tokens') or 0);manager_out+=int(m.get('output_tokens') or 0);manager_latency+=float(m.get('call_latency_seconds') or 0);manager_usage_incomplete+=not bool(m.get('usage_complete',True))
 epoch_wall=sum(float(load_json(path).get('epoch_wall_seconds') or 0) for path in target.glob('epochs/epoch_*/summary.json'))
 result={'epochs':history['current_epoch'],'attempts':len(history['attempts']),'reused_v1_signature_artifacts':reused_signature_artifacts,'manager_api_attempts':manager_api,'manager_failed_api_attempts':manager_failed_api,'manager_usage_incomplete_artifacts':manager_usage_incomplete,'manager_input_tokens':manager_in,'manager_output_tokens':manager_out,'manager_summed_latency_seconds':manager_latency,'evolution_epoch_wall_seconds':epoch_wall,'pairwise_provenance_files':len(provenance),'pairwise_api_attempts':pair_api,'pairwise_usage_incomplete_calls':pair_usage_incomplete,'pairwise_input_tokens':pair_in,'pairwise_output_tokens':pair_out,'pairwise_summed_latency_seconds':pair_latency}
 if reused_control_signature_artifacts:result['reused_control_v2_signature_artifacts']=reused_control_signature_artifacts
 return result
def _accepted_child_ids(rubric,history):
 ids=[]
 for root,state in history['root_states'].items():
  if state['status']=='accepted_locked':ids.extend(e.child_id for e in rubric.child_edges(root))
 return tuple(ids)

def require_heldout_treatment(rubric,history):
 ids=_accepted_child_ids(rubric,history)
 if not ids:raise RuntimeError('no_treatment: final rubric has no accepted children; heldout access is forbidden')
 return ids

def non_degenerate_acceptance_check(history):
 accepted=[x for x in history['attempts'] if x['decision']==ACCEPTED]
 return None if not accepted else all(x['accuracy_delta'] is not None and x['accuracy_delta']>=0 for x in accepted)
def heldout(config,output,protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 validate_policy(config,protocol);validate_phase5_lineage(output);target=experiment_root(output,protocol);history=_history(target);status=load_json(target/'stage_status.json')
 if status.get('heldout',{}).get('status')=='passed':print('split-evolution-heldout already passed; reusing frozen report');return
 if status.get('report',{}).get('status')!='passed':raise RuntimeError('run split-evolution-report first')
 final_rubric=StructuredRubric.load_json(target/'final'/'rubric.json');child_ids=require_heldout_treatment(final_rubric,history)
 heldout_path=base._path(config['heldout_dataset']);heldout_hash=file_sha256(heldout_path)
 if heldout_hash.lower()!=config['heldout_dataset_sha256'].lower():raise RuntimeError('heldout dataset hash mismatch')
 rows=load_jsonl_dataset(heldout_path,expected_count=500);base_path=output/'predictions/init_pairwise_p05_heldout500.json';base_pred=PairwisePredictionOutput.load_json(base_path);init_rubric=StructuredRubric.load_json(_epoch(target,0)/'rubric_committed.json');hdir=target/'heldout500'
 frozen={'schema_version':'1.0.0','final_rubric_sha256':final_rubric.rubric_sha256,'accepted_roots':{r:s['children'] for r,s in history['root_states'].items() if s['status']=='accepted_locked'},'pairwise_request_spec':base_pred.request_spec.to_dict(),'heldout_dataset_sha256':heldout_hash,'base_pairwise_sha256':canonical_sha256(base_pred.to_dict()),'metrics':['ACC','Coverage','correct_count','Wilson95CI','paired_corrected_harmed','exact_McNemar'],'selection_after_heldout_forbidden':True}
 if protocol.is_memory_treatment:frozen.update({'exploratory_reused_heldout':True,'control_experiment':protocol.control_experiment_dir})
 if (hdir/'frozen_manifest.json').exists() and load_json(hdir/'frozen_manifest.json')!=frozen:raise RuntimeError('heldout already frozen with different inputs')
 _write(hdir/'frozen_manifest.json',frozen)
 child_predictions=[]
 for node_id in child_ids:
  node=final_rubric.get_node(node_id);one=StructuredRubric(nodes={node_id:node},edges=(),root_ids=(node_id,));short_id='c'+canonical_sha256(node_id)[:12];artifact=hdir/'child_predictions'/f'{short_id}.json'
  if artifact.exists():p=PairwisePredictionOutput.load_json(artifact)
  else:
   p,generated,_=base._generate_pairwise(config,hdir,one,rows,f'heldout_{short_id}',execution_backend_pool=base._single_endpoint_execution_pool(config,PAIRWISE_ENDPOINT),request_backend_id=base_pred.request_spec.backend_id,request_level_progress=True);artifact.parent.mkdir(parents=True,exist_ok=True);p.save_json(artifact)
  child_predictions.append(p)
 combined=merge_predictions(base_pred,child_predictions,final_rubric);combined.save_json(hdir/'combined_pairwise.json');_,init_votes=execute_offline_m1(init_rubric,base_pred,rows);_,final_votes=execute_offline_m1(final_rubric,combined,rows)
 baseline=load_json(output/'reports/init_baseline.json')['heldout500'][base.StructuredSystemVariant.M1_ALL_ROOTS_CASCADE.value]
 replay=base._heldout_vote_metrics(init_votes,rows)
 if replay['accuracy']!=baseline['accuracy'] or replay['coverage']!=baseline['coverage']:raise RuntimeError('Init heldout M1 replay mismatch')
 per_root=[]
 for root,state in history['root_states'].items():
  if state['status']!='accepted_locked':continue
  child_root_ids=tuple(e.child_id for e in final_rubric.child_edges(root));parent_variant=StructuredRubric(nodes={root:final_rubric.get_node(root)},edges=(),root_ids=(root,));sub_nodes={root:final_rubric.get_node(root),**{i:final_rubric.get_node(i) for i in child_root_ids}};sub=StructuredRubric(nodes=sub_nodes,edges=tuple(e for e in final_rubric.edges if e.parent_id==root and e.child_id in child_root_ids),root_ids=(root,));_,pv=execute_offline_m1(parent_variant,project_pairwise_prediction(combined,parent_variant),rows);_,sv=execute_offline_m1(sub,project_pairwise_prediction(combined,sub),rows);parent_name=final_rubric.get_node(root).criterion.name;scope=[x[parent_name].parse_ok and x[parent_name].answer_valid and x[parent_name].vote.value in {'A','B'} for x in combined.node_outputs];indices=[i for i,x in enumerate(scope) if x];local_parent=[pv[i] for i in indices];local_spec=[sv[i] for i in indices];local_rows=[rows[i] for i in indices]
  parent_local=base._heldout_vote_metrics(local_parent,local_rows);spec_local=base._heldout_vote_metrics(local_spec,local_rows);heldout_delta=spec_local['accuracy']-parent_local['accuracy'];accepted_attempt=next(x for x in history['attempts'] if x['root_id']==root and x['decision']=='accepted');discovery_delta=accepted_attempt['accuracy_delta']
  per_root.append({'root_id':root,'parent_only_all500':base._heldout_vote_metrics(pv,rows),'parent_plus_children_all500':base._heldout_vote_metrics(sv,rows),'parent_scope_support':len(indices),'parent_scope_parent':parent_local,'parent_scope_specialized':spec_local,'discovery_accuracy_delta':discovery_delta,'heldout_accuracy_delta':heldout_delta,'discovery_to_heldout_delta_shift':heldout_delta-discovery_delta,'paired_parent_scope':base._paired_heldout_comparison(local_parent,local_spec,local_rows),'activation':_activation(combined,child_root_ids,final_rubric,scope)})
 report_value={'schema_version':'1.0.0','init_five_root_m1':replay,'final_split_evolved_m1':base._heldout_vote_metrics(final_votes,rows),'paired_init_to_final':base._paired_heldout_comparison(init_votes,final_votes,rows),'per_evolved_root':per_root,'heldout_accessed_once':True,'selection_or_retry_after_report_forbidden':True}
 if protocol.is_memory_treatment:report_value.update({'protocol':protocol.rubric_memory_mode,'exploratory_reused_heldout':True,'final_rubric_sha256':final_rubric.rubric_sha256,'combined_pairwise_sha256':canonical_sha256(combined.to_dict())})
 _write(hdir/'report.json',report_value);status['heldout']={'status':'passed'};_write(target/'stage_status.json',status);print(json.dumps(report_value['paired_init_to_final'],indent=2))

def _activation(pred,child_ids,rubric,scope):
 names=[rubric.get_node(i).criterion.name for i in child_ids];dist=defaultdict(int);overlap=conflict=0
 for ok,row in zip(scope,pred.node_outputs):
  if not ok:continue
  votes=[row[n].vote.value for n in names if row[n].parse_ok and row[n].answer_valid and row[n].vote.value in {'A','B'}];dist[len(votes)]+=1;overlap+=len(votes)>=2;conflict+=('A' in votes and 'B' in votes)
 return {'active_children_distribution':dict(dist),'overlap_samples':overlap,'conflict_samples':conflict}

def _semantic_near_duplicates(rubric):
 root_set=set(rubric.root_ids);items=[]
 for node_id in rubric.preorder_node_ids():
  if node_id in root_set:continue
  node=rubric.get_node(node_id);tokens=set(re.findall(r'[a-z0-9]+',(node.criterion.name+' '+node.criterion.description).lower()))
  items.append((node_id,node.criterion.name,tokens))
 pairs=[]
 for i,(left_id,left_name,left) in enumerate(items):
  for right_id,right_name,right in items[i+1:]:
   similarity=len(left&right)/len(left|right) if left|right else 1.0
   if similarity>=.35:pairs.append({'left_id':left_id,'left_name':left_name,'right_id':right_id,'right_name':right_name,'lexical_jaccard':similarity})
 return sorted(pairs,key=lambda x:(-x['lexical_jaccard'],x['left_id'],x['right_id']))

def _accepted_diagnostics(history):
 values=[]
 for record in history['attempts']:
  if record['decision']!=ACCEPTED:continue
  evaluation=load_json(Path(record['attempt_dir'])/'evaluation.json')
  values.append({'root_id':record['root_id'],'epoch':record['epoch'],'attempt':record['attempt'],'child_diagnostics':evaluation['child_diagnostics'],'subtree_diagnostic':evaluation['subtree_diagnostic']})
 return values

def _memory_ablation_comparison(config,output,target,discovery,held):
 control=output/CONTROL_EXPERIMENT_DIR;control_history=_history(control);treatment_history=_history(target)
 control_discovery=load_json(control/'final'/'discovery_report.json');control_held=load_json(control/'heldout500'/'report.json')
 _,_,discovery_rows,_,_=base._phase6_inputs(config,output)
 control_epoch=_epoch(control,control_history['current_epoch']);treatment_epoch=_epoch(target,treatment_history['current_epoch'])
 control_rubric=StructuredRubric.load_json(control/'final'/'rubric.json');treatment_rubric=StructuredRubric.load_json(target/'final'/'rubric.json')
 control_pred=PairwisePredictionOutput.load_json(control_epoch/'discovery_pairwise.json');treatment_pred=PairwisePredictionOutput.load_json(treatment_epoch/'discovery_pairwise.json')
 _,control_discovery_votes=execute_offline_m1(control_rubric,control_pred,discovery_rows);_,treatment_discovery_votes=execute_offline_m1(treatment_rubric,treatment_pred,discovery_rows)
 heldout_path=base._path(config['heldout_dataset']);heldout_rows=load_jsonl_dataset(heldout_path,expected_count=500)
 control_held_pred=PairwisePredictionOutput.load_json(control/'heldout500'/'combined_pairwise.json');treatment_held_pred=PairwisePredictionOutput.load_json(target/'heldout500'/'combined_pairwise.json')
 _,control_held_votes=execute_offline_m1(control_rubric,control_held_pred,heldout_rows);_,treatment_held_votes=execute_offline_m1(treatment_rubric,treatment_held_pred,heldout_rows)
 accepted=lambda history:[{'root_id':x['root_id'],'epoch':x['epoch'],'attempt':x['attempt'],'delta':x['accuracy_delta']} for x in history['attempts'] if x['decision']==ACCEPTED]
 return {
  'primary':{'metric':'heldout500_equal_weight_five_root_m1_accuracy','control':control_held['final_split_evolved_m1'],'treatment':held['final_split_evolved_m1'],'accuracy_delta':held['final_split_evolved_m1']['accuracy']-control_held['final_split_evolved_m1']['accuracy'],'paired':base._paired_heldout_comparison(control_held_votes,treatment_held_votes,heldout_rows)},
  'secondary':{'discovery90_control':control_discovery['discovery_m1']['final'],'discovery90_treatment':discovery['discovery_m1']['final'],'paired':base._paired_heldout_comparison(control_discovery_votes,treatment_discovery_votes,discovery_rows)},
  'acceptance_order':{'control':accepted(control_history),'treatment':accepted(treatment_history)},
  'diagnostics':{'treatment_child_count':len(treatment_rubric.nodes)-len(treatment_rubric.root_ids),'control_child_count':len(control_rubric.nodes)-len(control_rubric.root_ids),'treatment_semantic_near_duplicates':_semantic_near_duplicates(treatment_rubric),'control_semantic_near_duplicates':_semantic_near_duplicates(control_rubric),'treatment_accepted_child_diagnostics':_accepted_diagnostics(treatment_history),'control_accepted_child_diagnostics':_accepted_diagnostics(control_history),'root_activation':held['per_evolved_root'],'rubric_memory_epochs':discovery.get('rubric_memory_epochs',[])},
  'claim_status':('supports_memory_acc_gain' if held['final_split_evolved_m1']['accuracy']>control_held['final_split_evolved_m1']['accuracy'] else 'no_acc_gain' if held['final_split_evolved_m1']['accuracy']==control_held['final_split_evolved_m1']['accuracy'] else 'memory_reduced_acc'),
  'confirmatory':False,
  'caveat':'The same heldout-500 was used by prior experiments; this is an exploratory paired ablation, not a new unbiased confirmatory test.'}

def final_report(config,output,protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 validate_policy(config,protocol);validate_phase5_lineage(output);target=experiment_root(output,protocol);status=load_json(target/'stage_status.json')
 if status.get('heldout',{}).get('status')!='passed':raise RuntimeError('run split-evolution-heldout first')
 discovery=load_json(target/'final'/'discovery_report.json');held=load_json(target/'heldout500'/'report.json');history=_history(target)
 final_rubric=StructuredRubric.load_json(target/'final'/'rubric.json');held_manifest=load_json(target/'heldout500'/'frozen_manifest.json')
 if discovery['final_rubric_sha256']!=final_rubric.rubric_sha256 or held_manifest['final_rubric_sha256']!=final_rubric.rubric_sha256:
  raise RuntimeError('final Rubric hash differs across discovery and heldout artifacts')
 if protocol.is_memory_treatment:
  combined=PairwisePredictionOutput.load_json(target/'heldout500'/'combined_pairwise.json')
  if held.get('final_rubric_sha256')!=final_rubric.rubric_sha256 or held.get('combined_pairwise_sha256')!=canonical_sha256(combined.to_dict()):
   raise RuntimeError('Treatment heldout report hash differs from frozen predictions')
 m1=[{'split':'discovery90','system':'Init five-root M1',**discovery['discovery_m1']['init']},{'split':'discovery90','system':'Final split-evolved M1',**discovery['discovery_m1']['final']},{'split':'heldout500','system':'Init five-root M1',**held['init_five_root_m1']},{'split':'heldout500','system':'Final split-evolved M1',**held['final_split_evolved_m1']}]
 report_value={'schema_version':'1.0.0','tables':{'epoch_root_trajectory':discovery['trajectory'],'final_root_local_accuracy':discovery['final_roots'],'init_vs_final_m1':m1,'reject_retry':discovery['retry_history'],'cost_and_growth':{**discovery['cost'],'initial_nodes':5,'final_nodes':len(StructuredRubric.load_json(target/'final'/'rubric.json').nodes)}},'history_evidence':discovery['history_evidence'],'heldout_paired':held['paired_init_to_final'],'success_checks':{'all_discovery_accepts_non_degenerate':non_degenerate_acceptance_check(history),'heldout_accuracy_improved':held['final_split_evolved_m1']['accuracy']>held['init_five_root_m1']['accuracy'],'heldout_corrected_gt_harmed':len(held['paired_init_to_final']['corrected_sample_ids'])>len(held['paired_init_to_final']['harmed_sample_ids']),'coverage_not_materially_lower':held['final_split_evolved_m1']['coverage']>=held['init_five_root_m1']['coverage']-.01}}
 if protocol.is_memory_treatment:
  report_value['manager_memory_ablation']=_memory_ablation_comparison(config,output,target,discovery,held)
 _write(target/'final_report.json',report_value);t=report_value['tables'];md=['# Split-only Evolution Final Report','',discovery['history_evidence']['statement'],'','## 1. Epoch × root trajectory','',_md_table(['epoch','root','attempt','decision','parent ACC','specialized ACC','delta'],[(x.get('epoch'),x['root_id'],x.get('attempt','-'),x['decision'],x.get('parent_accuracy'),x.get('specialized_accuracy'),x.get('accuracy_delta')) for x in t['epoch_root_trajectory']]),'','## 2. Final root local results','',_md_table(['root','criterion','status','attempts','children','parent ACC','specialized ACC'],[(x['root_id'],x['criterion_name'],x['status'],x['attempts'],', '.join(x['children']),x['parent_accuracy'],x['specialized_accuracy']) for x in t['final_root_local_accuracy']]),'','## 3. Init vs Final M1','',_md_table(['split','system','ACC','Coverage','correct'],[(x['split'],x['system'],x['accuracy'],x['coverage'],x.get('correct_count','-')) for x in m1]),'','## 4. Reject → retry','',json.dumps(t['reject_retry'],ensure_ascii=False,indent=2),'','## 5. Cost and rubric growth','',json.dumps(t['cost_and_growth'],ensure_ascii=False,indent=2)]
 if protocol.is_memory_treatment:
  comparison=report_value['manager_memory_ablation'];primary=comparison['primary'];md.extend(['','## 6. Manager memory ablation','',comparison['caveat'],'',_md_table(['system','heldout ACC','Coverage','correct'],[('Control v2',primary['control']['accuracy'],primary['control']['coverage'],primary['control'].get('correct_count')),('Global-Rubric Memory',primary['treatment']['accuracy'],primary['treatment']['coverage'],primary['treatment'].get('correct_count'))]),'',f"Claim status: {comparison['claim_status']}; ACC delta={primary['accuracy_delta']:.6f}"])
 (target/'final_report.md').write_text('\n'.join(md),encoding='utf-8');status=load_json(target/'stage_status.json');status['final_report']={'status':'passed'};_write(target/'stage_status.json',status);print(str(target/'final_report.md'))

def _visual_source_records(source):
 history=load_json(source/'evolution_history.json')
 if not history.get('completed'):raise RuntimeError('Phase7 Split+Refine source is incomplete')
 records=[x for x in history['attempts']
          if x['root_id']==VISUAL_RETRY_ROOT and x['decision']==COMPETITION_REJECTED]
 if not records:raise RuntimeError('Phase7 source contains no rejected Visual Split')
 return sorted(records,key=lambda x:(x['epoch'],x['attempt']))

def _latest_visual_source_attempt(source):
 return Path(_visual_source_records(source)[-1]['attempt_dir'])

def _source_record_with_retry_diagnostics(source,record,rows):
 attempt=Path(record['attempt_dir'])
 candidate=SpecializeCandidate.from_dict(load_json(attempt/'candidate.json'))
 combined=PairwisePredictionOutput.load_json(attempt/'combined_pairwise.json')
 evaluation=SpecializeEvaluation.from_dict(load_json(attempt/'evaluation.json'))
 before=StructuredRubric.load_json(source/'epochs'/f"epoch_{record['epoch']-1:02d}"/'rubric_committed.json')
 diagnostics=build_child_retry_diagnostics(
  candidate=candidate,combined=combined,rows=rows,parent_node_id=VISUAL_RETRY_ROOT,
  parent_name=before.get_node(VISUAL_RETRY_ROOT).criterion.name,evaluation=evaluation)
 copied=dict(record);payload=dict(record.get('history_payload') or {})
 payload['child_retry_diagnostics']=diagnostics
 payload['preservation_directives']=diagnostics['preservation_directives']
 copied['history_payload']=payload;copied['source_attempt']=True
 return copied

def visual_retry_freeze(config,output):
 protocol=VISUAL_RETRY_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output)
 target=experiment_root(output,protocol);manifest_path=target/'frozen_manifest.json'
 if manifest_path.exists():print('split-retry-visual-freeze already passed');return
 source=output/protocol.source_experiment_dir
 records=_visual_source_records(source);latest=records[-1]
 source_epoch=latest['epoch']-1;source_epoch_dir=source/'epochs'/f'epoch_{source_epoch:02d}'
 required=[source_epoch_dir/x for x in ('rubric_committed.json','discovery_pairwise.json','feedback.json')]
 missing=[str(x) for x in required if not x.exists()]
 if missing:raise RuntimeError(f'Visual retry source artifacts missing: {missing}')
 rubric=StructuredRubric.load_json(required[0]);pred=PairwisePredictionOutput.load_json(required[1])
 _,_,rows,_,_=base._phase6_inputs(config,output)
 feedback=RubricFeedback.from_dict(load_json(required[2]));ctx=EvolutionContext(rubric,feedback)
 trigger=detect_specialize_trigger(ctx,VISUAL_RETRY_ROOT,config['evolution_policy']['trigger_thresholds'])
 if not trigger.triggered:raise RuntimeError('frozen Visual source is no longer Split-eligible')
 pool=base.BackendPoolSpec.from_dict(config['backend_pool'])
 if protocol.pairwise_endpoint not in {x.endpoint_id for x in pool.endpoints}:
  raise RuntimeError(f'backend_pool is missing {protocol.pairwise_endpoint}')
 managers,profiles,specs,ids=_managers(config,protocol)
 source_attempt=Path(latest['attempt_dir']);source_signatures=load_json(source_attempt/'error_signatures.json')
 source_spec=next(iter(source_signatures['outputs'].values()))['request_spec']
 scientific=('model','prompt_sha256','decoding_config','prompt_version','parser_version')
 if any(specs['error_signature'][key]!=source_spec[key] for key in scientific):
  raise RuntimeError('Phase7 Visual ErrorSignature scientific identity drift')
 seed=[_source_record_with_retry_diagnostics(source,x,rows) for x in records]
 latest_diag=seed[-1]['history_payload']['child_retry_diagnostics']
 manifest={'schema_version':'2.0.0','experiment':protocol.experiment_dir,
  'variant':'visual_grounding_child_diagnostic_v2','root_id':VISUAL_RETRY_ROOT,
  'source_experiment_dir':protocol.source_experiment_dir,'source_epoch':source_epoch,
  'source_visual_attempt_count':len(seed),'source_latest_attempt_dir':str(source_attempt),
  'source_latest_candidate_sha256':file_sha256(source_attempt/'candidate.json'),
  'source_latest_combined_pairwise_sha256':file_sha256(source_attempt/'combined_pairwise.json'),
  'source_error_signatures_sha256':file_sha256(source_attempt/'error_signatures.json'),
  'initial_rubric_sha256':rubric.rubric_sha256,'initial_trigger':trigger.to_dict(),
  'pairwise_request_spec':pred.request_spec.to_dict(),'pairwise_endpoint':protocol.pairwise_endpoint,
  'max_treatment_attempts':protocol.max_attempts,'retry_feedback_mode':protocol.retry_feedback_mode,
  'rubric_memory_mode':protocol.rubric_memory_mode,'manager_profiles':profiles,
  'manager_request_specs':specs,'manager_endpoint_identities':ids,
  'acceptance_rule':'specialized_accuracy >= parent_accuracy on frozen parent scope',
  'heldout_access':'not_implemented_for_b1_discovery_pilot',
 'preservation_implementation':latest_diag and _retry_feedback(
   _history_projection({'attempts':seed},VISUAL_RETRY_ROOT,True))['preservation_implementation']}
 _write(manifest_path,manifest);_snapshot(_epoch(target,0),rubric,pred,rows,manifest)
 _,memory_hash=freeze_rubric_memory(_epoch(target,0),rubric)
 manifest['epoch_00_rubric_memory_sha256']=memory_hash;_write(manifest_path,manifest)
 _write(_epoch(target,0)/'summary.json',{'epoch':0,'source_epoch':source_epoch,
  'root_id':VISUAL_RETRY_ROOT,'source_latest_diagnostics':latest_diag})
 state={VISUAL_RETRY_ROOT:{'status':'eligible','attempt_count':0,'accepted_epoch':None,'children':[]}}
 _save_history(target,{'schema_version':'2.0.0','current_epoch':0,'completed':False,
  'stop_reason':None,'root_states':state,'attempts':[],'seed_attempts':seed})
 _write(target/'stage_status.json',{'freeze':{'status':'passed'}})
 print(json.dumps({'target':str(target),'root_id':VISUAL_RETRY_ROOT,
  'source_failures':len(seed),'max_treatment_attempts':protocol.max_attempts},indent=2))

def visual_retry_audit(config,output):
 protocol=VISUAL_RETRY_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output)
 target=experiment_root(output,protocol)
 if not (target/'frozen_manifest.json').exists():raise RuntimeError('run split-retry-visual-freeze first')
 history=_history(target);latest=history['seed_attempts'][-1]
 diagnostics=latest['history_payload']['child_retry_diagnostics']
 strongest=max(diagnostics['children'],key=lambda x:(x['accuracy_delta_same_support'],x['accuracy']))
 expected={'parent_accuracy':62/89,'specialized_accuracy':61/89,'support':89}
 tolerance=1e-12
 if (abs(diagnostics['parent_accuracy']-expected['parent_accuracy'])>tolerance
     or abs(diagnostics['specialized_accuracy']-expected['specialized_accuracy'])>tolerance
     or diagnostics['parent_scope_support']!=expected['support']):
  raise RuntimeError('Visual source metrics differ from frozen B1 audit expectations')
 peripheral=next((x for x in diagnostics['children']
                  if x['criterion_name']=='peripheral_detail_verification_accuracy'),None)
 if peripheral is None or peripheral['recommended_action'] not in {'preserve_exact','refine'}:
  raise RuntimeError('audit failed to preserve/refine the known strong peripheral child')
 audit={'schema_version':'2.0.0','api_requests_sent':0,'source_attempt':latest['attempt'],
  'parent_accuracy':diagnostics['parent_accuracy'],
  'specialized_accuracy':diagnostics['specialized_accuracy'],
  'accuracy_delta':diagnostics['specialized_accuracy']-diagnostics['parent_accuracy'],
  'corrected_count':len((latest['history_payload']).get('corrected_sample_ids',[])),
  'harmed_count':len((latest['history_payload']).get('harmed_sample_ids',[])),
  'sibling_conflict_samples':len(set(x for pair in diagnostics['sibling_pairs']
                                    for x in pair['conflict_sample_ids'])),
  'strongest_child':strongest,'peripheral_child':peripheral,
  'preservation_directives':diagnostics['preservation_directives'],
  'hash_identical_reuse_claimed':False,'pairwise_prediction_reuse_claimed':False,
  'status':'passed'}
 _write(target/'offline_audit.json',audit);status=load_json(target/'stage_status.json')
 status['audit']={'status':'passed'};_write(target/'stage_status.json',status)
 print(json.dumps(audit,indent=2))

def visual_retry_run(config,output):
 target=experiment_root(output,VISUAL_RETRY_PROTOCOL)
 if not (target/'offline_audit.json').exists():raise RuntimeError('run split-retry-visual-audit first')
 return run(config,output,VISUAL_RETRY_PROTOCOL)

def visual_retry_report(config,output):
 protocol=VISUAL_RETRY_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output)
 target=experiment_root(output,protocol);history=_history(target)
 if not history.get('completed'):raise RuntimeError('run split-retry-visual-run first')
 attempts=[]
 for record in history['attempts']:
  diagnostics_path=Path(record['attempt_dir'])/'child_retry_diagnostics.json'
  diagnostics=load_json(diagnostics_path) if diagnostics_path.exists() else None
  attempts.append({'attempt':record['attempt'],'decision':record['decision'],
   'parent_accuracy':record['parent_accuracy'],'specialized_accuracy':record['specialized_accuracy'],
   'accuracy_delta':record['accuracy_delta'],
   'children':[] if diagnostics is None else diagnostics['children'],
   'sibling_pairs':[] if diagnostics is None else diagnostics['sibling_pairs'],
   'preservation_directives':[] if diagnostics is None else diagnostics['preservation_directives'],
   'diagnostics_available':diagnostics is not None,
   'structured_failure':(record.get('history_payload') or {}).get('structured_failure')})
 final_epoch=_epoch(target,history['current_epoch']);summary=load_json(final_epoch/'summary.json')
 report_value={'schema_version':'2.0.0','experiment':protocol.experiment_dir,
  'root_id':VISUAL_RETRY_ROOT,'source_failures_used':len(history.get('seed_attempts',[])),
  'treatment_attempt_count':len(attempts),'final_status':history['root_states'][VISUAL_RETRY_ROOT]['status'],
  'accepted':any(x['decision']==ACCEPTED for x in attempts),'attempts':attempts,
  'final_discovery_m1':summary['m1'],'pairwise_endpoint':protocol.pairwise_endpoint,
  'acceptance_rule_unchanged':True,'heldout_accessed':False,
  'preservation_mode':'manager_directive_only','hash_identical_reuse':False,
  'pairwise_prediction_reused':False}
 _write(target/'report.json',report_value);status=load_json(target/'stage_status.json')
 status['report']={'status':'passed'};_write(target/'stage_status.json',status)
 print(json.dumps({'accepted':report_value['accepted'],'attempts':len(attempts),
  'final_status':report_value['final_status']},indent=2))


# ---------------------------------------------------------------------------
# Split-retry v2: frozen clusters, exact locked children, and sample packets.
# This is intentionally a one-root discovery pilot rather than a new generic
# Split scheduler.  It keeps the v1 protocol replayable and makes every v2
# source artifact explicit.

PARTIAL_ACCEPT_LOCKED='partial_accept_locked'

def _v2_votes(parent_outputs, child_outputs, names):
 return tuple(aggregate_child_subtrees(parent_outputs[i].vote,
  [child_outputs[name][i].vote for name in names if _decisive(child_outputs[name][i])])
  for i in range(len(parent_outputs)))

def _v2_metrics(parent_outputs, child_outputs, names, rows):
 scope=[_decisive(value) for value in parent_outputs];support=sum(scope)
 if not support:raise ValueError('locked retry requires non-empty parent scope')
 votes=_v2_votes(parent_outputs,child_outputs,names)
 correct=sum(scope[i] and votes[i].value==str(rows[i]['answer']) for i in range(len(rows)))
 parent_correct=sum(scope[i] and parent_outputs[i].vote.value==str(rows[i]['answer'])
                    for i in range(len(rows)))
 corrected=[str(rows[i]['sample_id']) for i in range(len(rows)) if scope[i]
            and votes[i].value==str(rows[i]['answer'])
            and parent_outputs[i].vote.value!=str(rows[i]['answer'])]
 harmed=[str(rows[i]['sample_id']) for i in range(len(rows)) if scope[i]
          and votes[i].value!=str(rows[i]['answer'])
          and parent_outputs[i].vote.value==str(rows[i]['answer'])]
 return {'parent_scope_support':support,'parent_accuracy':parent_correct/support,
         'specialized_accuracy':correct/support,'accuracy_delta':(correct-parent_correct)/support,
         'corrected_sample_ids':corrected,'harmed_sample_ids':harmed}

def select_compatible_locked_children(diagnostics,combined,rows,parent_name,
                                     *,min_support=15,min_net_corrected=3):
 """Return the deterministic, collectively compatible strong-child subset."""
 parent_outputs=[row[parent_name] for row in combined.node_outputs]
 child_outputs={item['criterion_name']:[row[item['criterion_name']]
                for row in combined.node_outputs]
                for item in diagnostics['children']}
 # The explicit loops below deliberately use only the fixed parent scope; no
 # global M1 or heldout quantity can influence locking.
 candidates=[]
 for item in diagnostics['children']:
  target=item.get('target',{})
  if (item['support']>=min_support and item.get('single_child_specialized_delta',0)>0
      and item.get('net_corrected',0)>=min_net_corrected
      and target.get('net_corrected',0)>=0):
   candidates.append(item)
 candidates.sort(key=lambda x:(-x['net_corrected'],-x['support'],x['cluster_id'],x['criterion_name']))
 selected=[];current=_v2_metrics(parent_outputs,child_outputs,selected,rows)
 trace=[]
 for item in candidates:
  proposed=selected+[item['criterion_name']]
  metric=_v2_metrics(parent_outputs,child_outputs,proposed,rows)
  compatible=metric['specialized_accuracy']>current['specialized_accuracy']
  trace.append({'criterion_name':item['criterion_name'],'cluster_id':item['cluster_id'],
                'eligible_individually':True,'compatible_with_locked_set':compatible,
                'single_child_specialized_delta':item['single_child_specialized_delta'],
                'net_corrected':item['net_corrected'],'candidate_metrics':metric})
  if compatible:selected=proposed;current=metric
 return {'schema_version':'1.0.0','min_support':min_support,
         'min_net_corrected':min_net_corrected,'locked_criterion_names':selected,
         'locked_metrics':current,'selection_trace':trace,
         'eligible_children':[x['criterion_name'] for x in candidates]}

def _v2_packet(candidate,combined,rows,parent_name,child_name,signatures=None):
 """Build a deterministic <=8 sample packet without changing cluster IDs."""
 by_id={str(row['sample_id']):i for i,row in enumerate(rows)}
 child=next(x for x in candidate.children if x.criterion_name==child_name)
 cluster=next(x for x in candidate.cluster_proposal.clusters if x.cluster_id==child.cluster_id)
 parent=[row[parent_name] for row in combined.node_outputs]
 outputs=[row[child_name] for row in combined.node_outputs]
 target=[by_id[sid] for sid in cluster.sample_ids]
 target_wrong=[i for i in target if not _decisive(outputs[i])
               or outputs[i].vote.value!=str(rows[i]['answer'])]
 target_correct=[i for i in target if _decisive(outputs[i])
                and outputs[i].vote.value==str(rows[i]['answer'])]
 non_target_harmed=[i for i,row in enumerate(rows)
                    if str(row['sample_id']) not in set(cluster.sample_ids)
                    and _decisive(outputs[i])
                    and outputs[i].vote.value!=str(row['answer'])
                    and _decisive(parent[i]) and parent[i].vote.value==str(row['answer'])]
 groups=(('target_wrong_or_none',target_wrong,3),('target_correct',target_correct,2),
         ('non_target_harmed',non_target_harmed,3))
 selected=[]
 for role,indices,limit in groups:
  for index in indices[:limit]:
   selected.append({'role':role,'sample_id':str(rows[index]['sample_id']),
                    'gold':str(rows[index]['answer']),'parent_vote':parent[index].vote.value,
                    'child_vote':outputs[index].vote.value,
                    'worker_thought':getattr(outputs[index],'thought',None),
                    'question':rows[index]['question'],'A':rows[index]['A'],'B':rows[index]['B'],
                    'error_signature':None if not signatures or str(rows[index]['sample_id']) not in signatures
                    else signatures[str(rows[index]['sample_id'])].to_dict()})
 representative=[item['sample_id'] for item in selected
                 if item['role']!='non_target_harmed'][:3]
 if not representative:representative=list(cluster.sample_ids[:3])
 return {'schema_version':'1.0.0','cluster_id':cluster.cluster_id,
         'previous_child':{'criterion_name':child.criterion_name,'description':child.description,
                           'rationale':child.rationale},'samples':selected,
         'representative_sample_ids':representative}

def _v2_child_prediction(source_combined,source_after,source_candidate,locked_names):
 ids=[source_candidate.node_id_by_cluster[child.cluster_id] for child in source_candidate.children
      if child.criterion_name in locked_names]
 rubric=StructuredRubric(nodes={node_id:source_after.get_node(node_id) for node_id in ids},
                        edges=(),root_ids=tuple(ids))
 return project_pairwise_prediction(source_combined,rubric),rubric

def _v2_retry_feedback(*,lock,diagnostics,history,attempt):
 return {'schema_version':'3.0.0','mode':'fixed_cluster_locked_sample_v3','attempt':attempt,
  'locked_children':lock['locked_criterion_names'],'locked_metrics':lock['locked_metrics'],
  'selection_trace':lock['selection_trace'],'current_children':diagnostics['children'],
  'sibling_pairs':diagnostics['sibling_pairs'],
  'prior_failure_attributions':[x.get('failure_attribution') for x in history.get('attempts',[])
                                if x.get('failure_attribution')],
  'instructions':{'locked_children':'lock_exact','other_children':'rewrite_or_replace_in_fixed_cluster',
                  'clustering':'forbidden','heldout':'forbidden'}}

def _v2_source(target):
 root=target/'frozen_source'
 return (StructuredRubric.load_json(root/'before_rubric.json'),
         PairwisePredictionOutput.load_json(root/'before_pairwise.json'),
         SpecializeCandidate.from_dict(load_json(root/'candidate.json')),
         StructuredRubric.load_json(root/'candidate_rubric.json'),
         PairwisePredictionOutput.load_json(root/'combined_pairwise.json'),
         {sid:ErrorSignatureOutput.from_dict(value).signature
          for sid,value in load_json(root/'error_signatures.json')['outputs'].items()})

def locked_retry_v2_freeze(config,output):
 protocol=LOCKED_RETRY_V2_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output)
 target=experiment_root(output,protocol);manifest_path=target/'frozen_manifest.json'
 if manifest_path.exists():
  manifest=load_json(manifest_path)
  if 'split_failure_attribution' not in manifest.get('manager_request_specs',{}):
   _,_,specs,_=_managers(config,protocol)
   manifest['manager_request_specs']['split_failure_attribution']=specs['split_failure_attribution']
   manifest['freeze_schema_upgrade']='added_split_failure_attribution_request_spec_v1'
   _write(manifest_path,manifest)
   print('split-retry-v2-freeze upgraded frozen attribution request spec')
  else:print('split-retry-v2-freeze already passed')
  return
 _,_,rows,_,_=base._phase6_inputs(config,output);source=output/protocol.source_experiment_dir
 records=_visual_source_records(source);latest=records[-1];attempt=Path(latest['attempt_dir'])
 before=StructuredRubric.load_json(source/'epochs'/f"epoch_{latest['epoch']-1:02d}"/'rubric_committed.json')
 pred=PairwisePredictionOutput.load_json(source/'epochs'/f"epoch_{latest['epoch']-1:02d}"/'discovery_pairwise.json')
 candidate=SpecializeCandidate.from_dict(load_json(attempt/'candidate.json'))
 after=StructuredRubric.load_json(attempt/'candidate_rubric.json')
 combined=PairwisePredictionOutput.load_json(attempt/'combined_pairwise.json')
 signatures=load_json(attempt/'error_signatures.json')
 if candidate.edit_candidate.patch.base_rubric_sha256!=before.rubric_sha256:
  raise RuntimeError('Visual source candidate base Rubric drift')
 parent=before.get_node(VISUAL_RETRY_ROOT);evaluation=SpecializeEvaluation.from_dict(load_json(attempt/'evaluation.json'))
 diagnostics=build_child_retry_diagnostics(candidate=candidate,combined=combined,rows=rows,
  parent_node_id=VISUAL_RETRY_ROOT,parent_name=parent.criterion.name,evaluation=evaluation)
 lock=select_compatible_locked_children(diagnostics,combined,rows,parent.criterion.name)
 if 'peripheral_detail_verification_accuracy' not in lock['locked_criterion_names']:
  raise RuntimeError('v2 audit did not lock the known positive Visual child')
 managers,profiles,specs,ids=_managers(config,protocol)
 source_spec=next(iter(signatures['outputs'].values()))['request_spec']
 scientific=('model','prompt_sha256','decoding_config','prompt_version','parser_version')
 if any(specs['error_signature'][key]!=source_spec[key] for key in scientific):
  raise RuntimeError('frozen source ErrorSignature scientific identity drift')
 frozen=target/'frozen_source'
 for name,value in {'before_rubric.json':before.to_dict(),'before_pairwise.json':pred.to_dict(),
                    'candidate.json':candidate.to_dict(),'candidate_rubric.json':after.to_dict(),
                    'combined_pairwise.json':combined.to_dict(),'error_signatures.json':signatures,
                    'child_diagnostics.json':diagnostics,'lock_selection.json':lock}.items():_write(frozen/name,value)
 manifest={'schema_version':'3.0.0','experiment':protocol.experiment_dir,
  'variant':'visual_grounding_locked_sample_v3','root_id':VISUAL_RETRY_ROOT,
  'source_experiment_dir':protocol.source_experiment_dir,'source_attempt_dir':str(attempt),
  'source_candidate_sha256':canonical_sha256(candidate.to_dict()),
  'source_combined_sha256':canonical_sha256(combined.to_dict()),
  'source_signature_sha256':canonical_sha256(signatures),'base_rubric_sha256':before.rubric_sha256,
  'pairwise_request_spec':pred.request_spec.to_dict(),'pairwise_endpoint':protocol.pairwise_endpoint,
  'max_repair_attempts':protocol.max_attempts,'manager_profiles':profiles,
  'manager_request_specs':specs,'manager_endpoint_identities':ids,
  'lock_selection_sha256':canonical_sha256(lock),'heldout_access':'forbidden_discovery_pilot'}
 _write(manifest_path,manifest);_snapshot(_epoch(target,0),before,pred,rows,manifest)
 memory,memory_hash=freeze_rubric_memory(_epoch(target,0),before)
 _write(target/'v2_state.json',{'schema_version':'1.0.0','completed':False,'attempt_count':0,
  'decision':None,'locked_criterion_names':lock['locked_criterion_names'],
  'lock_metrics':lock['locked_metrics'],'attempts':[],'rubric_memory_sha256':memory_hash})
 _write(target/'stage_status.json',{'freeze':{'status':'passed'}})
 print(json.dumps({'target':str(target),'locked_children':lock['locked_criterion_names'],
  'locked_specialized_accuracy':lock['locked_metrics']['specialized_accuracy']},indent=2))

def locked_retry_v2_audit(config,output):
 protocol=LOCKED_RETRY_V2_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output)
 target=experiment_root(output,protocol)
 if not (target/'frozen_manifest.json').exists():raise RuntimeError('run split-retry-v2-freeze first')
 lock=load_json(target/'frozen_source'/'lock_selection.json');metric=lock['locked_metrics']
 if metric['parent_scope_support']!=89 or abs(metric['parent_accuracy']-62/89)>1e-12:
  raise RuntimeError('frozen Visual parent scope does not match expected 62/89')
 if abs(metric['specialized_accuracy']-68/89)>1e-12:
  raise RuntimeError('locked-only Visual sanity check does not match 68/89')
 audit={'schema_version':'1.0.0','api_requests_sent':0,'locked_criterion_names':lock['locked_criterion_names'],
        'locked_metrics':metric,'status':'passed'}
 _write(target/'offline_audit.json',audit);status=load_json(target/'stage_status.json');status['audit']={'status':'passed'};_write(target/'stage_status.json',status);print(json.dumps(audit,indent=2))

def _v2_current(target,state,source_candidate,source_after,source_combined,source_diagnostics):
 if not state['attempts']:return source_candidate,source_after,source_combined,source_diagnostics
 latest=state['attempts'][-1];d=Path(latest['attempt_dir'])
 return (SpecializeCandidate.from_dict(load_json(d/'candidate.json')),
         StructuredRubric.load_json(d/'candidate_rubric.json'),
         PairwisePredictionOutput.load_json(d/'combined_pairwise.json'),
         load_json(d/'child_retry_diagnostics.json'))

def locked_retry_v2_run(config,output):
 protocol=LOCKED_RETRY_V2_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output)
 target=experiment_root(output,protocol)
 if not (target/'offline_audit.json').exists():raise RuntimeError('run split-retry-v2-freeze and split-retry-v2-audit first')
 state=load_json(target/'v2_state.json')
 if state['completed']:print('split-retry-v2-run already completed');return
 manifest=load_json(target/'frozen_manifest.json');before,pred,source_candidate,source_after,source_combined,signatures=_v2_source(target)
 _,_,rows,_,_=base._phase6_inputs(config,output);feedback=RubricFeedback.from_dict(load_json(_epoch(target,0)/'feedback.json'))
 parent=before.get_node(VISUAL_RETRY_ROOT);parent_name=parent.criterion.name
 managers,_,specs,_=_managers(config,protocol)
 for stage in ('error_signature','child_generation','split_failure_attribution'):
  if specs[stage]!=manifest['manager_request_specs'][stage]:raise RuntimeError(f'v2 Manager request identity drift: {stage}')
 pool=base._single_endpoint_execution_pool(config,protocol.pairwise_endpoint)
 locked_names=list(state['locked_criterion_names']);locked_prediction,_=_v2_child_prediction(source_combined,source_after,source_candidate,locked_names)
 for attempt_no in range(state['attempt_count']+1,(protocol.max_attempts or 2)+1):
  current,current_after,current_combined,diagnostics=_v2_current(target,state,source_candidate,source_after,source_combined,load_json(target/'frozen_source'/'child_diagnostics.json'))
  d=target/'attempts'/f'attempt_{attempt_no:02d}';d.mkdir(parents=True,exist_ok=True)
  retry_feedback=_v2_retry_feedback(lock={'locked_criterion_names':locked_names,'locked_metrics':state['lock_metrics'],'selection_trace':load_json(target/'frozen_source'/'lock_selection.json')['selection_trace']},diagnostics=diagnostics,history=state,attempt=attempt_no)
  _write(d/'retry_feedback.json',retry_feedback);_write(d/'frozen_cluster_proposal.json',current.cluster_proposal.to_dict())
  rows_by_id={str(row['sample_id']):row for row in rows};children=[];generated=[];packets=[]
  for cluster in current.cluster_proposal.clusters:
   old=next(x for x in current.children if x.cluster_id==cluster.cluster_id)
   if old.criterion_name in locked_names:
    children.append(next(x for x in source_candidate.children if x.criterion_name==old.criterion_name));continue
   diag=next(x for x in diagnostics['children'] if x['criterion_name']==old.criterion_name)
   action='replace' if diag.get('net_corrected',0)<0 or diag.get('target',{}).get('net_corrected',0)<0 else 'rewrite'
   packet=_v2_packet(current,current_combined,rows,parent_name,old.criterion_name,signatures);packet['requested_action']=action;packets.append(packet)
   rep=[rows_by_id[sid] for sid in packet['representative_sample_ids']]
   extra=[rows_by_id[x['sample_id']] for x in packet['samples'] if x['sample_id'] not in packet['representative_sample_ids']]
   child_path=d/'children'/f'{cluster.cluster_id}.json'
   child=ChildCriterionProposal.from_dict(load_json(child_path)) if child_path.exists() else managers['child_generation'].generate_child(
    parent=parent,cluster=cluster,signatures=[signatures[x] for x in cluster.sample_ids],representative_rows=rep,
    supplemental_rows=extra,siblings=children,prior_failures=state['attempts'],
    rubric_memory=load_json(_epoch(target,0)/'rubric_memory.json'),retry_feedback=retry_feedback,
    repair_context=packet)
   if not child_path.exists():_write(child_path,child.to_dict())
   children.append(child);generated.append(child);print(f'split-retry-v2 child {len(generated)} root={VISUAL_RETRY_ROOT} cluster={cluster.cluster_id}',flush=True)
  _write(d/'sample_packets.json',packets)
  try:candidate=build_specialize_candidate(EvolutionContext(before,feedback),VISUAL_RETRY_ROOT,current.cluster_proposal,children,rows,signatures)
  except ValueError as exc:raise ProposalInvalid('fixed_cluster_child_generation',str(exc)) from exc
  _write(d/'candidate.json',candidate.to_dict());after=apply_rubric_patch(before,candidate.edit_candidate.patch);after.save_json(d/'candidate_rubric.json')
  predictions=[locked_prediction]
  if generated:
   ids=tuple(candidate.node_id_by_cluster[x.cluster_id] for x in generated);generated_rubric=StructuredRubric(nodes={i:after.get_node(i) for i in ids},edges=(),root_ids=ids)
   fresh,artifact,valid=base._generate_pairwise(config,d,generated_rubric,rows,'repaired_children',execution_backend_pool=pool,request_backend_id=pred.request_spec.backend_id,request_level_progress=True)
   predictions.append(fresh);_write(d/'pairwise_reuse.json',{'locked_prediction_reused':True,'locked_criterion_names':locked_names,'generated_criterion_names':[x.criterion_name for x in generated],'generated_artifact':str(artifact),'generated_valid_rate':valid})
  combined=merge_predictions(pred,predictions,after);combined.save_json(d/'combined_pairwise.json')
  child_rubric=StructuredRubric(nodes={node_id:after.get_node(node_id) for node_id in candidate.node_id_by_cluster.values()},edges=(),root_ids=tuple(candidate.node_id_by_cluster.values()))
  child_prediction=project_pairwise_prediction(combined,child_rubric)
  evaluation,_,execution=evaluate_specialize_candidate(before_rubric=before,after_rubric=after,combined_prediction=combined,child_prediction=child_prediction,dataset=rows,parent_node_id=VISUAL_RETRY_ROOT,cluster_proposal=candidate.cluster_proposal,candidate=candidate,policy=runtime_acceptance_policy(config))
  _write(d/'evaluation.json',evaluation.to_dict());records=base._split_parent_scope_predictions(context=EvolutionContext(before,feedback),candidate=candidate,combined_prediction=combined,after_execution=execution,rows=rows);_write(d/'specialized_predictions.json',records)
  new_diag=build_child_retry_diagnostics(candidate=candidate,combined=combined,rows=rows,parent_node_id=VISUAL_RETRY_ROOT,parent_name=parent_name,evaluation=evaluation);_write(d/'child_retry_diagnostics.json',new_diag)
  decision=ACCEPTED if evaluation.specialized_accuracy>=evaluation.parent_accuracy else COMPETITION_REJECTED
  attribution=None
  if decision==COMPETITION_REJECTED:
   attribution=_required_failure_attribution(managers['semantic_cluster'],d,parent=parent,signatures=tuple(signatures.values()),cluster_proposal=candidate.cluster_proposal.to_dict(),children=candidate.children,local_metrics=evaluation.to_dict(),changed_predictions=records,retry_feedback=retry_feedback)
  record={'attempt':attempt_no,'attempt_dir':str(d),'decision':decision,'parent_accuracy':evaluation.parent_accuracy,'specialized_accuracy':evaluation.specialized_accuracy,'accuracy_delta':evaluation.accuracy_delta,'failure_attribution':None if attribution is None else attribution['attribution'],'generated_children':[x.criterion_name for x in generated],'locked_children':locked_names}
  state['attempt_count']=attempt_no;state['attempts'].append(record);_write(target/'v2_state.json',state)
  if decision==ACCEPTED:
   _snapshot(target/'final',after,combined,rows,manifest);state.update({'completed':True,'decision':ACCEPTED});_write(target/'v2_state.json',state);break
 if not state['completed']:
  lock_metric=state['lock_metrics']
  if locked_names and lock_metric['specialized_accuracy']>=lock_metric['parent_accuracy']:
   source_nodes={source_candidate.node_id_by_cluster[x.cluster_id]:source_after.get_node(source_candidate.node_id_by_cluster[x.cluster_id]) for x in source_candidate.children if x.criterion_name in locked_names}
   nodes={**before.nodes,**source_nodes};edges=tuple(list(before.edges)+[edge for edge in source_after.edges if edge.parent_id==VISUAL_RETRY_ROOT and edge.child_id in source_nodes])
   partial=StructuredRubric(nodes=nodes,edges=edges,root_ids=before.root_ids);partial_combined=merge_predictions(pred,[locked_prediction],partial)
   _write(target/'partial_locked_evaluation.json',lock_metric);_snapshot(target/'final',partial,partial_combined,rows,manifest)
   state.update({'completed':True,'decision':PARTIAL_ACCEPT_LOCKED});_write(target/'v2_state.json',state)
  else:
   state.update({'completed':True,'decision':'reject_parent_only'});_write(target/'v2_state.json',state)
 status=load_json(target/'stage_status.json');status['run']={'status':'passed','details':{'decision':state['decision'],'attempt_count':state['attempt_count']}};_write(target/'stage_status.json',status);print(json.dumps({'decision':state['decision'],'attempt_count':state['attempt_count']},indent=2))

def locked_retry_v2_report(config,output):
 protocol=LOCKED_RETRY_V2_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output);target=experiment_root(output,protocol)
 state=load_json(target/'v2_state.json')
 if not state.get('completed'):raise RuntimeError('run split-retry-v2-run first')
 report={'schema_version':'1.0.0','experiment':protocol.experiment_dir,'root_id':VISUAL_RETRY_ROOT,
  'decision':state['decision'],'attempt_count':state['attempt_count'],'locked_children':state['locked_criterion_names'],
  'locked_metrics':state['lock_metrics'],'attempts':state['attempts'],'heldout_accessed':False,
  'claim_status':('full_accept' if state['decision']==ACCEPTED else 'partial_locked_only' if state['decision']==PARTIAL_ACCEPT_LOCKED else 'no_positive_result')}
 _write(target/'report.json',report);status=load_json(target/'stage_status.json');status['report']={'status':'passed'};_write(target/'stage_status.json',status);print(json.dumps({'decision':state['decision'],'attempt_count':state['attempt_count']},indent=2))


def _v2_heldout_target(output):
 return experiment_root(output,LOCKED_RETRY_V2_PROTOCOL)/LOCKED_RETRY_V2_HELDOUT_DIR


def _same_scientific_request(left,right):
 left=dict(left.to_dict() if hasattr(left,'to_dict') else left)
 right=dict(right.to_dict() if hasattr(right,'to_dict') else right)
 left.pop('backend_id',None);right.pop('backend_id',None)
 return left==right


def _v2_final_children(target):
 state=load_json(target/'v2_state.json')
 if state.get('decision')!=ACCEPTED:
  raise RuntimeError('v2 heldout diagnostic requires an accepted full child set')
 record=state['attempts'][-1];d=Path(record['attempt_dir'])
 candidate=SpecializeCandidate.from_dict(load_json(d/'candidate.json'))
 rubric=StructuredRubric.load_json(target/'final'/'rubric_committed.json')
 if candidate.edit_candidate.patch.base_rubric_sha256!=load_json(target/'frozen_manifest.json')['base_rubric_sha256']:
  raise RuntimeError('v2 final candidate base drift')
 return candidate,rubric,list(state['locked_criterion_names'])


def _v2_global_source(output):
 source=output/VISUAL_RETRY_SOURCE_DIR
 rubric=StructuredRubric.load_json(source/'final'/'rubric.json')
 prediction=PairwisePredictionOutput.load_json(source/'heldout500'/'combined_pairwise.json')
 return rubric,prediction


def _v2_child_variant_rubric(base_rubric,final_rubric,candidate,child_names):
 child_ids=[]
 for child in candidate.children:
  if child.criterion_name in child_names:
   child_ids.append(candidate.node_id_by_cluster[child.cluster_id])
 nodes=dict(base_rubric.nodes)
 nodes.update({node_id:final_rubric.get_node(node_id) for node_id in child_ids})
 edges=tuple(list(base_rubric.edges)+[
  edge for edge in final_rubric.edges
  if edge.parent_id==VISUAL_RETRY_ROOT and edge.child_id in child_ids])
 return StructuredRubric(nodes=nodes,edges=edges,root_ids=base_rubric.root_ids)


def _v2_visual_subtree(rubric,child_names):
 parent=rubric.get_node(VISUAL_RETRY_ROOT)
 child_ids=[edge.child_id for edge in rubric.child_edges(VISUAL_RETRY_ROOT)
            if rubric.get_node(edge.child_id).criterion.name in child_names]
 nodes={VISUAL_RETRY_ROOT:parent,**{node_id:rubric.get_node(node_id) for node_id in child_ids}}
 edges=tuple(edge for edge in rubric.edges if edge.parent_id==VISUAL_RETRY_ROOT and edge.child_id in child_ids)
 return StructuredRubric(nodes=nodes,edges=edges,root_ids=(VISUAL_RETRY_ROOT,))


def _v2_isolated_children(final_rubric,candidate,child_names):
 node_ids=[candidate.node_id_by_cluster[child.cluster_id] for child in candidate.children
           if child.criterion_name in child_names]
 return StructuredRubric(nodes={node_id:final_rubric.get_node(node_id) for node_id in node_ids},
                         edges=(),root_ids=tuple(node_ids))


def _v2_heldout_manifest(config,output):
 protocol=LOCKED_RETRY_V2_PROTOCOL;target=experiment_root(output,protocol)
 candidate,final_rubric,locked=_v2_final_children(target)
 source_rubric,source_prediction=_v2_global_source(output)
 rows=load_jsonl_dataset(base._path(config['heldout_dataset']),expected_count=500)
 expected=base._expected_pairwise_request_spec(config,rows)
 if not _same_scientific_request(source_prediction.request_spec,expected):
  raise RuntimeError('v2 heldout Worker scientific identity drift')
 v2_before=StructuredRubric.load_json(target/'frozen_source'/'before_rubric.json')
 source_parent=source_rubric.get_node(VISUAL_RETRY_ROOT).criterion
 before_parent=v2_before.get_node(VISUAL_RETRY_ROOT).criterion
 if source_parent.name!=before_parent.name or source_parent.description!=before_parent.description:
  raise RuntimeError('Visual parent is not identical in v2 and heldout source')
 child_names=[child.criterion_name for child in candidate.children]
 if set(locked)-set(child_names):raise RuntimeError('locked child missing from final v2 candidate')
 if len(child_names)!=4 or len(set(child_names))!=4:
  raise RuntimeError('v2 heldout diagnostic requires exactly four unique children')
 return {'schema_version':'1.0.0','experiment':'visual_split_retry_v2_heldout_diagnostic',
  'exploratory':True,'selection_after_heldout_forbidden':True,
  'v2_report_sha256':file_sha256(target/'report.json'),
  'v2_final_rubric_sha256':final_rubric.rubric_sha256,
  'v2_candidate_sha256':canonical_sha256(candidate.to_dict()),
  'v2_before_rubric_sha256':v2_before.rubric_sha256,
  'source_global_rubric_sha256':source_rubric.rubric_sha256,
  'source_global_prediction_path':str(output/VISUAL_RETRY_SOURCE_DIR/'heldout500'/'combined_pairwise.json'),
  'source_global_prediction_sha256':file_sha256(output/VISUAL_RETRY_SOURCE_DIR/'heldout500'/'combined_pairwise.json'),
  'heldout_dataset_sha256':file_sha256(base._path(config['heldout_dataset'])),
  'sample_ids':list(source_prediction.sample_ids),'sample_fingerprints':list(source_prediction.sample_fingerprints),
  'pairwise_request_spec':expected.to_dict(),'pairwise_endpoint':protocol.pairwise_endpoint,
  'locked_child_names':locked,'all_child_names':child_names,
  'child_descriptions_sha256':canonical_sha256({child.criterion_name:child.description for child in candidate.children}),
  'generated_request_count':len(child_names)*len(rows),
  'full_m1_context':'phase7_final_nonvisual_outputs_fixed_across_variants'}


def locked_retry_v2_heldout_freeze(config,output):
 protocol=LOCKED_RETRY_V2_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output)
 target=_v2_heldout_target(output);target.mkdir(parents=True,exist_ok=True);manifest=_v2_heldout_manifest(config,output);path=target/'frozen_manifest.json'
 if path.exists() and load_json(path)!=manifest:
  previous=load_json(path);status_path=target/'stage_status.json';status=load_json(status_path) if status_path.exists() else {}
  route_only=(set(previous)==set(manifest) and all(previous[key]==manifest[key] for key in manifest if key!='pairwise_endpoint'))
  if route_only and previous.get('pairwise_endpoint')=='vllm-8001' and not (target/'child_predictions.json').exists() and status.get('run',{}).get('status')!='passed':
   print('split-retry-v2-heldout-freeze updated execution route to vllm-8000 before online run')
  else:raise RuntimeError('v2 heldout manifest drift')
 _write(path,manifest);_write(target/'stage_status.json',{'freeze':{'status':'passed'}})
 print(json.dumps({'child_count':len(manifest['all_child_names']),'generated_request_count':manifest['generated_request_count']},indent=2))


def _v2_verify_heldout_freeze(config,output):
 target=_v2_heldout_target(output);path=target/'frozen_manifest.json'
 if not path.exists():raise RuntimeError('run split-retry-v2-heldout-freeze first')
 frozen=load_json(path)
 if frozen!=_v2_heldout_manifest(config,output):raise RuntimeError('v2 heldout inputs drifted after freeze')
 return target,frozen


def _v2_child_metrics(prediction,parent_outputs,rows,name):
 outputs=[row[name] for row in prediction.node_outputs]
 decisive=[_decisive(item) for item in outputs];indices=[i for i,x in enumerate(decisive) if x]
 correct=[i for i in indices if outputs[i].vote.value==str(rows[i]['answer'])]
 parent_correct=[i for i in indices if parent_outputs[i].vote.value==str(rows[i]['answer'])]
 corrected=[str(rows[i]['sample_id']) for i in indices if outputs[i].vote.value==str(rows[i]['answer']) and parent_outputs[i].vote.value!=str(rows[i]['answer'])]
 harmed=[str(rows[i]['sample_id']) for i in indices if outputs[i].vote.value!=str(rows[i]['answer']) and parent_outputs[i].vote.value==str(rows[i]['answer'])]
 return {'criterion_name':name,'support':len(indices),'coverage':len(indices)/len(rows),'accuracy':len(correct)/len(indices) if indices else 0.0,
  'parent_accuracy_same_support':len(parent_correct)/len(indices) if indices else 0.0,
  'accuracy_delta_same_support':(len(correct)-len(parent_correct))/len(indices) if indices else 0.0,
  'corrected_count':len(corrected),'harmed_count':len(harmed),'net_corrected':len(corrected)-len(harmed),
  'corrected_sample_ids':corrected,'harmed_sample_ids':harmed}


def _v2_conflicts(prediction,names,scope):
 result=[]
 for left_index,left in enumerate(names):
  for right in names[left_index+1:]:
   joint=conflict=0
   for use,row in zip(scope,prediction.node_outputs):
    if not use:continue
    a,b=row[left],row[right]
    if _decisive(a) and _decisive(b):
     joint+=1;conflict+=a.vote.value!=b.vote.value
   result.append({'left':left,'right':right,'joint_decisive':joint,'conflict_count':conflict,
                  'conflict_rate':conflict/joint if joint else 0.0})
 return result


def locked_retry_v2_heldout_run(config,output):
 protocol=LOCKED_RETRY_V2_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output)
 target,manifest=_v2_verify_heldout_freeze(config,output);status_path=target/'stage_status.json';status=load_json(status_path)
 if status.get('run',{}).get('status')=='passed':print('split-retry-v2-heldout-run already completed');return
 candidate,final_rubric,locked=_v2_final_children(experiment_root(output,protocol));source_rubric,source_prediction=_v2_global_source(output);rows=load_jsonl_dataset(base._path(config['heldout_dataset']),expected_count=500)
 generated_path=target/'child_predictions.json';generated=PairwisePredictionOutput.load_json(generated_path) if generated_path.exists() else None
 names=manifest['all_child_names'];node_ids=tuple(candidate.node_id_by_cluster[child.cluster_id] for child in candidate.children)
 child_rubric=StructuredRubric(nodes={node_id:final_rubric.get_node(node_id) for node_id in node_ids},edges=(),root_ids=node_ids)
 if generated is None:
  generated,_,valid=base._generate_pairwise(config,target,child_rubric,rows,'v2_children',execution_backend_pool=base._single_endpoint_execution_pool(config,protocol.pairwise_endpoint),request_backend_id=source_prediction.request_spec.backend_id,request_level_progress=True)
  generated.save_json(generated_path)
 else:valid=sum(item.parse_ok and item.answer_valid for row in generated.node_outputs for item in row.values())/(len(rows)*len(names))
 variants={'parent_only':[],'locked_only':locked,'full_v2':names};evaluations={};answers={};subtree_answers={}
 parent_name=source_rubric.get_node(VISUAL_RETRY_ROOT).criterion.name
 scope=[_decisive(row[parent_name]) for row in source_prediction.node_outputs];indices=[i for i,x in enumerate(scope) if x];local_rows=[rows[i] for i in indices]
 for label,child_names in variants.items():
  rubric=_v2_child_variant_rubric(source_rubric,final_rubric,candidate,child_names)
  child_pred=None if not child_names else project_pairwise_prediction(generated,_v2_isolated_children(final_rubric,candidate,child_names))
  combined=project_pairwise_prediction(source_prediction,source_rubric) if child_pred is None else merge_predictions(source_prediction,[child_pred],rubric)
  system_dir=target/'systems'/label
  system_dir.mkdir(parents=True,exist_ok=True)
  combined.save_json(system_dir/'combined_pairwise.json')
  execution,votes=execute_offline_m1(rubric,combined,rows);execution.save_json(system_dir/'m1_execution.json');answers[label]=votes
  subtree=_v2_visual_subtree(rubric,child_names);sub_execution,sub_votes=execute_offline_m1(subtree,project_pairwise_prediction(combined,subtree),rows);sub_execution.save_json(system_dir/'subtree_execution.json');subtree_answers[label]=sub_votes
  evaluations[label]={'m1':base._heldout_vote_metrics(votes,rows),'subtree_all500':base._heldout_vote_metrics(sub_votes,rows),'subtree_parent_scope':base._heldout_vote_metrics([sub_votes[i] for i in indices],local_rows)}
 parent_outputs=[row[parent_name] for row in source_prediction.node_outputs]
 child_metrics=[]
 for name in names:
  item=_v2_child_metrics(generated,parent_outputs,rows,name)
  one_rubric=_v2_child_variant_rubric(source_rubric,final_rubric,candidate,[name]);one=_v2_visual_subtree(one_rubric,[name]);one_combined=merge_predictions(source_prediction,[project_pairwise_prediction(generated,_v2_isolated_children(final_rubric,candidate,[name]))],one_rubric);_,one_votes=execute_offline_m1(one,project_pairwise_prediction(one_combined,one),rows)
  item['single_child_specialized_parent_scope']=base._heldout_vote_metrics([one_votes[i] for i in indices],local_rows)
  without=[x for x in names if x!=name];without_rubric=_v2_child_variant_rubric(source_rubric,final_rubric,candidate,without);without_subtree=_v2_visual_subtree(without_rubric,without);without_combined=merge_predictions(source_prediction,[project_pairwise_prediction(generated,_v2_isolated_children(final_rubric,candidate,without))],without_rubric);_,without_votes=execute_offline_m1(without_subtree,project_pairwise_prediction(without_combined,without_subtree),rows)
  item['leave_one_out_specialized_delta']=evaluations['full_v2']['subtree_parent_scope']['accuracy']-base._heldout_vote_metrics([without_votes[i] for i in indices],local_rows)['accuracy'];child_metrics.append(item)
 pairs={}
 for label in ('locked_only','full_v2'):
  pairs[f'{label}_vs_parent']={'subtree_parent_scope':base._paired_heldout_comparison([subtree_answers['parent_only'][i] for i in indices],[subtree_answers[label][i] for i in indices],local_rows),'m1_all500':base._paired_heldout_comparison(answers['parent_only'],answers[label],rows)}
 pairs['full_v2_vs_locked_only']={'subtree_parent_scope':base._paired_heldout_comparison([subtree_answers['locked_only'][i] for i in indices],[subtree_answers['full_v2'][i] for i in indices],local_rows),'m1_all500':base._paired_heldout_comparison(answers['locked_only'],answers['full_v2'],rows)}
 value={'schema_version':'1.0.0','exploratory':True,'metrics':evaluations,'paired':pairs,'children':child_metrics,'sibling_conflicts':_v2_conflicts(generated,names,scope),'parent_scope_support':len(indices),'generated_valid_rate':valid,'selection_after_heldout_forbidden':True}
 _write(target/'evaluations.json',value);status['run']={'status':'passed','details':{'generated_request_count':len(names)*len(rows),'generated_valid_rate':valid}};_write(status_path,status);print(json.dumps({'parent_scope_support':len(indices),'generated_valid_rate':valid},indent=2))


def locked_retry_v2_heldout_report(config,output):
 protocol=LOCKED_RETRY_V2_PROTOCOL;validate_policy(config,protocol);validate_phase5_lineage(output);target,manifest=_v2_verify_heldout_freeze(config,output)
 path=target/'evaluations.json'
 if not path.exists():raise RuntimeError('run split-retry-v2-heldout-run first')
 report={'schema_version':'1.0.0','experiment':'visual_split_retry_v2_heldout_diagnostic','frozen_manifest_sha256':file_sha256(target/'frozen_manifest.json'),'generated_request_count':manifest['generated_request_count'],**load_json(path)}
 _write(target/'report.json',report);lines=['# Visual Grounding Split-retry v2 Heldout Diagnostic','', 'Exploratory paired diagnostic; heldout was not used for selection.','', '| System | Subtree ACC (parent scope) | M1 ACC | M1 Coverage |','|---|---:|---:|---:|']
 for name in ('parent_only','locked_only','full_v2'):
  item=report['metrics'][name];lines.append(f"| {name} | {item['subtree_parent_scope']['accuracy']:.4f} | {item['m1']['accuracy']:.4f} | {item['m1']['coverage']:.4f} |")
 lines.extend(['','| Comparison | Subtree net corrected | M1 net corrected |','|---|---:|---:|'])
 for name,item in report['paired'].items():lines.append(f"| {name} | {item['subtree_parent_scope']['net_corrected']} | {item['m1_all500']['net_corrected']} |")
 (target/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');status=load_json(target/'stage_status.json');status['report']={'status':'passed'};_write(target/'stage_status.json',status);print(json.dumps(report['paired'],indent=2))

def repair_audit(config,output):
 """Replay v1 candidates locally without mutating v1 or contacting any API."""
 validate_policy(config);validate_phase5_lineage(output);runtime_acceptance_policy(config);source=output/'phase6_split_only_evolution_v1';target=experiment_root(output)
 if not source.exists():raise RuntimeError(f'v1 source run not found: {source}')
 _,_,rows,_,_=base._phase6_inputs(config,output);replays=[]
 for candidate_path in sorted(source.glob('epochs/epoch_*/roots/*/attempt_*/candidate.json')):
  d=candidate_path.parent;combined_path=d/'combined_pairwise.json'
  if not combined_path.exists():continue
  epoch_no=int(d.parents[2].name.split('_')[1]);before=StructuredRubric.load_json(source/'epochs'/f'epoch_{epoch_no-1:02d}'/'rubric_committed.json');candidate=SpecializeCandidate.from_dict(load_json(candidate_path));after=StructuredRubric.load_json(d/'candidate_rubric.json');combined=PairwisePredictionOutput.load_json(combined_path);ids=tuple(candidate.node_id_by_cluster.values());child_rubric=StructuredRubric(nodes={i:after.get_node(i) for i in ids},edges=(),root_ids=ids);child=project_pairwise_prediction(combined,child_rubric);evaluation,_,_=evaluate_specialize_candidate(before_rubric=before,after_rubric=after,combined_prediction=combined,child_prediction=child,dataset=rows,parent_node_id=candidate.parent_node_id,cluster_proposal=candidate.cluster_proposal,candidate=candidate,policy=runtime_acceptance_policy(config));decision=ACCEPTED if evaluation.specialized_accuracy>=evaluation.parent_accuracy else COMPETITION_REJECTED;replays.append({'epoch':epoch_no,'root_id':candidate.parent_node_id,'attempt_dir':str(d),'decision':decision,'parent_accuracy':evaluation.parent_accuracy,'specialized_accuracy':evaluation.specialized_accuracy,'accuracy_delta':evaluation.accuracy_delta,'corrected_sample_ids':list(evaluation.subtree_diagnostic.corrected_sample_ids),'harmed_sample_ids':list(evaluation.subtree_diagnostic.harmed_sample_ids)})
 audit={'schema_version':'1.0.0','source_run':'phase6_split_only_evolution_v1','source_status':'invalid_due_to_orchestrator_bug','reason':'runtime config was passed to a versioned artifact deserializer; program errors were recorded as Split rejections','v1_mutated':False,'api_requests_sent':0,'replayed_competitions':len(replays),'replays':replays};_write(target/'repair_audit_v1.json',audit);print(json.dumps({'replayed_competitions':len(replays),'output':str(target/'repair_audit_v1.json')},indent=2))
def run_stage(config,output,stage):
 actions={'split-evolution-repair-audit':repair_audit,'split-evolution-freeze':freeze,'split-evolution-smoke':smoke,'split-evolution-run':run,'split-evolution-report':report,'split-evolution-heldout':heldout,'split-evolution-final-report':final_report,
 'split-memory-freeze':lambda c,o:freeze(c,o,MEMORY_PROTOCOL),'split-memory-smoke':lambda c,o:smoke(c,o,MEMORY_PROTOCOL),'split-memory-run':lambda c,o:run(c,o,MEMORY_PROTOCOL),'split-memory-report':lambda c,o:report(c,o,MEMORY_PROTOCOL),'split-memory-heldout':lambda c,o:heldout(c,o,MEMORY_PROTOCOL),'split-memory-final-report':lambda c,o:final_report(c,o,MEMORY_PROTOCOL),
 'split-retry-visual-freeze':visual_retry_freeze,'split-retry-visual-audit':visual_retry_audit,
 'split-retry-visual-run':visual_retry_run,'split-retry-visual-report':visual_retry_report,
 'split-retry-v2-freeze':locked_retry_v2_freeze,'split-retry-v2-audit':locked_retry_v2_audit,
 'split-retry-v2-run':locked_retry_v2_run,'split-retry-v2-report':locked_retry_v2_report,
 'split-retry-v2-heldout-freeze':locked_retry_v2_heldout_freeze,
 'split-retry-v2-heldout-run':locked_retry_v2_heldout_run,
 'split-retry-v2-heldout-report':locked_retry_v2_heldout_report}
 actions[stage](config,output)

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
 SpecializeManagerFailure, StructuredCriterionSnapshot, StructuredRubric,
 apply_rubric_patch, assemble_specialized_pairwise_prediction,
 build_specialize_candidate, detect_specialize_trigger,
 evaluate_specialize_candidate, execute_offline_m1, extract_rubric_feedback,
 project_pairwise_prediction)
from critiq.structured.aggregation import aggregate_flat_votes
from .experiment_utils import atomic_write_json, canonical_sha256, load_json, load_jsonl_dataset
from .rubric_factory import file_sha256
from . import run_rubric_evolution as base

EXPERIMENT_DIR='phase6_split_only_evolution_v2'
MEMORY_EXPERIMENT_DIR='phase6_split_only_evolution_global_memory_v1'
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

 @property
 def is_memory_treatment(self):return self.rubric_memory_mode=='global_rubric_v1'


CONTROL_PROTOCOL=EvolutionProtocol(EXPERIMENT_DIR,'split-evolution')
MEMORY_PROTOCOL=EvolutionProtocol(
 MEMORY_EXPERIMENT_DIR,'split-memory','global_rubric_v1',CONTROL_EXPERIMENT_DIR,True)


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
 if protocol.is_memory_treatment and config.get('split_manager_memory_ablation')!=MEMORY_CONFIG_V1:
  raise ValueError('split_manager_memory_ablation must equal frozen global-rubric v1 protocol')
 t=config['evolution_policy']['trigger_thresholds']
 if t['tau_split']!=.70 or t['tau_cov_high']!=.80:raise ValueError('trigger must be ACC < .70 and Coverage > .80')
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
 return PairwisePredictionOutput(base_prediction.sample_ids,base_prediction.sample_fingerprints,tuple(StructuredCriterionSnapshot(c.name,c.description) for c in ordered),rows,answers,base_prediction.request_spec)

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
  if p['model']!=MANAGER_MODEL:raise RuntimeError(f'{stage} must use {MANAGER_MODEL}')
  managers[stage]=m;profiles[stage]=p;specs[stage]=m.request_specs()[stage].to_dict();identities[stage]=ids
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
 if len(pool.endpoints)!=1 or pool.endpoints[0].endpoint_id!=PAIRWISE_ENDPOINT:raise RuntimeError('Pairwise must use only vllm-8001')
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

def _history_projection(history,root):
 items=[]
 for record in history['attempts']:
  if record['root_id']!=root or record['decision']==ACCEPTED:continue
  payload=record.get('history_payload') or {};failure=payload.get('structured_failure') or {};details=failure.get('details') or {}
  compact_failure={'code':failure.get('code'),'stage':failure.get('stage'),'details':{k:details[k] for k in ('type','message','parse_error','attempt_count','collisions') if k in details}}
  items.append({'attempt':record['attempt'],'outcome':record['decision'],'failure':compact_failure,'cluster_summary':payload.get('cluster_summary'),'children_summary':payload.get('children_summary',[]),'local_metrics':payload.get('local_metrics'),'corrected_sample_ids':payload.get('corrected_sample_ids',[]),'harmed_sample_ids':payload.get('harmed_sample_ids',[]),'natural_language_attribution':payload.get('natural_language_attribution')})
 return items

def _prior(history,root):return _history_projection(history,root)

def _require_split_clusters(cluster):
 if len(cluster.clusters)<2:
  raise ProposalInvalid('semantic_cluster','Split requires at least two valid clusters',{
   'reason':'insufficient_cluster_count','valid_clusters':len(cluster.clusters),
   'required_minimum':2,'cluster_ids':[x.cluster_id for x in cluster.clusters],
   'unclustered_count':len(cluster.unclustered_sample_ids)})

def _signatures(target,manager,parent,trigger,node_feedback,rows_by_id,identity,spec,
                protocol:EvolutionProtocol=CONTROL_PROTOCOL):
 cache=target/'signature_cache'/root_shard(parent.node_id)/identity[:16];cache.mkdir(parents=True,exist_ok=True);v1_cache=target.parent/'phase6_split_only_evolution_v1'/'signature_cache'/root_shard(parent.node_id)/identity[:16]
 control_cache=target.parent/CONTROL_EXPERIMENT_DIR/'signature_cache'/root_shard(parent.node_id)/identity[:16]
 errors={x.sample_id:x for x in node_feedback.errors if x.outcome=='wrong'};values={};total=len(trigger.decisive_wrong_sample_ids);reuse={'same_run':0,'v1_identity_match':0,'generated':0};pending=[]
 if protocol.read_only_control_signatures:reuse['control_v2_exact_identity']=0
 def accept(sid,value,shard,source_kind):
  if value.signature is None:
   details={'sample_id':sid,'parse_error':value.parse_error,'metrics':value.metrics.to_dict()}
   if value.raw_response is None and value.metrics.api_attempts>0 and value.metrics.error_count>=value.metrics.api_attempts:raise TransportFailed('error_signature',f'transport failed for signature {sid}',details)
   raise ProposalInvalid('error_signature',f'invalid signature {sid}: {value.parse_error}',details)
  if value.request_spec.to_dict()!=spec:raise RuntimeError('signature request identity mismatch')
  if not shard.exists():
   source_payload={'source':source_kind,'source_run':None if source_kind=='generated' else 'phase6_split_only_evolution_v1'}
   if source_kind=='control_v2_exact_identity':source_payload={'source':source_kind,'source_run':CONTROL_EXPERIMENT_DIR,'source_sha256':file_sha256(control_cache/base._safe_artifact_name(sid))}
   _write(shard,value.to_dict());_write(shard.with_suffix('.source.json'),source_payload)
  values[sid]=value
 for sid in trigger.decisive_wrong_sample_ids:
  shard=cache/base._safe_artifact_name(sid);v1_shard=v1_cache/base._safe_artifact_name(sid);control_shard=control_cache/base._safe_artifact_name(sid)
  if shard.exists():accept(sid,ErrorSignatureOutput.from_dict(load_json(shard)),shard,'same_run');reuse['same_run']+=1
  elif protocol.read_only_control_signatures:
   if not control_shard.exists():raise RuntimeError(f'Control signature missing: {parent.node_id}/{sid}')
   accept(sid,ErrorSignatureOutput.from_dict(load_json(control_shard)),shard,'control_v2_exact_identity');reuse['control_v2_exact_identity']+=1
  elif v1_shard.exists():accept(sid,ErrorSignatureOutput.from_dict(load_json(v1_shard)),shard,'v1_identity_match');reuse['v1_identity_match']+=1
  else:pending.append((sid,shard))
 completed=len(values)
 if completed:print(f'split-evolution signatures cached={completed}/{total} root={parent.node_id}',flush=True)
 if protocol.read_only_control_signatures and pending:
  raise RuntimeError('Treatment may not generate ErrorSignatures online')
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
 thresholds=config['evolution_policy']['trigger_thresholds'];prior=_prior(history,root);_write(d/'history_projection.json',{'schema_version':'1.0.0','root_id':root,'attempt':attempt_no,'history':prior,'projection_sha256':canonical_sha256(prior)})
 cluster_path=d/'cluster_proposal.json'
 if protocol.is_memory_treatment:
  if rubric_memory is None or rubric_memory_sha256!=canonical_sha256(rubric_memory):raise RuntimeError('missing or invalid epoch rubric memory')
  _write(d/'rubric_memory_ref.json',{'rubric_memory_mode':protocol.rubric_memory_mode,'rubric_memory_sha256':rubric_memory_sha256,'control_signature_source':CONTROL_EXPERIMENT_DIR,'semantic_cluster_request_spec':specs['semantic_cluster'],'child_generation_request_spec':specs['child_generation']})
 cluster=ClusterProposal.from_dict(load_json(cluster_path)) if cluster_path.exists() else managers['semantic_cluster'].cluster(tuple(signatures.values()),criterion_name=pname,min_cluster_size=thresholds['N_min_cluster'],max_clusters=t.remaining_capacity,prior_failures=prior,rubric_memory=rubric_memory)
 if not cluster_path.exists():_write(cluster_path,cluster.to_dict())
 _require_split_clusters(cluster)
 children=[]
 for i,c in enumerate(cluster.clusters,1):
  reps=c.sample_ids[:3];child_path=d/'children'/f'{c.cluster_id}.json'
  child=ChildCriterionProposal.from_dict(load_json(child_path)) if child_path.exists() else managers['child_generation'].generate_child(parent=parent,cluster=c,signatures=[signatures[x] for x in c.sample_ids],representative_rows=[rows_by_id[x] for x in reps],siblings=children,prior_failures=prior,rubric_memory=rubric_memory)
  children.append(child)
  if not child_path.exists():_write(child_path,child.to_dict())
  print(f'split-evolution children {i}/{len(cluster.clusters)} root={root}',flush=True)
 candidate=build_specialize_candidate(ctx,root,cluster,children,rows,signatures);_write(d/'candidate.json',candidate.to_dict());after=apply_rubric_patch(rubric,candidate.edit_candidate.patch);after.save_json(d/'candidate_rubric.json')
 return {'root_id':root,'decision':'prepared','attempt_dir':d,'candidate':candidate,'after_rubric':after,'signatures':signatures}

def _failure(result,code,stage='competition',attribution=None,details=None):
 c=result.get('candidate');e=result.get('evaluation')
 cluster_summary=None if c is None else [{'cluster_id':x.cluster_id,'label':x.label,'sample_ids':list(x.sample_ids)} for x in c.cluster_proposal.clusters]
 children_summary=[] if c is None else [{'cluster_id':x.cluster_id,'criterion_name':x.criterion_name,'description':x.description[:800]} for x in c.children]
 metrics=None if e is None else {'parent_accuracy':e.parent_accuracy,'specialized_accuracy':e.specialized_accuracy,'accuracy_delta':e.accuracy_delta}
 compact_attribution=None if attribution is None else attribution.get('attribution',attribution)
 return {'structured_failure':{'code':code,'stage':stage,'details':dict(details or {})},'cluster_summary':cluster_summary,'children_summary':children_summary,'local_metrics':metrics,'corrected_sample_ids':[] if e is None else list(e.subtree_diagnostic.corrected_sample_ids),'harmed_sample_ids':[] if e is None else list(e.subtree_diagnostic.harmed_sample_ids),'natural_language_attribution':compact_attribution}

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

def _evaluate(config,epoch_dir,rows,rubric,pred,result,pool,manager,feedback):
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
 if decision==COMPETITION_REJECTED:
  attribution=_required_failure_attribution(manager,result['attempt_dir'],parent=rubric.get_node(result['root_id']),signatures=tuple(result['signatures'].values()),cluster_proposal=c.cluster_proposal.to_dict(),children=c.children,local_metrics=evaluation.to_dict(),changed_predictions=records)
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
 _,_,rows,_,_=base._phase6_inputs(config,output);managers,profiles,specs,_=_managers(config,protocol);pool=base._single_endpoint_execution_pool(config,PAIRWISE_ENDPOINT);max_epochs=POLICY_V1['max_epochs']
 if protocol.is_memory_treatment:
  specs['error_signature']=manifest['manager_request_specs']['error_signature']
  for stage in ('semantic_cluster','child_generation'):
   if specs[stage]!=manifest['manager_request_specs'][stage]:raise RuntimeError(f'Treatment Manager request identity drift after freeze: {stage}')
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
   try:results[root]=_evaluate(config,epoch_dir,rows,rubric,pred,results[root],pool,managers['semantic_cluster'],feedback)
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
  if should_stop_after_epoch(epoch_no,history) or epoch_no==max_epochs:
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
  result=_prepare(config,target,smoke_dir,root,1,rubric,pred,feedback,rows,history,managers,specs,protocol,rubric_memory,rubric_memory_sha256);result=_evaluate(config,smoke_dir,rows,rubric,pred,result,pool,managers['semantic_cluster'],feedback)
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
 'split-memory-freeze':lambda c,o:freeze(c,o,MEMORY_PROTOCOL),'split-memory-smoke':lambda c,o:smoke(c,o,MEMORY_PROTOCOL),'split-memory-run':lambda c,o:run(c,o,MEMORY_PROTOCOL),'split-memory-report':lambda c,o:report(c,o,MEMORY_PROTOCOL),'split-memory-heldout':lambda c,o:heldout(c,o,MEMORY_PROTOCOL),'split-memory-final-report':lambda c,o:final_report(c,o,MEMORY_PROTOCOL)}
 actions[stage](config,output)

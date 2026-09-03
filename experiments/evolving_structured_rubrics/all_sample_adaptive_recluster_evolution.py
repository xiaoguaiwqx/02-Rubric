"""Phase22 all-sample atomic root competition with adaptive reclustering.

The execution engine is the frozen Phase21 implementation. This module binds
the Phase22 protocol identity and the two planned semantic changes while
leaving Phase21's triggers, candidate construction, budgets, synchronous
commit, and diagnostic-only Global Arbiter unchanged.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping

from . import discovery_v2_prompt_v2_evolution as phase17
from . import unified_subtree_bundle_evolution as phase21
from .evolution_protocol import EvolutionProtocol


EXPERIMENT_DIR = "phase22_all_sample_subtree_adaptive_recluster_evolution_v1"
PROTOCOL_VERSION = "all-sample-root-subtree-adaptive-recluster-evolution-v1"
CONFIG_KEY = "all_sample_adaptive_recluster_evolution_v1_experiment"
STAGES = (
    "all-sample-adaptive-evolution-freeze",
    "all-sample-adaptive-evolution-audit",
    "all-sample-adaptive-evolution-smoke",
    "all-sample-adaptive-evolution-run",
    "all-sample-adaptive-evolution-report",
    "all-sample-adaptive-evolution-heldout",
    "all-sample-adaptive-evolution-final-report",
)
SETTINGS = {
    "protocol_version": PROTOCOL_VERSION,
    "source_protocol": phase17.PROTOCOL_VERSION,
    "source_experiment": phase21.EXPERIMENT_DIR,
    "optimization_unit": "complete_root_subtree",
    "scope_source": "epoch_start_unified_subtree_A_or_B",
    "error_source": "unified_subtree_scope_mismatch_only",
    "acceptance_metric": "all_sample_unified_net_gain_gt_0",
    "split_acceptance": "all_sample_unified_net_gain_gt_0",
    "refine_acceptance": "all_sample_unified_net_gain_gt_0",
    "split_retry": "failure_aware_recluster_or_reuse_clusters",
    "recluster_failure_types": ["cluster_or_decomposition_error"],
    "technical_failure_policy": "pause_without_scientific_retry_action",
    "strong_child_locking": False,
    "partial_acceptance": False,
    "specialized_acc_role": "post_commit_read_only_diagnostic",
    "candidate_commit": "all_independently_passing_roots_synchronous",
    "global_arbiter_role": "post_commit_diagnostic_only",
    "split_trigger": dict(phase21.SETTINGS["split_trigger"]),
    "bundle_refine_trigger": dict(phase21.SETTINGS["bundle_refine_trigger"]),
    "min_epochs": phase21.SETTINGS["min_epochs"],
    "max_epochs": phase21.SETTINGS["max_epochs"],
    "internal_k": phase21.SETTINGS["internal_k"],
    "temperature": phase21.SETTINGS["temperature"],
    "max_tokens": phase21.SETTINGS["max_tokens"],
    "max_parse_retries": phase21.SETTINGS["max_parse_retries"],
    "generation_seed_policy": phase21.SETTINGS["generation_seed_policy"],
    "pairwise_prompt_mode": phase21.SETTINGS["pairwise_prompt_mode"],
    "pairwise_prompt_version": phase21.SETTINGS["pairwise_prompt_version"],
    "dev_policy": phase21.SETTINGS["dev_policy"],
    "heldout_access": phase21.SETTINGS["heldout_access"],
    "vl_rewardbench_required": True,
}


PROTOCOL = EvolutionProtocol.from_settings(
    experiment_dir=EXPERIMENT_DIR, version=PROTOCOL_VERSION,
    config_key=CONFIG_KEY, stages=STAGES, settings=SETTINGS,
)


def _target(output: Path) -> Path:
    return output / EXPERIMENT_DIR


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    return PROTOCOL.validate(config)


def split_retry_action(primary_failure_type: str) -> str:
    return phase21.split_retry_action(primary_failure_type, protocol=PROTOCOL)


def split_retry_source(
    retry: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    return phase21.split_retry_source(retry)


def _delegate(name: str, config: Mapping[str, Any], output: Path) -> None:
    _settings(config)
    getattr(phase21, name)(config, output, protocol=PROTOCOL)


def freeze(config: Mapping[str, Any], output: Path) -> None:
    _delegate("freeze", config, output)


def audit(config: Mapping[str, Any], output: Path) -> None:
    _delegate("audit", config, output)


def smoke(config: Mapping[str, Any], output: Path) -> None:
    _delegate("smoke", config, output)


def run(config: Mapping[str, Any], output: Path) -> None:
    _delegate("run", config, output)


def report(config: Mapping[str, Any], output: Path) -> None:
    _delegate("report", config, output)


def heldout(config: Mapping[str, Any], output: Path) -> None:
    _delegate("heldout", config, output)


def final_report(config: Mapping[str, Any], output: Path) -> None:
    _delegate("final_report", config, output)


def run_stage(config: Mapping[str, Any], output: Path, stage: str) -> None:
    actions: dict[str, Callable[[Mapping[str, Any], Path], None]] = {
        STAGES[0]: freeze,
        STAGES[1]: audit,
        STAGES[2]: smoke,
        STAGES[3]: run,
        STAGES[4]: report,
        STAGES[5]: heldout,
        STAGES[6]: final_report,
    }
    if stage not in actions:
        raise ValueError(f"unsupported Phase22 stage: {stage}")
    started = time.monotonic()
    actions[stage](config, output)
    print(f"{stage} elapsed_seconds={time.monotonic() - started:.1f}")

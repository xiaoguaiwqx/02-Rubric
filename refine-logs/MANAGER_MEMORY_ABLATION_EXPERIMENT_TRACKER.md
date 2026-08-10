# Manager Global-Rubric Memory Ablation Tracker

## Implementation

| Item | Status | Evidence |
|---|---|---|
| Treatment-specific clustering and child prompts | Complete | Separate prompt versions; Control hashes unchanged |
| Optional `global_rubric_v1` Manager mode | Complete | Memory required for clustering and child generation only |
| Deterministic epoch Rubric snapshot | Complete | `rubric_memory.json` plus canonical hash |
| Exact read-only v2 ErrorSignature reuse | Complete | Freeze source manifest; `generated=0`; missing source fails |
| Six `split-memory-*` stages | Complete | CLI and shared orchestrator protocol |
| Discovery/heldout paired Control comparison | Complete | Treatment final report includes primary/secondary paired results |
| Offline regression tests | Complete | 191 full-suite tests pass; compile and diff checks pass |
| Real API experiment | Complete | Five epochs and heldout-500 completed under `phase6_split_only_evolution_global_memory_v1` |

## Execution

| Stage | Status | Notes |
|---|---|---|
| `split-memory-freeze` | Complete | Reused 157 Control signatures; generated 0 |
| `split-memory-smoke` | Complete | Discovery-only smoke completed |
| `split-memory-run` | Complete | Five epochs, 17 attempts, 4 accepted roots |
| `split-memory-report` | Complete | Final 17-node Rubric frozen |
| `split-memory-heldout` | Complete | Exploratory heldout-500 completed after freeze |
| `split-memory-final-report` | Complete | Control/Treatment paired comparison generated |

## Result and protocol decision

| System | Heldout ACC | Coverage | Correct | Final children / nodes |
|---|---:|---:|---:|---:|
| Init five-root M1 | 65.00% | 97.20% | 325 | 0 / 5 |
| Control v2 | 69.40% | 99.40% | 347 | 14 / 19 |
| Global-Rubric Memory | **70.00%** | 98.60% | **350** | **12 / 17** |

Treatment produced the highest equal-weight M1 ACC with a smaller final Rubric. Lexical near-duplicate pairs decreased from 14/91 possible Control pairs (15.4%) to 7/66 Treatment pairs (10.6%). The direct Treatment-Control difference is exploratory (`corrected=30`, `harmed=27`, McNemar `p=0.791`) and is not treated as a separate research contribution.

Split v1 is frozen with its trigger, Manager generation stages, local Specialized Accuracy competition, whole-set acceptance/fallback, required failure-attribution history, and `global_rubric_v1` memory contract. Future Split, Refine, and other Manager experiments use `global_rubric_v1` by default; changing frozen Split semantics requires a new protocol version and experiment directory.

## Review checklist

- Confirm every attempt reports `generated: 0` for ErrorSignatures.
- Confirm every root in one epoch has the same `rubric_memory_sha256`.
- Confirm epoch N accepted children first appear in epoch N+1 memory.
- Confirm failure history and Rubric memory both appear in clustering and child prompts.
- Confirm ErrorSignature and failure-attribution request identities match Control behavior.
- Confirm no heldout artifact exists before `split-memory-heldout`.
- Interpret final ACC using equal-weight five-root M1 only.

# Split-only Evolution v2 Repair Plan

**Date:** 2026-08-07
**Status:** Draft for review; no implementation or new API run is authorized by this document.
**Scope:** Repair experiment validity and orchestration. Do not change Split trigger, clustering objective, child-generation objective, Pairwise Worker, or local acceptance rule.

## 1. Why a repair is required

The v1 run is not a negative Split result. It is an invalid experiment run:

| Observed v1 fact | Interpretation |
|---|---|
| 20 root attempts were recorded | Four eligible roots were repeatedly scheduled for five epochs |
| 10 child Pairwise artifacts exist | Candidate generation and Worker evaluation sometimes completed |
| 0 `evaluation.json` files exist | No Parent-versus-Specialized competition completed |
| 0 accepted children; final rubric has five nodes | The reported Init=Final comparison has no treatment |
| Attempts were marked `rejected`/`exhausted` | Program/API failures were incorrectly converted into scientific Split failures |

The deterministic blocker is the construction of `CandidateAcceptancePolicy` from a plain config dictionary through an artifact deserializer that requires `schema_version`. The exception is then swallowed by a broad `except (ValueError, RuntimeError)` path and recorded as rejection. Reporting subsequently treats retries caused by this failure as natural reject-to-retry evidence and permits final/heldout reporting with no completed competitions.

Therefore v1 must be preserved and labeled `invalid_due_to_orchestrator_bug`. Its Init=Final heldout result cannot support or refute Split generalization.

## 2. Claims retained for v2

| Claim | Minimum valid evidence |
|---|---|
| C1: Split can improve eligible initial roots under the frozen local acceptance rule | At least one persisted, replayable Parent/Specialized competition; every acceptance satisfies `specialized_accuracy >= parent_accuracy`; rejected roots retry only after a valid competition failure |
| C2: Discovery improvements can transfer to heldout-500 | A non-empty final treatment rubric, frozen before heldout access, compared with the identical Init five-root baseline using paired metrics |
| Anti-claim: apparent failure/success is caused by orchestration or reporting | Typed failures, transactional state transitions, invariant gates, and exact offline replay agree |

The scientific rule remains unchanged: children vote first, ties/all-None fall back to the parent, and the complete child set is accepted iff local Specialized Accuracy on the fixed parent scope is not lower than Parent Accuracy.

## 3. Required corrections

### P0. Correct policy construction and add a regression test

- Construct runtime policy with `CandidateAcceptancePolicy(**config[...])`.
- Reserve `from_dict` for serialized artifacts that contain `schema_version`.
- Add a test using the real experiment config shape and execute one saved candidate through `_evaluate`.

**Why:** This is the direct cause of zero completed competitions. A unit test must reproduce the exact config/artifact boundary, not merely test the dataclass in isolation.

### P0. Introduce a typed attempt state machine

Use mutually exclusive outcomes:

| Outcome | Meaning | Consumes a scientific attempt? | Enters Split failure history? | Run action |
|---|---|---:|---:|---|
| `accepted` | Complete competition and non-negative local delta | Yes | No | Lock root |
| `competition_rejected` | Complete competition and negative local delta | Yes | Yes | Retry next valid epoch |
| `proposal_invalid` | Cluster/child output is structurally invalid | Yes | Yes, as proposal failure | Retry with compact diagnostic |
| `transport_failed` | API unavailable/timeouts after retry budget | No | No | Pause; resume same stage/attempt |
| `program_error` | Schema, invariant, filesystem, or implementation error | No | No | Hard abort; do not mutate scientific history |
| `cross_root_collision` | Accepted candidates conflict during synchronous commit | Yes | Yes, structural reason | Reject all involved candidates |

Only a persisted `evaluation.json` may produce `accepted` or `competition_rejected`. Broad exception handlers must not translate unknown `ValueError`/`RuntimeError` into a rejection.

**Why:** Scientific failure, model-output failure, service failure, and code failure answer different questions. Combining them creates false evidence and wastes retries.

### P0. Make epoch execution transactional

For each epoch:

1. Freeze the epoch-start rubric/hash and eligible root list.
2. Run each root through resumable stage shards: signatures, cluster, children, Pairwise, local evaluation.
3. Persist and validate every completed candidate result.
4. Resolve name collisions without root-order dependence.
5. Commit all accepted patches once, then generate M1/feedback and advance the epoch.

If a program invariant fails, the epoch remains uncommitted. Restarting resumes from the last valid shard and must not resend completed requests.

**Why:** A partial exception must not leave history, rubric, and predictions describing different states.

### P0. Add finalization and heldout gates

Before `split-evolution-report`:

- Recompute all local decisions from stored predictions.
- Verify accepted roots have an evaluation artifact and non-negative local delta.
- Verify attempt counts, root states, rubric node changes, and epoch summaries agree.
- Distinguish `max_valid_attempts_reached`, `no_retryable_roots`, `paused_transport_failure`, and `aborted_program_error`.

Before `split-evolution-heldout`:

- Require a valid discovery report and frozen final rubric hash.
- Require at least one accepted child set. If the rubric is unchanged, report `no_treatment` and do not access heldout.
- Verify the Init M1 replay against the frozen baseline.

**Why:** v1 should have been stopped before heldout. These gates make an invalid or empty treatment impossible to present as an experiment result.

### P1. Preserve full history but inject a compact semantic projection

Keep all raw attempt artifacts on disk. Give the Manager all prior attempts through a bounded structured projection containing:

- attempt number and outcome;
- cluster labels and assigned sample IDs;
- child name and concise definition;
- local Parent/Specialized metrics plus corrected/harmed summaries;
- categorized failure cause and actionable natural-language attribution.

Do not inject raw API responses, request specifications, full prediction arrays, token metadata, or repeated prompt text. Save the exact projected history and its hash with each new request. A completed competition rejection receives 397B attribution; parse failures receive a concise structural diagnostic; transport/program failures do not become semantic Split history.

**Why:** v1 cluster input grew from roughly 7–10K to 20–24K tokens, followed by repeated 22-attempt cluster failures. Raw history retention and prompt history are separate concerns.

### P1. Add API circuit breaking and honest retry accounting

- Keep the Manager pipeline fixed at 397B and Pairwise fixed at 8001.
- After repeated endpoint-wide failures, pause the run rather than exhausting every root.
- Report successful and failed calls separately, including `manager_failure.json` calls.
- Report wall-clock time and summed request latency separately.

**Why:** Nine observed 22-attempt Manager failures alone imply 198 failed calls that the v1 cost table did not represent correctly. Service instability must not alter the evolutionary trajectory.

### P1. Repair report semantics

- `natural_reject_retry_observed=true` only when a completed `competition_rejected` attempt is followed by another valid attempt for the same root.
- Empty predicates return `not_applicable`, not vacuous success.
- Show counts of scheduled, proposal-valid, Pairwise-complete, evaluation-complete, accepted, and rejected attempts.
- Label invalid/no-treatment runs prominently and suppress scientific success tables where comparison is tautological.

**Why:** Reports must describe evidence that actually exists, not infer it from directory counts or repeated scheduling.

## 4. Verification before any full API rerun

| Gate | Test | Required result |
|---|---|---|
| G0 | Real config constructs runtime acceptance policy | Pass without `schema_version` error |
| G1 | Offline replay of saved v1 candidate artifacts | Writes `evaluation.json`; metrics match an independent calculation |
| G2 | Injected program error during evaluation | Hard abort; no rejection/history/epoch advance |
| G3 | Injected transport exhaustion | Paused resumable state; no scientific attempt consumed |
| G4 | Negative local delta | `competition_rejected`, attribution saved, next request contains projected history |
| G5 | Non-negative local delta | Whole child set accepted and root locked |
| G6 | Crash between Pairwise and evaluation | Resume performs evaluation without new API calls |
| G7 | Two-root name collision under reversed iteration order | Identical rejection set and committed rubric |
| G8 | Zero accepted candidates | Heldout stage refuses access with `no_treatment` |
| G9 | Report with zero acceptances/retries | Uses `not_applicable`; no vacuous success/natural-history claim |
| G10 | Cost audit | Successful + failed calls reconcile with request artifacts |

As an additional offline sanity check, the saved v1 artifacts should reproduce the previously observed candidate behavior rather than the official empty result: some candidate sets improve locally while visual-grounding attempts regress. These values are diagnostic only and must not be promoted to official v2 results.

## 5. v2 run protocol

### Milestone M0 — Offline repair rehearsal (must run)

- Implement P0 changes and tests.
- Replay saved v1 Pairwise artifacts through local evaluation and synchronous commit logic.
- No Manager, Worker, or heldout calls.

**Go gate:** all G0–G10 tests pass and stored decisions are exactly replayable.

### Milestone M1 — Small live smoke test (must run)

- Use a separate smoke output directory.
- Exercise one eligible root through one complete attempt.
- Verify request provenance, stage resume, local evaluation, and report generation.
- Do not access heldout.

**Go gate:** one end-to-end valid decision exists and independent replay gives the same result.

### Milestone M2 — Official five-root v2 evolution (must run)

- New output: `phase6_split_only_evolution_v2/`; never overwrite v1.
- Preserve the original trigger, seed, 3–5 valid epochs, synchronous commit, acceptance rule, and model assignments.
- Reuse immutable ErrorSignatures from v1 only when dataset, parent prediction, decisive-wrong IDs, prompt/model/request identity, and artifact hashes all match. Record cross-run provenance. Re-run clustering and child generation; content-addressed Pairwise cache reuse is allowed only for an exact request identity.
- Transport pauses do not consume an epoch. A root is exhausted only after five valid scientific attempts.

**Go gate:** final discovery report passes all invariants and contains a non-empty treatment before heldout can be unlocked.

### Milestone M3 — Final heldout evaluation (conditional must run)

- Freeze final rubric, accepted child sets, request identity, dataset hash, and metrics.
- Generate only missing accepted-child predictions through 8001.
- Compare Init five-root M1 versus Final split-evolved M1 and each evolved root's parent-only versus specialized result.
- Report ACC, Coverage, Wilson interval, corrected/harmed, exact McNemar, and discovery-to-heldout local delta.
- No post-heldout child, epoch, or history selection.

**Go gate:** either report the frozen result, including a negative result, or invalidate the run for a documented protocol violation.

## 6. Heldout contamination decision

v1 accessed heldout but had no accepted children, so it exposed only an Init=Final null comparison and provided no treatment-selection signal. Two defensible options exist:

1. **Recommended when data permit:** use a newly reserved heldout-500-v2 for the official claim and keep the old heldout solely as a debugging split.
2. **Pragmatic fallback:** reuse the same heldout-500, explicitly disclose the v1 null access, freeze v2 before access, and make no design choice using the old heldout number.

This choice must be recorded before M2 begins.

## 7. Success and failure interpretation

- C1 is supported if valid local competitions occur and accepted child sets satisfy the frozen non-degradation rule. Natural history benefit additionally requires a valid reject-to-later-improvement sequence; it is not implied by retries alone.
- C2 is supported only if the frozen final treatment improves paired heldout M1 with corrected greater than harmed. Without McNemar significance, describe it as pilot evidence.
- If local gains disappear on heldout, the acceptance estimator overfits discovery; do not blame orchestration.
- If Manager proposals remain invalid after API stability and compact history are verified, the bottleneck is clustering/child quality and should become the next method experiment.
- If no child set is accepted, report a valid negative C1 result and skip C2 rather than manufacturing an Init=Final comparison.

## 8. Explicit non-goals for this repair

- Do not change `ACC < 0.70` and `Coverage > 0.80` triggers.
- Do not change Specialized Accuracy aggregation or introduce weighted Fitness.
- Do not independently prune weak children.
- Do not add global rubric memory to Manager.
- Do not tune prompts, clustering, epoch count, or child count using heldout outcomes.

These are later ablations only after the orchestration produces a valid baseline.

## 9. Review verdict

**Recommendation: approve with one pre-run decision—the heldout choice in Section 6.** The plan fixes the causal bug, prevents false evolutionary evidence, and preserves the scientific Split definition. Its main remaining risks are API reliability, history-induced prompt growth, and the fact that one seed/trajectory provides pilot rather than robustness evidence. A three-seed replication should be considered only after one valid v2 trajectory establishes that the mechanism is worth the cost.

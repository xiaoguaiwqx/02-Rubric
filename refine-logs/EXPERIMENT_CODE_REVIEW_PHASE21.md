# Phase21 Experiment Code Review

Date: 2026-08-31\
Scope: `unified_subtree_bundle_evolution.py`, `subtree_bundle_manager.py`,
`vl_rewardbench_unified_subtree_bundle_evolution.py`, CLI/config/tests.\
Review mode: **local-only**. The experiment-bridge workflow requested a fresh-agent
review, but no additional agent capacity was available in this session. This file
therefore does not claim independent-review status.

## Result-impacting findings fixed

1. **Epoch baseline overwrite** — committed root reports were initially written over
   `root_baselines`. They now use separate `root_baselines_before/` and
   `committed_root_reports/` directories.
2. **Candidate/commit identity** — reused calls acquire an `incremental_reuse` runtime
   flag. The scientific-call hash now excludes only that operational flag and verifies
   that each committed report is otherwise identical to its candidate/baseline source.
3. **Exhausted Split rescheduling** — an exhausted root could satisfy the ordinary
   numeric threshold and become eligible again. Triggering now distinguishes
   `pending`, `retryable`, and `exhausted` states.
4. **Rejected-candidate name collision** — child-name collision checking originally
   included rejected Split proposals. It now checks only candidates that independently
   passed local competition and can actually enter the synchronous commit.
5. **Bundle Refine mutation scope** — edited nodes initially acquired new lineage
   metadata, violating “description only”. Bundle application now preserves IDs,
   names, scores, examples, lineage, edges, roots, and topology; only selected child
   descriptions change.
6. **Failure attribution modality** — harmed/corrected image paths were present in text
   evidence but images were not attached to the Manager request. The attribution call
   now attaches up to six harmed and three corrected images.
7. **Representative Refine evidence** — Refine attached images without including the
   corresponding question/A/B/gold records. Those fields are now included in the
   prompt in the same deterministic order as the images.
8. **Unified runtime drift** — endpoint identity alone could not detect prompt/parser
   changes. The manifest now freezes model, decoding, runtime protocol, subtree and
   Arbiter prompt versions/hashes, and parser source hashes; audit/run/VLRB verify them.
9. **Custom parser identity** — Bundle Refine/attribution request specs inherited the
   legacy Refine parser version. They now freeze the Phase21 bundle parser version.
10. **Missing process telemetry** — attempt records now preserve root before/after
    ACC/Coverage/None, corrected/harmed/net, Manager usage, candidate Unified usage,
    attribution type, and local-improvement/global-regression interaction cases.
11. **Fresh-freeze directory creation** — the freeze stage now creates the epoch-00
    scaffold before writing `rubric_initial.json`, so it works in a clean output
    directory without relying on a previous run.
12. **Missing epoch-00 committed Rubric** — initialization now also writes
    `epochs/epoch_00/rubric_committed.json`. The evolution loop loads this artifact
    as the next epoch's baseline; without it, smoke passed but the first real
    `run` failed before any operator was scheduled.

## Verified scientific contracts

- scope is formed only by epoch-start Unified A/B outputs;
- ErrorSignatures use only scope-contained Unified mismatches;
- candidate `None` is wrong inside the frozen scope;
- Split has no strong-child lock or partial acceptance path;
- Bundle Refine may edit a strict subset of children but commits atomically;
- both operators accept only when `corrected - harmed > 0`;
- every independently accepted root is merged in one synchronous commit;
- commit calls Global Arbiter with frozen root reports and regenerates zero roots;
- Specialized diagnostics are written after commit and are absent from trigger and
  acceptance function signatures;
- Global Arbiter diagnostics cannot change or roll back local acceptance.

## Verification

```text
python -m py_compile <all Phase21 modules>                     PASSED
python -m unittest tests.structured.test_unified_subtree_bundle_evolution
                                                               12/12 PASSED
python -m unittest tests.structured.test_local_unified_subtree_evolution
                                                                8/8 PASSED
python -m unittest discover -s tests -p "test_*.py"           438/438 PASSED
git diff --check                                               PASSED
```

The existing Phase20 Initial VLRB artifact was also checked offline against the
Phase21 request construction and parser: **22,446/22,446 calls verified**, with
no new model requests.

## Review outcome

Local review outcome: **approve for live freeze/audit/smoke**. A fresh independent
review remains desirable before interpreting final experimental results, but there is
no known result-affecting blocker in the implemented Phase21 path.

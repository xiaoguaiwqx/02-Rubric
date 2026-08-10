# Refine v1 Experiment Tracker

| Item | Status | Evidence |
|---|---|---|
| Terminology and `OperatorKind.REFINE` | Implemented | No DEFINE schema or CLI added |
| Trigger and forced-smoke provenance | Implemented | Focused unit tests |
| Strict four-section proposal parser | Implemented | Name/no-op/length/decision-rule tests |
| Identity/topology-preserving patch | Implemented | Focused unit test |
| Own-support selective-ACC competition | Implemented | Strict improvement/tie/support tests |
| Subtree and full-M1 diagnostics | Implemented | `node_evaluation.json`, diagnostic artifacts |
| Required rejection attribution | Implemented | History commit guard and resumable artifact |
| Global Rubric memory | Implemented | Frozen hash and request identity |
| Forced smoke stages | Implemented, not run | Awaiting 397B and 8001 experiment |
| Split+Refine multi-epoch stages | Implemented, gated | Requires smoke report `go` |
| Heldout-500 diagnostic | Implemented, not run | Explicit heldout stage only |
| Focused + Split regression tests | Passed | 52 tests during implementation |
| Full unittest suite | Passed | 197 tests; compile checks passed |

## Scientific gate

Do not interpret forced smoke as automatic-trigger evidence. Do not run the full trajectory when discovery node ACC fails to strictly improve or heldout shows material reverse degradation. A prompt/evidence change after failure requires a new protocol version and output directory.

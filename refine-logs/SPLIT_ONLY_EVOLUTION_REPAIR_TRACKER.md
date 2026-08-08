# Split-only Evolution v2 Repair Tracker

| ID | Milestone | Purpose | Priority | Status | Gate / evidence |
|---|---|---|---|---|---|
| R00 | Evidence freeze | Label v1 invalid and preserve all artifacts | MUST | DONE | v1 unchanged; `repair_audit_v1.json` records invalid reason |
| R01 | M0 | Fix config-to-policy construction | MUST | DONE | Plain-config regression passes |
| R02 | M0 | Implement typed attempt outcomes and hard-abort path | MUST | DONE | Pause/abort/outcome tests pass |
| R03 | M0 | Make epoch commit transactional and resumable | MUST | DONE | Commit remains after all candidate evaluations; collision-order test passes |
| R04 | M0 | Add discovery/final/heldout invariant gates | MUST | DONE | Evaluation replay audit and no-treatment gate implemented |
| R05 | M0 | Add compact full-history projection and hash | MUST | DONE | All attempts projected; raw response/metrics excluded |
| R06 | M0 | Repair reporting and cost accounting | MUST | DONE | Non-vacuous checks, failed calls and epoch wall time added |
| R07 | M0 | Offline replay saved v1 candidate artifacts | MUST | DONE | 10 competitions replayed locally; metrics agree |
| R08 | M1 | One-root live smoke test, no heldout | MUST | READY | Requires user-run API smoke stage |
| R09 | Protocol | Freeze heldout reuse/new-split decision | MUST | REVIEW | Decision recorded before official v2 |
| R10 | M2 | Run official five-root v2 evolution | MUST | BLOCKED | Valid non-empty treatment and frozen discovery report |
| R11 | M3 | Run one-shot heldout evaluation | CONDITIONAL | BLOCKED | M2 valid and at least one child set accepted |
| R12 | Analysis | Produce result-to-claim report | MUST | BLOCKED | Separate C1, history evidence, and C2 conclusions |

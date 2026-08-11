# Split–Refine Improvement Tracker

| Run ID | Milestone | Purpose | Variant | Data | Priority | Status | Decision note |
|---|---|---|---|---|---|---|---|
| SR-M0-01 | M0 | Control replay and schema tests | new flags disabled | offline | MUST | DONE | 212 tests pass; frozen Control history projection unchanged |
| SR-M0-02 | M0 | Child diagnostic calculation tests | parent-on-child-scope, conflict matrix, leave-one-out | offline | MUST | DONE | Visual audit reproduces 62/89 vs 61/89 and 41 conflict samples |
| SR-B1-01 | M1 | Visual retry smoke | child diagnostics + preservation guidance, seed 42 | discovery-90 | MUST | RUNNING | Isolated 8000 Worker track; directive-only, not hash-identical reuse |
| SR-B1-02 | M1 | Visual retry stability | treatment, 3 Manager seeds | discovery-90 | CONDITIONAL | TODO | Run only if B1-01 is positive |
| SR-B2-01 | M2 | Trigger eligibility audit | uniform vs role-aware | frozen Split rubric | MUST | DONE | 4 uniform, 11 role-aware, 7 newly eligible children |
| SR-B2-02 | M2 | Refine-only role-aware evolution | child trigger v2 | discovery-90 | MUST | RUNNING | Isolated 8001 Worker track, 2–3 epochs |
| SR-B2-03 | M2 | Frozen heldout diagnostic | discovery-selected final rubric | heldout-500 | CONDITIONAL | TODO | Run only after B2 discovery freeze |
| SR-B3-01 | M3 | Integrated final trajectory | feedback v2 + trigger v2 | discovery-90 | MUST | TODO | Five roots, 3–5 epochs |
| SR-B3-02 | M3 | Integrated heldout report | frozen B3 final rubric | heldout-500 | MUST | TODO | Exploratory paired comparison |
| SR-B4-01 | M4 | Stuck-root scheduling ablation | 3 failures -> Refine parent -> Split | Visual Grounding | NICE | TODO | Defer if B1 already succeeds |

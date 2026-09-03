# Phase20 Code Review

**Experiment:** Local Unified-Subtree competition evolution\
**Protocol:** `local-unified-subtree-competition-evolution-v1`\
**Review date:** 2026-08-29

## Initial review findings and fixes

The fresh implementation review identified two trajectory-level blockers:

1. Split lock diagnostics were created before the Phase20 root-scope decision.
   The implementation now creates the strong-child retry lock only after the
   root guard rejects the candidate. A root-accepted candidate cannot acquire a
   legacy retry lock.
2. Root-level Refine failures did not carry paired root evidence into later
   Manager context. The implementation now persists the full paired sample-ID
   sets, representative before/after cases, root evidence, and root evaluation
   in the attempt and history projections used by both Split and Refine retry
   Managers.

## Verification

- `py_compile` passes for all new and modified modules.
- Focused Phase20 and related evolution tests pass: **62 tests**.
- Full repository suite passes: **426 tests**.
- `git diff --check` passes (only normal Git LF/CRLF warnings).
- Freeze and offline audit pass.
- Live 20-sample smoke passes with zero root technical failures; both
  `vllm-8000` and `vllm-8001` receive requests after interleaved worker startup.

The attempted follow-up fresh-agent review could not refresh its external
review token. The blocker findings above were independently rechecked against
the shared worktree and covered by focused tests.

## Verdict

**Ready for the user-controlled full run.** No remaining known issue is expected
to change the scientific trajectory or acceptance semantics.

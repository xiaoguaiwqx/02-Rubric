# Split-only Evolution Experiment Tracker

| Stage | Status | Output | Gate |
|---|---|---|---|
| Freeze | READY | `phase6_split_only_evolution_v1/frozen_manifest.json` | Exactly four roots trigger |
| Epoch evolution | READY | `epochs/epoch_01..05/` | Local specialized ACC only |
| Discovery report | READY | `final/discovery_report.json` | Final rubric hash frozen |
| Heldout-500 | BLOCKED ON REPORT | `heldout500/report.json` | One-shot access, 8001 only |
| Final report | BLOCKED ON HELDOUT | `final_report.json`, `final_report.md` | Five tables complete |

Implementation sanity checks are local-only; no Manager or Pairwise requests are launched during implementation.
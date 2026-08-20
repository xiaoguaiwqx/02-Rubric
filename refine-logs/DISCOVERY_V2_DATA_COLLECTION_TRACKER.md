# Discovery-v2 Data Collection Tracker

| Item | Status | Evidence |
|---|---|---|
| Independent runner and resumable CLI stages | implemented | `discovery_data_v2.py` |
| Immutable HF revision lock and gated VisionArena failure | implemented | source-lock stage/tests |
| Six source adapters and schema failures | implemented | focused tests |
| Content-addressed images and benchmark exclusion | implemented | ingest stage |
| Dev frozen before Worker screening | implemented | ingest ordering/assertions |
| Prompt-v2 original/swap five-root screening | implemented | screen stages |
| Metadata-only Dev150 and Coverage candidates150 | completed | `selection_v2/selection_manifest.json` |
| Criterion-agnostic original/swap Generic screening | completed | `selection_v2/generic_screen/report.json` |
| Source-balanced Hard candidates90 excluding Coverage | completed | `selection_v2/hard_candidates_90.jsonl` |
| Criterion-agnostic 397B double-order pre-adjudication | implemented; offline freeze completed | `adjudication_v2/frozen_manifest.json` and new five-stage protocol |
| Blind review and reconciliation export v2 | implemented, pending online adjudication | `discovery-v2-review-v2-export` |
| Coverage70 + Hard30 Discovery100 Demo selector | completed; two low-generality MM-RLHF records quality-replaced | `discovery100_demo_single_order_v3/frozen_discovery100.jsonl` |
| Discovery100 Demo 397B balanced single-order path | v2 completed; v3 needs only two replacement judgments | 98 cached unchanged + 2 new; 50 original + 50 swapped |
| Discovery100 Demo blind review/finalize | implemented, pending human review | `discovery-v2-demo-review-export/finalize` |
| Strict 75/25/150 deterministic finalize | implemented | production-size offline test |
| Compact report and dataset card | implemented | report/finalize stages |
| Live source lock, ingest and offline selection | completed | `discovery_data_v2/` artifacts |
| Human review | pending | generated after adjudication |

Runtime data, images, caches, model outputs, and review queues remain under the
ignored experiment output tree and are not intended for Git.

# Wuji v9: frozen-pad preflight and first A-arm checkpoint

**Interim record, 2026-10-19.** The user accepted the highlighted v8 transfer contacts as local improvement examples and authorized the short v9 training pilot. This record uses the already opened fixed validation seed `2026101701` for all three policies, 24 CPU episodes each. No unseen seed was opened and no natural-turn success is claimed. The v9 [contact map](../../../../assets/wuji_hand2_palm_up_human_demo_v9/pad_regions.json) and [pilot contract](../../../../experiments/wuji_hand2/palm_up_human_demo_v9/PLAN.md) were frozen before the first v9 policy update.

| Diagnostic policy | Strict passes | Main remaining failures | First-episode three-view video | Raw 24-episode metrics |
| --- | ---: | --- | --- | --- |
| v7 iteration 9000, unchanged policy under v9 pad map | 0/24 | non-pad 24, two-pad 17, opposing 8 | [v7 video](baseline_v7_pad/rollout.mp4) | [v7 metrics](baseline_v7_pad/metrics.json) |
| v8 iteration 996, unchanged policy under v9 pad map | 2/24 | two-pad 20, non-pad 6 | [v8 video](baseline_v8_pad/rollout.mp4) | [v8 metrics](baseline_v8_pad/metrics.json) |
| v9 A iteration 100, **early training checkpoint** | 4/24 | non-pad 16, two-pad 8; five episodes incomplete | [A100 video](validation_0000100/rollout.mp4) | [A100 metrics](validation_0000100/metrics.json) |

The v8 result remains an improvement candidate, not an accepted full turn. The 2/24 calibrated result supersedes neither its original **0/24 sensor-only** assessment nor the motion review requirement. A100 is an early diagnostic point: its higher pass count comes with incomplete episodes, so it cannot be selected as a final model. The same frozen 23/24 entry threshold applies to later checkpoints.

For each case the directory also contains uncut first-episode [frame metrics](baseline_v8_pad/frame_metrics.json), [raw contacts](baseline_v8_pad/contacts.json), and [state replay](baseline_v8_pad/rollout.npz); the corresponding filenames exist under each row's directory. The [warm-start wrapper provenance](provenance.json) records unchanged v7/v8 policy weights and exact source hashes. The [A100 checkpoint](arm_a_checkpoint_0000100.pt) is preserved only as a stage weight, with its training source manifest in the live run. The videos use the copied v8 renderer and retain its `V8` overlay; the on-frame task name and these metrics distinguish the v9 A run. This inherited overlay is a display-label issue, not a change in policy or physics.

The first-episode videos are diagnostic examples. They cannot substitute for all 24 episode metrics or the human check of finger crossing, contact location, release and handover, and final regrasp. A and B continue under separate run directories; the unseen seed `2026101703` remains sealed until fixed validation and human review qualify a candidate.

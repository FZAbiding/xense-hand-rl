# Wuji v1–v7 checkpoint index

The six `.pt` files in this directory are byte-for-byte copies of historical local source checkpoints. [manifest.json](manifest.json) gives each original path, SHA256, size, and qualification status. Their inclusion protects the experiment history; **none of the archived v1–v5 checkpoints is a qualified natural Triangle Pass model**. Restoring an old checkpoint also requires its matching scene, grasp bank, observation/action specification, and source version.

| Stage | Archived weight | Evidence and status |
| --- | --- | --- |
| v1 corrected palm-up | [source checkpoint](v1_palm_up_checkpoint_000550.pt) | Historical parent of v2; task orientation changed from legacy. |
| v2 opposing pinch | [policy](v2_pinch_policy.pt) | [Report](../palm_up_pinch_v2/REPORT.md): 0/24 strict. |
| v3 tripod | [policy](v3_tripod_policy.pt) | [Report](../palm_up_tripod_v3/REPORT.md): 0/24 strict. |
| v4 stable hold | [policy](v4_hold_policy.pt) | [Report](../palm_up_human_demo_v4/REPORT.md): hold only, not accepted handover. |
| v4 failed handover | [policy](v4_handover_policy.pt) | Rejected crossing/failed handover diagnostic; never use as approved imitation. |
| v5 long-run final | [checkpoint](v5_long_final_checkpoint_0064902.pt) | 24-hour final, 0/24 strict; [inheritance report](../../../experiments/wuji_hand2/palm_up_human_demo_v6/long_training/PLAN.md). The separate tracked [v5 model](../../../models/wuji_hand2/v5_20261012_final.pt) has a different SHA and must not be silently substituted for this final checkpoint. |
| v6b | [10-second fixed-validation model](../palm_up_human_demo_v6b/snapshot_20261014_42500/quality10/model.pt), [6-second fixed-validation model](../palm_up_human_demo_v6b/snapshot_20261014_42500/fast6_validation/model.pt), [3-second failure](../palm_up_human_demo_v6b/snapshot_20261014_42500/faster3_latest/model.pt) | Already tracked with exact [report, videos, and metrics](../palm_up_human_demo_v6b/REPORT.md). The final 3-second holdout failed 0/24. |
| v7 | [best iteration 9000](../palm_up_human_demo_v7/progress_20261015/turn_3s_best_iteration_9000/model.pt), [final iteration 51074](../palm_up_human_demo_v7/progress_20261015/turn_3s_final_holdout_iteration_51074/model.pt) | Already tracked with [holdout videos, raw metrics, and contact audit](../palm_up_human_demo_v7/stage_a_20261016/REVIEW.md). Original gate 24/24 and 23/24 respectively; pad-location and human review open. |

To verify copied files: `sha256sum *.pt` and compare with [manifest.json](manifest.json). Do not select a model from an unseen holdout or assign one checkpoint's score to another.

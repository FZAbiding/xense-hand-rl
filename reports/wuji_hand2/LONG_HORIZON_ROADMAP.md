# Wuji continuous spinning roadmap (proposal only)

Status: planning on 2026-10-17; **no new training is authorized or started by this document**. v7's selected 3-second turn is promising under its original metric, but its 5.5-second rollout is not a continuous 30-second result. The [v7 stage A review](palm_up_human_demo_v7/stage_a_20261016/REVIEW.md) and [next-stage contact plan](palm_up_human_demo_v7/NEXT_STAGE_PLAN.md) define the immediate gate.

The subsequent [collision-mesh contact map](palm_up_human_demo_v7/contact_geometry_20261017/REVIEW.md) and [isolated v8 preflight plan](../../experiments/wuji_hand2/palm_up_human_demo_v8/PLAN.md) make that gate concrete while the human contact verdict remains pending.

The [completed v8 short pilot](palm_up_human_demo_v8/pilot_20261017/REVIEW.md) improved the sensor-only diagnostic signals, but its selected checkpoint passed **0/24** fixed-validation strict episodes. Its unseen seed remains unopened. Treat the v8 weight as diagnostic only; calibrate the real acceptable pad surface and review the motion before another short isolated ablation. The [proposed v9 plan](../../experiments/wuji_hand2/palm_up_human_demo_v9/PLAN.md) specifies that sequence for user review. The 10/30/60/120-second stages below remain plans and have not been started.

## Entry gate: correct the short-cycle evidence

Calibrate the actual thumb/index/middle finger-pad regions on collision geometry and separate pad, distal-side, other-body, and idle-finger forces. Freeze this criterion in both training observation/reward and independent CPU evaluation. Review synchronized v7 video against the approved human reference for real release, handover, finger-chain order, and end regrasp. If the motion is rejected, keep its failure evidence and do not imitate it. In a new isolated version, reproduce a 3-second full turn with at least 23/24 fixed-validation and 23/24 independent holdout passes under the **new** contact and posture gate, plus an explicit favorable human review. The specific threshold is a proposed stage gate, not a claim that v7 already meets it.

## Duration progression

| Stage | Target and approximate cycles at 3 seconds/turn | Required evidence before promotion |
| --- | --- | --- |
| Bridge | 10–12 seconds, roughly 3–4 turns | Establish more than one genuine release/regrasp cycle; 24 uncut independent CPU episodes and a separate fixed validation set, no internal reset or grip-loss spin. |
| 30 seconds | Roughly 10 turns | At least 23/24 strict passes on both sets, complete 30-second survival, stable cycle-by-cycle contact and angle, no drift accumulation or disguised terminal failure; human review of beginning, middle, and end. |
| 60 seconds | Roughly 20 turns | Same gates over a full minute with independent disturbances and per-cycle failure timing; preserve worst-case episodes and all failures. |
| 120 seconds | Roughly 40 turns | Same gates over two minutes, with robust recovery from allowed small disturbances and no hidden policy reset or video repetition. |

Cycle counts are planning estimates. Report actual continuous signed turns, direction, per-cycle time, release/contact events, and pen-end identity. Keep the full uncut episode and per-frame telemetry; do not infer long-duration success from a single good cycle or a survival-only hold. For each stage, predeclare the evaluation duration, allowed disturbances, exact strict thresholds, fixed/holdout seeds, and human review criteria in its own `PLAN.md`.

## Proposed training method

1. Make an isolated successor only after the entry gate. Warm-start only from a checkpoint whose behavior the user accepts, with its SHA and scene/observation compatibility recorded.
2. Expand the episode horizon gradually while preserving the successful short-cycle task. Train for stable release and recovery, not merely terminal angle; add cumulative drift and idle-finger penalties only after observing their isolated effect in short pilots.
3. Compare one intervention per pilot: horizon curriculum, contact reward, recovery examples, or policy memory. Keep a frozen baseline and early-stop on strict validation regression. Save stage-best and final weights independently.
4. At each duration, use multiple training seeds if resources allow. Select on fixed validation, then run the separately frozen holdout once. Investigate every early end and every apparent angle gain with lost contact.

Compute and wall-clock budgets should be estimated from short pilots after the entry gate, rather than assuming v7's 24-hour run scales linearly. Each duration is a separate approval and experiment, with a reviewable executable config, seed list, stop rule, and expected artifact list.

## Longer-term improvements to plan, not execute now

- **Contact fidelity:** calibrate pad geometry and friction/force parameters against hardware measurements; add uncertainty bands where the model is uncalibrated.
- **Perception and feedback:** evaluate whether tactile and proprioceptive signals can detect true release and recover an incipient slip; compare recurrent and memoryless policies with identical observations.
- **Robustness:** widen object mass, friction, size, initial pose, and perturbations only after nominal strict quality holds; report the tested domain and worst cases.
- **Motion quality:** obtain multi-view or instrumented human reference for occluded handovers, then compare cycle phase, finger-order, and contact sequence rather than visual similarity alone.
- **Transfer readiness:** quantify sim-to-real gaps and safety limits before any hardware attempt; this roadmap makes no real-hand performance claim.

# Dexterous manipulation training SOP

This workflow distills the Wuji Hand 2 palm-up pen-spinning v1–v7 experiments. It is a reusable **process**, not a claim that a particular hand, motion, contact threshold, or policy is generally valid. Read the [experiment review and presentation SOP](EXPERIMENT_REVIEW_SOP.md) before simulation, evaluation, or delivery. The [Wuji evidence and lessons](../reports/wuji_hand2/V1_V7_LESSONS.md) explain why each gate exists.

## 1. Define the task and isolate the version

1. Write a versioned `PLAN.md` with task geometry, palm normal, active fingers, reference motion, pen-end identity, angle direction, target timing, control/physics rates, rewards, hard constraints, training budget, validation seeds, unseen-test seeds, and explicit pass/fail gates.
2. Give each version separate `experiments/`, `assets/`, `runs/`, `artifacts/`, and `reports/` directories. Record the source commit, parent checkpoint SHA256, scene and grasp-bank hashes, and every changed setting. Never silently reuse an old scene, initial-state bank, or metric definition.
3. Verify the robot's joint axes, limits, collision geometry, actuator mapping, palm orientation, object dimensions/mass, and actual control period with a short deterministic rollout. A plausible video from the wrong orientation is not a valid task result.
4. Build and certify stable initial states over the intended hold duration and disturbances. A 2-second grasp search cannot establish a 60-second baseline.

## 2. Validate the motion before reward optimization

1. Keep the human source clip, provenance, time range, and hash. Mark occluded contacts as unknown; a 2D projection does not reveal 3D forces.
2. Inspect colored and numbered fingers from synchronized top, palm-side, and side views. Check the whole index/middle chains, crossing, side splay, joint signs, self-collision, and the release and regrasp sequence.
3. Map contact points to calibrated pad surfaces on the actual collision geometry. Log contact body, local/world position, normal/tangential force, and non-pad support separately. A merged distal-body plus tip-sensor channel must not be called proven fingertip contact.
4. Freeze a machine-readable quality gate before training: survival, no drop, workspace/drift, joint and self-contact rules, continuous signed angle with pen-end identity, contact pair and release sequence, and human motion review. Do not promote a candidate from angle or survival alone.
5. If the user rejects a motion, preserve the video and raw metrics, withdraw its naturalness claim, and exclude it as a qualified demonstration or imitation target.

## 3. Train one hypothesis at a time

1. Start from the shortest physically valid action and increase amplitude, speed, and duration in separate stages. Preserve the last accepted checkpoint before each change.
2. Use small, isolated ablations for reward and curriculum changes. Keep configurations, assets, seeds, and runs distinct. In Wuji v7, course-only and combined reward/course pilots produced materially different results.
3. Run fixed validation periodically with frozen seeds and gates. Save **best selected** and **latest/final** checkpoints separately. Select on the fixed validation set before opening a fresh holdout; never pick a model after seeing that holdout.
4. Watch for regressions and failure modes, not only mean reward: early termination, inertial object motion after lost grip, excessive side or idle-finger support, command clipping, finger crossing, bad terminal regrasp, or speed-induced collapse.
5. Record wall-clock and effective GPU time, updates, environment steps, interruptions, and supervisor state. Before resuming a long run, inspect the existing process/supervisor; do not start a duplicate trainer.

## 4. Evaluate independently and publish evidence

1. Replay the frozen network in an independent CPU physics evaluator for complete, uncut episodes, with no automatic reset inside the target interval. Use both fixed validation and a separately frozen unseen set, including declared perturbations.
2. Report the denominator and per-episode failures. Keep raw trajectories, per-frame contacts, metrics JSON, model SHA256, and synchronized two-view video. A video must link to the exact checkpoint and seed that produced it.
3. Distinguish physical survival, angle tracking, actual contact/release, posture, robustness, and human approval. A passing old metric is not retroactively a pass under a corrected contact definition.
4. Publish the offline HTML card review required by [the presentation SOP](EXPERIMENT_REVIEW_SOP.md), including reference, candidate, baseline, and clear failed examples. Preserve rejected examples and corrected conclusions.
5. Advance only after the current stage passes both machine and human gates. A faster single turn is not evidence for a long continuous sequence; each duration needs its own uninterrupted evaluation.

## 5. Handoff checklist for another agent

Before the next agent trains, supply: the versioned plan and inheritance chain; executable config and source commit; scene/grasp-bank/model hashes; validation and unseen seeds; exact gate implementation and limitations; best and final weights; raw metrics and synchronized videos; human verdict; known failure cases; and the next experiment's one stated hypothesis. If any item is unknown, mark it unknown rather than inventing a success.

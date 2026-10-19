# Wuji v10 overnight robustness preflight

**2026-10-20, before v10 PPO training.** The warm-start policy is the preserved [v9 A900 model](../../palm_up_human_demo_v9/pilot_20261019/arm_a_best_0000900/model.pt), unchanged in the [v10 evaluation wrapper](a900_wrapped_for_v10_020.pt). The [provenance record](provenance.json) contains both hashes. The v10 scene, grasp bank, and pad map are byte-identical to v9. The sealed unseen seed `2026101703` has not been evaluated.

| Fixed CPU condition | Strict result | Evidence |
| --- | ---: | --- |
| Standard 5.5 s turn, seed `2026101701` | **24/24** | [three-view video](baseline_standard/rollout.mp4), [all 24 metrics](baseline_standard/metrics.json), [first-episode contacts](baseline_standard/contacts.json) |
| ±0.02 N push, diagnostic seed `2026102001` | **24/24** | [all 24 metrics](calibration/push_002.metrics.json) |
| ±0.08 N push, same diagnostic seed | **24/24** | [all 24 metrics](calibration/push_008.metrics.json) |
| **±0.2 N push, same diagnostic seed** | **18/24** | [uncut first failed episode](baseline_push_020/rollout.mp4), [all 24 metrics](baseline_push_020/metrics.json), [contacts](baseline_push_020/contacts.json) |

At ±0.2 N, failed episode IDs were **0, 2, 10, 14, 21, 22**. Three ended before 5.5 s and three exceeded the non-pad-force fraction; episode 2 additionally failed rotation progress and final regrasp. The first video is therefore an actual failed rollout, not a representative success clip. The perturbation is 0.1 s every 2 s with each horizontal force component independently uniform in ±0.2 N. It is a strong stress test on the 0.025 kg pen. A900 passing standard validation does not establish that the motion is natural or robust to this challenge.

The [frozen v10 plan](../../../../experiments/wuji_hand2/palm_up_human_demo_v10/PLAN.md) uses ±0.2 N as the diagnostic target and keeps the standard no-push gates unchanged. Training will start only under the overnight GPU readiness rule. Even if v10 improves, its selected policy remains diagnostic until the user reviews the handover and the sealed unseen seed is later evaluated.

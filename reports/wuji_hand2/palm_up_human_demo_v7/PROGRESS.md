# Wuji v7 正式长训阶段进度（2026-10-15）

三组短训的[同种子比较、失败视频与入选模型](PILOTS.md)已归档。正式训练从“只改课程”第 500 轮权重开始，按训练预算扣除三组短训后的约 23 小时有效 GPU 时间运行。最新流水线状态和实时训练指标分别保存在 `runs/wuji_hand2/palm_up_human_demo_v7/pipeline_20261015/status.json` 与 `runs/wuji_hand2/palm_up_human_demo_v7/long_training_20261015/training_status.json`；以下只记录已完成的独立 CPU 评估，训练仍在进行。

| 阶段 | 检查点 | 固定验证种子严格通过 | 失败原因 | 状态 |
| --- | ---: | ---: | --- | --- |
| 360°/4 秒，目标后保持约 2 秒 | 第 100 轮 | **23/24** | 1 回合提前结束，并出现姿态/反弯与终点回夹失败 | 达到 ≥23/24，已进入 3.5 秒课程 |
| 360°/3.5 秒，目标后保持约 2 秒 | 第 500 轮 | **24/24** | 无严格失败 | 已达到 ≥23/24，进入 3 秒课程 |
| 360°/3 秒，目标后保持约 2 秒 | 第 1000 轮 | **23/24** | 1 回合提前结束，伴随姿态/反弯、自穿透、相向夹持和终点回夹失败 | 达到 ≥23/24；继续训练，最终保留种子与人工审阅待完成 |
| 360°/3 秒，当前最佳固定验证 | 第 3000 轮 | **24/24** | 无严格失败 | 已归档最佳模型，平均角误差约 1.52° |
| 360°/3 秒，最近一次固定验证 | 第 6500 轮 | **23/24** | 1 回合提前结束，转角进度和终点回夹失败 | 训练仍在进行；不覆盖第 3000 轮最佳结论 |

4 秒第 100 轮[独立评估视频](progress_20261015/turn_4s_iteration_100/rollout.mp4)、[逐回合指标](progress_20261015/turn_4s_iteration_100/metrics.json)、[接触记录](progress_20261015/turn_4s_iteration_100/contacts.json)和[模型权重](progress_20261015/turn_4s_iteration_100/model.pt)已固定归档。模型 SHA256 为 `37e03ee82085cbab8393b9b304a2ef4c297e7ada9db563ed54239547e2185017`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_4s_iteration_100/rollout.mp4"></video>

3.5 秒第 500 轮[独立评估视频](progress_20261015/turn_3p5s_iteration_500/rollout.mp4)、[逐回合指标](progress_20261015/turn_3p5s_iteration_500/metrics.json)、[接触记录](progress_20261015/turn_3p5s_iteration_500/contacts.json)和[模型权重](progress_20261015/turn_3p5s_iteration_500/model.pt)已固定归档。模型 SHA256 为 `92fde883ed666fd9e472797c911993e45e239e0c7930e5a18261df802c3d69e6`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3p5s_iteration_500/rollout.mp4"></video>

3 秒第 1000 轮[独立评估视频](progress_20261015/turn_3s_iteration_1000/rollout.mp4)、[逐回合指标及失败标记](progress_20261015/turn_3s_iteration_1000/metrics.json)、[接触记录](progress_20261015/turn_3s_iteration_1000/contacts.json)和[模型权重](progress_20261015/turn_3s_iteration_1000/model.pt)已固定归档。24 回合平均实际转角约 355.45°、平均时长约 5.48/5.5 秒；平均值不替代上述逐回合严格门槛。模型 SHA256 为 `797c12fb8d09bb11ff4ad44b4e366c89de1a7c87464e590837cc80f9f6d2257f`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_iteration_1000/rollout.mp4"></video>

3 秒第 3000 轮是截至第 6500 轮的**最佳固定验证模型**：[视频](progress_20261015/turn_3s_best_iteration_3000/rollout.mp4) · [逐回合指标](progress_20261015/turn_3s_best_iteration_3000/metrics.json) · [接触记录](progress_20261015/turn_3s_best_iteration_3000/contacts.json) · [模型权重](progress_20261015/turn_3s_best_iteration_3000/model.pt)。24/24 严格通过，模型 SHA256 为 `36f02a04fb33a274dc8d40524785e2600cc52c8460d36cdc117ca483edfea4e5`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_best_iteration_3000/rollout.mp4"></video>

第 6500 轮是本次归档时的**最近评估模型**：[视频](progress_20261015/turn_3s_latest_iteration_6500/rollout.mp4) · [逐回合指标及失败标记](progress_20261015/turn_3s_latest_iteration_6500/metrics.json) · [接触记录](progress_20261015/turn_3s_latest_iteration_6500/contacts.json) · [模型权重](progress_20261015/turn_3s_latest_iteration_6500/model.pt)。它为 23/24；模型 SHA256 为 `199e23b98cab8c50e533cbf43f7ab08644c989b3b2c1a524c391a9f92a01b9a0`。第 2500、4000 轮验证曾降到 20/24，说明训练中的性能仍有波动；因此持续保留最佳合格权重。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_latest_iteration_6500/rollout.mp4"></video>

视频为闭环策略的同步三视角；真人参考与训练前诊断见 [BASELINE.md](BASELINE.md)。严格物理通过率与人工动作相似性分开记录，后者仍**待审阅**。主训练配置不含 60 秒连续课程；须先完成 3 秒动作的人工审阅，再单独建立连续训练版本。最终未见种子只在选模结束后使用。

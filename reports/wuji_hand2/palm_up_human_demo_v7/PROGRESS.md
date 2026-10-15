# Wuji v7 正式长训阶段进度（2026-10-15）

三组短训的[同种子比较、失败视频与入选模型](PILOTS.md)已归档。正式训练从“只改课程”第 500 轮权重开始，按训练预算扣除三组短训后的约 23 小时有效 GPU 时间运行。最新流水线状态和实时训练指标分别保存在 `runs/wuji_hand2/palm_up_human_demo_v7/pipeline_20261015/status.json` 与 `runs/wuji_hand2/palm_up_human_demo_v7/long_training_20261015/training_status.json`；以下只记录已完成的独立 CPU 评估，训练仍在进行。

| 阶段 | 检查点 | 固定验证种子严格通过 | 失败原因 | 状态 |
| --- | ---: | ---: | --- | --- |
| 360°/4 秒，目标后保持约 2 秒 | 第 100 轮 | **23/24** | 1 回合提前结束，并出现姿态/反弯与终点回夹失败 | 达到 ≥23/24，已进入 3.5 秒课程 |
| 360°/3.5 秒 | 尚未归档合格模型 | 待评估 | 待评估 | 训练中 |

4 秒第 100 轮[独立评估视频](progress_20261015/turn_4s_iteration_100/rollout.mp4)、[逐回合指标](progress_20261015/turn_4s_iteration_100/metrics.json)、[接触记录](progress_20261015/turn_4s_iteration_100/contacts.json)和[模型权重](progress_20261015/turn_4s_iteration_100/model.pt)已固定归档。模型 SHA256 为 `37e03ee82085cbab8393b9b304a2ef4c297e7ada9db563ed54239547e2185017`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_4s_iteration_100/rollout.mp4"></video>

视频为闭环策略的同步三视角；真人参考与训练前诊断见 [BASELINE.md](BASELINE.md)。严格物理通过率与人工动作相似性分开记录，后者仍**待审阅**。3 秒单圈通过及人工审阅完成前，60 秒连续课程不会自动启动；最终未见种子只在选模结束后使用。

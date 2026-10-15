# Wuji v7 启动基线与 v6b 最终失败证据

**状态（2026-10-15）：** v6b 的 24 小时墙钟预算已结束，第 47,173 轮最终独立保留种子严格通过 **0/24**。v7 以已归档的 [6 秒固定验证模型](../palm_up_human_demo_v6b/snapshot_20261014_42500/fast6_validation/model.pt)作为候选热启动；它在 v7 新未见种子上 **23/24**，在轻微水平外力扰动组 **24/24**。这是训练前 CPU 诊断，不能冒充 v7 训练成果或人工认可的自然动作。

两组 v7 基线均测试 360°/6 秒，并在目标转角完成后继续约 2 秒，共 8.5 秒。新增终点回夹检查后，48 回合均跑满时长并在终帧保持相向双指接触；未见种子唯一失败回合缺少第三次稳定交接。扰动为每 2 秒施加 0.1 秒水平外力、每轴最大 0.02 N，逐回合方向独立采样。所用源权重 SHA256 为 `c97c54d6fa62a83dbafa264098e1c924ce3c293996865d443320a6fab8ad7881`；另一个 23/24 的 v6b 诊断模型是不同哈希，未混用。

| CPU 评估 | 严格通过 | 平均实际转角 | 平均角误差 | 平均回合时长 | 原始指标 |
| --- | ---: | ---: | ---: | ---: | --- |
| v6b 最终 3 秒整圈，保留种子 | **0/24** | 343.19° | 1.54° | 3.88/5 秒 | [metrics.json](baseline_20261015/v6b_final_failure/metrics.json) |
| v7 训练前 6 秒基线，新未见种子 | **23/24** | 355.07° | 1.94° | 8.50/8.5 秒 | [metrics.json](baseline_20261015/v7_unseen/metrics.json) |
| v7 训练前 6 秒基线，外力扰动 | **24/24** | 355.08° | 1.94° | 8.50/8.5 秒 | [metrics.json](baseline_20261015/v7_perturbed/metrics.json) |

v6b 最终失败组中 24/24 未达相向夹持与至少三次交接，22/24 提前结束，21/24 双指尖接触不足。失效分项可重叠，不能以接近整圈或低角误差替代交接与持笔验收。v7 下一阶段针对这一问题，按 [训练计划](../../../experiments/wuji_hand2/palm_up_human_demo_v7/PLAN.md)开展独立的课程和接触奖励消融。

## 视频

以下机器人视频均为真实闭环策略 rollout，保留同一时间步的三视角和彩色手指标识；[真人参考 A](baseline_20261015/human_A.mp4)仅作动作审阅参照。文件及 SHA256 在 [baseline_manifest.json](baseline_manifest.json)，[离线卡片页](baseline_review.html)可并排观看。

**v6b 最终 3 秒失败：** [下载 MP4](baseline_20261015/v6b_final_failure/rollout.mp4) · [逐帧指标](baseline_20261015/v6b_final_failure/frame_metrics.json) · [接触](baseline_20261015/v6b_final_failure/contacts.json)

<video controls preload="metadata" playsinline width="960" src="baseline_20261015/v6b_final_failure/rollout.mp4"></video>

**v7 训练前 6 秒未见种子：** [下载 MP4](baseline_20261015/v7_unseen/rollout.mp4) · [逐帧指标](baseline_20261015/v7_unseen/frame_metrics.json) · [接触](baseline_20261015/v7_unseen/contacts.json) · [原始轨迹](baseline_20261015/v7_unseen/rollout.npz)

<video controls preload="metadata" playsinline width="960" src="baseline_20261015/v7_unseen/rollout.mp4"></video>

**v7 训练前 6 秒扰动组：** [下载 MP4](baseline_20261015/v7_perturbed/rollout.mp4) · [原始指标](baseline_20261015/v7_perturbed/metrics.json)

三视角抽查未见明显食指/中指交叉，整条骨链和所有回合的指腹接触仍需人工与接触数据共同复核。人工动作相似性状态为**待审阅**，物理验收通过不表示已达到真人示范效果。

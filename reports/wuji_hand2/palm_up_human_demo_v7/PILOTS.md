# Wuji v7 三组短训与选模（2026-10-15）

三组独立试验各运行约 20 分钟有效 GPU 训练时间，均从已归档的 v6b 6 秒固定验证权重（SHA256 `c97c54d6fa62a83dbafa264098e1c924ce3c293996865d443320a6fab8ad7881`）初始化。固定验证种子为 `2026101501`，每项 24 回合。下表使用**各权重分别在 6 秒和 4.5 秒任务上的同种子 CPU 复评**，不把训练时的回合完成率等同于严格通过率。

| 试验与检查点 | 6 秒严格通过 | 4.5 秒严格通过 | 结论 |
| --- | ---: | ---: | --- |
| 只改课程，第 500 轮 | **23/24** | **24/24** | 唯一两级均达到 ≥23/24；选为正式训练起点 |
| 只改课程，最新第 790 轮 | 19/24 | 24/24 | 6 秒保持能力退化，不选 |
| 只改接触奖励，最新第 776 轮 | 23/24 | 22/24 | 4.5 秒未过门槛 |
| 课程与奖励同时改，最新第 1040 轮 | 24/24 | 8/24 | 4.5 秒明显失败 |
| 课程与奖励同时改，第 1000 轮 | 23/24 | 11/24 | 4.5 秒未过门槛 |

选择记录与原始逐回合指标见 [selection.json](pilot_20261015/selection.json)。所选模型为 [course_only_500.pt](pilot_20261015/course_only_500.pt)，SHA256 `ecc7511ed9fd7733dd151f869aa05823d0eb7a4aeb2033ae6b5216b56ab7b00f`。它是短训第 500 轮模型，与“最新第 790 轮”不同。正式长训从这个权重开始，以 4 秒单圈为当前级，总 24 小时有效 GPU 预算扣除三组试验共约 1 小时；60 秒连续任务需在 3 秒单圈通过且人工审阅后另行启动。

## 严格评估视频与原始证据

视频为闭环策略评估的同步三视角。真人参考与训练前基线见 [BASELINE.md](BASELINE.md)。机器人动作与真人示范的相似性仍**待人工审阅**，物理门槛通过不表示已认可自然动作。

**入选权重，第 500 轮，4.5 秒 24/24：** [视频](pilot_20261015/course_only_500_4p5/rollout.mp4) · [逐回合指标](pilot_20261015/course_only_500_4p5/metrics.json) · [接触记录](pilot_20261015/course_only_500_4p5/contacts.json)

<video controls preload="metadata" playsinline width="960" src="pilot_20261015/course_only_500_4p5/rollout.mp4"></video>

**同一入选权重，6 秒 23/24：** [视频](pilot_20261015/course_only_500_6s/rollout.mp4) · [逐回合指标](pilot_20261015/course_only_500_6s/metrics.json)

<video controls preload="metadata" playsinline width="960" src="pilot_20261015/course_only_500_6s/rollout.mp4"></video>

**最新权重回退示例，第 790 轮，6 秒 19/24：** [视频](pilot_20261015/course_only_latest_6s/rollout.mp4) · [逐回合指标](pilot_20261015/course_only_latest_6s/metrics.json)

<video controls preload="metadata" playsinline width="960" src="pilot_20261015/course_only_latest_6s/rollout.mp4"></video>

**接触奖励单独消融，4.5 秒 22/24：** [视频](pilot_20261015/contact_only_latest_4p5/rollout.mp4) · [逐回合指标](pilot_20261015/contact_only_latest_4p5/metrics.json)

<video controls preload="metadata" playsinline width="960" src="pilot_20261015/contact_only_latest_4p5/rollout.mp4"></video>

**组合消融，4.5 秒 8/24：** [视频](pilot_20261015/combined_latest_4p5/rollout.mp4) · [逐回合指标](pilot_20261015/combined_latest_4p5/metrics.json)

<video controls preload="metadata" playsinline width="960" src="pilot_20261015/combined_latest_4p5/rollout.mp4"></video>

不合格模型及视频保留为失败证据；它们不作为成功演示或模仿目标。正式长训仍在进行，当前级与后续结果以独立 CPU 严格评估为准。

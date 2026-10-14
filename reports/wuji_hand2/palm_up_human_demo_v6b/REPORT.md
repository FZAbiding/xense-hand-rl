# Wuji v6b 阶段评估快照（2026-10-14，第 42,500 轮）

**结论：** 360°/10 秒与 360°/6 秒策略在各自固定验证集上均为 24/24 严格通过；最新归档的 360°/3 秒策略为 **0/24，未通过**。这些是 CPU MuJoCo 物理验收结果，真人动作相似性仍待人工审阅。v6b 训练在本快照建立时仍在运行，本报告不是最终保留集结论。

本目录保存每个视频对应的独立模型、24 回合原始指标、逐帧指标、接触记录、轨迹和首帧。文件来源与 SHA256 见 [snapshot_manifest.json](snapshot_manifest.json)；[离线三视角评审页](review.html)可并排查看真人参考和候选动作。视频是实际闭环策略 rollout，画面中的 `V5` 文字沿用旧渲染器；评估任务、权重和指标均为 v6b。

| 阶段与证据 | CPU 严格通过 | 平均实际转角 | 平均角误差 | 平均回合时长 | 权重 |
| --- | ---: | ---: | ---: | ---: | --- |
| [10 秒整圈，固定验证](snapshot_20261014_42500/quality10/metrics.json)（第 1,000 轮） | 24/24 | 357.85° | 1.31° | 12.00/12 秒 | [model.pt](snapshot_20261014_42500/quality10/model.pt) |
| [6 秒整圈，固定验证](snapshot_20261014_42500/fast6_validation/metrics.json)（第 1,500 轮） | 24/24 | 356.20° | 1.78° | 8.00/8 秒 | [model.pt](snapshot_20261014_42500/fast6_validation/model.pt) |
| [6 秒整圈，额外未见种子诊断](snapshot_20261014_42500/fast6_unseen/metrics.json)（独立权重） | 23/24 | 351.53° | 1.85° | 7.84/8 秒 | [model.pt](snapshot_20261014_42500/fast6_unseen/model.pt) |
| [3 秒整圈，最新已评估快照](snapshot_20261014_42500/faster3_latest/metrics.json)（第 42,500 轮） | **0/24** | 352.10° | 1.90° | 4.10/5 秒 | [model.pt](snapshot_20261014_42500/faster3_latest/model.pt) |

6 秒的额外未见种子诊断权重 SHA256 为 `2c96285f...`，与固定验证所选权重 `c97c54d6...` **不同**；23/24 不能直接归给后者。额外诊断唯一失败回合提前结束，并触发漂移、接触和闲置指门槛。10 秒、6 秒通过物理门槛也不等于动作已获人工批准。

## 最新 3 秒模型：失败视频

[下载第 42,500 轮三视角视频](snapshot_20261014_42500/faster3_latest/rollout.mp4) · [逐帧指标](snapshot_20261014_42500/faster3_latest/frame_metrics.json) · [原始接触](snapshot_20261014_42500/faster3_latest/contacts.json) · [轨迹](snapshot_20261014_42500/faster3_latest/rollout.npz)

<video controls preload="metadata" playsinline width="960" src="snapshot_20261014_42500/faster3_latest/rollout.mp4"></video>

第 42,500 轮的 24 回合中，相向双指夹持和至少三次稳定交接均 **24/24 未达标**；双指尖接触 23/24 未达标，完整 5 秒 22/24 未达标，6/24 出现指序或反弯违规。一个回合可以同时触发多项失败，不能相加。接近 360° 的转角不能抵消脱手或换指失败；[早期逐帧诊断](../palm_up_human_demo_v7/v6b_failure_analysis.md)也记录了脱手后靠惯性继续转动的例子。

## 已达到的阶段与真人参考

- [10 秒整圈视频](snapshot_20261014_42500/quality10/rollout.mp4) · [模型](snapshot_20261014_42500/quality10/model.pt)
- [6 秒整圈固定验证视频](snapshot_20261014_42500/fast6_validation/rollout.mp4) · [模型](snapshot_20261014_42500/fast6_validation/model.pt)
- [6 秒整圈额外未见种子视频](snapshot_20261014_42500/fast6_unseen/rollout.mp4) · [对应独立模型](snapshot_20261014_42500/fast6_unseen/model.pt)
- [已批准的真人参考片段](snapshot_20261014_42500/human_A.mp4)：仅作动作审阅参考；未经时相或三维接触标定，不声称机器人与真人同步或等效。

三个机器人视频均保留同一时间步的多视角和彩色手指标识。人工审阅仍需检查食指/中指整条骨链、实际指腹接触位置、释放与交接，以及是否贴近批准的真人示范。下一阶段应先复核 6 秒候选，再针对提速后失夹持和收尾回夹设计独立实验；不得用放宽门槛替代改进动作。

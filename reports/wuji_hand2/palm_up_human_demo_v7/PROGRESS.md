# Wuji v7 正式长训阶段进度（更新至 2026-10-16）

三组短训的[同种子比较、失败视频与入选模型](PILOTS.md)已归档。正式训练从“只改课程”第 500 轮权重开始，主训练已完成约 23 小时有效 GPU 预算；连同短训约 1 小时，总计约 24 小时。流水线状态保存在 `runs/wuji_hand2/palm_up_human_demo_v7/pipeline_20261015/status.json`。以下区分固定验证、最终未见集和仍待完成的人工动作审阅。

| 阶段 | 检查点 | 固定验证种子严格通过 | 失败原因 | 状态 |
| --- | ---: | ---: | --- | --- |
| 360°/4 秒，目标后保持约 2 秒 | 第 100 轮 | **23/24** | 1 回合提前结束，并出现姿态/反弯与终点回夹失败 | 达到 ≥23/24，已进入 3.5 秒课程 |
| 360°/3.5 秒，目标后保持约 2 秒 | 第 500 轮 | **24/24** | 无严格失败 | 已达到 ≥23/24，进入 3 秒课程 |
| 360°/3 秒，目标后保持约 2 秒 | 第 1000 轮 | **23/24** | 1 回合提前结束，伴随姿态/反弯、自穿透、相向夹持和终点回夹失败 | 达到 ≥23/24；继续训练，最终保留种子与人工审阅待完成 |
| 360°/3 秒，历史最佳固定验证 | 第 3000 轮 | **24/24** | 无严格失败 | 截至第 6500 轮的最佳，平均角误差约 1.52° |
| 360°/3 秒，历史最近固定验证 | 第 6500 轮 | **23/24** | 1 回合提前结束，转角进度和终点回夹失败 | 第一次阶段归档时的最近结果 |
| 360°/3 秒，当前最佳固定验证 | 第 9000 轮 | **24/24** | 无严格失败 | 截至第 13000 轮的最佳，平均角误差约 0.55° |
| 360°/3 秒，历史最近固定验证 | 第 13000 轮 | **23/24** | 1 回合提前结束，姿态/反弯和终点回夹失败 | 2026-10-15 16:50 的阶段快照 |
| 360°/3 秒，严重退化证据 | 第 33000 轮 | **9/24** | 13 回合提前结束，双指尖接触 9、姿态/反弯 8、自穿透 6 回合失败 | 不合格，不能作为成功演示或后续模仿目标 |
| 360°/3 秒，历史退化验证 | 第 33500 轮 | **14/24** | 8 回合提前结束，相向夹持 9、终点回夹 9、自穿透 5 回合失败 | 不合格；最佳仍为第 9000 轮 |
| 360°/3 秒，本次归档的最新固定验证 | 第 40000 轮 | **20/24** | 4 回合未通过严格门槛，逐回合原因见原始指标 | 不合格；最佳仍为第 9000 轮 |
| 360°/3 秒，训练结束前固定验证 | 第 51000 轮 | **22/24** | 2 回合未通过严格门槛，逐回合原因见原始指标 | 不合格；最佳仍为第 9000 轮 |
| 360°/3 秒，最后权重最终未见集 | 第 51074 轮 | **23/24** | 第 23 回合提前结束，未达到转角、接触和终点回夹等门槛 | 仅为最后权重的物理终评；自然动作待人审 |

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

第 9000 轮更新了 3 秒任务的**最佳固定验证模型**：[视频](progress_20261015/turn_3s_best_iteration_9000/rollout.mp4) · [逐回合指标](progress_20261015/turn_3s_best_iteration_9000/metrics.json) · [接触记录](progress_20261015/turn_3s_best_iteration_9000/contacts.json) · [模型权重](progress_20261015/turn_3s_best_iteration_9000/model.pt)。24/24 严格通过、平均角误差约 0.55°，模型 SHA256 为 `98ad3e008ad1100e29550662acdb975eae4d4d062cdb1a91db7c51169904b9a4`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_best_iteration_9000/rollout.mp4"></video>

第 13000 轮是 2026-10-15 16:50（北京时间）归档时的**最近评估模型**：[视频](progress_20261015/turn_3s_latest_iteration_13000/rollout.mp4) · [逐回合指标及失败标记](progress_20261015/turn_3s_latest_iteration_13000/metrics.json) · [接触记录](progress_20261015/turn_3s_latest_iteration_13000/contacts.json) · [模型权重](progress_20261015/turn_3s_latest_iteration_13000/model.pt)。它为 23/24；模型 SHA256 为 `e16878bdbc36a0758ea4d7f5d221db1ffef6483238367638adb3ade3e483468e`。最近第 11500 轮曾降到 22/24，说明波动尚未完全消失。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_latest_iteration_13000/rollout.mp4"></video>

截至 2026-10-16 02:52（北京时间），训练仍在 3 秒单圈课程，已用约 16.11/23 小时主训练预算。第 25500–33500 轮的 17 次固定验证平均约 17.82/24，通过数范围为 9–23，**没有一次达到 24/24**。训练继续并不改变已归档的第 9000 轮最佳权重；最终未见种子尚未评估。

**第 33000 轮严重失败：** [视频](progress_20261015/turn_3s_failure_iteration_33000/rollout.mp4) · [逐回合失败门槛](progress_20261015/turn_3s_failure_iteration_33000/metrics.json) · [接触记录](progress_20261015/turn_3s_failure_iteration_33000/contacts.json) · [模型权重](progress_20261015/turn_3s_failure_iteration_33000/model.pt)。9/24 严格通过，模型 SHA256 为 `bb7cc95e97041f4d80c7c9479a88258c3db465839bd73c34a0c16c53588c915d`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_failure_iteration_33000/rollout.mp4"></video>

**第 33500 轮最近失败：** [视频](progress_20261015/turn_3s_latest_iteration_33500/rollout.mp4) · [逐回合失败门槛](progress_20261015/turn_3s_latest_iteration_33500/metrics.json) · [接触记录](progress_20261015/turn_3s_latest_iteration_33500/contacts.json) · [模型权重](progress_20261015/turn_3s_latest_iteration_33500/model.pt)。14/24 严格通过，模型 SHA256 为 `adc5d5b5bab34e19bc2766fdc4adbbc24e6e5a2ca99c5531d3282e421cdce33c`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_latest_iteration_33500/rollout.mp4"></video>

**第 40000 轮阶段快照：** [同步三视角视频](progress_20261015/turn_3s_latest_iteration_40000/rollout.mp4) · [逐回合指标及失败标记](progress_20261015/turn_3s_latest_iteration_40000/metrics.json) · [逐帧指标](progress_20261015/turn_3s_latest_iteration_40000/frame_metrics.json) · [接触记录](progress_20261015/turn_3s_latest_iteration_40000/contacts.json) · [原始轨迹](progress_20261015/turn_3s_latest_iteration_40000/rollout.npz) · [模型权重](progress_20261015/turn_3s_latest_iteration_40000/model.pt)。固定验证为 20/24；模型 SHA256 为 `6e2079056724ab9d3d6aa6c1069890d7ad50c01b191f127ccf7a197cd69f8c4a`。这是当前阶段归档的最新完成评估，不是合格演示。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_latest_iteration_40000/rollout.mp4"></video>

**第 9000 轮最佳模型的独立轻微水平扰动诊断：** 使用预留种子 `2026101502`、每 2 秒施加 0.1 秒水平外力、每轴最大 0.02 N，24 回合严格通过 **23/24**；唯一失败回合提前结束，并失去终点回夹。[视频](progress_20261015/turn_3s_best9000_push/rollout.mp4) · [逐回合指标](progress_20261015/turn_3s_best9000_push/metrics.json) · [接触记录](progress_20261015/turn_3s_best9000_push/contacts.json)。诊断对应的模型 SHA256 仍为 `98ad3e008ad1100e29550662acdb975eae4d4d062cdb1a91db7c51169904b9a4`；最终保留种子尚未使用。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_best9000_push/rollout.mp4"></video>

退化分析和待确认的收尾方案见 [NEXT_STAGE_PLAN.md](NEXT_STAGE_PLAN.md)。

2026-10-16 约 06:08（北京时间）原流水线全部进程意外退出，日志未记录正常完成或明确退出原因。到退出前主训练累计约 19.37/23 小时；没有 `STOP` 文件，终评尚未开始。10:55 在确认进程和锁均不在、检查点及源码清单存在后，恢复**同一**流水线与原预算；训练器从第 40400 轮检查点接续，重新计算未保存的约 90 轮。新训练进程 PID 2452708，心跳与迭代随后正常更新。原定预算没有提前停止；中断的空闲时间不计入有效 GPU 训练。中断原因仍未查明。

## 24 小时预算完成后的结果

流水线和监督器均已正常结束：第 51074 轮，主训练有效时间 `82797.858` 秒（约 23 小时），监督器本次恢复后重启数为 0。**第 9000 轮仍是按固定验证预选的最佳权重**，不能依据最终未见集反向换模。

训练结束前第 51000 轮固定验证为 **22/24**：[三视角视频](progress_20261015/turn_3s_final_validation_iteration_51000/rollout.mp4) · [逐回合指标与失败标记](progress_20261015/turn_3s_final_validation_iteration_51000/metrics.json) · [逐帧指标](progress_20261015/turn_3s_final_validation_iteration_51000/frame_metrics.json) · [接触记录](progress_20261015/turn_3s_final_validation_iteration_51000/contacts.json) · [原始轨迹](progress_20261015/turn_3s_final_validation_iteration_51000/rollout.npz) · [评估权重](progress_20261015/turn_3s_final_validation_iteration_51000/model.pt)。权重 SHA256 `cbfd24c7ab26c2762f1bd35d7569991ab2927b522576b980cdb0c640325170db`。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_final_validation_iteration_51000/rollout.mp4"></video>

现有收尾器按原定流程对**最后权重**第 51074 轮运行最终未见种子 `2026101503`，严格通过 **23/24**：[三视角视频](progress_20261015/turn_3s_final_holdout_iteration_51074/rollout.mp4) · [逐回合指标与失败标记](progress_20261015/turn_3s_final_holdout_iteration_51074/metrics.json) · [逐帧指标](progress_20261015/turn_3s_final_holdout_iteration_51074/frame_metrics.json) · [接触记录](progress_20261015/turn_3s_final_holdout_iteration_51074/contacts.json) · [原始轨迹](progress_20261015/turn_3s_final_holdout_iteration_51074/rollout.npz) · [**该次评估的精确权重**](progress_20261015/turn_3s_final_holdout_iteration_51074/model.pt) · [训练终态](progress_20261015/turn_3s_final_holdout_iteration_51074/training_status.json) · [源码哈希清单](progress_20261015/turn_3s_final_holdout_iteration_51074/source_manifest.json)。评估权重 SHA256 `74aae3bf2a2e953ab069ddbcf2a3a4fb186398302a5e117c2b19cda59c5b501d`。第 23 回合在 2.9 秒提前结束，实际约 200.8°，目标进度约 288°；同时出现漂移、自穿透、双指尖和相向接触、交接及终点回夹门槛失败。不能把 23/24 表述为全回合成功。

<video controls preload="metadata" playsinline width="960" src="progress_20261015/turn_3s_final_holdout_iteration_51074/rollout.mp4"></video>

## 阶段 A：预选模型最终未见集与接触位置复核

用户确认后，第 9000 轮预选权重在同一最终未见种子 `2026101503` 上按**原 v7 严格指标 24/24**，全部回合运行满 5.5 秒。对应[视频](stage_a_20261016/holdout_best9000/rollout.mp4)、[逐回合指标](stage_a_20261016/holdout_best9000/metrics.json)、[接触记录](stage_a_20261016/holdout_best9000/contacts.json)和[原始轨迹](stage_a_20261016/holdout_best9000/rollout.npz)已归档，精确权重仍是上述 SHA256 `98ad3e008ad1100e29550662acdb975eae4d4d062cdb1a91db7c51169904b9a4`。最后权重的 23/24 终评仍作为单独对照，未用于选模。

<video controls preload="metadata" playsinline width="960" src="stage_a_20261016/holdout_best9000/rollout.mp4"></video>

**接触部位问题：** 原训练和评估将手指末节本体与指尖传感区域合并计入“指尖”力。精确重放全部 24 回合后，只计指尖传感区域的双接触帧比例每回合仅 **75.0%–78.2%**；若沿用 95% 门槛，则 **0/24**。末节本体承力超过 0.05 N 的帧比例为每回合 24.5%–30.0%。详见[逐回合审计、全部接触点及离线真人并排审阅页](stage_a_20261016/REVIEW.md)。原 24/24 只说明**原 v7 门槛通过**，不能证明真实指腹接触或自然 Triangle Pass 已通过。真人动作一致性仍**待用户审阅**；60 秒连续训练尚未启动。[下一阶段计划](NEXT_STAGE_PLAN.md)已按此结果修订。

视频为闭环策略的同步三视角；真人参考与训练前诊断见 [BASELINE.md](BASELINE.md)。严格物理通过率与人工动作相似性分开记录，后者仍**待审阅**。主训练配置不含 60 秒连续课程；须先完成 3 秒动作的人工审阅，再单独建立连续训练版本。最终未见种子只在选模结束后使用。

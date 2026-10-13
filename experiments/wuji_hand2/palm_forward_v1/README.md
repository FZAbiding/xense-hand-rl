# Wuji Hand 2 竖直、掌心朝前实验 v1

此实验独立于掌心朝下与掌心朝上的历史实验。底座网格最低点位于桌面 Z=0；官方手指伸展方向为世界 +Z，掌心法线为世界 -X。只共用训练、接触计量和评估代码，不读取历史场景、抓握库或权重。

入口配置为 `config.json`。抓握搜索的参数分别保存在 `calibration_search_v2.json`、`calibration_search_v3.json` 和 `control_refinement.json`。失败搜索和认证分别保留，最终用静态控制预紧量搜索出的候选做认证。`control_refinement_v5.json` 是未执行的备用配置，不属于本次成果。

新资产：`assets/wuji_hand2_palm_forward_v1/`。新训练及评估：`runs/wuji_hand2/palm_forward_v1/`。正式归档：`artifacts/wuji_hand2/palm_forward_v1/`。

在全新目录中运行：

```bash
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/build.py
.venv-warp/bin/python experiments/wuji_hand2/calibrate.py \
  --orientation palm_forward_v1 \
  --search-config experiments/wuji_hand2/palm_forward_v1/calibration_search_v2.json \
  --bank-name grasp_search_v2.npz --max-candidates 729 --settle-seconds 12
.venv-warp/bin/python experiments/wuji_hand2/certify_grasps.py \
  --orientation palm_forward_v1 --source-bank grasp_search_v2.npz \
  --seconds 60 --perturbations 4 --seed 20261009
```

以上是会失败的初轮复现。最终可用抓握的复现继续执行：

```bash
.venv-warp/bin/python experiments/wuji_hand2/calibrate.py \
  --orientation palm_forward_v1 \
  --search-config experiments/wuji_hand2/palm_forward_v1/calibration_search_v3.json \
  --bank-name grasp_search_v3.npz --max-candidates 729 --settle-seconds 12 \
  --save-settled --retain-grasps 32
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/refine_controls.py
.venv-warp/bin/python experiments/wuji_hand2/certify_grasps.py \
  --orientation palm_forward_v1 --source-bank grasp_controls_v4.npz \
  --report-name controls_v4_60s_certification.json \
  --seconds 60 --perturbations 4 --seed 20261009
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/run.py
```

构建、训练和评估拒绝覆盖已有资产或同名结果。只有 `grasp_version=palm_forward_v1` 能读取新抓握库，历史权重因版本与资产校验不能恢复到新实验。

最终静态认证在 32 个候选中选出 4 个（原候选索引 3、4、24、26），每个都通过无扰动加 4 次扰动的 60 秒验收。认证器对本布局使用与训练/CPU评估一致的 8 次求解迭代，`certified_bank_manifest.json` 固定认证报告与场景/抓握库 SHA-256，训练会核对这些签名。

任务仍为相同 180 mm × 18 mm、25 g 自由笔杆，在水平面内往复转动。课程从 ±0.2 rad/6 s 开始，目标为 ±0.6 rad/4 s；从小幅长回合开始，只有验证种子下 8×60 秒严格全通过，才扩大幅度。命令裁剪惩罚为 2.0，原实验为 0.2，因此这是同任务验收，非完全相同训练超参数的控制变量实验。

启动实验留出评估采用种子 20261011，24×60 秒，不自动复位。所有失败指标仍公开；小幅阶段的结果不能当成最终同任务对比。启动与中幅实验已生成的视频为前 12 秒，最终共同任务视频覆盖完整 60 秒。

启动实验的小幅留出结果为 23/24 持笔、0/24 严格通过。独立的 `matched_task_config.json` 从本布局的小幅第 500 轮检查点继续训练，探索标准差上限降为 0.1，课程扩大到 ±0.35 rad/6 s 与 ±0.6 rad/4 s。此配置按验证集全回合持笔/工作空间存活决定课程推进，严格质量门槛保持不变；验证种子为 20261015，最终三手共同留出种子为 20261016。目录分别为 `runs/wuji_hand2/palm_forward_v1/matched_task/` 与 `artifacts/wuji_hand2/palm_forward_v1/matched_task/`，不会覆盖启动实验。

```bash
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/run.py \
  --config experiments/wuji_hand2/palm_forward_v1/matched_task_config.json
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/archive.py \
  --config experiments/wuji_hand2/palm_forward_v1/matched_task_config.json
```

中幅验证最佳模型为第 700 轮（6/8 持笔，0/8 严格通过），没有满足该配置的课程推进门槛。为完成最终同任务失败基准，`final_task_config.json` 单独从它训练 ±0.6 rad/4 s 任务，探索标准差上限为 0.05；验证种子 20261017，最终共同留出种子 20261018。新输出目录为 `runs/wuji_hand2/palm_forward_v1/final_task/` 和 `artifacts/wuji_hand2/palm_forward_v1/final_task/`，其结果不覆盖前两阶段。

```bash
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/run.py \
  --config experiments/wuji_hand2/palm_forward_v1/final_task_config.json
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/archive.py \
  --config experiments/wuji_hand2/palm_forward_v1/final_task_config.json
```

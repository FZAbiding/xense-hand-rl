# Revo3-V2.0 实验

V1.0 已原样归档至 `artifacts/revo3/v1.0/`，小幅策略在其 `small/` 子目录；LEAP 不变。
V2.0 代码、资产、训练日志分别位于 `experiments/revo3_v2/`、`assets/revo3_v2/`、`runs/revo3_v2/`。
通过严格验收前，不将候选检查点命名为最终成功策略。小幅650轮和中幅1000轮已分别归档至 `artifacts/revo3/v2.0/small/`、`medium/`，两者均通过CPU 12×60s、新扰动种子24×60s、GPU 128×60s。后续运行状态见 `artifacts/revo3/v2.0/experiment.json`。

无名指/小拇指稳定性的具体改进、控制公式、运动数据和归因边界见 [改进说明](../../reports/revo3_v2/IMPROVEMENTS.md)。

后续已完成±0.6rad/6s（1400轮）和±0.6rad/4s（1700轮），两阶段均通过CPU12×60s、新扰动种子CPU24×60s及GPU128×60s；分别归档至 `wide_6s/` 和 `wide_4s/`。详情见 [大幅与提速记录](../../reports/revo3_v2/WIDE_STAGE.md)；最终同任务对比见 [V1/V2数据](../../reports/revo3_v2/matched_task_1700/README.md)。

## 已完成的实验与当前证据

完整参数、分项 loss、所有失败及成功评估见 [自动数据表](../../reports/revo3_v2/DATA.md)；原始 JSON/轨迹保留在每个 run 目录。

| 阶段 | 累计 PPO 轮数 | 新增环境步 | 设置变化 | 独立验证结果 |
|---|---:|---:|---|---|
| 从头先导 | 250 | 8,192,000 | 新抓姿/动作/观测/奖励集成 | CPU 12×12s 不掉笔，但关节裁剪 29.67%，严格通过 0/12 |
| 裁剪惩罚续训 | 400 | 4,915,200 | command clip penalty 0.2→3 | CPU 12×12s 通过 8/12；12×60s 仅 1/12，通过短时不代表长时稳定 |
| 长回合续训 | 650 | 8,192,000 | episode 12→60s，其余任务/奖励不变 | CPU 12×60s、另一扰动种子 24×60s、GPU 128×60s 全部通过 |
| 中幅课程 | 900 | 8,192,000 | amplitude 0.2→0.35，period 仍为6s | GPU 128/128通过，但CPU 8/12，第三抓姿51–57s峰值不足 |
| 中幅精度分支 | 1050 | 4,915,200 | tracking sigma 0.18→0.10，任务与物理不变 | 最后1050轮有掉笔；选1000轮，CPU 12/12 + 新扰动24/24、GPU128/128均通过60s |

小幅策略累计 21,299,200 个环境步，网络训练种子仍只有 1 个。三个验证种子是初始状态扰动种子，**不是三个独立训练种子**。
第 650 轮在 CPU 12×60s 的 MAE=0.273°、平均倾斜=0.429°、命令裁剪=0.871%、法向力标量和=0.614N；无测得的非远端指节或掌心承载帧。

先导→裁剪惩罚分支还包含额外优化和学习率重启，不能把改善全部归因于惩罚系数。长回合分支保持动作/物理/奖励，学习率从上一末值延续，但仍需等预算短回合对照才能作严格因果判断。
长时失败轨迹证据：第400轮在0–12s的MAE为0.70°、裁剪1.76%；24–36s升至2.39°、6.72%，之后逐渐失去接触并掉落。这与短回合未覆盖的控制状态漂移相符，不是“硅胶已被证明是问题”。

当前结论仅适用于刚体仿真、三组抓姿和小范围初始扰动。未完成软硅胶辨识、质量/摩擦/时延随机化、独立训练种子复现或真机验收。

中幅精度分支第1050轮在CPU中有2/12掉笔、GPU中有13/128掉笔，多发生于4–10s；即使训练loss更低，也不作为交付策略。一次针对1050轮的24回合CPU测试异常退出（退出码139，无完整报告），没有计入任何通过率，其原因尚未确定。所有已完成失败测试均保留。

## 第一组：集成先导实验

`runs/revo3_v2/pilot_quality_seed1`，从头训练，不加载 V1.0 权重。
这是一组集成可行性实验，联合验证新抓姿、增量动作、质量奖励和 PPO 稳定性修正；**不能据此单独归因某项改动的收益**。后续受控消融须在相同观测、预算和种子设置下逐项比较。

- 沿用 V1.0 官方 MJCF 的物理参数、碰撞、质量、关节轴和限位，没有人为扩大运动范围，也没有假设硅胶材料参数。
- 从 96 组候选中选出 3 个抓姿。控制目标限位余量至少约 3.6°，静态两秒测试末段的平均倾斜约 3.6–6.6°，无测得的非远端指节承载。
- “指尖”目前仍按远端刚体归类，尚未细分真实软指腹接触区域；这是待校核项，不宣称已经实现硅胶层。
- 21 维动作：相对上一位置目标的增量，按关节范围缩放，最大约 0.012–0.035 rad/控制步；环指、小指尺度再减半，仍保留全部关节及碰撞。
- 目标命令保留软限位余量，记录实际裁剪率；不以隐藏裁剪来通过质量验收。
- 109 维仿真特权观测：原 81 维 + 21 维当前目标状态 + 7 维接触法向力统计。
- 新奖励明确鼓励水平姿态；不持续奖励第 3 个接触；惩罚非指尖受力、掌心受力和过大的法向力标量和。
- 第一期为 ±0.2 rad（±11.5°）、6 秒周期，尚不能与 V1.0 的 ±34.4°直接比较绝对误差。
- 1,024 个并行环境、H=32，计划 250 轮，即 8,192,000 个环境步；训练种子 1。
- 初始学习率 1e-4，KL 驱动调整；完整采样批次 KL 检查，异常更新回退。优化时 reward 缩放 0.25，日志中的 episode return 和 reward 分项保留原始尺度。
- 每轮保存的日志含 policy/value/entropy、完整批次 KL、clip fraction、explained variance、分项 reward、倾斜、接触力和命令裁剪。`clip_fraction` 是 PPO 比例裁剪；`command_clip_fraction` 是关节命令裁剪，两者不能混用。
- 每个训练目录保存代码/资产哈希与快照，后续改动不覆盖历史实验。

## 命令

```bash
# 首次生成抓姿；已有 bank 时会拒绝覆盖，避免使权重对应的环境失效。
.venv-warp/bin/python experiments/revo3_v2/calibrate.py

.venv-warp/bin/python experiments/revo3_v2/train.py --run runs/revo3_v2/pilot_quality_seed1 --num-envs 1024 --iterations 250 --amplitude .2 --period 6 --seed 1

.venv-warp/bin/python experiments/revo3_v2/evaluate.py --checkpoint runs/revo3_v2/pilot_quality_seed1/checkpoint_000250.pt --output runs/revo3_v2/eval_pilot250 --episodes 12 --seconds 12
```

训练入口拒绝覆盖已有日志目录。恢复训练必须使用新的 `--run`，通过 `--resume` 指定 V2.0 检查点；不能直接恢复 V1.0 的 81 维策略。
`--video` 会生成 CPU 闭环轨迹的视频；GPU 训练结束后再启用渲染，避免并发 GPU 作业。

## 统一严格验收

每回合同时满足：

- 不掉笔、不离开工作空间；测试不自动复位。
- 平均 yaw 误差≤3°；每个完整半周期实际峰值达到目标幅度至少 90%。
- 平均倾斜≤10°、倾斜 P95≤15°。
- 关节目标裁剪比例<5%。
- 非指尖法向力>0.05 N 的帧<5%；掌心法向力>0.05 N 的帧<1%。
- 至少两个远端指节各有>0.02 N 接触力的帧>80%。

这些力阈值是本实验操作定义，不是已辨识的实物材料/安全标准。法向力标量和不是合力。
物理掉落（中心高度低于 15 mm 或位移超过 75 mm）与训练工作空间失效（45 mm 位移、25 mm 高度下降或40°倾斜）分别报告，避免把姿态超界混称为掉笔。

## 后续顺序

1. 独立 CPU 验证中间检查点，查明尚未通过的质量项；达标前不升级幅度。
2. 小幅、中幅、大幅6s及大幅4s的60秒验收均已通过；下一步做独立训练种子复现和更广扰动验证，不把单训练种子的名义仿真成功当作真机可靠性证明。
3. 按控制变量原则比较抓姿、动作、奖励的消融，不把集成先导实验当成因果证明。
4. 确認硬件版本、零位、触觉及软指腹配置，再做材料/执行器辨识；目前不启动真机操作。

分析依据见 [V1.0 对比诊断](../../reports/hand_comparison/REPORT.md)。

## 长回合、GPU 验收与分阶段归档

```bash
.venv-warp/bin/python experiments/revo3_v2/train.py --run runs/revo3_v2/long60_seed1 --resume runs/revo3_v2/clip_penalty3_seed1/checkpoint_000400.pt --iterations 250 --episode-seconds 60 --command-clip-penalty 3 --lr .000056723625
.venv-warp/bin/python experiments/revo3_v2/evaluate_gpu.py --checkpoint runs/revo3_v2/long60_seed1/checkpoint_000650.pt --output runs/revo3_v2/eval_long650_gpu60s --num-envs 128 --seconds 60
.venv-warp/bin/python experiments/revo3_v2/render.py --evaluation runs/revo3_v2/eval_long650_60s
.venv-warp/bin/python experiments/revo3_v2/summarize.py
```

示例训练/评估目录已经存在；重跑时使用新目录，避免覆盖。`render.py` 从保存的 CPU 轨迹渲染完整60秒，默认20fps、真实时间播放，不拼接或复位。
CPU/GPU共用同一质量门槛函数。`package.py` 要求 CPU、GPU、另一扰动种子各至少60秒、严格通过率≥95%且全部存活/保持工作空间，才创建阶段包；拒绝覆盖旧包。包内的 `evaluator_*.py` 是审计快照，运行入口仍在 `experiments/revo3_v2/`。

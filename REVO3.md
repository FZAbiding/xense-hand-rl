# 强脑 Revo3-V1.0 指尖往复转笔 policy

本页记录已归档的 V1.0；新实验见 [Revo3-V2.0](experiments/revo3_v2/README.md)。V1.0 的旧成功判据不等于 V2.0 的动作质量验收。

使用强脑官方 Revo3 右手资产，从头训练 PPO；不是将 LEAP 权重直接套在 Revo3 上。
Revo3 策略有 21 维动作和 81 维状态观测，LEAP 原来的 16 维动作策略保持不变。

## 最终交付

`artifacts/revo3/v1.0/policy.pt`：目标幅度 **±0.6 rad（±34.4°）**，周期 **4 秒**。
选择 `runs/revo3_medium/checkpoint_000550.pt`：先 300 轮小幅学习，再 250 轮中幅微调，共 **18,022,400 个环境步**；不含失败分支与丢弃的后续小幅训练。

| 独立测试 | 成功率 | 平均角度误差 | 至少两根指尖真实接触的时间占比 |
|---|---:|---:|---:|
| GPU，128 次 × 12 秒，种子 777 | 100% | 3.31° | 99.93% |
| CPU，32 次 × 60 秒，新种子 20260922 | 100% | 3.37° | 99.99% |

GPU 每次完成 6 次正负交替阈值穿越；CPU 每次完成 30 次，平均 2.93 根指尖接触。GPU 平均实际角度跨度约 70.3°，正反向峰值并非完全对称。

- [最终演示视频](artifacts/revo3/v1.0/demo.mp4)
- [最终策略权重](artifacts/revo3/v1.0/policy.pt)
- [角度跟踪曲线](artifacts/revo3/v1.0/tracking.png)
- [完整评估报告](artifacts/revo3/v1.0/results.json)
- [接触几何审查](artifacts/revo3/v1.0/contact_audit.json)

这不是全程仅指尖接触的策略：对 GPU 保存的前 4 条轨迹、共 1,920 帧进行 CPU 接触几何复核，约 38.4% 的帧还接触其他指节或掌心，其中掌心接触约 8.0%。主要夹持接触仍为拇指、食指、中指远端。该审查不是接触力测量，也不是新的独立动力学实验。

## 已验证的小幅基线

`artifacts/revo3/v1.0/small/policy.pt`：目标幅度 ±0.35 rad（±20.1°），周期 4 秒。

| 独立测试 | 成功率 | 平均角度误差 | 至少两根指尖真实接触的时间占比 |
|---|---:|---:|---:|
| GPU，128 次 × 12 秒 | 100% | 2.18° | 99.81% |
| CPU，32 次 × 60 秒 | 100% | 1.85° | 99.97% |

对应 `runs/revo3_fingertip/checkpoint_000300.pt`，经历 300 轮、9,830,400 个环境步。
这里的 100% 是有限样本结果，不代表任意初始条件和扰动下的可靠性。
训练继续到后期曾出现性能退化，因此检查点根据独立测试选择，而非直接使用 latest。

- [小幅演示视频](artifacts/revo3/v1.0/small/demo.mp4)
- [小幅策略权重](artifacts/revo3/v1.0/small/policy.pt)
- [小幅评估报告](artifacts/revo3/v1.0/small/results.json)
- [论文、模型来源与失败实验记录](REVO3_RESEARCH.md)

## 播放与复测

```bash
.venv-warp/bin/python play.py --checkpoint artifacts/revo3/v1.0/policy.pt
.venv-warp/bin/python evaluate.py --checkpoint artifacts/revo3/v1.0/policy.pt --output runs/revo3_recheck_gpu --num-envs 128 --seconds 12 --video
.venv-warp/bin/python evaluate_cpu.py --checkpoint artifacts/revo3/v1.0/policy.pt --output runs/revo3_recheck_cpu --episodes 32 --seconds 60 --seed 20260922
```

`play.py` 打开 CPU MuJoCo 实时窗口；无桌面环境用 `evaluate.py --video`。
为避免 GPU 训练与渲染并发的潜在冲突，GPU 作业建议串行；CPU 测试可以与 GPU 训练并行。

## 从头复现

环境安装见根目录 [README.md](README.md)，依赖版本见 `requirements.txt`。

```bash
.venv-warp/bin/python bootstrap_revo3.py
.venv-warp/bin/python calibrate_revo3.py
.venv-warp/bin/python validate_assets.py
.venv-warp/bin/python train.py --robot revo3 --objective fingertip --run runs/new_revo3_small --num-envs 1024 --iterations 300 --amplitude .35
.venv-warp/bin/python train.py --robot revo3 --objective fingertip --run runs/new_revo3_medium --resume runs/new_revo3_small/checkpoint_000300.pt --num-envs 1024 --iterations 250 --amplitude .6 --lr .0001
.venv-warp/bin/python evaluate.py --checkpoint runs/new_revo3_medium/checkpoint_000550.pt --output runs/new_revo3_gpu --num-envs 128 --seconds 12 --video
.venv-warp/bin/python evaluate_cpu.py --checkpoint runs/new_revo3_medium/checkpoint_000550.pt --output runs/new_revo3_cpu --episodes 32 --seconds 60 --seed 20260922
```

资产固定到官方 commit `f332a6f0dc944e26b82976b637074b03f7ee8a2c`。保留原始网格、关节轴、限位、质量和惯量；结构回归测试会核对这些字段。浮点接触计算存在平台差异，复训应重新验收。
已有权重依赖当前 `assets/revo3/grasp.npz` 与 `scene.xml`，使用权重时不要重新校准后混用。
打包结果中的 `task_manifest.json` 记录相应任务文件哈希。

## 控制和验收约定

每 25 ms 执行一次策略；每个控制步含 5 个 5 ms 的接触动力学子步。
动作是相对已校准抓姿的关节位置残差：`target = clip(home + 0.65 * clip(action, -1, 1), joint_limits)`。
21 个动作的关节顺序见 `assets/revo3/provenance.json`。
81 维输入依次为：21 维关节位置误差、21 维关节速度、3 维笔位置误差、3 维笔轴、6 维笔速度、6 维角度/目标/相位、21 维上一动作；缩放与归一化以 `pen_env.py` 和 checkpoint 为准。

成功必须同时满足：

- 全程不掉笔；笔中心高度不能低于初始高度 18 mm，空间偏移不超过 45 mm。
- 每周期实际依次穿越正负目标幅度的 60%，12 秒至少 6 次，60 秒至少 30 次。
- 平均角度绝对误差小于 0.25 rad。
- 至少两个指尖在超过 80% 的帧靠近笔轴段（24 mm 内），且至少两根指尖在超过 80% 的帧有真实物理接触。

CPU 评估逐步运行网络与动力学，不是回放，测试中没有自动复位。初始关节噪声为 0.005 rad，笔位置噪声为 0.4 mm；训练种子 1，GPU 测试种子 777，小幅与中间检查点 CPU 测试种子 901，最终中幅 CPU 验收使用新种子 20260922。

## 范围与限制

笔是自行构建的 180 mm 长、18 mm 直径、25 g 训练杆，自由刚体、正常重力，无辅助外力或绑定。手掌固定，开始时已夹住笔。控制增益和摩擦参数尚未做真机辨识。
当前任务为仿真中指尖夹持的往复旋转，不包含从桌面抓取、360° finger-pass 或真机部署。
观测使用仿真提供的物体状态，不能直接接上硬件使用。原小红书视频未能读取，不宣称逐帧复现参考视频。
官方 Revo3 资产在该 commit 尚无明确许可证；详见 [NOTICE.md](NOTICE.md)。

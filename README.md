# Xense Hand RL

面向多种灵巧手与操作任务的强化学习项目。

项目原创代码与内容采用 [MIT License](LICENSE)。第三方代码与资产保留各自许可证，具体来源和限制见 [NOTICE.md](NOTICE.md)。

## 当前案例：LEAP / Revo3 指尖往复转笔

**新增强脑 Revo3 策略：** 独立的 21 关节训练、真实指尖接触验收和播放方式见 [REVO3.md](REVO3.md)。两种手的资产与策略分别保存，下面继续记录原 LEAP 案例。

已完成从公开手模型下载、笔 3D 建模、初始抓姿搜索、PPO 训练到独立物理评估。
最终策略使笔在指尖绕掌面法向做 **±57.3°、周期 4 秒** 的往复旋转。
笔为自由刚体，运动由手指接触产生。手掌固定，策略直接控制 16 个手指关节。

- [演示视频](artifacts/leap/demo.mp4)
- [角度跟踪曲线](artifacts/leap/tracking.png)
- [训练曲线](artifacts/leap/learning_curves.png)
- [最终权重](artifacts/leap/policy.pt)
- [实验结果与原始数据入口](artifacts/leap/results.json)
- [论文阅读及实现选择](RESEARCH.md)

## 成果目录

`artifacts` 按手型归档；Revo3 进一步按实验版本及课程阶段归档，保留旧权重。

```text
artifacts/
├── leap/              # LEAP 最终策略、视频、评估与校验清单
└── revo3/
    ├── v1.0/          # 已归档的 Revo3 V1.0 成果
    │   └── small/     # V1.0 小幅基线
    └── v2.0/          # V2.0 实验状态与通过验收后的成果
        ├── small/     # ±11.5°、6s 周期，CPU/GPU 60s 严格验收通过
        ├── medium/    # ±20.1°、6s 周期，CPU/GPU 60s 严格验收通过
        ├── wide_6s/   # ±34.4°、6s 周期，CPU/GPU 60s 严格验收通过
        └── wide_4s/   # ±34.4°、4s 周期，CPU/GPU 60s 严格验收通过
```

## 已测结果

Revo3 V2.0 正在独立实验，进度与命令见 [V2.0 实验记录](experiments/revo3_v2/README.md)。V1.0 权重未被覆盖。
V2 小幅与中幅阶段均已通过 CPU 12+24 回合、GPU 128 环境的60秒验证：[中幅完整一分钟视频](artifacts/revo3/v2.0/medium/demo_60s.mp4)、[V2实验报告](reports/revo3_v2/REPORT.md)、[参数与分项 loss 数据](reports/revo3_v2/DATA.md)。以下表格仍为原 LEAP 验收标准，不与 V2 严格成功率直接混用。

V2后续大幅提速也已通过同样规模长时验收：**±34.4°、4s周期，第1700轮**。[最新完整视频](artifacts/revo3/v2.0/wide_4s/demo_60s.mp4)、[改进点说明](reports/revo3_v2/IMPROVEMENTS.md)、[同幅同速V1/V2对比](reports/revo3_v2/matched_task_1700/README.md)。未验证真机或独立训练种子复现。

| 控制方式 | 幅度 | 测试 | 成功率 | 平均角度误差 |
|---|---:|---|---:|---:|
| 静止抓姿 | ±34.4° | GPU，64 次 × 12 秒 | 0% | 58.3° |
| 运动学参考、无学习 | ±34.4° | GPU，64 次 × 12 秒 | 0% | 47.0° |
| PPO，第 550 轮 | ±34.4° | GPU，128 次 × 12 秒 | 100% | 1.66° |
| 同上 | ±34.4° | GPU，64 次 × 60 秒 | 100% | 1.82° |
| 同上 | ±34.4° | CPU，32 次 × 12 秒 | 100% | 1.73° |
| **最终 PPO，第 800 轮** | **±57.3°** | **GPU，128 次 × 12 秒** | **100%** | **2.58°** |
| **最终 PPO，第 800 轮** | **±57.3°** | **CPU，32 次 × 60 秒** | **100%** | **2.80°** |

这些是有限样本的实测成功率，不是任意扰动下的可靠性保证。CPU 测试每步重新执行神经网络动作和接触动力学，不是轨迹回放；最终策略平均实际接触笔的手指数为 2.48。

评估使用训练外种子（GPU 777/2026，CPU 901），关节初始噪声标准差 0.005 rad，笔初始位置噪声标准差 0.4 mm。评估中不自动复位。
成功要求：全程不掉落、每周期实际依次跨过正负 60% 幅度阈值、平均绝对角度误差小于 0.25 rad、至少两个指尖在超过 80% 的时刻位于笔轴段 24 mm 内。
12 秒要求 6 次交替阈值穿越，60 秒要求 30 次。指尖距离只是评估代理；CPU 报告另外记录真实接触手指数。

## 在当前机器播放

在项目目录执行：

```bash
.venv-warp/bin/python play.py
```

默认打开 MuJoCo 实时窗口，运行已训练策略 120 秒。关闭窗口即可退出，掉笔会停止且不自动复位。播放器运行 CPU 动力学，训练使用 RTX 4090。

无桌面时生成视频并重新评估：

```bash
.venv-warp/bin/python evaluate.py --checkpoint artifacts/leap/policy.pt --output runs/recheck --num-envs 128 --video
.venv-warp/bin/python evaluate_cpu.py --checkpoint artifacts/leap/policy.pt --output runs/recheck_cpu --episodes 32 --seconds 60
```

当前 `.venv-warp` 通过 `--system-site-packages` 复用本机已有 microduck 环境的库，未修改该环境。移植机器时应按下一节创建独立环境。

首次克隆时请同时获取固定版本的第三方依赖：

```bash
git clone --recurse-submodules https://github.com/FZAbiding/xense-hand-rl.git
cd xense-hand-rl
```

## 从头复现

需要 Linux、Python 3.12、支持 CUDA 的 NVIDIA GPU。实际训练依赖版本列在 `requirements.txt`，本次机器为 RTX 4090 24GB。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python bootstrap.py
.venv/bin/python calibrate_grasp.py
.venv/bin/python train.py --run runs/new_small --num-envs 1024 --iterations 550
.venv/bin/python train.py --run runs/new_wide --resume runs/new_small/checkpoint_000550.pt --num-envs 1024 --iterations 250 --amplitude 1.0 --period 4 --lr 0.00015
.venv/bin/python evaluate.py --checkpoint runs/new_wide/latest.pt --output runs/new_eval --num-envs 128 --video
```

`bootstrap.py` 按固定 commit 获取 LEAP 手资产和 Playground 手模型 XML，保留已有 checkout，遇到版本不一致会报错。`build_assets.py` 生成带颜色材质的 `assets/pen.obj`、`pen.mtl` 和物理场景 `scene.xml`。
最终交付策略实际经历 550 轮小幅训练和 250 轮大幅微调，共 **26,214,400 个环境步**。训练吞吐约 7–8 万环境步/秒，所选权重的两阶段训练约 6 分钟，另有编译、抓姿校准和独立验证时间。浮点接触计算与机器配置可能使重新训练结果存在差异。

训练命令的 `--iterations` 表示本次追加轮数。模型每 50 轮保存，`latest.pt` 可用于恢复；交付选用经过完整验证的第 800 轮，而非最后一次训练权重。

可选对照：

```bash
.venv-warp/bin/python build_reference.py
.venv-warp/bin/python evaluate.py --reference --output runs/ik_recheck --num-envs 64
.venv-warp/bin/python train.py --reference --run runs/ppo_residual
```

已运行静止基线与运动学参考基线；残差 PPO 接口已实现，但未训练该分支。最终策略是直接 PPO，没有使用运动学参考、示范动作或对笔施加辅助力。

## 实现与资产

`pen_env.py`：MuJoCo Warp 批量接触动力学、66 维状态观测、目标相位、奖励与终止条件。
`train.py`：PPO、GAE、观测归一化、超时 bootstrap、断点保存。
`evaluate.py` / `evaluate_cpu.py`：独立种子、无自动复位的 GPU/CPU 评估。
`calibrate_grasp.py`：指尖逆运动学及 2 秒重力下的抓姿筛选。
`artifacts/leap/manifest.json`：最终权重及演示文件 SHA-256、源资产 commit 和训练步数。

笔长 180 mm、直径 18 mm、质量 25 g，使用圆滑 capsule 碰撞、独立 OBJ 渲染。它是较粗的转笔训练杆；没有验证普通 8 mm 直径细笔。动力学为正常重力 -9.81 m/s²、5 ms 时间步，策略控制频率 40 Hz。
状态包含仿真提供的笔姿态和速度，因此当前是仿真中的特权状态策略。
手模型来源与许可证见 [NOTICE.md](NOTICE.md)，原始许可随资产保留。

## 参考动作与范围

用户给定的小红书页面未能读取到视频，所以动作依据文字“指尖来回往复旋转笔”实现。**当前结果不代表逐帧复现原视频，也不是 360° finger-pass 或真机部署结果。**

实验中曾尝试 JAX/Brax 安装路线，因新依赖下载慢改用本机已有 Warp/PyTorch。大幅训练在约第 870 轮出现一次 exit 139，原因未确定；完整第 800 轮权重不受影响，随后从第 850 轮独立恢复训练 100 轮并正常保存。不同 GPU 训练/渲染作业建议串行运行；并发是否导致该异常尚未证实。
早期视频评估在最终接触数 JSON 序列化时因 NumPy 整数报错，指标和视频已生成；该问题已修复，最终评估与打包正常完成。

# Wuji Hand 2 与 LEAP、Revo3 同任务实验

使用 [Wuji 官方 `wuji-description`](https://github.com/wuji-technology/wuji-description) 的 Hand 2 Beta 2 右手 MJCF/网格，源版本固定在 `third_party/wuji-description` 子模块。目标与现有 LEAP、Revo3 一致：固定掌部，纯手指接触控制自由笔杆，目标偏航角 `0.6 × sin(2πt/4)` rad；笔长 180 mm、直径 18 mm、质量 25 g，控制周期 25 ms，物理步长 5 ms。

`build_wuji_hand2.py` 保留官方关节限位、连杆质量/惯量和执行器力矩范围；为了与 Revo3 的仿真位置控制器对齐，将 `kp` 设置为 3，保留官方 `kv`。默认场景在 `assets/wuji_hand2_palm_up/`：基座水平、掌心法线为世界 `+Z`。原 `assets/wuji_hand2/` 的掌心法线为 `-Z`，属于朝向错误的历史实验。摩擦、接触求解器和控制器属于任务适配，不代表已标定的 Wuji 真机动力学。

新朝向的[模型预览图](../../assets/wuji_hand2_palm_up/orientation_preview.png)显示掌心朝上；`validate_assets.py` 会在关节、质量和执行器参数之外检查掌心法线方向。

## 复现

```bash
git submodule update --init third_party/wuji-description
.venv-warp/bin/python build_wuji_hand2.py
.venv-warp/bin/python experiments/wuji_hand2/validate_assets.py
```

新朝向场景的训练初态必须通过 12 秒搜索和额外的 60 秒、含扰动认证；训练代码只读取 `grasp_bank_60s.npz`。在首轮 8 个 12 秒合格候选中，仅 1 个通过了无扰动和 4 次随机扰动的 60 秒认证，见 `assets/wuji_hand2_palm_up/grasp_bank_60s_certification.json`。旧权重不能用于新场景。须用 `--grasp-version palm_up_v1` 重新训练，再独立 CPU 评估；当前不应发布与 LEAP/Revo3 的新分数。校准器和认证器拒绝覆盖已有同名 `.npz`，以免训练资产被静默替换。

旧朝下场景保留两版抓握库：`grasp_bank.npz` 是只验证 2 秒的早期初态，`grasp_bank_long_initial.npz` 由 12 秒搜索生成。旧权重、配置、日志、评估和演示归档于 [`artifacts/wuji_hand2/legacy_palm_down/`](../../artifacts/wuji_hand2/legacy_palm_down/)。

```bash
.venv-warp/bin/python experiments/wuji_hand2/calibrate.py \
  --local-refine --max-candidates 729 --settle-seconds 12 \
  --bank-name grasp_bank.npz
.venv-warp/bin/python experiments/wuji_hand2/certify_grasps.py
```

PPO 与 Revo3 V2 使用相同的观察/奖励结构、1024 并行环境、32 步 rollout、4 epoch × 8 minibatch、40 Hz 控制和严格接触门槛；Wuji 有 20 个执行器，因而观察为 105 维、动作为 20 维。旧场景的训练日志、配置、源文件快照、检查点和评估原始轨迹位于 `runs/wuji_hand2/`，正式归档于 `artifacts/wuji_hand2/legacy_palm_down/`。旧课程顺序为 ±0.2 rad/6 s 短回合、±0.2 rad/6 s 长回合、±0.35 rad/6 s、±0.6 rad/4 s；其权重不能算作新朝向成果。

独立 CPU 评估会在整个 60 秒内重新运行神经网络和 MuJoCo 接触动力学，不自动复位；同一验收要求包括存活、工作空间、平均角度误差 ≤3°、倾角、指尖接触、非指尖受力、命令裁剪和每半周期转角。旧三手对比报告中的 Wuji 行已因朝向错误撤回；LEAP 的严格分项指标由扩展后的 `evaluate_cpu.py` 生成，Revo3/Wuji 使用各自 V2 评估程序，共享同一 `quality_pass` 判据。

本实验只比较这个仿真中的固定掌部往复转笔任务，不证明任何一款真实灵巧手的优劣；官方模型中的驱动增益尚待真机标定，也未做跨训练种子统计。

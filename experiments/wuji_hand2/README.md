# Wuji Hand 2 与 LEAP、Revo3 同任务实验

使用 [Wuji 官方 `wuji-description`](https://github.com/wuji-technology/wuji-description) 的 Hand 2 Beta 2 右手 MJCF/网格，源版本固定在 `third_party/wuji-description` 子模块。目标与现有 LEAP、Revo3 一致：固定掌部，纯手指接触控制自由笔杆，目标偏航角 `0.6 × sin(2πt/4)` rad；笔长 180 mm、直径 18 mm、质量 25 g，控制周期 25 ms，物理步长 5 ms。

`build_wuji_hand2.py` 保留官方关节限位、连杆质量/惯量和执行器力矩范围；为了与 Revo3 的仿真位置控制器对齐，将 `kp` 设置为 3，保留官方 `kv`。摩擦、接触求解器、手掌朝向属于任务适配，不代表已标定的 Wuji 真机动力学。具体变更见 `assets/wuji_hand2/provenance.json` 和根目录 `NOTICE.md`。

## 复现

```bash
git submodule update --init third_party/wuji-description
.venv-warp/bin/python build_wuji_hand2.py
.venv-warp/bin/python experiments/wuji_hand2/validate_assets.py
```

仓库保留两版抓握库。`grasp_bank.npz` 是只验证 2 秒的早期初态，后续 60 秒复测会掉笔；`grasp_bank_long_initial.npz` 由 12 秒细网格搜索生成，并保存经整段验证的起始姿态。为避免覆盖已训练策略的输入资产，校准器拒绝覆盖已有同名 `.npz`；重新搜索时指定新的 `--bank-name`。

```bash
.venv-warp/bin/python experiments/wuji_hand2/calibrate.py \
  --local-refine --max-candidates 729 --settle-seconds 12 \
  --bank-name grasp_bank_reproduced.npz
```

PPO 与 Revo3 V2 使用相同的观察/奖励结构、1024 并行环境、32 步 rollout、4 epoch × 8 minibatch、40 Hz 控制和严格接触门槛；Wuji 有 20 个执行器，因而观察为 105 维、动作为 20 维。训练日志、配置、源文件快照、检查点和评估原始轨迹位于 `runs/wuji_hand2/`。课程顺序为 ±0.2 rad/6 s 短回合、±0.2 rad/6 s 长回合、±0.35 rad/6 s、±0.6 rad/6 s、±0.6 rad/4 s。每段的实际 `--resume`、学习率和源文件哈希均写在对应 `config.json`、`source_manifest.json` 中。

独立 CPU 评估会在整个 60 秒内重新运行神经网络和 MuJoCo 接触动力学，不自动复位；同一验收要求包括存活、工作空间、平均角度误差 ≤3°、倾角、指尖接触、非指尖受力、命令裁剪和每半周期转角。最终三手对比见 `reports/wuji_hand2/REPORT.md`；LEAP 的严格分项指标由扩展后的 `evaluate_cpu.py` 生成，Revo3/Wuji 使用各自 V2 评估程序，共享同一 `quality_pass` 判据。

本实验只比较这个仿真中的固定掌部往复转笔任务，不证明任何一款真实灵巧手的优劣；官方模型中的驱动增益尚待真机标定，也未做跨训练种子统计。

# Wuji Hand 2 旧朝向归档（不参与有效三手对比）

本目录的 `scene.xml` 将掌心法线指向世界 `-Z`；原本要求的是水平基座、掌心指向 `+Z`。`policy.pt` 依赖本目录相同朝向的 `grasp_bank.npz`，不可在修正后的场景中使用。`holdout_evaluation.json` 中的 24×60 秒指标、`holdout_rollout.npz` 和 `demo.mp4` 仅反映此旧场景。

训练配置和原始指标分别见 `training_config.json`、`training_metrics.jsonl`；训练输入哈希见 `training_source_manifest.json`，评估输入哈希见 `holdout_manifest.json`。`runs/` 中仍保留分阶段检查点和完整源码快照。本目录是公开成果归档，不改写历史检查点。

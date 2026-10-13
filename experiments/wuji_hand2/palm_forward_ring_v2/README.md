# 掌心向前，无名指修正续训

本轮配置仅在此目录。旧版实验与已认证资源不修改。复用原认证场景和握持库，明确迁移 checkpoint 800 的策略权重；四个无名指输出重新初始化，优化器状态重置。机械参数不变。

运行新实验（已有输出时会拒绝覆盖）：

```bash
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/run.py --config experiments/wuji_hand2/palm_forward_ring_v2/config.json
.venv-warp/bin/python experiments/wuji_hand2/evaluate.py --checkpoint runs/wuji_hand2/palm_forward_v1/final_task/matched/checkpoint_000800.pt --output runs/wuji_hand2/palm_forward_ring_v2/baseline_cpu60 --episodes 24 --seconds 60 --seed 20261020
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_ring_v2/archive.py
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_ring_v2/report.py
```

新增 controller 位于 `experiments/wuji_hand2/ring_control.py`。训练和 CPU 评估共用这一实现；新版评估校验它与训练快照一致。默认 `legacy` 保留旧模型行为。限幅统计包含软件边界触发。20°/秒限制作用于控制命令，实际关节轨迹单独测量。稳态姿态指标排除前 3 秒初始认证姿态过渡；任务指标使用完整 60 秒。

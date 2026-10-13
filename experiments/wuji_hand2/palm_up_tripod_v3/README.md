# 收起无名指和小拇指的三指转笔

掌心向上，保持 ±0.6 rad / 6 秒目标，网络仅输出前三指的 12 个动作。两条收起手指的命令由新认证初态固定为 [70,0,95,60] 度；物理关节与碰撞保留。旧五指权重仅用于显式迁移前三指输出，优化器重置，新握持库重新建立。

独立配置为 config.json。重建成功的初态流程（已有文件会拒绝覆盖）：

```bash
PYTHONPATH=. .venv-warp/bin/python experiments/wuji_hand2/palm_up_tripod_v3/policy_seeds.py
PYTHONPATH=. .venv-warp/bin/python experiments/wuji_hand2/palm_up_tripod_v3/refine.py
PYTHONPATH=. .venv-warp/bin/python experiments/wuji_hand2/palm_up_tripod_v3/certify_clearance.py
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/run.py --config experiments/wuji_hand2/palm_up_tripod_v3/config.json
.venv-warp/bin/python experiments/wuji_hand2/evaluate.py --checkpoint runs/wuji_hand2/palm_up_pinch_v2/wide_6s/checkpoint_001000.pt --output runs/wuji_hand2/palm_up_tripod_v3/baseline_cpu60 --episodes 24 --seconds 60 --seed 2026101003
.venv-warp/bin/python experiments/wuji_hand2/palm_up_tripod_v3/archive_report.py
```

build.py 的原始随机预紧搜索、两个 IK 搜索及最初 60 秒认证失败，记录保留在资产目录；训练只读后来经 refine.py 和 certify_clearance.py 认证的 canonical grasp_bank_60s.npz，不使用失败候选。最后的全连杆筛选比仅指尖分类严格，避免遗漏近端碰笔。

限幅指标现在以 12 个主动关节为分母，不能与旧 20 关节指标直接作等效比较。命令保持收指姿态不等于锁死物理关节。严格任务成功还必须跟踪足够转角、保持前三指夹持、两指触笔占比低于 1%，而不只是笔不掉落。

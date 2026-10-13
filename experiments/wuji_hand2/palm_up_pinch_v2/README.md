# 掌心向上夹持转笔，独立配置 v2

最新目标以 Revo3 `wide_6s/demo_60s.mp4` 为参考：掌心向上、手指夹持自由笔杆、±0.6 rad / 6 秒往复转动。既有掌心向前实验保留。此实验仅从早期掌心向上 checkpoint 550 继续训练。

本轮新增相向夹持奖励和门槛，区别于两个手指同向托笔；`pinch.py` 被训练与评估共同调用。自然无名指控制来自 `ring_control.py`，不冻结手指，机械范围不变。原始认证资源只读复用，初态在新判据和控制器下已重新经过 60 秒 × 5 次扰动检查。

```bash
PYTHONPATH=. .venv-warp/bin/python experiments/wuji_hand2/palm_up_pinch_v2/certify_pinch.py
.venv-warp/bin/python experiments/wuji_hand2/palm_forward_v1/run.py --config experiments/wuji_hand2/palm_up_pinch_v2/config.json
.venv-warp/bin/python experiments/wuji_hand2/palm_up_pinch_v2/archive_report.py
```

运行器是共享工具，显式传入本轮配置；会拒绝覆盖已有训练/认证结果。课程进阶使用原 CPU 生存门槛，严格结果还要求夹持、跟踪、半周期幅度和原质量门槛。未通过不等于已经实现参考视频效果。

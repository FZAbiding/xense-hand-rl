# 真人示范驱动的三指转笔：v4 独立实验

用户已批准 A / Triangle Pass 示范，本轮独立训练与诊断已执行。2026-10-12 用户否定食指、中指交叉的 v4 局部轨迹，撤回局部动作通过结论。约 58° 与 7/24 仅保留作旧数值门槛下的历史记录，不可用作合格示范。静态夹持 24/24 × 60 秒仅证明旧指标下稳定性，自然姿态须重新审阅。v5 先提交计划，用户确认前不启动训练。

配置、初态库、权重、运行和结果均使用 v4 独立目录，不继承 v3 策略。`task_spec.draft.json` 是任务规格记录，不是训练器配置；训练配置为 `config.json`、`config_hold_refine.json`、`config_handover.json`。阶段门槛没有降低。

完整说明：`reports/wuji_hand2/palm_up_human_demo_v4/REPORT.md`。视频索引：`artifacts/wuji_hand2/palm_up_human_demo_v4/review.html`。复现入口包括 `build.py`、`train.py`、`evaluate.py`、`optimize_reference.py`、`check_reference.py`；有结果的目录拒绝覆盖。

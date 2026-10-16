# 实验协作规范

涉及灵巧手仿真、转笔训练、评估或结果展示时，先阅读并遵循
[实验评审与展示 SOP](docs/EXPERIMENT_REVIEW_SOP.md) 和
[灵巧操作训练 SOP](docs/DEXTEROUS_RL_TRAINING_SOP.md)。
Wuji v1–v7 的结果、失败经验和权重入口见
[阶段总结](reports/wuji_hand2/V1_V7_LESSONS.md)；
30/60/120 秒路线见[长期规划](reports/wuji_hand2/LONG_HORIZON_ROADMAP.md)。

- 每个新版本使用独立的配置、资产、运行和结果目录，记录继承来源，不混用旧版权重或初态库。
- 默认保留 SOP 中的离线 HTML 卡片布局、真人参考、双视角视频、原始指标和明确的失败标注。
- 关节角合法、无明显穿透、不掉笔、笔角度变化均不足以证明动作合理。检查食指/中指交叉、接触位置、释放与交接，以及与批准示范的动作一致性。
- 用户否定的动作必须保留失败证据并撤回相应成功结论，不能继续当作合格参考或模仿目标。
- v5 的 24 小时长训已完成，历史方案见 `experiments/wuji_hand2/palm_up_human_demo_v5/long_training/PLAN.md`。当前用户只要求规划后续长时训练，不启动新训练；继续保留姿态硬约束、独立验收与失败标注。

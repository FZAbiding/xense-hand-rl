# V1 / V2 同幅同速 CPU 对比

目标±34.38°、周期4s；各12回合×60s，种子20260925，不自动复位。

抓姿、控制接口、奖励、观测与训练预算不同，因此这是系统级结果对比，不是单变量因果证明。关节运动来自40Hz采样；P95/最大单步变化等分指指标均为各回合对应统计的平均。

| 指标 | V1 | V2 |
|---|---:|---:|
| quality_success_rate | 0 | 1 |
| survival_rate | 1 | 1 |
| workspace_survival_rate | 0.666667 | 1 |
| mae_deg | 3.3467 | 0.242998 |
| mean_tilt_deg | 30.3801 | 0.298054 |
| p95_tilt_deg | 35.6355 | 0.765408 |
| command_clip_fraction | 0.407255 | 0.00137235 |
| normal_force_sum_n | 9.97448 | 0.940934 |
| non_tip_force_frame_fraction | 0.361389 | 0 |
| palm_force_frame_fraction | 0.0788889 | 0 |
| thumb_mean_abs_velocity_deg_s | 47.2166 | 13.5261 |
| thumb_rms_acceleration_deg_s2 | 5061.04 | 77.3742 |
| thumb_p95_step_deg | 3.71629 | 0.903055 |
| thumb_max_step_deg | 7.34744 | 1.47297 |
| index_mean_abs_velocity_deg_s | 71.631 | 4.13954 |
| index_rms_acceleration_deg_s2 | 6490.86 | 46.5631 |
| index_p95_step_deg | 5.03905 | 0.317911 |
| index_max_step_deg | 6.79119 | 0.620209 |
| middle_mean_abs_velocity_deg_s | 71.5717 | 5.51619 |
| middle_rms_acceleration_deg_s2 | 6777.47 | 71.1781 |
| middle_p95_step_deg | 5.25244 | 0.483209 |
| middle_max_step_deg | 7.63786 | 1.39585 |
| ring_mean_abs_velocity_deg_s | 71.1888 | 7.56554 |
| ring_rms_acceleration_deg_s2 | 6516.1 | 53.3372 |
| ring_p95_step_deg | 4.44425 | 0.523727 |
| ring_max_step_deg | 6.62699 | 0.990005 |
| little_mean_abs_velocity_deg_s | 68.1349 | 4.95253 |
| little_rms_acceleration_deg_s2 | 7057.44 | 36.9068 |
| little_p95_step_deg | 4.23927 | 0.355575 |
| little_max_step_deg | 6.6495 | 0.726343 |

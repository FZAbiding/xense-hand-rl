# 相关研究与本案例的选择

## 阅读来源

1. Charlesworth & Montana, ICML 2021, *Solving Challenging Dexterous Manipulation Tasks With Trajectory Optimisation and Reinforcement Learning*.
   https://proceedings.mlr.press/v139/charlesworth21a/charlesworth21a.pdf
   方法部分结合轨迹优化与 TD3，通过示范状态初始化改善探索。本案例据此保留物理抓姿搜索与可选运动学参考路径；并未声称复现 TOPDM/TD3。
2. Ma et al., ICLR 2024, *Eureka: Human-Level Reward Design via Coding Large Language Models*.
   https://eureka-research.github.io/ 和 https://openreview.net/pdf?id=IEduRUO55F
   参考其课程学习思路，从保持物体和小幅旋转开始，以独立测量验证奖励设计。本案例为人工迭代奖励，不是调用 Eureka 自动奖励生成系统。
3. Wang et al., CoRL 2024, *Lessons from Learning to Spin “Pens”*.
   https://arxiv.org/html/2407.18902v2 和 https://github.com/HaozhiQi/penspin
   参考物体状态特权观测、抓姿初始化、多点指尖接触及保持笔水平的奖励。论文采用 Allegro，本文实现采用 LEAP。仿真结果不代表真机迁移成功。
4. MuJoCo Playground 官方实现与 LEAP 模型。
   https://github.com/google-deepmind/mujoco_playground
   https://github.com/google-deepmind/mujoco_menagerie/tree/main/leap_hand
   使用公开模型结构、网格和碰撞几何，保留原始仓库与许可证；具体 commit 记录在 assets/provenance.json。

## 动作定义与边界

原始小红书视频链接无法通过网页工具读取，因此暂按用户文字定义实现：笔在指尖绕掌面法向正反往复旋转。当前首阶段命令为 ±0.6 rad，周期 4 秒，单次评估 12 秒。它不是完整 360° finger-pass，也不声称已匹配原视频。

固定手掌，16 个受控手指关节。笔长 180 mm、半径 9 mm、质量 25 g，有自由关节、重力、惯性和接触摩擦。碰撞使用 capsule，渲染使用独立生成的 OBJ，笔无驱动器、无 weld、无外力辅助、无位置重放。半径 9 mm 是第一阶段的粗笔；普通细笔需要后续尺寸课程验证。

PPO 观测包含关节、笔的位置/朝向/速度、目标相位、前次动作。直接策略输出相对初始抓姿的关节目标。可选参考模式输出相对运动学参考的残差；两种模式需要分别报告，不能把参考控制称为纯 RL。

## 验证规则

评估使用不同随机种子、确定性策略、初始姿态扰动、不自动重置。成功要求完整时段不掉落、有足够次数的实际正反阈值穿越、平均角度误差较小且持续接近多个指尖。只有 reward 上升不算成功。真实接触数另外用 MuJoCo CPU 对渲染轨迹计算，近邻指尖数不能代替真实接触数。

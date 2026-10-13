"""Package measured successes and failures without claiming a completed spin."""
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .build import ROOT
from tools.render_experiment_review import render_review


def main():
    relative = Path('artifacts/wuji_hand2/palm_up_human_demo_v4')
    art = ROOT/relative
    report = ROOT/'reports/wuji_hand2/palm_up_human_demo_v4'
    report.mkdir(exist_ok=True, parents=True)
    load = lambda p: json.loads(p.read_text())
    hold = load(art/'hold_verified/metrics.json')
    handover = load(art/'handover/metrics.json')
    partial = load(art/'partial_rotation/metrics.json')
    human_review = load(art/'human_review.json')
    runs = ROOT/'runs/wuji_hand2/palm_up_human_demo_v4'
    counts = {}
    for name in ['hold','hold_refine','handover']:
        rows = [json.loads(s) for s in (runs/name/'metrics.jsonl').read_text().splitlines() if s.strip()]
        counts[name] = max(x['iteration'] for x in rows)
    summary = dict(experiment_id='wuji_hand2_palm_up_human_demo_v4',
        status='user_rejected_finger_crossing',
        full_rotation_achieved=False, human_like_handover_achieved=False,
        ppo_iterations=counts, total_ppo_iterations=sum(counts.values()),
        ppo_environment_steps=sum(counts.values())*1024*32,
        hold=dict(episodes=24, seconds_per_episode=60, passed_under_previous_physical_gates=24,
            natural_posture_status='requires_re_review', metrics=str((art/'hold_verified/metrics.json').relative_to(ROOT))),
        learned_handover=dict(episodes=24, passed=0, metrics=str((art/'handover/metrics.json').relative_to(ROOT))),
        open_loop_partial_rotation=dict(nominal_angle_deg=partial['episodes'][0]['actual_final_angle_deg'],
            perturbed_episodes=24, partial_passed=0,
            previous_numeric_gate_passed=sum(x['partial_rotation_passed'] for x in partial['episodes'][1:]),
            natural_motion_status='rejected_finger_crossing',
            handover_passed=0, metrics=str((art/'partial_rotation/metrics.json').relative_to(ROOT))),
        human_review=human_review,
        next_required_milestone='Wait for v5 plan approval; calibrate natural finger ordering and review initial poses before short physical handover previews and training.')
    (report/'summary.json').write_text(json.dumps(summary,indent=2))
    z = np.load(art/'partial_rotation/rollout.npz')
    q, tr, forces = z['qpos_0'], z['metrics_0'], z['forces_0']
    t = (np.arange(len(q))+1)*.025
    fig, axes = plt.subplots(3,1,figsize=(11,10),sharex=True)
    axes[0].plot(t,np.degrees(tr[:,0]),label='Actual directed angle')
    axes[0].plot(t,np.degrees(tr[:,1]),'--',label='Partial target (-60 deg)')
    axes[0].set_ylabel('Pen angle (deg)');axes[0].legend()
    for i,label in [(6,'Index PIP'),(7,'Index DIP'),(10,'Middle PIP'),(11,'Middle DIP')]:
        axes[1].plot(t,np.degrees(q[:,i]),label=label)
    axes[1].axhline(0,color='red',ls='--');axes[1].set_ylabel('Joint angle (deg)');axes[1].legend(ncol=2)
    for i,label in enumerate(['Thumb distal','Index distal','Middle distal','Ring distal','Pinky distal','Other hand links','Palm']):
        axes[2].plot(t,forces[:,i],label=label)
    axes[2].set_ylabel('Pen contact force (N)');axes[2].set_xlabel('Time (s)');axes[2].legend(ncol=3)
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('V4 rejected: index/middle crossing. Historical numeric gates: 7/24; accepted natural motion: 0')
    fig.tight_layout();fig.savefig(art/'partial_rotation/diagnostics.png',dpi=150);fig.savefig(report/'partial_rotation_diagnostics.svg');plt.close(fig)
    fig, axes = plt.subplots(3,4,figsize=(14,8),sharex=True)
    for i,ax in enumerate(axes.flat):
        ax.plot(t,np.degrees(q[:,i]));ax.set_title(f'{["Thumb","Index","Middle"][i//4]} {["MCP flex","MCP lateral","PIP","DIP"][i%4]}')
        ax.set_ylabel('deg');ax.grid(alpha=.25)
    fig.suptitle('All 12 active joints: nominal partial-rotation trajectory')
    fig.tight_layout();fig.savefig(art/'partial_rotation/all_active_joints.png',dpi=150);plt.close(fig)
    md = f'''# 真人示范驱动转笔 v4：执行结果

已按用户批准的 A / Triangle Pass 示范执行独立实验，使用新配置、新初态库、新策略和独立目录，没有继承 v3 权重。**完整转笔尚未实现**。

## 2026-10-12 人工审阅更正

**用户指出食指、中指交叉，已否定此动作。撤回“局部转动通过”的任务结论。** 原约 58° 转角及 7/24 数字只描述旧物理/角度指标，不代表姿态自然或任务通过。自然动作通过结论为 0；该轨迹禁止用于后续模仿参考或成功初态。关节不反弯、没有明显自穿透，并不能排除指骨交叉和异常侧摆。原静态夹持也只保留旧指标下的稳定性结论，初态自然性须重新审阅。

原始视频、轨迹、指标保持不变，人工意见和用户截图见 `human_review.json` 与 `human_review/`。下一版只做计划草案，用户确认前不启动训练。展示规范已沉淀到仓库 `docs/EXPERIMENT_REVIEW_SOP.md`。

## 本次实际结果

| 项目 | 实测结果 | 判断 |
|---|---|---|
| PPO 训练 | hold 300 + hold_refine 200 + handover 400 = 900 轮，29,491,200 环境步 | 已执行 |
| 静态夹持 | 旧物理指标未见种子 24/24，通过每次 60 秒测试 | 自然姿态待重新审阅 |
| PPO 60° 转动及换指 | 24/24 不掉笔，但换指验收 0/24，实际转角接近 0° | 未通过 |
| 全物理轨迹优化 | 12 代 × 64 条真实驱动轨迹，名义条件实际转动 {partial['episodes'][0]['actual_final_angle_deg']:.2f}° | 食指/中指交叉，人工否定 |
| 局部旋转扰动复测 | 旧数值门槛 7/24；17/24 完成 7 秒仿真；换指 0/24 | 自然动作通过结论撤回 |
| 180° / 360° / 连续循环 | 前置换指阶段失败，因此未进入 | 未实现 |

### 关节和接触

静态夹持 24 组测试：食指、中指 PIP/DIP 反弯帧占比均为 0；非指尖手部接触力 > 0.05 N 帧占比为 0；无名指、小拇指全部连杆接触笔的合力 > 0.02 N 帧占比为 0。两个活跃指尖接触帧占比为 100%，相向夹持约 99.998%。中指 PIP 平均运动幅度 0.83°。

名义局部旋转：中指 PIP 约 9.87–24.47°、幅度 14.60°，DIP 约 14.97–42.76°；没有外翻/负角度，也没有靠无名指、小拇指或中间指节支撑笔。指尖相向夹持覆盖整段。笔最大位移 9.60 mm，目标角跟踪 MAE 5.54°。这段是**开环手指控制轨迹，不是已训练出的转笔策略**，也没有完成换指。

验收力阈值会忽略低于阈值的微小接触，不等同于数学意义的零接触力；所有原始接触位置、力和轨迹均保留。

### 真人参考和任务定义

批准的视频已复制到本实验 `reference/human_A.mp4`，SHA256 为 `5ffe284d73e0a07d24beff5b8b3d955cfcb8f2b0a38cd829cf498be973ec720a`。对原片 139–141.5 秒做了 6 个带笔两端身份的手工标注。142 秒附近有镜头切换，未把剪辑拼成完整循环。图像中的二维投影角度不作为三维关节、接触力或完整 360° 运动的真值。

采用掌心向上、拇指/食指/中指参与，无名指/小拇指收拢。物理模型保留所有 20 个关节和真实碰撞，控制网络仅输出 12 个活跃关节。任务中的 TI→IM→MT 接触顺序是待验证的机器人规划假设；没有把被遮挡的人手接触顺序说成已知真值。

### 为什么目前没有达到示范效果

静态策略学会了安全夹持，但在转动奖励下仍停留在三指锁笔状态，无法主动释放一个指头并把接触交给下一对。仅用几何 IK 可以找到笔姿态，但部分初始解存在严重手指自穿透，加入真实自碰撞约束后，许多解又会滑到拇指中节或收拢的指节上。所有这些解都被保留为失败诊断，没有作为成功抓取库。

后续的连续全物理优化确实产生约 58° 转动，但依然以三指共同夹持为主；关节微扰标准差 0.005 rad、笔位置微扰标准差 0.4 mm 就使局部动作成功率降至 7/24。因此瓶颈是接触交接及其鲁棒性，不能通过继续放大角度目标或只检查“不掉笔”解决。

还独立测试了直径 9 mm、长 16 cm、重 12 g 的细笔假设；该尺寸不是对真人视频的测量，未通过动力学验收，也未替换主实验的原始笔模型。主结果仍使用直径 18 mm、长 18 cm、重 25 g 的原模型。

## 查看产物

- [离线视频索引](../../../{relative}/review.html)：真人参考、名义局部旋转、稳定夹持、失败 PPO 交接各自注明状态；没有伪同步或循环拼接。
- [局部旋转双视角视频](../../../{relative}/partial_rotation/rollout.mp4)
- [60 秒稳定夹持视频](../../../{relative}/hold_verified/rollout.mp4)
- [局部旋转角度、关节与接触曲线](../../../{relative}/partial_rotation/diagnostics.png)
- [全部 12 个活跃关节](../../../{relative}/partial_rotation/all_active_joints.png)
- [机器可读结果](summary.json)

策略文件、配置、训练时源文件清单、模型资产、原始关节轨迹和逐连杆接触记录均在各阶段目录。`hold_verified/policy.pt` 为 checkpoint 100；选择依据是独立验证集，24 组未见种子用于最终验收。`handover/policy.pt` 是失败尝试，不能作为可用转笔策略。静态跟踪误差不能与旧版的摆动任务误差直接比较。

## 下一步明确门槛

先在当前真实碰撞模型中实现一次可重复的释放与交接，显式约束重心附近的接触位置、接触力矩、释放指头的间隙，并对初态扰动一起优化。必须通过原有接触交接和姿态验收，再将合格参考轨迹用于模仿初始化/强化学习，最后进入 180°、360° 和连续循环。当前没有足够证据把几何可达解当作动力学可行示范，也没有证明任务在该手模型上不可实现。

本轮训练与诊断已结束；完整任务仍未完成。源文件在 `experiments/wuji_hand2/palm_up_human_demo_v4/`，运行记录在 `runs/wuji_hand2/palm_up_human_demo_v4/`，不覆盖旧版本。
'''
    (report/'REPORT.md').write_text(md)
    (art/'README.md').write_text(md.replace('../../../'+str(relative)+'/', './').replace('(summary.json)', '(../../../reports/wuji_hand2/palm_up_human_demo_v4/summary.json)'))
    manifest = dict(title='Wuji v4：食指/中指交叉动作已被人工否定',
        summary='2026-10-12 更正：约 58° 转角与旧数值门槛 7/24 不代表自然动作成功。撤回局部通过结论，保留失败证据。静态初态须重新审阅。v5 等待计划确认，尚未训练。',
        cards=[
            dict(title='真人参考：Triangle Pass',status='reference',video='reference/human_A.mp4',description='用户批准的参考动作。批准参考不等于批准机器人动作。'),
            dict(title='被否定的局部旋转：食指、中指交叉',status='rejected',video='partial_rotation/rollout.mp4',description='仅保留作失败诊断；禁止用作合格模仿参考。原始转角数据未改写。',links=[dict(label='人工审阅与撤回记录',path='human_review.json')]),
            dict(title='静态夹持：自然姿态待重新审阅',status='pending',video='hold_verified/rollout.mp4',description='旧物理指标下 24/24 × 60 秒稳定持笔；不能据此认定姿态自然。'),
            dict(title='失败 PPO 转动/交接',status='failed',video='handover/rollout.mp4',description='换指验收 0/24；不是可用转笔策略。')],
        images=[dict(label='用户指出的交叉姿态：视角 1',path='human_review/finger_crossing_view_1.png'),
                dict(label='用户指出的交叉姿态：视角 2',path='human_review/finger_crossing_view_2.png'),
                dict(label='历史实测曲线（对应被否定轨迹）',path='partial_rotation/diagnostics.png'),
                dict(label='全部活跃关节（角度合法不足以排除交叉）',path='partial_rotation/all_active_joints.png')],
        links=[dict(label='完整更正说明',path='README.md'),dict(label='原始物理复测指标',path='partial_rotation/metrics.json'),
               dict(label='下一版计划（待确认）',path='../../../experiments/wuji_hand2/palm_up_human_demo_v5/PLAN.md'),
               dict(label='展示与验收 SOP',path='../../../docs/EXPERIMENT_REVIEW_SOP.md')])
    (art/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    render_review(manifest, art/'review.html')
    spec_path=Path(__file__).with_name('task_spec.draft.json')
    spec=load(spec_path);spec['status']='user_rejected_finger_crossing'
    spec['acceptance']['finger_crossing']='reject; calibrate complete finger chains and palm-local ordering before next training'
    spec['acceptance']['human_simulation_motion_accepted']=False
    spec['execution_result']=summary
    spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(spec_path, art/spec_path.name)
    (runs/'execution_summary.json').write_text(json.dumps(summary,indent=2))
    (Path(__file__).parent/'README.md').write_text('''# 真人示范驱动的三指转笔：v4 独立实验

用户已批准 A / Triangle Pass 示范，本轮独立训练与诊断已执行。2026-10-12 用户否定食指、中指交叉的 v4 局部轨迹，撤回局部动作通过结论。约 58° 与 7/24 仅保留作旧数值门槛下的历史记录，不可用作合格示范。静态夹持 24/24 × 60 秒仅证明旧指标下稳定性，自然姿态须重新审阅。v5 先提交计划，用户确认前不启动训练。

配置、初态库、权重、运行和结果均使用 v4 独立目录，不继承 v3 策略。`task_spec.draft.json` 是任务规格记录，不是训练器配置；训练配置为 `config.json`、`config_hold_refine.json`、`config_handover.json`。阶段门槛没有降低。

完整说明：`reports/wuji_hand2/palm_up_human_demo_v4/REPORT.md`。视频索引：`artifacts/wuji_hand2/palm_up_human_demo_v4/review.html`。复现入口包括 `build.py`、`train.py`、`evaluate.py`、`optimize_reference.py`、`check_reference.py`；有结果的目录拒绝覆盖。
''')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for base in [art,report,Path(__file__).parent] for p in sorted(base.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name!='sha256_manifest.json'}
    for name in ['AGENTS.md', 'docs/EXPERIMENT_REVIEW_SOP.md', 'tools/render_experiment_review.py',
                 'experiments/wuji_hand2/palm_up_human_demo_v5/PLAN.md']:
        manifest[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (report/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

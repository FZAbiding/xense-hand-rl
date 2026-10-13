"""Compare the continued policy and original policy on identical fresh seeds."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
CFG=Path(__file__).with_name('config.json')


def summarize(metrics):
    rows=metrics['per_episode']
    result={k:metrics[k] for k in ['episodes','seconds','seed','quality_success_rate','survival_rate','mae_deg','command_clip_fraction','non_tip_force_frame_fraction','two_tip_fraction']}
    result['ring_pose']={key:float(np.mean([r['ring_pose'][key] for r in rows])) for key in [
        'steady_dip_hyperextension_fraction','steady_mcp_hyperextension_fraction',
        'steady_abduction_outside_20deg_fraction','steady_coordination_mae_deg']}
    result['ring_joint_min_deg']=np.min([r['ring_pose']['joint_min_deg'] for r in rows],axis=0).tolist()
    result['ring_joint_max_deg']=np.max([r['ring_pose']['joint_max_deg'] for r in rows],axis=0).tolist()
    result['ring_motion']={k:float(np.mean([r['finger_motion']['ring'][k] for r in rows])) for k in rows[0]['finger_motion']['ring']}
    result['ring_clip_fraction']=float(np.mean([v for r in rows for k,v in r['joint_clip_fractions'].items() if 'ring_finger' in k]))
    return result


def main():
    cfg=json.loads(CFG.read_text());base=ROOT/cfg['runs'];out=ROOT/'reports/wuji_hand2/palm_forward_ring_v2';out.mkdir(parents=True,exist_ok=True)
    before=json.loads((base/'baseline_cpu60/metrics.json').read_text())
    after=json.loads((base/'holdout_cpu60/metrics.json').read_text())
    for key in ['seed','episodes','seconds','amplitude_rad','period_seconds']:
        assert before[key]==after[key],key
    status=json.loads((base/'status.json').read_text())
    a,b=summarize(before),summarize(after)
    result=dict(before=a,after=b,selected_checkpoint=status['selected_checkpoint'],
        settling_note='Steady ring pose metrics exclude first 3 seconds; all task and clipping metrics include the full 60 seconds.',
        scope='Same certified scene/grasp bank, seeds, target, duration, perturbations and gates. Controller, ring output initialization, reward and optimizer were changed together; this is not an isolated reward ablation.')
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    lines=['# 无名指修正续训结果','',
        '独立配置：`experiments/wuji_hand2/palm_forward_ring_v2/config.json`。从上一轮选中的 checkpoint 800 追加 600 轮（19,660,800 环境步）。复用原始已认证掌心向前场景和握持库，旧配置、旧权重、旧运行结果不覆盖。',
        '', '无名指采用软件范围 MCP 屈曲 0–65°、外展 ±15°、PIP 0–85°、DIP 0–60°，控制命令最高 20°/秒。机械范围仍为官方原值。初始姿态保留原认证状态并平滑过渡；无名指保留四关节独立输出，协调关系只是奖励先验。仅重置无名指输出层四行，并重置优化器状态。',
        '',f'选择：`{status["selected_checkpoint"]}`。在固定验证种子上比较各检查点，先按严格成功率、再按工作空间生存率、最后按跟踪误差选择。新留出测试种子 {after["seed"]}，24 次 × 60 秒，目标 ±0.6 rad / 4 秒；旧模型也使用同一测试种子复测。',
        '', '| 指标 | 旧模型复测 | 新模型 |','|---|---:|---:|',
        f'| 60 秒不掉笔 | {a["survival_rate"]*24:.0f}/24 | {b["survival_rate"]*24:.0f}/24 |',
        f'| 严格通过 | {a["quality_success_rate"]*24:.0f}/24 | {b["quality_success_rate"]*24:.0f}/24 |',
        f'| 跟踪 MAE | {a["mae_deg"]:.2f}° | {b["mae_deg"]:.2f}° |',
        f'| 无名指限幅占比 | {a["ring_clip_fraction"]:.1%} | {b["ring_clip_fraction"]:.1%} |',
        f'| 无名指 DIP 反弯超过 5°（稳态） | {a["ring_pose"]["steady_dip_hyperextension_fraction"]:.1%} | {b["ring_pose"]["steady_dip_hyperextension_fraction"]:.1%} |',
        f'| 无名指 MCP 反弯超过 5°（稳态） | {a["ring_pose"]["steady_mcp_hyperextension_fraction"]:.1%} | {b["ring_pose"]["steady_mcp_hyperextension_fraction"]:.1%} |',
        f'| 无名指外展超过 ±20°（稳态） | {a["ring_pose"]["steady_abduction_outside_20deg_fraction"]:.1%} | {b["ring_pose"]["steady_abduction_outside_20deg_fraction"]:.1%} |',
        f'| DIP/PIP 协调偏差（稳态） | {a["ring_pose"]["steady_coordination_mae_deg"]:.2f}° | {b["ring_pose"]["steady_coordination_mae_deg"]:.2f}° |',
        '', '稳态姿态统计排除前 3 秒原始认证姿态过渡；任务成功率、限幅、跟踪误差仍统计完整 60 秒。命令限速不是实际关节速度保证，实际运动见 comparison.json。软件约束触发也计入限幅，严格成功门槛未放宽。',
        '', '复测比较同时改变控制器、奖励、输出初始化和优化器，不能将差异归因于单一因素。严格通过为零时，模型仍属于实验结果，不能宣称已完成转笔任务。',
        '', '视频：`artifacts/wuji_hand2/palm_forward_ring_v2/rollout.mp4`；轨迹：`ring_comparison.png`。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for label,sub in [('Previous','baseline_cpu60'),('Continued','holdout_cpu60')]:
        data=np.load(base/sub/'rollout.npz');q=np.degrees(data['qpos'][0,:,12:16]);ts=(np.arange(len(q))+1)*float(data['dt'])
        for j,ax in enumerate(axs):ax.plot(ts,q[:,j],label=label)
    for j,(ax,name,lo,hi) in enumerate(zip(axs,['MCP flexion','MCP abduction','PIP','DIP'],[0,-15,0,0],[65,15,85,60])):
        ax.axhline(lo,color='gray',linestyle='--');ax.axhline(hi,color='gray',linestyle='--');ax.axvspan(0,3,color='gray',alpha=.12);ax.set_ylabel(name+' (deg)')
    axs[0].legend();axs[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(out/'ring_comparison.png');plt.close(fig)
    dest=ROOT/cfg['artifacts']
    for file in ['REPORT.md','comparison.json','ring_comparison.png']:(dest/file).write_bytes((out/file).read_bytes())
    manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
    (dest/'archive_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

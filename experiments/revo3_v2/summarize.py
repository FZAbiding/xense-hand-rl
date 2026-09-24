"""Regenerate V2 experiment tables from raw logs, without hand-entered scores."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/'reports/revo3_v2';out.mkdir(parents=True,exist_ok=True)
    training=[];evaluations=[];histories=[]
    for path in sorted((ROOT/'runs/revo3_v2').glob('*/metrics.jsonl')):
        cfg=json.loads((path.parent/'config.json').read_text())
        history=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not history:continue
        last=history[-1];histories.append((path.parent.name,history))
        row={k:cfg.get(k) for k in ['seed','num_envs','horizon','amplitude','period','episode_seconds','command_clip_penalty','tracking_sigma','resume']}
        row.update(run=path.parent.name,**{k:last.get(k) for k in ['iteration','steps','seconds','total_loss','policy_loss','value_loss','entropy','kl','lr','command_clip_fraction','tilt_deg']})
        training.append(row)
    for path in sorted((ROOT/'runs/revo3_v2').glob('*/metrics.json')):
        r=json.loads(path.read_text());row={k:v for k,v in r.items() if k not in ['per_episode','quality_definition']}
        row['simulation_valid']=not (path.parent/'INVALIDATED.json').exists() and r.get('simulation_valid',True)
        row['evaluation']=path.parent.name
        row['worst_episode_minimum_half_cycle_gain']=min(e['minimum_half_cycle_gain'] for e in r['per_episode'])
        evaluations.append(row)
    for name,rows in [('training',training),('evaluation',evaluations)]:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    lines=['# V2 原始日志汇总','',
           '由 `experiments/revo3_v2/summarize.py` 自动生成。不同幅度、周期、回合长度不能直接比较绝对误差或回报。',
           'P95 tilt 是各回合 P95 的平均；gain 列给出所有回合的最差半周期峰值比。训练 steps 是该运行新增步数，不含 resume 的历史。','',
           '| 评估 | 后端 | 幅度 rad | 秒/回合 | 回合数 | 严格通过 | 存活 | MAE° | tilt° | 命令裁剪 | 最差 gain |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in evaluations:
        link=f'../../runs/revo3_v2/{r["evaluation"]}/metrics.json'
        validity='' if r['simulation_valid'] else ' [INVALID: capacity overflow]'
        lines.append(f'| [{r["evaluation"]}]({link}){validity} | {"GPU" if "GPU" in r["backend"] else "CPU"} | {r["amplitude_rad"]:.2f} | {r["seconds"]:g} | {r["episodes"]} | {r["quality_success_rate"]:.1%} | {r["survival_rate"]:.1%} | {r["mae_deg"]:.3f} | {r["mean_tilt_deg"]:.3f} | {r["command_clip_fraction"]:.2%} | {r["worst_episode_minimum_half_cycle_gain"]:.3f} |')
    lines+=['','## 训练末轮分项 loss','',
            '总 loss = policy + 0.5 × value − 0.002 × entropy。负的高斯微分熵是有效值。各组 reward/episode 分布不同，loss 不是效果排行榜。','',
            '| 运行 | 末轮 | 新增步数 | 用时 s | policy loss | value loss | entropy | 总 loss | KL |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in training:
        lines.append(f'| {r["run"]} | {r["iteration"]} | {r["steps"]:,} | {r["seconds"]:.1f} | {r["policy_loss"]:.5f} | {r["value_loss"]:.5f} | {r["entropy"]:.5f} | {r["total_loss"]:.5f} | {r["kl"]:.5f} |')
    lines+=['','![训练曲线](training.png)','']
    (out/'DATA.md').write_text('\n'.join(lines))
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for name,history in histories:
        x=[r['iteration'] for r in history]
        for ax,key,scale in [(axs[0,0],'error_rad',180/np.pi),(axs[0,1],'tilt_deg',1),(axs[1,0],'command_clip_fraction',100),(axs[1,1],'value_loss',1)]:
            ax.plot(x,[r[key]*scale for r in history],label=name)
    for ax,title in zip(axs.flat,['Training yaw MAE (deg)','Training tilt (deg)','Joint command clipping (%)','Value loss']):
        ax.set_title(title);ax.set_xlabel('Cumulative PPO iteration');ax.grid(alpha=.2)
    axs[1,1].set_yscale('symlog');axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'training.png');plt.close(fig)
    print(out/'DATA.md')


if __name__=='__main__':main()

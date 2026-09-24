"""Compare CPU evaluations only when target, duration, count and seed match."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',default='reports/revo3_v2/matched_task');args=p.parse_args()
    records=[json.loads((ROOT/name/'metrics.json').read_text()) for name in [args.baseline,args.candidate]]
    for key in ['amplitude_rad','period_seconds','seconds','episodes','seed','backend']:
        assert records[0][key]==records[1][key],f'Not matched: {key}'
    assert 'CPU' in records[0]['backend']
    rows=[]
    for metric in ['quality_success_rate','survival_rate','workspace_survival_rate','mae_deg','mean_tilt_deg',
                   'p95_tilt_deg','command_clip_fraction','normal_force_sum_n','non_tip_force_frame_fraction','palm_force_frame_fraction']:
        rows.append(dict(metric=metric,v1=records[0][metric],v2=records[1][metric]))
    for finger in ['thumb','index','middle','ring','little']:
        for metric in ['mean_abs_velocity_deg_s','rms_acceleration_deg_s2','p95_step_deg','max_step_deg']:
            values=[float(np.mean([r['finger_motion'][finger][metric] for r in record['per_episode']])) for record in records]
            rows.append(dict(metric=f'{finger}_{metric}',v1=values[0],v2=values[1]))
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    result=dict(baseline=args.baseline,candidate=args.candidate,
                matched={k:records[0][k] for k in ['amplitude_rad','period_seconds','seconds','episodes','seed']},
                caveat='Same command and evaluation protocol, not a single-variable ablation: grasp, controller, reward, observations and training budget differ. Joint motion sampled at 40Hz; finger metrics averaged over episode summaries.',metrics=rows)
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['metric','v1','v2']);w.writeheader();w.writerows(rows)
    lines=['# V1 / V2 同幅同速 CPU 对比','',
        f'目标±{np.degrees(records[0]["amplitude_rad"]):.2f}°、周期{records[0]["period_seconds"]:g}s；各{records[0]["episodes"]}回合×{records[0]["seconds"]:g}s，种子{records[0]["seed"]}，不自动复位。',
        '', '抓姿、控制接口、奖励、观测与训练预算不同，因此这是系统级结果对比，不是单变量因果证明。关节运动来自40Hz采样；P95/最大单步变化等分指指标均为各回合对应统计的平均。',
        '', '| 指标 | V1 | V2 |','|---|---:|---:|']
    for row in rows:lines.append(f'| {row["metric"]} | {row["v1"]:.6g} | {row["v2"]:.6g} |')
    (out/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

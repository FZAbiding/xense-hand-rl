"""Archive the folded-finger 12-action experiment and matched-seed comparison."""
import hashlib,json,os,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def main():
    cfg_path=Path(__file__).with_name('config.json');cfg=json.loads(cfg_path.read_text());base=ROOT/cfg['runs'];dest=ROOT/cfg['artifacts'];assets=ROOT/cfg['assets']
    status=json.loads((base/'status.json').read_text())
    if 'holdout' not in status:raise RuntimeError('Wait for full evaluation/video')
    checkpoint=ROOT/status['selected_checkpoint'];metrics=status['holdout'];baseline=json.loads((base/'baseline_cpu60/metrics.json').read_text())
    for k in ['seed','episodes','seconds','amplitude_rad','period_seconds']:assert metrics[k]==baseline[k],k
    shutil.copytree(checkpoint.parent/'snapshot',dest/'snapshot',dirs_exist_ok=True)
    shutil.copytree(cfg_path.parent,dest/'experiment',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(assets,dest/'assets',dirs_exist_ok=True)
    for name in ['config.json','metrics.jsonl','source_manifest.json','warm_start_lineage.json']:
        if (checkpoint.parent/name).exists():shutil.copy2(checkpoint.parent/name,dest/('training_'+name))
    for name in ['env.py','train.py','evaluate.py','motion_audit.py','pinch.py','ring_control.py']:
        target=dest/'evaluation_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'experiments/wuji_hand2'/name,target)
    shutil.copy2(assets/'scene.xml',dest/'original_scene.xml')
    scene=ET.parse(assets/'scene.xml')
    for mesh in scene.getroot().findall('./asset/mesh'):mesh.set('file',os.path.relpath((assets/mesh.get('file')).resolve(),dest))
    scene.write(dest/'scene.xml',encoding='utf-8',xml_declaration=True)
    shutil.copy2(base/'baseline_cpu60/metrics.json',dest/'baseline_metrics.json')
    validate=dest/'validation';validate.mkdir(exist_ok=True)
    for p in base.glob('eval_*/metrics.json'):shutil.copy2(p,validate/(p.parent.name+'.json'))
    rows=metrics['per_episode'];failure=dict(dropped=sum(not r['survived'] for r in rows),tracking=sum(r['mae_deg']>3 for r in rows),
        amplitude=sum(r['minimum_half_cycle_gain']<.9 for r in rows),inactive_contact=sum(r['inactive_finger_contact_fraction']>=.01 for r in rows),
        pinch=sum(r['opposing_pinch_fraction']<=.8 for r in rows),clipping=sum(r['command_clip_fraction']>=.05 for r in rows),
        non_tip=sum(r['non_tip_force_frame_fraction']>=.05 for r in rows))
    comp=dict(before={k:v for k,v in baseline.items() if k!='per_episode'},after={k:v for k,v in metrics.items() if k!='per_episode'},
        failure_counts=failure,note='Matched scene geometry, target, seeds and perturbation magnitude; bank, controller/action space and tracking sigma changed together. Clipping denominator is 12 active joints now versus 20 before.')
    report=ROOT/'reports/wuji_hand2/palm_up_tripod_v3';report.mkdir(parents=True,exist_ok=True)
    lines=['# 掌心向上、收起两指的三指转笔实验','',
       '已按要求把无名指和小拇指收向掌部，固定位置命令为 MCP 屈曲 70°、外展 0°、PIP 95°、DIP 60°。策略只输出拇指、食指、中指的 12 个动作；两指的 8 个策略输出及探索噪声均移除。模型仍有官方 20 个物理执行器，所有手指几何、质量、机械限位、力矩范围与碰撞均保留。掌心仍朝上。',
       '', '参考 [Steve the Juggler 的转笔示例](https://www.jugglingworld.biz/tricks/prop-manipulation-tricks/pen-spinning-tricks/)中由拇指和中指主导的动作，并查看了 [收起无名指和小拇指的教程](https://mh.co.za/how-to-spin-a-pen-in-your-fingers/)。这些人体示例用于选取动作结构；本机器人目标保持 ±0.6 rad、6 秒周期往复转动。',
       '', '旧握持不能直接删掉小拇指：移除其支撑后会滑落。新初态由旧策略真实轨迹提取并细化前三指预紧，初始 3 个候选再经全连杆接触筛选，最终仅保留 1 个；完整 60 秒 × 无扰动及 4 次扰动均通过。失败搜索和被剔除候选全部留存。',
       '', '本轮独立配置：`experiments/wuji_hand2/palm_up_tripod_v3/config.json`，追加 600 轮（19,660,800 环境步）。显式从上一轮掌心向上模型迁移前三指输出，重置优化器，重新认证新握持库；不覆盖旧配置和旧结果。跟踪奖励宽度由 0.1 调整为 0.18，以帮助扩大转动幅度。',
       '', f'选择模型：`{status["selected_checkpoint"]}`。留出测试为 24 次 × 60 秒，种子 {metrics["seed"]}；旧模型以相同种子重新评估。',
       '', '| 指标 | 旧模型复测 | 收指三指模型 |','|---|---:|---:|',
       f'| 不掉笔 | {baseline["survival_rate"]*24:.0f}/24 | {metrics["survival_rate"]*24:.0f}/24 |',
       f'| 严格成功 | {baseline["quality_success_rate"]*24:.0f}/24 | {metrics["quality_success_rate"]*24:.0f}/24 |',
       f'| 无名指/小拇指全连杆触笔占比（合计法向力 >0.02 N） | {baseline["ring_pinky_contact_fraction"]:.2%} | {metrics["ring_pinky_contact_fraction"]:.2%} |',
       f'| 相向夹持帧占比 | {baseline["opposing_pinch_fraction"]:.2%} | {metrics["opposing_pinch_fraction"]:.2%} |',
       f'| 跟踪 MAE | {baseline["mae_deg"]:.2f}° | {metrics["mae_deg"]:.2f}° |',
       f'| 最小半周期增益的回合均值 | {baseline["minimum_half_cycle_gain"]:.3f} | {metrics["minimum_half_cycle_gain"]:.3f} |',
       f'| 非指尖受力帧占比 | {baseline["non_tip_force_frame_fraction"]:.2%} | {metrics["non_tip_force_frame_fraction"]:.2%} |',
       f'| 命令限幅占比 | {baseline["command_clip_fraction"]:.2%}（20 关节） | {metrics["command_clip_fraction"]:.2%}（12 关节） |',
       '', f'新模型的各项失败回合数（可重叠）：{failure}。',
       '', '验收还要求仅前三指的相向受力夹持 >80%，两条收起手指全部连杆触笔占比 <1%，并保留原跟踪误差、工作空间、倾角、非指尖接触和转角幅度门槛。收起手指接触会受到惩罚，但没有通过删除碰撞来隐藏接触。低触笔占比必须与存活和跟踪一起判断；笔已掉落时低触笔不能算改善。',
       '', '新旧初态、动作维度和奖励同时改变，不能把差异单独归因于收指。软件保持的是命令，真实关节仍有重力和接触响应。严格成功为零时，结果仍是实验模型。',
       '', '完整视频：`artifacts/wuji_hand2/palm_up_tripod_v3/rollout.mp4`；精确 12 动作权重：`policy.pt`，不能用 20 动作旧部署器直接加载。']
    text='\n'.join(lines)+'\n';(report/'REPORT.md').write_text(text);(dest/'REPORT.md').write_text(text)
    (report/'comparison.json').write_text(json.dumps(comp,indent=2));(dest/'comparison.json').write_text(json.dumps(comp,indent=2))
    (dest/'README.md').write_text('# 三指转笔，收起无名指与小拇指\n\n配置和结果见 REPORT.md，完整视频为 rollout.mp4。policy.pt 有 12 个动作，使用本轮控制器；收起两指的真实执行器、连杆和碰撞保留。snapshot 为精确训练快照，assets 包含完整新初态搜索/认证证据；实际结果以 metrics.json 为准。scene.xml 使用仓库内相对网格路径。\n')
    manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
    (dest/'archive_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(comp,indent=2))


if __name__=='__main__':main()

"""Archive and report the independent palm-up opposing-pinch experiment."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def main():
    cfg_path=Path(__file__).with_name('config.json');cfg=json.loads(cfg_path.read_text())
    base=ROOT/cfg['runs'];dest=ROOT/cfg['artifacts'];assets=ROOT/cfg['assets']
    status=json.loads((base/'status.json').read_text())
    if 'holdout' not in status:raise RuntimeError('Wait for final evaluation and video')
    selected=ROOT/status['selected_checkpoint'];train_cfg=json.loads((selected.parent/'config.json').read_text())
    metrics=status['holdout'];rows=metrics['per_episode']
    failures=dict(tracking_mae=sum(r['mae_deg']>3 for r in rows),
                  half_cycle_amplitude=sum(r['minimum_half_cycle_gain']<.9 for r in rows),
                  command_clipping=sum(r['command_clip_fraction']>=.05 for r in rows),
                  opposing_pinch=sum(r['opposing_pinch_fraction']<=.8 for r in rows),
                  non_tip_contact=sum(r['non_tip_force_frame_fraction']>=.05 for r in rows),
                  dropped=sum(not r['survived'] for r in rows))
    certification=json.loads((base/'pinch_grasp_certification.json').read_text())
    if not certification['passed']:raise RuntimeError('Pinch initial grasp not certified')
    shutil.copytree(selected.parent/'snapshot',dest/'snapshot',dirs_exist_ok=True)
    shutil.copytree(cfg_path.parent,dest/'experiment',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['config.json','metrics.jsonl','source_manifest.json']:
        shutil.copy2(selected.parent/name,dest/('training_'+name))
    for name in ['env.py','train.py','evaluate.py','pinch.py','ring_control.py','motion_audit.py']:
        target=dest/'evaluation_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'experiments/wuji_hand2'/name,target)
    for name in ['provenance.json','grasp_bank_60s.npz','grasp_bank_60s_certification.json']:
        shutil.copy2(assets/name,dest/name)
    shutil.copy2(assets/'scene.xml',dest/'original_scene.xml')
    scene=ET.parse(assets/'scene.xml')
    for mesh in scene.getroot().findall('./asset/mesh'):
        mesh.set('file',os.path.relpath((assets/mesh.get('file')).resolve(),dest))
    scene.write(dest/'scene.xml',encoding='utf-8',xml_declaration=True)
    shutil.copy2(base/'pinch_grasp_certification.json',dest/'pinch_grasp_certification.json')
    validation=dest/'validation';validation.mkdir(exist_ok=True)
    for p in base.glob('eval_*/metrics.json'):shutil.copy2(p,validation/(p.parent.name+'.json'))
    reference=ROOT/cfg['reference_video']
    sheet=Path('/tmp/revo3_reference_contact_sheet.png')
    if sheet.exists():shutil.copy2(sheet,dest/'reference_contact_sheet.png')
    executed=sum(json.loads((ROOT/s['selected_checkpoint']).parent.joinpath('config.json').read_text())['iterations'] for s in status['stages'])
    lineage=dict(reference_video=cfg['reference_video'],reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                 initial_checkpoint=cfg['initial_checkpoint'],initial_sha256=hashlib.sha256((ROOT/cfg['initial_checkpoint']).read_bytes()).hexdigest(),
                 selected_checkpoint=status['selected_checkpoint'],selected_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
                 executed_new_iterations=executed,executed_environment_steps=executed*1024*32,
                 scene_sha256=hashlib.sha256((assets/'scene.xml').read_bytes()).hexdigest(),
                 bank_sha256=hashlib.sha256((assets/'grasp_bank_60s.npz').read_bytes()).hexdigest(),
                 palm_normal_world=[0,0,1],palm_forward_weights_used=False)
    (dest/'lineage.json').write_text(json.dumps(lineage,indent=2))
    (dest/'failure_counts.json').write_text(json.dumps(failures,indent=2))
    report=ROOT/'reports/wuji_hand2/palm_up_pinch_v2';report.mkdir(parents=True,exist_ok=True)
    task_matched=metrics['amplitude_rad']==.6 and metrics['period_seconds']==6
    lines=['# Wuji 掌心向上、夹持转笔实验','',
      '已按最新要求改回掌心朝上。参考 `artifacts/revo3/v2.0/wide_6s/demo_60s.mp4` 的多个时刻和对应配置：目标 ±0.6 rad（±34.38°），周期 6 秒，手指弯曲夹持自由笔杆，随目标往复转动。参考视频不是连续 360°绕指旋转。',
      '', '独立配置：`experiments/wuji_hand2/palm_up_pinch_v2/config.json`。使用既有掌心向上模型的精确场景与已认证初态，仅迁移早期掌心向上 checkpoint 550；没有使用掌心向前权重，历史实验不覆盖。掌心世界法线 +Z，腕部四元数 [0.5,0.5,-0.5,-0.5]。',
      '', '新增夹持定义：至少两个不同手指的末端接触各具有 >0.02 N 的净法向力，方向余弦 ≤−0.3。同向托住笔的两个手指不算夹持；训练奖励鼓励相向受力，严格验收还要求相向夹持帧占比 >80%。保留原跟踪误差、半周期幅度、指尖接触、非指尖受力和限幅门槛。自由笔杆保留重力、自由关节和接触动力学，没有焊接、轨迹脚本或额外支撑。',
      '', '新初态认证：60 秒常值握持控制，含无扰动与 4 次随机扰动，全 5 次均通过，相向夹持占比 100%。无名指使用 natural_v1 软件控制范围、命令限速和协调奖励；机械极限不改变。',
      '', f'实际追加训练 {executed} 轮（{executed*1024*32:,} 环境步）。选择 `{status["selected_checkpoint"]}`，使用固定验证种子 {cfg["evaluation"]["validation_seed"]}；另以种子 {metrics["seed"]} 做 24 次 × 60 秒留出测试。',
      '', f'本次留出目标：±{metrics["amplitude_rad"]} rad，周期 {metrics["period_seconds"]} 秒。与参考视频幅度/周期匹配：{task_matched}。流水线状态：`{status["outcome"]}`。',
      '', '| 指标 | 留出结果 |','|---|---:|',
      f'| 不掉笔 | {metrics["survival_rate"]*24:.0f}/24 |',
      f'| 工作空间内保持 | {metrics["workspace_survival_rate"]*24:.0f}/24 |',
      f'| 严格夹持转动成功 | {metrics["quality_success_rate"]*24:.0f}/24 |',
      f'| 相向夹持帧占比 | {metrics["opposing_pinch_fraction"]:.2%} |',
      f'| 最差一回合夹持帧占比 | {min(r["opposing_pinch_fraction"] for r in rows):.2%} |',
      f'| 跟踪 MAE | {metrics["mae_deg"]:.3f}° |',
      f'| 指尖双指接触占比 | {metrics["two_tip_fraction"]:.2%} |',
      f'| 非指尖受力帧占比 | {metrics["non_tip_force_frame_fraction"]:.2%} |',
      f'| 命令限幅占比 | {metrics["command_clip_fraction"]:.2%} |',
      f'| 各回合最小半周期幅度增益的均值 | {metrics["minimum_half_cycle_gain"]:.3f} |',
      '', f'24 次测试的失败项计数（可重叠）：{failures}。',
      '', '不能以不掉笔或相向接触单独宣称任务完成。严格通过不足时，还需要解决跟踪幅度、失夹、非指尖接触或限幅等实际失败项。GPU 训练分数不能代替 CPU 闭环留出结果。若在小角度阶段停止，则这次结果未完成参考视频的 ±0.6 rad 目标。',
      '', '完整视频：`artifacts/wuji_hand2/palm_up_pinch_v2/rollout.mp4`。权重、配置、训练源码快照、接触认证、每个检查点验证结果和原始留出轨迹一起保存。']
    text='\n'.join(lines)+'\n';(report/'REPORT.md').write_text(text);(dest/'REPORT.md').write_text(text)
    (report/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (dest/'README.md').write_text('# 掌心向上夹持转笔\n\n本轮独立配置和结果见 REPORT.md。policy.pt 需要本轮 natural_v1 与 opposing-pinch 配置；不要与掌心向前权重混用。scene.xml 使用仓库内相对网格路径；snapshot 保存精确训练代码，evaluation_source 保存评估代码。完整演示为 rollout.mp4；实际任务是否成功以 metrics.json 为准。\n')
    manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
    (dest/'archive_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(lineage=lineage,result={k:v for k,v in metrics.items() if k!='per_episode'}),indent=2))


if __name__=='__main__':main()

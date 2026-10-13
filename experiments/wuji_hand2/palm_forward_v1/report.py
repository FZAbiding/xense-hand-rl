"""Write the new upright comparison from the frozen, held-out evaluations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    folder=ROOT/'reports/wuji_hand2/palm_forward_v1'
    comparison=json.loads((folder/'comparison.json').read_text())
    status=json.loads((ROOT/'runs/wuji_hand2/palm_forward_v1/final_task/status.json').read_text())
    wuji=status['holdout']; rows=wuji['per_episode']; n=wuji['episodes']
    table=['| 手型 | 全程持笔 | 工作空间存活 | 严格通过 | 平均角误差 | 命令裁剪 | 非指尖受力帧 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in comparison['evaluations'].items():
        table.append(f"| {name} | {round(row['survival_rate']*n)}/{n} | {round(row['workspace_survival_rate']*n)}/{n} | {round(row['quality_success_rate']*n)}/{n} | {row['mae_deg']:.2f}° | {row['command_clip_fraction']:.1%} | {row['non_tip_force_frame_fraction']:.1%} |")
    checks={'掉笔':lambda r:not r['survived'], '工作空间':lambda r:not r['within_workspace'],
            '平均角误差 >3°':lambda r:r['mae_deg']>3,
            '平均倾角 >10° 或 P95 >15°':lambda r:r['mean_tilt_deg']>10 or r['p95_tilt_deg']>15,
            '命令裁剪 ≥5%':lambda r:r['command_clip_fraction']>=.05,
            '非指尖受力帧 ≥5%':lambda r:r['non_tip_force_frame_fraction']>=.05,
            '掌部受力帧 ≥1%':lambda r:r['palm_force_frame_fraction']>=.01,
            '双指尖接触帧 ≤80%':lambda r:r['two_tip_fraction']<=.8,
            '任一半周期幅度 <90%':lambda r:r['minimum_half_cycle_gain']<.9}
    failures='\n'.join(f'- {name}：{sum(check(r) for r in rows)}/{n} 回合' for name,check in checks.items())
    trajectories=[]; iterations=0
    for path,config in [('runs/wuji_hand2/palm_forward_v1','config.json'),
                        ('runs/wuji_hand2/palm_forward_v1/matched_task','matched_task_config.json'),
                        ('runs/wuji_hand2/palm_forward_v1/final_task','final_task_config.json')]:
        state=json.loads((ROOT/path/'status.json').read_text())
        cfg=json.loads((ROOT/'experiments/wuji_hand2/palm_forward_v1'/config).read_text())
        for stage in state['stages']:
            params=next(s for s in cfg['training']['stages'] if s['name']==stage['name'])
            iterations+=params['iterations']
            metrics=stage['metrics']
            trajectories.append(f"- `{path}/{stage['name']}`：新增 {params['iterations']} 轮，±{params['amplitude']} rad/{params['period']} s；验证严格通过 {metrics['quality_success_rate']:.0%}，持笔 {metrics['survival_rate']:.0%}；选中 `{stage['selected_checkpoint']}`。")
    passed=round(wuji['quality_success_rate']*n)
    text=f'''# Wuji Hand 2 竖直、掌心朝前：独立同任务实验

本轮已完成新场景、抓握认证、从零训练及扩幅提速、独立 CPU 评估与成果归档。最终 Wuji 为 **{passed}/{n} 严格通过**、{round(wuji['survival_rate']*n)}/{n} 全程持笔，平均角误差 {wuji['mae_deg']:.2f}°。这是当前仿真适配、初始抓握与训练策略的结果，不能推断真实硬件优劣。

## 布局与隔离

底座网格最低点在桌面 Z=0；手指伸展方向 +Z；掌心法线 -X。官方几何、质量/惯量、关节限位与执行器力矩范围已逐项校验；位置增益 kp=3 为任务适配。见 [布局预览](../../../assets/wuji_hand2_palm_forward_v1/orientation_preview.png)。旧掌心朝下、掌心朝上场景、抓握库和权重均未用于本轮。

布局与启动训练配置为 [config.json](../../../experiments/wuji_hand2/palm_forward_v1/config.json)，扩幅配置为 [matched_task_config.json](../../../experiments/wuji_hand2/palm_forward_v1/matched_task_config.json)，最终任务配置为 [final_task_config.json](../../../experiments/wuji_hand2/palm_forward_v1/final_task_config.json)。各阶段采用新目录，失败数据保留；恢复训练只使用本布局的检查点，并核对场景/抓握库签名。

## 抓握和训练谱系

首轮 729 个候选无合格抓握。第二轮搜索有 2 个 12 秒候选，但完整扰动认证失败；局部第三轮有 6 个松弛候选，仍未通过完整认证。静态预紧控制搜索的 1024 个候选中有 64 个通过 12 秒筛选，保留前 32 个后，4 个通过无扰动加 4 次扰动的 60 秒认证。最终抓握库包含这 4 个初态，认证采用与训练/CPU 评估一致的 8 次求解迭代。场景和认证库签名见 `assets/wuji_hand2_palm_forward_v1/certified_bank_manifest.json`。

GPU 静态诊断中 4 个初态均持笔 60 秒，其中 1 个的接触分项未达到 CPU 认证口径，说明后端接触差异仍需关注。

{chr(10).join(trajectories)}

本轮累计执行 {iterations} 轮、{iterations*1024*32:,} 个环境步，包含最终未被选中的训练分支。所有策略使用训练种子 1，没有跨训练种子统计。启动配置的裁剪惩罚为 2.0；后续探索标准差上限分别为 0.1、0.05。扩幅阶段未满足全回合存活门槛，故另用最终任务配置完成同任务失败基准；这些调整没有放宽最终严格验收门槛。

## 固定共同任务与最终结果

三手共同控制 180 mm ×18 mm、25 g 自由笔杆；40 Hz 策略控制、5 ms 物理步长；目标偏航角 `0.6 sin(2πt/4)` rad。每只手在留出种子 20261018 上做 24×60 秒 CPU 闭环测试，关节扰动标准差 0.005 rad、笔位置扰动 0.4 mm，中途不自动复位。最终 Wuji 检查点先在验证种子 20261017 上固定，再执行留出测试，未根据最终测试改选模型。

LEAP 和 Revo3 复测已有的共同任务策略；三手训练算法细节、初态和预算并不完全相同，因此这是同物理任务与同验收口径的比较。

{chr(10).join(table)}

严格门槛：完整存活且不出工作空间、MAE≤3°、平均倾角≤10°、P95倾角≤15°、命令裁剪<5%、非指尖受力帧<5%、掌部受力帧<1%、双指尖接触帧>80%，每半周期幅度至少达到目标的90%。Wuji 不合格分项如下（同一回合可有多项失败）：

{failures}

掉笔后仍记录至完整 60 秒，平均角误差和接触比例包含失败后的轨迹；不能单独用低非指尖接触比例宣称表现好。严格通过率是共同判据。

## 复核与归档

最终权重、训练配置/日志、源码快照、认证、60 秒轨迹与视频在 [artifacts/wuji_hand2/palm_forward_v1/final_task/](../../../artifacts/wuji_hand2/palm_forward_v1/final_task/)。前两阶段分别保留在根目录和 `matched_task/`，实际任务幅度以其配置为准。最终视频是第一个回合完整 60 秒轨迹，不自动复位，画面标注了质量验收结果。

共同摘要和检查点/场景 SHA-256 在 [comparison.json](comparison.json)，最终逐回合数据见 [Wuji metrics.json](../../../runs/wuji_hand2/palm_forward_v1/final_task/holdout_cpu60/metrics.json)。LEAP/Revo3 原始结果在 `runs/comparison/palm_forward_v1/*_cpu60_seed20261018/`。

下一步应优先检查 CPU/Warp 的逐步接触一致性、抓握在完整 ±0.6 rad 工作空间中的可达性，以及控制限位附近的策略行为；本轮尚不能作为通过验收的 Wuji 策略发布。
'''
    (folder/'REPORT.md').write_text(text)
    print(folder/'REPORT.md')


if __name__=='__main__':main()

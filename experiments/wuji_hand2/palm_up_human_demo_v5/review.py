"""Render and archive V5 calibration and physical previews for human review."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from tools.render_experiment_review import render_review
from .common import ROOT,ASSETS,RUNS,ART,FINGERS,order_measure,chain
from .visualize import render


def main():
    reports=ROOT/'reports/wuji_hand2/palm_up_human_demo_v5';reports.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    initial=read(ART/'initial_settled/metrics.json');motion=read(ART/'motion_settled/metrics.json');handover=read(ART/'handover_probe/metrics.json')
    calibration=read(RUNS/'calibration.json')
    reference=ART/'reference';reference.mkdir(exist_ok=True)
    src=ROOT/'artifacts/wuji_hand2/palm_up_human_demo_v4/reference'
    for name in ['human_A.mp4','annotation.json','annotated_00.jpg','annotated_02.jpg','annotated_04.jpg']:
        shutil.copy2(src/name,reference/name)
    stages=read(Path(__file__).with_name('reference_stages.json'))
    shutil.copy2(Path(__file__).with_name('reference_stages.json'),reference/'stages.json')
    initial_pose=np.load(ART/'initial_settled/candidate_pose.npz')['qpos']
    render([initial_pose],ART/'initial_settled/initial_pose.png','settled candidate; awaiting human review',video=False)
    for stage,title in [('initial_settled','fixed control: long-duration failure retained'),('motion_settled','15 deg physical preview; robustness failed'),('handover_probe','release/transfer failed; not a handover')]:
        z=np.load(ART/stage/'rollout.npz');rows=read(ART/stage/'frame_metrics.json')
        render(z['qpos'],ART/stage/'rollout.mp4',title,metrics=rows)
    z=np.load(ART/'initial_settled/rollout.npz');rows=read(ART/'initial_settled/frame_metrics.json')
    render(z['qpos'][:480],ART/'initial_settled/short_preview.mp4','first 12 s of nominal hold; not full 60 s pass',metrics=rows[:480])
    model=mujoco.MjModel.from_xml_path(str(ASSETS/'scene.xml'));data=mujoco.MjData(model)
    positive=model.qpos0.copy();positive[:20]=np.radians([45,-25,30,25,35,0,35,30,35,0,35,30,75,0,95,65,75,0,95,65]);positive[20:23]=[0,0,-1]
    render([positive],ART/'natural_curl_calibration.png','neutral parallel curl: geometric calibration only',video=False)
    negative=np.load(ROOT/'artifacts/wuji_hand2/palm_up_human_demo_v4/partial_rotation/rollout.npz')['qpos_0'][0]
    render([negative],ART/'rejected_v4_calibration.png','rejected V4 crossing: NEGATIVE example only',video=False)
    z=np.load(ART/'motion_settled/rollout.npz');q=z['qpos'];rows=read(ART/'motion_settled/frame_metrics.json');t=(np.arange(len(q))+1)*.025
    fig,axes=plt.subplots(4,1,figsize=(11,11),sharex=True)
    axes[0].plot(t,[np.degrees(r['angle']) for r in rows],label='Actual directed pen angle');axes[0].axhline(15,ls='--',label='15 degree endpoint target');axes[0].legend();axes[0].set_ylabel('deg')
    axes[1].plot(t,[r['order']['tip_order_gap_m']*1000 for r in rows],label='Index - middle tip order gap');axes[1].axhline(6,ls='--',color='red',label='Conservative task minimum');axes[1].set_ylabel('mm');axes[1].legend()
    for j,label in [(6,'Index PIP'),(7,'Index DIP'),(10,'Middle PIP'),(11,'Middle DIP')]:axes[2].plot(t,np.degrees(q[:,j]),label=label)
    axes[2].set_ylabel('deg');axes[2].legend(ncol=2)
    f=np.array([r['forces'] for r in rows])
    for j,label in enumerate(['Thumb','Index','Middle','Ring','Pinky','Other links','Palm']):axes[3].plot(t,f[:,j],label=label)
    axes[3].legend(ncol=4);axes[3].set_ylabel('N');axes[3].set_xlabel('Time (s)')
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('V5 natural-order candidate: nominal 15 deg; 16/24 angle passes; human review pending');fig.tight_layout();fig.savefig(ART/'motion_settled/diagnostics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(14,8),sharex=True)
    for j,ax in enumerate(axes.flat):ax.plot(t,np.degrees(q[:,j]));ax.set_title(model.joint(j).name);ax.set_ylabel('deg');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(ART/'motion_settled/all_active_joints.png',dpi=150);plt.close(fig)
    # Audit actual contact locations along the directed pen axis and volar-pad local y.
    cc=read(ART/'motion_settled/contacts.json');locations={}
    for pose,contacts in zip(q,cc):
        data.qpos[:]=pose;mujoco.mj_forward(model,data);axis=data.xmat[model.body('pen').id].reshape(3,3)[:,2]
        for c in contacts:
            if c['normal_n']<=.02:continue
            r=locations.setdefault(c['body'],dict(axial_m=[],local_y_m=[]))
            r['axial_m'].append(float(np.dot(np.array(c['world_position_m'])-pose[20:23],axis)))
            r['local_y_m'].append(c['local_position_m'][1])
    audit={name:dict(axial_range_mm=[min(r['axial_m'])*1000,max(r['axial_m'])*1000],local_y_range_mm=[min(r['local_y_m'])*1000,max(r['local_y_m'])*1000],contact_samples=len(r['axial_m'])) for name,r in locations.items()}
    (ART/'motion_settled/contact_location_audit.json').write_text(json.dumps(audit,indent=2))
    summary=dict(status='physical_previews_complete_awaiting_human_posture_review',training_started=False,human_review='pending',
        full_task_achieved=False,calibration_negative_frames_detected=calibration['rejected_v4_detected'],
        selected_candidate=52,settled_initial_pen_depth_mm=initial['initial_pen_contact_depth_m']*1000,
        initial_12s_holdout_passed=initial['holdout_passed'],initial_holdout_episodes=24,
        nominal_long_hold_seconds=initial['nominal_60s']['seconds'],nominal_60s_passed=False,
        small_motion_nominal_deg=motion['nominal']['final_angle_deg'],small_motion_physical_passed=motion['physical_holdout_passed'],
        small_motion_angle_passed=motion['angle_holdout_passed'],motion_holdout_episodes=24,
        contact_region_status='thumb contact includes side-edge locations; not accepted as a volar-pad reference',
        handover_holdout_passed=handover['holdout_passed'],handover_holdout_episodes=24,
        next_action='Human review of initial posture and short preview; then improve long hold and release/transfer before any training reference is accepted.')
    for path in [reports/'summary.json',RUNS/'status.json',ART/'summary.json']:path.write_text(json.dumps(summary,indent=2))
    text=f'''# Wuji v5：自然指序标定与物理预演

用户已批准执行 v5 计划。本轮完成独立标定、初态搜索、短转动与释放/交接探测；**尚未进行 PPO 或模仿训练，所有姿态仍待人工审阅**。

## 当前结果

| 检查 | 结果 | 结论 |
|---|---|---|
| 旧交叉动作回归检查 | v4 的 280/280 帧被新约束检出 | 该反例得到覆盖，不能据此证明覆盖所有异常姿态 |
| 新初态搜索 | 160 个候选，66 个通过 2 秒指标，前 8 名中 3 个通过 8 秒 | 独立新姿态，未用旧初态库或权重 |
| 原始 IK 初态候选 52 | 8/8 验证、24/24 × 12 秒扰动测试通过 | 含约 2.15 mm 初始接触压缩，不作为推荐初态 |
| 物理稳定后初态 | 真实仿真 1 秒后的状态，初始笔接触压缩 {summary['settled_initial_pen_depth_mm']:.3f} mm | 推荐审阅这个低压缩初态；重置速度为 0 |
| 稳定后初态的 12 秒扰动复测 | {initial['holdout_passed']}/24 | 低于 23/24 门槛，未达标 |
| 请求 60 秒的名义固定控制 | 约 {initial['nominal_60s']['seconds']:.2f} 秒后因倾斜超限停止 | 失败视频完整保留；没有补齐成 60 秒 |
| 4 秒小角度物理预演 | 实际 {motion['nominal']['final_angle_deg']:.2f}°；最大位移 {motion['nominal']['max_drift_m']*1000:.2f} mm | 开环控制候选；不是已训练策略 |
| 小角度扰动复测 | 物理夹持 {motion['physical_holdout_passed']}/24，达到 13.5–18° 为 {motion['angle_holdout_passed']}/24 | 未达到动作鲁棒性门槛 |
| TI→TM 交接探测 | 16 条释放安排，独立扰动验收 {handover['holdout_passed']}/24 | 未实现合格交接 |

新初态及名义 15° 预演中未检出指序违规、食指/中指反弯或活跃手指自穿透；短预演非指尖/闲置手指接触均未超既定力阈值。注意：这是有限任务约束下的检测结果，**不能代替人工确认自然动作**。

接触位置审计也发现了尚未解决的问题：拇指触点在其传感器局部 y 约 -0.88–0.23 mm，包含侧缘位置，不能把“接触的是指尖连杆”直接说成“已经实现正面指腹夹持”。食指、中指局部 y 分别约 2.71–4.27 mm、3.29–3.95 mm。各触点沿笔轴相对重心的范围见 `motion_settled/contact_location_audit.json`。拇指接触区域须继续核对并约束，当前候选不作为已合格指腹参考。

## v5 与旧版的差别

使用原 `assets/wuji_hand2_palm_up/scene.xml` 的独立逐字节副本；不修改碰撞、笔尺寸/质量、机械限位和接触参数。掌面局部 +x 为中指到食指方向，-z 为远端方向，+y 为掌面法线。对整条指骨在掌面投影的相交情况与对应骨点的横向次序同时检查，并对食指/中指侧摆给出暂定 ±10° 软件范围。6 mm 指序间隙是基于当前中立模型的保守任务约束，不是人体普遍阈值。

所有候选从新 IK 搜索生成。v4 轨迹只用作拒绝反例，从未参与正样本初始化、优化目标或权重继承。笔只在各回合开始放置，之后运动来自实际手指执行器与自由笔的接触动力学。

拇指=金色，食指=橙色，中指=绿色，其他手指=灰色。画面只隐藏碰撞几何的渲染覆盖，不关闭物理碰撞。三个视角为同一仿真时刻，无剪辑循环或伪同步。

## 示范分解和交接限制

已批准短片的可观察阶段与机器人假设单独记录在 `reference/stages.json`。139–141.5 秒是连续观测段；142 秒镜头切换不用于补齐循环。接触顺序有遮挡，不能当作精确动作捕捉真值。本轮分别检查拇指让开，以及中指让开→重新接近→食指让开的三段控制；真实接触对需要连续 4 帧（0.1 秒）建立才记录交接。当前没有检测到合格 TI→TM 转移。

主要问题已从明显的指序交叉转为自然工作空间内的长期稳定和接触释放：固定控制会逐渐倾斜并接触闲置指；小角度转动对微小初态扰动仍敏感；释放动作容易使笔倾斜，未形成稳定的新接触对。不能把 15° 转动当作换指成功，也不能将失败候选作为训练参考。

## 人工审阅与后续门槛

请先检查 `review.html` 中低压缩初态和 15° 预演的手指姿态是否符合预期，尤其食指/中指是否并排、指尖屈曲是否自然、拇指是否合理夹持。当前不请求把失败交接批准为合格动作。

人工认可姿态后，先改善长时夹持、转角鲁棒性和明确释放，再做预演复查；满足既定门槛后才生成训练库和开始分阶段学习。依据已确认的 `PLAN.md`：**“短预演须经人工确认动作合理后，才允许作为模仿或强化学习参考。”**

数据入口：各阶段的 `metrics.json`、`frame_metrics.json`、`contacts.json`、`rollout.npz`、`controller.npz`/`candidate_pose.npz`。逐接触记录包括位置、法向力和接触坐标系力；本轮没有把它当成完整力矩平衡优化。代码、配置假设、源文件哈希与原模型来源均保留在独立 v5 目录。
'''
    (ART/'README.md').write_text(text);(reports/'REPORT.md').write_text(text)
    manifest=dict(title='Wuji v5：自然指序候选与真实物理短预演',
        summary=f'姿态待人工审阅，尚未训练。稳定后初态 12 秒扰动 {initial["holdout_passed"]}/24；15° 预演物理夹持 24/24、转角 16/24；交接 0/24。长时夹持与鲁棒性仍失败。',
        cards=[dict(title='真人参考：用户已批准',status='reference',video='reference/human_A.mp4',description='仅作动作意图参考，不把遮挡接触猜测当作真值。'),
               dict(title='新初态：请检查自然指序',status='pending',video='initial_settled/short_preview.mp4',description='名义固定控制前 12 秒片段；金=拇指、橙=食指、绿=中指。完整长时失败见下方。'),
               dict(title='15° 真实物理短预演',status='pending',video='motion_settled/rollout.mp4',description='开环执行器控制；名义 15.02°，扰动角度验收仅 16/24。拇指含侧缘接触，尚未成为合格指腹参考，姿态待人工审阅。',links=[dict(label='逐回合指标',path='motion_settled/metrics.json'),dict(label='实际接触位置审计',path='motion_settled/contact_location_audit.json')]),
               dict(title='失败：接触释放与交接探测',status='failed',video='handover_probe/rollout.mp4',description='未形成合格新接触对，倾斜超限提前结束。不能当作转笔成功。'),
               dict(title='失败：长时固定控制',status='failed',video='initial_settled/rollout.mp4',description=f'请求 60 秒，在 {initial["nominal_60s"]["seconds"]:.2f} 秒结束。保存真实失败过程，不补帧。')],
        images=[dict(label='低压缩初态三视角（待人工审阅）',path='initial_settled/initial_pose.png'),
                dict(label='中立并排屈曲：仅几何标定',path='natural_curl_calibration.png'),
                dict(label='旧版交叉反例：已被新检测拒绝',path='rejected_v4_calibration.png'),
                dict(label='新短预演：转角、指序、关节和接触',path='motion_settled/diagnostics.png'),
                dict(label='全部 12 个活跃关节',path='motion_settled/all_active_joints.png')],
        links=[dict(label='完整结果和限制',path='README.md'),dict(label='机器可读摘要',path='summary.json'),
               dict(label='已批准工作计划',path='../../../experiments/wuji_hand2/palm_up_human_demo_v5/PLAN.md'),
               dict(label='展示 SOP',path='../../../docs/EXPERIMENT_REVIEW_SOP.md'),dict(label='真人可见证据与待验证假设',path='reference/stages.json')])
    (ART/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));render_review(manifest,ART/'review.html')
    (Path(__file__).parent/'README.md').write_text('# Wuji v5\n\n已批准计划的标定与物理预演已完成，姿态等待人工审阅；没有开始训练。结果见 `artifacts/wuji_hand2/palm_up_human_demo_v5/review.html` 和 `reports/wuji_hand2/palm_up_human_demo_v5/REPORT.md`。\n\n执行入口依次为 `prepare.py`、`verify_initial.py`、`settle_initial.py`、`search_motion.py --initial initial_settled --output motion_settled --seed 2026101207`、`probe_handover.py`、`review.py`。拒绝覆盖已有实验产物。\n')
    archive=RUNS/'source_snapshot';archive.mkdir(exist_ok=True)
    for f in Path(__file__).parent.iterdir():
        if f.is_file():shutil.copy2(f,archive/f.name)
    files=[p for directory in [ART,reports,ASSETS,RUNS,Path(__file__).parent] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sha256_manifest.json']
    (reports/'sha256_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

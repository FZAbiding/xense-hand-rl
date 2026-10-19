"""Compare frozen v7/v8 first-episode pen contacts on the same collision meshes.

This is an evidence plot, not an anatomical pad classifier or a new pass gate.
Both inputs were produced with the v8 scene and fixed seed 2026101701.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets/wuji_hand2_palm_up_human_demo_v8/scene.xml"
SOURCES = {
    "v7 baseline": ROOT / "reports/wuji_hand2/palm_up_human_demo_v8/preflight_20261017/baseline_v7_legacy_obs_sensor_gate/contacts.json",
    "v8 iteration 996": ROOT / "reports/wuji_hand2/palm_up_human_demo_v8/pilot_20261017/final_0000996/fixed_validation/contacts.json",
}
OUTPUT = ROOT / "reports/wuji_hand2/palm_up_human_demo_v9/gate0_20261019"
FINGERS = ("thumb", "index_finger", "middle_finger")
PARTS = ("distal", "tip_sensor_frame")
COLORS = {"distal": "#e67633", "tip_sensor_frame": "#2477b9"}


def mesh_vertices_in_body(model, body_name):
    body_id = model.body(body_name).id
    vertices = []
    for geom_id in np.flatnonzero(model.geom_bodyid == body_id):
        if model.geom_group[geom_id] != 2 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[geom_id]
        begin = model.mesh_vertadr[mesh_id]
        end = begin + model.mesh_vertnum[mesh_id]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        vertices.append(model.mesh_vert[begin:end] @ rotation.reshape(3, 3).T + model.geom_pos[geom_id])
    if not vertices:
        raise ValueError(f"No collision mesh on {body_name}")
    return np.concatenate(vertices)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    for finger in FINGERS:
        sensor = model.body(f"r_{finger}_tip_sensor_frame")
        if not np.allclose(model.body_pos[sensor.id], 0):
            raise ValueError(f"Sensor origin changed for {finger}; transform contact coordinates")

    source_data = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    if any(len(frames) != 220 for frames in source_data.values()):
        raise ValueError("Expected 220 frames in each fixed-validation first episode")
    summary = {
        "scope": "First fixed-validation episode only; contact surface anatomy and motion acceptance remain pending.",
        "seed": 2026101701,
        "force_filter_n": 0.02,
        "scene": str(SCENE.relative_to(ROOT)),
        "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        "sources": {},
    }
    points_by_source = {}
    force_by_source = {}
    for name, frames in source_data.items():
        path = SOURCES[name]
        points = defaultdict(list)
        force = defaultdict(lambda: np.zeros(220))
        for frame_no, contacts in enumerate(frames):
            for contact in contacts:
                body = contact["body"]
                if body not in {f"r_{finger}_{part}" for finger in FINGERS for part in PARTS}:
                    continue
                normal = float(contact["normal_n"])
                force[body][frame_no] += normal
                if normal > 0.02:
                    points[body].append([*contact["local_position_m"], normal, frame_no])
        body_summary = {}
        for finger in FINGERS:
            for part in PARTS:
                body = f"r_{finger}_{part}"
                array = np.asarray(points[body]).reshape(-1, 5)
                body_summary[body] = {
                    "contacts_gt_0p02_n": len(array),
                    "active_frames_gt_0p02_n": int(np.count_nonzero(force[body] > 0.02)),
                    "normal_force_sum_n": float(force[body].sum()),
                    "median_local_xyz_mm": (np.median(array[:, :3], axis=0) * 1000).tolist() if len(array) else None,
                    "local_y_5_95_mm": (np.percentile(array[:, 1], [5, 95]) * 1000).tolist() if len(array) else None,
                }
        summary["sources"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frames": len(frames),
            "bodies": body_summary,
        }
        points_by_source[name] = points
        force_by_source[name] = force

    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)
    for row, finger in enumerate(FINGERS):
        for col, name in enumerate(SOURCES):
            ax = axes[row, col]
            counts = []
            for part in PARTS:
                body = f"r_{finger}_{part}"
                vertices = mesh_vertices_in_body(model, body) * 1000
                ax.scatter(vertices[:, 1], vertices[:, 2], s=0.5, alpha=0.08,
                           color=COLORS[part], rasterized=True)
                contacts = np.asarray(points_by_source[name][body]).reshape(-1, 5)
                if len(contacts):
                    ax.scatter(contacts[:, 1] * 1000, contacts[:, 2] * 1000, s=5, alpha=0.25,
                               color=COLORS[part], rasterized=True)
                counts.append(f"{part.replace('tip_sensor_frame', 'sensor')} {len(contacts)}")
            ax.set(title=f"{name}: {finger}\n{', '.join(counts)}", xlabel="Body-local y (mm)", ylabel="Body-local z (mm)")
            ax.grid(alpha=0.2)
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Same v8 mesh and fixed seed: first episode contact locations (>0.02 N)\nBlue: tip sensor body; orange: distal body. Colors do not define anatomical pad.")
    fig.savefig(OUTPUT / "mesh_contact_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    times = np.arange(220) * 0.025
    series = (
        ("r_thumb_tip_sensor_frame", "Thumb sensor", "#8661b0"),
        ("r_thumb_distal", "Thumb distal", "#b38031"),
        ("r_index_finger_tip_sensor_frame", "Index sensor", "#27935c"),
        ("r_middle_finger_tip_sensor_frame", "Middle sensor", COLORS["tip_sensor_frame"]),
        ("r_middle_finger_distal", "Middle distal", COLORS["distal"]),
    )
    for ax, name in zip(axes, SOURCES):
        for body, label, color in series:
            ax.plot(times, force_by_source[name][body], label=label, color=color, linewidth=1.3)
        for start, end in ((1.25, 1.65), (2.75, 3.30)):
            ax.axvspan(start, end, color="#f0cf43", alpha=0.15)
        ax.set(title=name, ylabel="Summed normal force (N)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("First episode time (s); shaded windows require human handover review")
    fig.suptitle("Fixed-seed v7/v8 contact-force evidence; no pad acceptance inferred")
    fig.savefig(OUTPUT / "contact_force_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

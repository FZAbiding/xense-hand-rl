"""Map the frozen Wuji v7 holdout contacts onto the collision mesh frames.

Read-only analysis of the archived 24-episode replay. This script does not
re-evaluate a policy, change a gate, or declare which distal faces are pads.
"""

import argparse
import gzip
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
DEFAULT_CONTACTS = ROOT / "reports/wuji_hand2/palm_up_human_demo_v7/stage_a_20261016/contact_audit/contact_points_all_episodes.json.gz"
DEFAULT_SCENE = ROOT / "assets/wuji_hand2_palm_up_human_demo_v7_long/scene.xml"
DEFAULT_OUTPUT = ROOT / "reports/wuji_hand2/palm_up_human_demo_v7/contact_geometry_20261017"
FINGERS = ("thumb", "index_finger", "middle_finger")
PARTS = ("distal", "tip_sensor_frame")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.contacts, "rt") as file:
        episodes = json.load(file)
    if len(episodes) != 24 or any(len(ep) != 220 for ep in episodes):
        raise ValueError("Expected the frozen 24 x 220-frame v7 holdout")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    contacts = defaultdict(list)
    per_episode = []
    for episode_id, episode in enumerate(episodes):
        episode_rows = defaultdict(list)
        for frame in episode:
            for contact in frame["contacts"]:
                body = contact["body"]
                if contact["normal_n"] <= 0.02 or not any(
                    body == f"r_{finger}_{part}" for finger in FINGERS for part in PARTS
                ):
                    continue
                row = [*contact["local_position_m"], contact["normal_n"]]
                contacts[body].append(row)
                episode_rows[body].append(row)
        for body, rows in episode_rows.items():
            points = np.asarray(rows)
            per_episode.append({
                "episode": episode_id,
                "body": body,
                "contacts_gt_0p02_n": len(points),
                "median_local_y_m": float(np.median(points[:, 1])),
                "normal_force_sum_n": float(points[:, 3].sum()),
            })
    body_summaries = {}
    for body, rows in sorted(contacts.items()):
        points = np.asarray(rows)
        body_summaries[body] = {
            "contacts_gt_0p02_n": len(points),
            "normal_force_sum_n": float(points[:, 3].sum()),
            "local_xyz_quantiles_m": {
                str(q): np.percentile(points[:, :3], q, axis=0).tolist()
                for q in (5, 25, 50, 75, 95)
            },
        }
    summary = {
        "source_contacts": str(args.contacts.relative_to(ROOT)),
        "source_contacts_sha256": hashlib.sha256(args.contacts.read_bytes()).hexdigest(),
        "scene": str(args.scene.relative_to(ROOT)),
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "episodes": len(episodes),
        "frames_per_episode": 220,
        "contact_force_filter_n": 0.02,
        "interpretation": "Body-local position statistics and collision mesh overlay only; pad surfaces are not yet human-calibrated.",
        "bodies": body_summaries,
        "per_episode": per_episode,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"distal": "#e67633", "tip_sensor_frame": "#2477b9"}
    for row, finger in enumerate(("middle_finger", "thumb")):
        for part in PARTS:
            body = f"r_{finger}_{part}"
            if part == "tip_sensor_frame" and not np.allclose(model.body_pos[model.body(body).id], 0):
                raise ValueError("Sensor body offset changed; transform contact points before overlay")
            vertices = mesh_vertices_in_body(model, body) * 1000
            points = np.asarray(contacts[body])[:, :3] * 1000
            for column, coords in enumerate(((1, 2), (0, 1))):
                ax = axes[row, column]
                ax.scatter(vertices[:, coords[0]], vertices[:, coords[1]], s=0.4, alpha=0.08,
                           color=colors[part], rasterized=True)
                ax.scatter(points[:, coords[0]], points[:, coords[1]], s=2, alpha=0.12,
                           color=colors[part], label=f"{part} contact" if row == column == 0 else None,
                           rasterized=True)
        for column, (x, y) in enumerate((("local y (mm)", "local z (mm)"), ("local x (mm)", "local y (mm)"))):
            ax = axes[row, column]
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{finger}: {x.split()[1]}–{y.split()[1]} plane")
            ax.grid(alpha=0.2)
            ax.set_aspect("equal", adjustable="box")
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.96), markerscale=4)
    fig.suptitle("Wuji v7 holdout: contact positions vs collision meshes (24 episodes; >0.02 N)")
    fig.savefig(args.output / "contact_geometry.png", dpi=180)
    plt.close(fig)

    first = episodes[0]
    times = np.asarray([row["seconds"] for row in first])
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for body, color in (
        ("r_middle_finger_distal", colors["distal"]),
        ("r_middle_finger_tip_sensor_frame", colors["tip_sensor_frame"]),
        ("r_index_finger_tip_sensor_frame", "#27935c"),
        ("r_thumb_tip_sensor_frame", "#8661b0"),
    ):
        force = [row["normal_force_n_by_body"].get(body, 0) for row in first]
        ax.plot(times, force, label=body.removeprefix("r_"), color=color, linewidth=1.5)
    ax.set(xlabel="episode time (s)", ylabel="summed contact normal force (N)",
           title="Wuji v7 best checkpoint: first holdout episode contact bodies")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(args.output / "contact_timeline_ep0.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "contact_bodies": len(body_summaries)}))


if __name__ == "__main__":
    main()

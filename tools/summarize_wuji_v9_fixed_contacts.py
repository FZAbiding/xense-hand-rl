"""Summarize paired 24-episode v7/v8 fixed-seed contact-body occupancy.

Collision bodies remain unclassified anatomically; this does not compute success.
"""

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/wuji_hand2/palm_up_human_demo_v9/gate0_20261019"
LABELS = ("v7_baseline", "v8_iteration_996")
FINGERS = ("thumb", "index_finger", "middle_finger")
PARTS = ("distal", "tip_sensor_frame")
FILTER_N = 0.02


def main():
    summary = {
        "scope": "24 paired fixed-validation episodes, seed 2026101701; collision-body occupancy only, no pad labels or new pass gate",
        "filter_per_contact_normal_n_gt": FILTER_N,
        "sources": {},
    }
    for label in LABELS:
        path = OUTPUT / f"{label}_all_contacts.json.gz"
        with gzip.open(path, "rt") as file:
            episodes = json.load(file)
        if len(episodes) != 24 or any(len(ep) != 220 for ep in episodes):
            raise ValueError(f"Expected 24 full 220-frame episodes for {label}")
        by_body = defaultdict(list)
        per_episode = []
        for episode in episodes:
            points = defaultdict(list)
            active = defaultdict(int)
            for frame in episode:
                seen = set()
                for contact in frame["contacts"]:
                    body = contact["body"]
                    if not any(body == f"r_{finger}_{part}" for finger in FINGERS for part in PARTS):
                        continue
                    normal = float(contact["normal_n"])
                    if normal <= FILTER_N:
                        continue
                    points[body].append([*contact["local_position_m"], normal])
                    by_body[body].append([*contact["local_position_m"], normal])
                    seen.add(body)
                for body in seen:
                    active[body] += 1
            row = {}
            for finger in FINGERS:
                for part in PARTS:
                    body = f"r_{finger}_{part}"
                    row[body] = {"active_frames": active[body], "active_fraction": active[body] / 220,
                                 "contacts_gt_filter": len(points[body])}
            per_episode.append(row)
        bodies = {}
        for finger in FINGERS:
            for part in PARTS:
                body = f"r_{finger}_{part}"
                points = np.asarray(by_body[body]).reshape(-1, 4)
                fractions = [row[body]["active_fraction"] for row in per_episode]
                bodies[body] = {
                    "contact_samples_gt_filter": len(points),
                    "active_frame_fraction": float(np.mean(fractions)),
                    "active_frame_fraction_min_max": [float(np.min(fractions)), float(np.max(fractions))],
                    "normal_force_sum_n": float(points[:, 3].sum()),
                    "median_local_xyz_mm": (np.median(points[:, :3], axis=0) * 1000).tolist() if len(points) else None,
                    "local_y_5_95_mm": (np.percentile(points[:, 1], [5, 95]) * 1000).tolist() if len(points) else None,
                }
        summary["sources"][label] = {
            "contact_archive": str(path.relative_to(ROOT)),
            "contact_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "episodes": len(episodes),
            "frames_per_episode": 220,
            "bodies": bodies,
            "per_episode": per_episode,
        }
    (OUTPUT / "full_24_contact_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    chosen = (
        ("r_middle_finger_distal", "Middle distal body"),
        ("r_thumb_distal", "Thumb distal body"),
        ("r_middle_finger_tip_sensor_frame", "Middle sensor body"),
    )
    for ax, (body, title) in zip(axes, chosen):
        for label, color in zip(LABELS, ("#e67633", "#2477b9")):
            fractions = [row[body]["active_fraction"] for row in summary["sources"][label]["per_episode"]]
            ax.plot(np.arange(24), fractions, marker="o", markersize=3, linewidth=1.2,
                    color=color, label=f"{label}: mean {np.mean(fractions):.1%}")
        ax.set(title=title, ylabel="Active fraction of 220 frames", ylim=(0, 1))
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Paired fixed-validation episode index")
    fig.suptitle("Wuji v7/v8 contact-body occupancy, >0.02 N per contact\nA collision body is not an approved anatomical pad")
    fig.savefig(OUTPUT / "paired_24_contact_occupancy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

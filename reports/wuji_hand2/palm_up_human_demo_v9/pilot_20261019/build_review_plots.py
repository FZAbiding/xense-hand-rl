"""Rebuild first-episode contact and joint plots from archived CPU rollout data."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
plt.style.use("dark_background")

for name in ("arm_a_best_0000900", "arm_b_best_observed_0000800"):
    folder = ROOT / name
    frames = json.loads((folder / "frame_metrics.json").read_text())
    qpos = np.load(folder / "rollout.npz")["qpos"]
    time = np.arange(1, len(frames) + 1) * 0.025
    force = np.asarray([frame["forces"] for frame in frames])

    fig, ax = plt.subplots(figsize=(12, 4.3), layout="constrained")
    for channel, label, color in (
        (0, "thumb pad", "#f5bb43"),
        (1, "index pad", "#ef674d"),
        (2, "middle pad", "#3dd6ae"),
        (5, "non-pad", "#b7a8e9"),
    ):
        ax.plot(time, force[:, channel], label=label, color=color, lw=1.6)
    for start, end in ((1.25, 1.65), (2.75, 3.30)):
        ax.axvspan(start, end, color="#f5bb43", alpha=0.12)
    ax.axhline(0.05, color="#b7a8e9", lw=0.8, ls="--", alpha=0.8)
    ax.set(xlabel="Time (s)", ylabel="Normal force (N)", xlim=(0, time[-1]), title=f"{name}: first episode, CPU contact forces")
    ax.legend(ncol=4, loc="upper center")
    fig.savefig(folder / "contact_curve.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.3), layout="constrained")
    for index, label, color in (
        (6, "index PIP", "#ef674d"),
        (7, "index DIP", "#f49a82"),
        (10, "middle PIP", "#3dd6ae"),
        (11, "middle DIP", "#9debd4"),
    ):
        ax.plot(time, np.degrees(qpos[:, index]), label=label, color=color, lw=1.6)
    for start, end in ((1.25, 1.65), (2.75, 3.30)):
        ax.axvspan(start, end, color="#f5bb43", alpha=0.12)
    ax.set(xlabel="Time (s)", ylabel="Joint angle (deg)", xlim=(0, time[-1]), title=f"{name}: first episode, index/middle flexion")
    ax.legend(ncol=4, loc="upper center")
    fig.savefig(folder / "joint_curve.svg")
    plt.close(fig)
    for filename in ("contact_curve.svg", "joint_curve.svg"):
        path = folder / filename
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")

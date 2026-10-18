"""Regenerate first-episode joint and contact curves from archived v8 rollout data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "final_0000996" / "fixed_validation"
rollout = np.load(ROOT / "rollout.npz")
frames = json.loads((ROOT / "frame_metrics.json").read_text())
qpos = rollout["qpos"]
time = np.arange(len(frames)) * float(rollout["dt"])
assert len(qpos) == len(frames) == 220

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
for index, label in [(6, "Index PIP"), (7, "Index DIP"), (10, "Middle PIP"), (11, "Middle DIP")]:
    axes[0].plot(time, np.degrees(qpos[:, index]), label=label)
axes[0].set(ylabel="Joint angle (degrees)", title="Selected v8 diagnostic: first fixed-validation episode")
axes[0].legend(ncol=4)

forces = np.array([row["forces"][:3] for row in frames])
for index, label in enumerate(("Thumb sensor", "Index sensor", "Middle sensor")):
    axes[1].plot(time, forces[:, index], label=label)
axes[1].set(ylabel="Normal force (N)")
axes[1].legend(ncol=3)

axes[2].plot(time, np.degrees([row["angle"] for row in frames]), label="Actual signed angle")
axes[2].plot(time, np.degrees([row["goal"] for row in frames]), label="Goal angle", linestyle="--")
axes[2].set(xlabel="Time (s)", ylabel="Angle (degrees)")
axes[2].legend()

for ax in axes:
    ax.grid(alpha=0.25)
fig.savefig(Path(__file__).with_name("first_episode_curves.png"), dpi=160)

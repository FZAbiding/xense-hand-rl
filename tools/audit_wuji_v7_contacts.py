"""Replay a completed Wuji v7 CPU evaluation and audit actual pen contact bodies.

This leaves the frozen evaluator and policy unchanged. It records the contact
body after every control step and compares the replay with an existing result.
Run from the repository root with ``python -m tools.audit_wuji_v7_contacts``.
"""

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from experiments.wuji_hand2.palm_up_human_demo_v5.common import pen_contacts
from experiments.wuji_hand2.palm_up_human_demo_v7.long_training import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-metrics", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    task = cfg["levels"][checkpoint["curriculum_level"]]
    if task.get("continuous") or task["angle_deg"] == 0:
        raise ValueError("This audit expects the finite 3-second Wuji v7 task")
    frames_per_episode = round(task["episode_seconds"] / .025)
    expected_calls = cfg["validation_episodes"] * frames_per_episode * 2
    original = json.loads(args.original_metrics.read_text())
    if original["checkpoint_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
        raise ValueError("Original evaluation and checkpoint hashes differ")
    if not original["holdout"]:
        raise ValueError("Original metrics are not a holdout evaluation")

    calls = 0
    contact_frames = [[] for _ in range(cfg["validation_episodes"])]
    original_forces = evaluate.contact_forces

    def audited_forces(model, data):
        nonlocal calls
        forces = original_forces(model, data)
        if calls % 2:
            episode = calls // (frames_per_episode * 2)
            frame = calls // 2 % frames_per_episode
            contacts = pen_contacts(model, data)
            per_body = defaultdict(float)
            for contact in contacts:
                per_body[contact["body"]] += contact["normal_n"]
            contact_frames[episode].append({
                "frame": frame,
                "seconds": round((frame + 1) * .025, 3),
                "normal_force_n_by_body": dict(per_body),
                "contacts": contacts,
            })
        calls += 1
        return forces

    evaluate.contact_forces = audited_forces
    evaluate.render = lambda *unused_args, **unused_kwargs: None
    evaluate.update_page = lambda *unused_args, **unused_kwargs: None
    art = Path(cfg["artifacts"])
    global_files = [art / name for name in ("final_result.json", "latest_evaluation.json")]
    previous = {path: path.read_bytes() if path.exists() else None for path in global_files}
    sys.argv = [
        "evaluate", "--checkpoint", str(args.checkpoint), "--holdout",
        "--output-name", args.output_name,
    ]
    try:
        evaluate.main()
    finally:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
    if calls != expected_calls or any(len(frames) != frames_per_episode for frames in contact_frames):
        raise RuntimeError(f"Incomplete contact replay: {calls} calls; expected {expected_calls}")

    out = art / args.output_name
    replay = json.loads((out / "metrics.json").read_text())
    if replay != original:
        raise RuntimeError("Replay metrics differ from the original holdout")

    summaries = []
    for episode, frames in enumerate(contact_frames):
        sensor_two = 0
        distal_gt_005 = 0
        middle_distal_gt_005 = 0
        thumb_distal_gt_005 = 0
        side_like_gt_005 = 0
        for row in frames:
            body = row["normal_force_n_by_body"]
            sensor_two += sum(
                force > .02 for name, force in body.items() if name.endswith("tip_sensor_frame")
            ) >= 2
            distal_gt_005 += any(
                force > .05 for name, force in body.items() if name.endswith("_distal")
            )
            middle_distal_gt_005 += body.get("r_middle_finger_distal", 0) > .05
            thumb_distal_gt_005 += body.get("r_thumb_distal", 0) > .05
            side_like_force = sum(
                contact["normal_n"] for contact in row["contacts"]
                if any(finger in contact["body"] for finger in
                       ("r_thumb", "r_index_finger", "r_middle_finger"))
                and contact["local_position_m"][1] < .001
                and contact["normal_n"] > .02
            )
            side_like_gt_005 += side_like_force > .05
        summaries.append({
            "episode": episode,
            "legacy_two_tip_fraction": replay["episodes"][episode]["gates"]["two_tip_fraction"],
            "sensor_only_two_tip_fraction": sensor_two / frames_per_episode,
            "distal_body_gt_0p05_fraction": distal_gt_005 / frames_per_episode,
            "middle_distal_gt_0p05_fraction": middle_distal_gt_005 / frames_per_episode,
            "thumb_distal_gt_0p05_fraction": thumb_distal_gt_005 / frames_per_episode,
            "side_like_force_gt_0p05_fraction": side_like_gt_005 / frames_per_episode,
        })
    audit = {
        "checkpoint_sha256": original["checkpoint_sha256"],
        "original_metrics_sha256": hashlib.sha256(args.original_metrics.read_bytes()).hexdigest(),
        "seed": original["seed"],
        "replay_exactly_matches_original_metrics": True,
        "note": "The legacy force vector groups distal finger bodies with tip sensor frames. Sensor-only fractions are a separate contact-location diagnostic, not a retrospective change to the frozen legacy gate.",
        "episodes": summaries,
    }
    (out / "contact_audit.json").write_text(json.dumps(audit, indent=2))
    summary_frames = [[{k: v for k, v in row.items() if k != "contacts"} for row in frames]
                      for frames in contact_frames]
    (out / "contact_frames.json").write_text(json.dumps(summary_frames, indent=2))
    with gzip.open(out / "contact_points_all_episodes.json.gz", "wt") as file:
        json.dump(contact_frames, file)
    print(json.dumps({"episodes": len(summaries), "audit": str(out / "contact_audit.json")}))


if __name__ == "__main__":
    main()

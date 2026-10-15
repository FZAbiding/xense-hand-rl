"""Run the approved isolated v7 pilots, select a baseline, and supervise training."""
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
MODULE = "experiments.wuji_hand2.palm_up_human_demo_v7"
CONFIG_DIR = ROOT / "experiments/wuji_hand2/palm_up_human_demo_v7/long_training"
PIPELINE_DIR = ROOT / "runs/wuji_hand2/palm_up_human_demo_v7/pipeline_20261015"
PILOTS = ("course_only", "contact_only", "combined")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def run_logged(command, name):
    path = PIPELINE_DIR / f"{name}.log"
    with path.open("a", buffering=1) as log:
        log.write(f"\nLAUNCH unix={time.time()} command={command}\n")
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{name} exited {process.returncode}; see {path}")


def pilot_config(name):
    return CONFIG_DIR / "pilots" / f"{name}.json"


def pilot_run(name):
    return ROOT / read_json(pilot_config(name))["runs"]


def pilot_artifacts(name):
    return ROOT / read_json(pilot_config(name))["artifacts"]


def evaluate_at(name, level, checkpoint_name="latest.pt"):
    run = pilot_run(name)
    art = pilot_artifacts(name)
    suffix = "" if checkpoint_name == "latest.pt" else f"_{Path(checkpoint_name).stem}"
    output_name = f"cross_level_{level:02d}{suffix}"
    evidence = art / output_name / "metrics.json"
    if evidence.exists():
        return read_json(evidence)
    source = run / checkpoint_name
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint["curriculum_level"] = level
    candidate = run / f"{output_name}.pt"
    torch.save(checkpoint, candidate)
    write_json(run / f"{output_name}_provenance.json", {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": source_sha,
        "evaluation_checkpoint": str(candidate.relative_to(ROOT)),
        "evaluation_checkpoint_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "level": level,
        "policy_weights_unchanged": True,
    })
    run_logged([
        sys.executable, "-m", f"{MODULE}.long_training.evaluate",
        "--checkpoint", str(candidate), "--seed", "2026101501",
        "--output-name", output_name, "--no-feedback",
    ], f"evaluate_{name}_{output_name}")
    return read_json(evidence)


def gate_score(metrics):
    episodes = metrics["episodes"]
    strict = sum(e["strict_passed"] for e in episodes)
    contact = sum("opposing_fraction" not in e["failed_gates"] and
                  "two_tip_fraction" not in e["failed_gates"] and
                  "contact_handovers" not in e["failed_gates"] for e in episodes)
    duration = sum("full_duration" not in e["failed_gates"] for e in episodes)
    mean_seconds = sum(e["seconds"] for e in episodes) / len(episodes)
    return (strict, contact, duration, mean_seconds)


def candidate_checkpoints(name):
    run = pilot_run(name)
    art = pilot_artifacts(name)
    names = ["latest.pt"]
    for evidence in sorted(art.glob("validation_*/metrics.json")):
        metrics = read_json(evidence)
        if gate_score(metrics)[0] < 23:
            continue
        checkpoint_name = f"checkpoint_{metrics['iteration']:07d}.pt"
        if (run / checkpoint_name).exists() and checkpoint_name not in names:
            names.append(checkpoint_name)
    return names


def select_main_config():
    main_path = CONFIG_DIR / "config.json"
    base = read_json(main_path)
    records = {}
    for name in PILOTS:
        for checkpoint_name in candidate_checkpoints(name):
            quality = evaluate_at(name, 0, checkpoint_name)
            speed = evaluate_at(name, 1, checkpoint_name)
            suffix = "" if checkpoint_name == "latest.pt" else f"_{Path(checkpoint_name).stem}"
            key = f"{name}:{checkpoint_name}"
            source = pilot_run(name) / checkpoint_name
            records[key] = {
                "pilot": name,
                "checkpoint": str(source.relative_to(ROOT)),
                "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "quality": str((pilot_artifacts(name) / f"cross_level_00{suffix}/metrics.json").relative_to(ROOT)),
                "speed": str((pilot_artifacts(name) / f"cross_level_01{suffix}/metrics.json").relative_to(ROOT)),
                "quality_score": gate_score(quality),
                "speed_score": gate_score(speed),
            }
    eligible = [key for key, record in records.items()
                if record["quality_score"][0] >= 23 and record["speed_score"][0] >= 23]
    winner = max(eligible, key=lambda key: (records[key]["speed_score"], records[key]["quality_score"])) if eligible else None
    if winner is None:
        write_json(PIPELINE_DIR / "selection.json", {"pilots": records, "eligible": [], "winner": None,
                    "decision": "No checkpoint passed both 6-second and 4.5-second CPU gates; diagnose before long training."})
        raise RuntimeError("No pilot checkpoint passed both strict stage gates")
    selected = records[winner]
    base["init_checkpoint"] = selected["checkpoint"]
    base["starting_level"] = 2
    pilot_settings = read_json(pilot_config(selected["pilot"]))
    base["reward_variant"] = pilot_settings["reward_variant"]
    base["curriculum_mode"] = pilot_settings["curriculum_mode"]
    base["selection_note"] = f"Selected {winner} after same-seed 6-second and 4.5-second CPU comparison."
    active = sum(float(read_json(pilot_run(n) / "training_status.json")["active_training_seconds"]) for n in PILOTS)
    base["active_training_hours"] = (24 * 3600 - active) / 3600
    base["pilot_active_training_seconds"] = active
    base["selected_pilot"] = selected["pilot"]
    base["selected_candidate"] = winner
    base["selected_source_sha256"] = hashlib.sha256((ROOT / base["init_checkpoint"]).read_bytes()).hexdigest()
    write_json(main_path, base)
    write_json(PIPELINE_DIR / "selection.json", {
        "pilots": records, "eligible": eligible, "winner": winner,
        "main_init_checkpoint": base["init_checkpoint"],
        "main_init_sha256": base["selected_source_sha256"],
        "main_starting_level": base["starting_level"],
        "main_reward_variant": base["reward_variant"],
        "pilot_active_training_seconds": active,
        "main_active_training_hours": base["active_training_hours"],
    })


def main():
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    with (PIPELINE_DIR / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for name in PILOTS:
                cfg_path = pilot_config(name)
                run = pilot_run(name)
                status = run / "training_status.json"
                if status.exists() and read_json(status).get("status") == "budget_finished":
                    continue
                write_json(PIPELINE_DIR / "status.json", {"phase": f"pilot_{name}", "unix": time.time()})
                run_logged([sys.executable, "-m", f"{MODULE}.long_training.supervise", "--config", str(cfg_path)], f"pilot_{name}")
                if not status.exists() or read_json(status).get("status") != "budget_finished":
                    raise RuntimeError(f"Pilot {name} did not finish its active-time budget")
            main_cfg = read_json(CONFIG_DIR / "config.json")
            main_run = ROOT / main_cfg["runs"]
            if not (main_run / "source_manifest.json").exists() and not (PIPELINE_DIR / "selection.json").exists():
                write_json(PIPELINE_DIR / "status.json", {"phase": "pilot_comparison", "unix": time.time()})
                select_main_config()
            write_json(PIPELINE_DIR / "status.json", {"phase": "main_training", "unix": time.time()})
            finalizer = subprocess.Popen([sys.executable, "-m", f"{MODULE}.long_training_finalize"], cwd=ROOT,
                stdout=(PIPELINE_DIR / "finalizer.log").open("a"), stderr=subprocess.STDOUT)
            run_logged([sys.executable, "-m", f"{MODULE}.long_training.supervise"], "main_supervisor")
            finalizer.wait()
            write_json(PIPELINE_DIR / "status.json", {"phase": "finished", "unix": time.time(),
                "main_training_status": read_json(main_run / "training_status.json"),
                "finalization_status": read_json(main_run / "finalization_status.json")})
        except BaseException as exc:
            write_json(PIPELINE_DIR / "status.json", {"phase": "failed", "unix": time.time(), "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()

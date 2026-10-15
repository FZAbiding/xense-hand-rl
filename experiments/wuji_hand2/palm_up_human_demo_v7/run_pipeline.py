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


def evaluate_at(name, level):
    run = pilot_run(name)
    art = pilot_artifacts(name)
    output_name = f"cross_level_{level:02d}"
    evidence = art / output_name / "metrics.json"
    if evidence.exists():
        return read_json(evidence)
    latest = run / "latest.pt"
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    source_sha = hashlib.sha256(latest.read_bytes()).hexdigest()
    checkpoint["curriculum_level"] = level
    candidate = run / f"cross_level_{level:02d}.pt"
    torch.save(checkpoint, candidate)
    write_json(run / f"cross_level_{level:02d}_provenance.json", {
        "source": str(latest.relative_to(ROOT)),
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
    ], f"evaluate_{name}_level_{level}")
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


def select_main_config():
    main_path = CONFIG_DIR / "config.json"
    base = read_json(main_path)
    records = {}
    for name in PILOTS:
        quality = evaluate_at(name, 0)
        speed = evaluate_at(name, 1)
        records[name] = {
            "quality": str((pilot_artifacts(name) / "cross_level_00/metrics.json").relative_to(ROOT)),
            "speed": str((pilot_artifacts(name) / "cross_level_01/metrics.json").relative_to(ROOT)),
            "quality_score": gate_score(quality),
            "speed_score": gate_score(speed),
            "latest_checkpoint": str((pilot_run(name) / "latest.pt").relative_to(ROOT)),
        }
    eligible = [n for n in PILOTS if records[n]["quality_score"][0] >= 23]
    winner = max(eligible, key=lambda n: records[n]["speed_score"]) if eligible else None
    if winner is None:
        base["starting_level"] = 0
        base["selection_note"] = "All pilots regressed on 6-second validation; restart from archived v6b source."
    else:
        base["init_checkpoint"] = records[winner]["latest_checkpoint"]
        base["starting_level"] = 2 if records[winner]["speed_score"][0] >= 23 else 1
        base["reward_variant"] = "course_only" if winner == "course_only" else "combined"
        base["curriculum_mode"] = "progressive"
        base["selection_note"] = f"Selected {winner} after same-seed 6-second and 4.5-second CPU comparison."
    active = sum(float(read_json(pilot_run(n) / "training_status.json")["active_training_seconds"]) for n in PILOTS)
    base["active_training_hours"] = (24 * 3600 - active) / 3600
    base["pilot_active_training_seconds"] = active
    base["selected_pilot"] = winner
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

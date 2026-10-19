#!/usr/bin/env bash
set -euo pipefail

cd /home/xense/src/xense-hand-rl
run_dir=runs/wuji_hand2/palm_up_human_demo_v10/overnight_robustness_20261020
mkdir -p "$run_dir"
ready_count=0

while (( 10#$(date +%H%M) < 500 )); do
  if pgrep -af 'experiments.wuji_hand2.palm_up_human_demo_v10.long_training.(supervise|train)' >/dev/null; then
    printf '%s duplicate v10 trainer detected; refusing to launch\n' "$(date -Is)" >> "$run_dir/launcher.log"
    exit 1
  fi

  sample=$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | head -n 1)
  IFS=, read -r util free_mib <<< "$sample"
  util=${util//[[:space:]]/}
  free_mib=${free_mib//[[:space:]]/}
  allow_parallel=0
  [[ -f "$run_dir/ALLOW_PARALLEL" ]] && allow_parallel=1

  if (( free_mib >= 12288 )) && { (( util < 15 )) || (( allow_parallel == 1 )); }; then
    ((ready_count+=1))
  else
    ready_count=0
  fi
  printf '%s gpu_util=%s%% free_mib=%s allow_parallel=%s ready_samples=%s\n' \
    "$(date -Is)" "$util" "$free_mib" "$allow_parallel" "$ready_count" >> "$run_dir/launcher.log"

  if (( ready_count >= 5 )) || { (( allow_parallel == 1 )) && (( free_mib >= 12288 )); }; then
    printf '%s launching v10 supervisor\n' "$(date -Is)" >> "$run_dir/launcher.log"
    exec /home/xense/src/xense-hand-rl/.venv-warp/bin/python -m \
      experiments.wuji_hand2.palm_up_human_demo_v10.long_training.supervise \
      --config experiments/wuji_hand2/palm_up_human_demo_v10/long_training/config.json
  fi
  sleep 60
done

printf '%s skipped: GPU did not meet overnight readiness before 05:00\n' "$(date -Is)" >> "$run_dir/launcher.log"

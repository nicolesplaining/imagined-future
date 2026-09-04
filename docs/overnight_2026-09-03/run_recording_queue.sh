#!/usr/bin/env bash
set -euo pipefail

remote_host="$1"
shift

for environment_seed in "$@"; do
  output_dir="/research/results/overnight_2026_09_03/recordings/seed_${environment_seed}"
  log_file="/research/results/overnight_2026_09_03/logs/recording_seed_${environment_seed}.log"
  /workspace/isaaclab/isaaclab.sh -p /research/scripts/run_cosmos3_seeded_screen.py \
    --headless \
    --device cuda:0 \
    --remote-host "$remote_host" \
    --remote-port 8001 \
    --environment-seed "$environment_seed" \
    --num-envs 1 \
    --num-runs 1 \
    --video-mode none \
    --output-folder-name "$output_dir" \
    --task \
      BananaInBowlTask \
      RubiksCubeTask \
      MustardInLeftBinTask \
      SpoonInMugTask \
      MarkerInMugTask \
      SmartphoneInBinTask \
    >"$log_file" 2>&1
done

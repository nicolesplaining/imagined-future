# Cosmos 3 future-strength dose v2: launch and completion receipt

Status: powered cohort launched after independent GO. Monitor only process,
completed-file count, and fatal/error text until exact 30/30 completion.

Frozen identities:

- Manifest: `1a37bdba796b580c905970c441aededccf4193c51fe370f0f3557eccd5a9ee7d`
- GO audit: `742665605e2a3ceb9ee45b8ef3bd09a3fa6501c482abddd3d8d0078aae9792d8`
- Runner: `f0db63e65e920239ba13837bfec4f352d2514885df708745030ef140bf9009ee`
- Launcher: `968fb2199188c25a3aaa0841b91e868d2df2925477932bcd810b3f00856dbeff`
- Analyzer: `6b7a642a7687f2176c17b4cf2050da6be3de01361540b58179c25c0b83d68f86`
- Completion packager: `b9a11108958ceb2a79578cc9b07afe5692679ce87683066285da6820752c3cc3`

Runtime:

- Host: `ubuntu@68.209.73.251`
- Container: `if-cosmos-dose-v2-eval-8003`
- GPU/port: GPU 2, `localhost:8003`
- Launcher host PID: `169798`
- NFS root:
  `/lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2`

The exact in-container launcher invocation is:

```bash
/workspace/.venv/bin/python \
  /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/snapshot/scripts/launch_cosmos3_future_strength_dose_response.py \
  --manifest /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/manifest.json \
  --expected-manifest-sha256 1a37bdba796b580c905970c441aededccf4193c51fe370f0f3557eccd5a9ee7d \
  --runner /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/snapshot/scripts/run_cosmos3_future_strength_dose_response.py \
  --analyzer /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/snapshot/scripts/summarize_cosmos3_future_strength_dose_response.py \
  --audit-report /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/audit_tools/cosmos3_future_strength_dose_v2_go_audit.json \
  --expected-audit-sha256 742665605e2a3ceb9ee45b8ef3bd09a3fa6501c482abddd3d8d0078aae9792d8 \
  --screen-root /research/external/RoboLab/output/cosmos3_population_screen_v1 \
  --output-root /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/evaluation/states \
  --host localhost --port 8003 --shard-index 0 --shard-count 1
```

Count/error-only monitoring command:

```bash
ssh ubuntu@68.209.73.251 '
root=/lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2
pid=$(cat "$root/logs/launcher.pid")
if kill -0 "$pid" 2>/dev/null; then echo process=RUNNING; else echo process=EXITED; fi
printf "completed="; find "$root/evaluation/states" -maxdepth 1 -type f -name "*.json" | wc -l
printf "fatal_or_error_lines="; grep -Eic "traceback|error|exception|failed" "$root/logs/launcher.log" || true
'
```

Only after the launcher emits exact `assigned=30`, `completed=30`,
`resume_skipped=0` and the directory contains exactly the 30 manifest-derived
files, run the frozen completion utility on the host:

```bash
sudo python3 \
  /lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/audit_tools/freeze_cosmos3_dose_output_package.py \
  --manifest /lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/manifest.json \
  --expected-manifest-sha256 1a37bdba796b580c905970c441aededccf4193c51fe370f0f3557eccd5a9ee7d \
  --output-root /lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/evaluation/states \
  --inventory-output /lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/packaging/output_inventory.json
```

Only after the inventory reports exact 30 files, mode 0444, and post-chmod
hash equality, run the frozen analyzer inside the same exact container:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/snapshot/src:/source \
/workspace/.venv/bin/python \
  /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/snapshot/scripts/summarize_cosmos3_future_strength_dose_response.py \
  --manifest /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/manifest.json \
  --expected-manifest-sha256 1a37bdba796b580c905970c441aededccf4193c51fe370f0f3557eccd5a9ee7d \
  --input-root /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/evaluation/states \
  --output /research/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/evaluation/analysis/summary.json \
  --bootstrap-samples 10000 --bootstrap-seed 20260903
```

Hash and mode-freeze the completed analysis artifact before interpretation,
then hand the 30-file inventory, analysis hash, and raw outputs to an
independent auditor. Do not inspect or summarize partial scientific payloads.


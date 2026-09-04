# Cosmos 3 overnight analysis

Run the analyzers from the repository root after per-state `summary.json` files
arrive on the shared filesystem. Both commands accept incomplete result trees,
report the missing frozen state IDs, and reject completed summaries that violate
the frozen schema or exact replay controls.

## Per-state runner templates

Start the Cosmos research server with `--attention-instrumentation` for the K/V
lane. For each manifest state, substitute its task, branch point, recording,
server address, and output directory in these templates.

Selection-free action grid plus fixed-recipient physical battery:

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/run_cosmos3_robolab_branches.py \
  --headless --device cuda:0 \
  --task TASK \
  --remote-host COSMOS_HOST --remote-port COSMOS_PORT \
  --recorded-hdf5 RECORDING/run_0.hdf5 \
  --output-dir OUTPUT_STATE_DIR \
  --branch-step BRANCH_STEP \
  --target-object-name TARGET_OBJECT_NAME \
  --branch-seeds 211 223 227 229 \
  --frozen-recipient-seed 211 --frozen-donor-seed 223 \
  --gaussian-seed 1223 --restore-strategy fresh_replay \
  --multi-donor --all-recipient-action-grid \
  --study-id STUDY_STATE_ID
```

Minimal K/V factorial on the fresh 24-state subset:

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/run_cosmos3_robolab_branches.py \
  --headless --device cuda:0 \
  --task TASK \
  --remote-host COSMOS_HOST --remote-port COSMOS_PORT \
  --recorded-hdf5 RECORDING/run_0.hdf5 \
  --output-dir OUTPUT_STATE_DIR \
  --branch-step BRANCH_STEP \
  --target-object-name TARGET_OBJECT_NAME \
  --branch-seeds 211 223 \
  --frozen-recipient-seed 211 --frozen-donor-seed 223 \
  --gaussian-seed 1223 --restore-strategy fresh_replay \
  --attention-kv-patch-layers \
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \
    18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 \
  --attention-kv-factorial --minimal-kv-factorial \
  --study-id STUDY_STATE_ID
```

The minimal K/V run needs only seeds 211 and 223; the four-way branch set remains
specific to the selection-free lane. The task-object names are `banana`,
`rubiks_cube`, `mustard`, `spoon_big`, `marker`, and `smartphone`, respectively,
for the six frozen tasks in config order.

## Selection-free multi-donor study

```bash
PYTHONPATH=src python scripts/summarize_cosmos3_selection_free.py \
  results/overnight_2026_09_03/selection_free \
  --manifest results/overnight_2026_09_03/frozen_manifest.json \
  --output-dir results/overnight_2026_09_03/analysis/selection_free
```

Action estimates come from the all-12-ordered-pair grid. Physical endpoint
estimates come from the fixed recipient-211-to-three-donors battery. The output
includes donor- and state-level CSVs, JSON, a compact LaTeX table, task-to-state
bootstrap intervals (10,000 resamples), native-separation quartiles,
leave-one-task-out estimates, and PNG/PDF plots.

## Minimal K/V factorial

```bash
PYTHONPATH=src python scripts/summarize_cosmos3_kv_factorial.py \
  results/overnight_2026_09_03/kv_factorial \
  --manifest results/overnight_2026_09_03/frozen_manifest.json \
  --config configs/overnight_2026-09-03.toml \
  --output-dir results/overnight_2026_09_03/analysis/kv_factorial
```

This analyzer verifies the task-balanced 24-state subset from the frozen config,
the config SHA-256, the seven-arm physical execution allowlist, all action-only
arms, and exact recipient/donor K/V record-replay. It emits cell and contrast
CSVs, replay audits, JSON, a compact LaTeX table, task-to-state bootstrap
intervals, leave-one-task-out estimates, and PNG/PDF plots.

Use `--no-plots` when Matplotlib is unavailable. Use
`--bootstrap-resamples 100` only for a quick pipeline smoke test; confirmatory
outputs retain the default 10,000 resamples.

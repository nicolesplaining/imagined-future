# FastWAM Optional-IDM external-WAM intervention

This package tests whether distinct native imagined futures directionally select
their associated actions in a released non-Cosmos WAM. It uses FastWAM's
Optional-IDM checkpoint, whose IDM path denoises a future video and then freezes
that video while denoising the action from its per-layer K/V cache.

## Frozen upstream

- Repository: `https://github.com/yuantianyuan01/FastWAM.git`
- Commit: `7faa71108368fbb3b6885649f112af607427a2d4`
- Commit date/message: `2026-08-20`, `Optimize IDM action-only inference`
- Checkpoint repository: `yuanty/fastwam`
- Checkpoint: `libero_optional_idm_2cam224.pt` (12.04 GB)
- Checkpoint SHA-256: `26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4`
- Dataset statistics: `libero_optional_idm_2cam224_dataset_stats.json`
- Statistics SHA-256: `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`

The local patch in
`third_party/patches/fastwam_optional_idm_interventions.patch` adds independent
video/action generators, final-video-latent replay, K/V replay, and optional
intermediate returns. It does not alter model weights, schedules, attention, or
the denoising equations.

Both FastWAM and the pinned LIBERO checkout are MIT-licensed. Wan/T5 components
and the released checkpoint retain their upstream model licenses.

## Conditions

For native branch `A`, donor `B`, and a third branch `C`, all sampled from the
same fixed LIBERO state:

| Condition | Future-side input | Action noise |
|---|---|---|
| `native` | generated `F_A`, native `KV_A` | `eps_A` |
| `self_latent` | replayed `F_A`, recomputed `KV_A` | `eps_A` |
| `self_cache` | replayed `F_A`, replayed `KV_A` | `eps_A` |
| `donor_latent` | replayed `F_B`, recomputed `KV_B` | `eps_A` |
| `donor_cache` | recipient latent `F_A`, replayed `KV_B` | `eps_A` |
| `wrong_latent` | replayed third future `F_C` | `eps_A` |
| `shuffled_cache` | `F_A`, donor keys with donor values token-permuted | `eps_A` |
| `first_frame` | no future-video route; donor `B` is a logical label only | `eps_A` |

The value-only token permutation is intentional. Applying the same token
permutation to paired K and V can be attention-invariant when every future token
is visible to the action query.

`first_frame` is a same-checkpoint architectural negative control. It cannot
receive a future transplant because that inference path contains no future
route. The runner emits all 12 ordered recipient-donor rows, varies the
registered donor video seed while holding recipient action noise fixed, and
requires exact agreement across the three donor labels for each recipient.

## Freeze the smoke manifest

```bash
python scripts/build_fastwam_optional_idm_manifest.py \
  --config configs/fastwam_optional_idm_smoke.toml \
  --output results/fastwam_optional_idm_smoke_v1/manifest.json
```

The manifest ID is a SHA-256 digest of all scientific choices: state IDs,
branches, independent seeds, inference settings, conditions, checkpoint name,
and upstream commit. Local paths are excluded. The runner refuses a modified
manifest or a FastWAM checkout at another commit.

The smoke config fixes eight states before generation: four initial states from
LIBERO-Spatial task 0 and four from LIBERO-Long task 0, with four native branches
per state. No state or future may be removed because its output is unattractive
or insufficiently separated. Mechanical failures are recorded rather than
silently filtered.

## Install

The setup script clones the pinned FastWAM and LIBERO commits, applies the patch,
and creates an isolated Python 3.10 environment. It deliberately does not
download model weights.

```bash
FASTWAM_DIR=/lambda/nfs/imagined-future/external/FastWAM \
LIBERO_DIR=/lambda/nfs/imagined-future/external/LIBERO \
FASTWAM_VENV=/home/ubuntu/venvs/fastwam \
PYTHON_BIN=/home/ubuntu/.local/bin/python3.10 \
bash scripts/setup_fastwam_optional_idm.sh
```

Download only the official Optional-IDM checkpoint and matching statistics into
the shared cache using the official Hugging Face CLI. Wan2.2 components are
resolved by FastWAM under `DIFFSYNTH_MODEL_BASE_PATH` during model construction.

Headless LIBERO rendering requires the NVIDIA EGL user-space library matching
the installed driver. On the Lambda image used for this study (driver
`580.105.08`), the required package was:

```bash
sudo apt-get install libnvidia-gl-580-server=580.105.08-0lambda0.24.04.1
```

Match the package version to `nvidia-smi` on a different image; do not copy that
version blindly. A successful preflight creates and renders one LIBERO state
with `MUJOCO_GL=egl` before model inference begins.

## Launch

```bash
export FASTWAM_ROOT=/lambda/nfs/imagined-future/external/FastWAM
export FASTWAM_LIBERO_ROOT=/lambda/nfs/imagined-future/external/LIBERO
export FASTWAM_CHECKPOINT=/lambda/nfs/imagined-future/checkpoints/fastwam_release/libero_optional_idm_2cam224.pt
export FASTWAM_DATASET_STATS=/lambda/nfs/imagined-future/checkpoints/fastwam_release/libero_optional_idm_2cam224_dataset_stats.json
export FASTWAM_MANIFEST=/lambda/nfs/imagined-future/results/fastwam_optional_idm_smoke_v1/manifest.json
export FASTWAM_OUTPUT_ROOT=/lambda/nfs/imagined-future/results/fastwam_optional_idm
export FASTWAM_PYTHON=/home/ubuntu/venvs/fastwam/bin/python
export LIBERO_CONFIG_PATH=/lambda/nfs/imagined-future/overnight_runtime_2026_09_03/libero_config
export DIFFSYNTH_MODEL_BASE_PATH=/lambda/nfs/imagined-future/checkpoints/fastwam_components
export FASTWAM_GPUS=0
bash scripts/launch_fastwam_optional_idm.sh
```

Initialize `LIBERO_CONFIG_PATH` once using the official LIBERO configuration
prompt, choosing its default benchmark paths. The launcher points Python at the
pinned LIBERO checkout explicitly because that revision's namespace package is
not exposed correctly by modern editable installers.

Set `FASTWAM_GPUS=0,1,...` only when those devices are explicitly assigned to
this lane. States are deterministically sharded by manifest order.

Each completed run has an atomic `.npz` array artifact and a `.json` completion
record under `<output>/<manifest_id>/<state_id>/runs/`. The JSON file is written
last, so it is the completion marker. Native final video latents are always
saved; full K/V persistence is optional because it can exceed 1 GB per branch.
K/V is always retained and replayed in memory, and cache shapes, dtypes, and
layerwise norms are recorded. A restart reconstructs cache from the saved future
latent if necessary.

## Gates before scaling

1. `self_latent` and `self_cache` reproduce `native` within the frozen numerical
   tolerance; exact-cache replay should be effectively exact.
2. The four native future latents are nonidentical.
3. Correct-donor retrieval exceeds its 25% four-way chance level in the smoke
   sample.
4. Donor attraction is stronger than wrong-future and shuffled-cache controls.
5. `first_frame` is invariant to video seed when action seed is fixed.

Only after these gates pass should the state/task grid be expanded. State is the
independent unit; branch directions and diffusion seeds are within-state
measurements.

The frozen analyzer enforces those gates and refuses to issue a scale decision
unless all eight states contain all 72 registered arms and valid completion
summaries:

```bash
PYTHONPATH=src python scripts/summarize_fastwam_optional_idm.py \
  --manifest "$FASTWAM_MANIFEST" \
  --output-root "$FASTWAM_OUTPUT_ROOT" \
  --summary-dir "$FASTWAM_OUTPUT_ROOT/$MANIFEST_ID/analysis"
```

It writes an explicit completeness/missingness audit before loading outcomes.
For a complete matrix it additionally writes run-level and state-level CSVs,
state-bootstrap aggregate JSON/CSV, a compact LaTeX table, and a four-panel PNG.
Every degenerate recipient--donor action axis remains in the run table and is
counted in each state and aggregate; only metrics that are mathematically
undefined on that axis are left empty. The mechanical scale decision requires
both correct-latent and correct-cache retrieval above four-way chance, correct
latent attraction above wrong-latent attraction, correct-cache attraction above
shuffled-cache attraction, exact replay and first-frame controls, finite arrays,
and distinct native video latents.

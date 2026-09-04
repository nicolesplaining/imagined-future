# Cross-region A10 recovery kit

Prepared before any instance replacement. Do not terminate or launch an
instance without the user's explicit authorization. H100 is unsupported by
Isaac Sim; Lambda's available graphics-capable recovery type is
`gpu_1x_a10`, currently in `us-east-1` or `us-west-1`, not Georgia. Therefore
two A10 nodes provide two physical lanes, each on GPU 0, without the Georgia
NFS mount. Prefer `us-east-1` for lower latency to the Cosmos server.

The broader diagnosis is recorded in
`docs/overnight_2026-09-03/isaac_h100_diagnosis_a10_recovery.md`. This file is
the exact transfer, provenance, and launch checklist.

## Frozen inputs and runtime provenance

| Input | Georgia source path | SHA-256 |
| --- | --- | --- |
| Branch runner | `/lambda/nfs/imagined-future/scripts/run_cosmos3_robolab_branches.py` | `a231d03b38e23e5e278e926f2b8941fe8de57e156e64d7642b3de7748ed5cbe7` |
| Study config | `/lambda/nfs/imagined-future/configs/overnight_2026-09-03.toml` | `0d454c53d11f228762f65711c966621ca6e2390cd4d3d99977f8bd5077150705` |
| Frozen manifest | `/lambda/nfs/imagined-future/docs/overnight_2026-09-03/frozen_manifest.json` | `7b2e4582e423da8766e6a2b38c087b863b591cae779a85cb46c23ed3383d9754` |
| Recording queue | `/lambda/nfs/imagined-future/docs/overnight_2026-09-03/run_recording_queue.sh` | `0f9fec89c9e243887cb380b7866e9b04ae2d08584b7bf3b3b3149856bb2d772a` |
| Selection analyzer | `/lambda/nfs/imagined-future/scripts/summarize_cosmos3_selection_free.py` | `c3d51eeb43241309604910ec9f214cbc28915822ea4b1da49b98d0ef5e488cc4` |
| K/V analyzer | `/lambda/nfs/imagined-future/scripts/summarize_cosmos3_kv_factorial.py` | `4ed3f86d35e64f6d49be250a48defeb06dc10ccb715e6a4737f7d134a4638736` |

The runtime is `robolab:overnight-2026-09-03`, image ID
`sha256:ea2b318e344149b2acdd2a1c8b51f30e549a366166d23e9ee8c6f4975a2b2a67`,
built from RoboLab `9db0aaf09d9fe5d4f37b168320788258c7012463`, OpenPI client
`aa6420561529593114160d05e5ad155792b272f3`, and IsaacLab base digest
`sha256:b4d8e96cbfb9a6c40067bec6cc5ee180e36d4c0164b25f7215c5f47e31897b94`.
The pinned RoboLab checkout is already copied into `/workspace/robolab` inside
the image; cross-region clients must not replace it with an unfrozen checkout.

The verified image archive on the Georgia NFS is:

```text
/lambda/nfs/imagined-future/runtime/images/robolab_overnight-2026-09-03_ea2b318e.tar.zst
```

It is 18,304,716,347 bytes with SHA-256
`a47a9e97620c0ba46746f867d68f5788ec4388a78f1f6f16adbd738c47722b5c`.
Portable `.sha256` and Docker `.inspect.json` sidecars sit beside it.

The eight exact runtime inputs in the table, plus the seeded-screen runner and
its two protocol modules, are packaged as:

```text
/lambda/nfs/imagined-future/runtime/images/cosmos3_a10_inputs_a231d03b.tar.gz
```

That 24,637-byte bundle has SHA-256
`ccd3c699511d48b88b34dff951884d5962a3f5d5e4eaa3731496f04b5de6fd12`
and a portable `.sha256` sidecar.

## Cross-region SSH path

The public Cosmos ports 8001 and 8002 time out; Georgia-private
`172.26.133.114` works only inside the Georgia region. On each A10, generate a
dedicated Ed25519 key. Append only its public key to the Georgia server's
`~/.ssh/authorized_keys`; never copy a private key between machines. Restrict
that key to the required forwards with `permitopen` options.

Run this supervised tunnel on each A10. The container uses host networking, so
it can reach the host-local ports. Lane 0 needs 8001; lane 1 may forward both
8001 and the dedicated, non-concurrent K/V service on 8002:

```bash
nohup bash -c 'while true; do
  ssh -NT -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new \
    -i ~/.ssh/if_georgia_ed25519 \
    -L 127.0.0.1:8001:127.0.0.1:8001 \
    -L 127.0.0.1:8002:127.0.0.1:8002 \
    ubuntu@68.209.73.251
  sleep 2
done' </dev/null >~/cosmos_tunnel.log 2>&1 &

nc -vz 127.0.0.1 8001
nc -vz 127.0.0.1 8002
```

Record the ephemeral public key and tunnel PID in the run ledger. Remove the
authorization after the study. Do not expose the Cosmos ports publicly.

## Transfer and verify each A10

Copy the exact image and input bundles from Georgia. The image already contains
RoboLab, so no external RoboLab bind is required.

```bash
mkdir -p ~/runtime_artifacts ~/imagined-future

rsync -a --partial ubuntu@68.209.73.251:/lambda/nfs/imagined-future/runtime/images/robolab_overnight-2026-09-03_ea2b318e.tar.zst\* ~/runtime_artifacts/
rsync -a --partial ubuntu@68.209.73.251:/lambda/nfs/imagined-future/runtime/images/cosmos3_a10_inputs_a231d03b.tar.gz\* ~/runtime_artifacts/

cd ~/runtime_artifacts
sha256sum -c robolab_overnight-2026-09-03_ea2b318e.tar.zst.sha256
sha256sum -c cosmos3_a10_inputs_a231d03b.tar.gz.sha256
zstd -dc robolab_overnight-2026-09-03_ea2b318e.tar.zst | sudo docker load
tar -xzf cosmos3_a10_inputs_a231d03b.tar.gz -C ~/imagined-future
sudo docker image inspect robolab:overnight-2026-09-03 --format '{{.Id}}'

sha256sum \
  ~/imagined-future/scripts/run_cosmos3_robolab_branches.py \
  ~/imagined-future/configs/overnight_2026-09-03.toml \
  ~/imagined-future/docs/overnight_2026-09-03/frozen_manifest.json \
  ~/imagined-future/docs/overnight_2026-09-03/run_recording_queue.sh
```

The image ID and four file hashes must match the table above before any
frozen run.

## Create one client per A10

Set `lane=0` on the first node and `lane=1` on the second. Both use that node's
only GPU 0 and isolated local caches/results.

```bash
lane=0  # lane=1 on the second A10
root="/var/lib/if-overnight/robolab-a10-${lane}"
sudo mkdir -p "$root/cache/kit" "$root/cache/ov"
sudo docker run -d \
  --name "if-robolab-a10-${lane}" \
  --runtime nvidia --gpus 'device=0' \
  --network host --ipc host --security-opt label=disable \
  -e PYTHONPATH=/research/src \
  -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  -v "$root/cache/kit:/isaac-sim/kit/cache" \
  -v "$root/cache/ov:/root/.cache/ov" \
  -v /home/ubuntu/imagined-future:/research \
  robolab:overnight-2026-09-03 -lc 'sleep infinity'
```

Before any frozen run, require the exact image ID, an A10 in `nvidia-smi`, and
a successful headless renderer launch on each lane:

```bash
sudo docker exec "if-robolab-a10-${lane}" nvidia-smi
sudo docker exec "if-robolab-a10-${lane}" \
  /workspace/isaaclab/isaaclab.sh -p -c \
  'from isaaclab.app import AppLauncher; x=AppLauncher(headless=True); print("ISAAC_APP_OK"); x.app.close()'
```

## Two-lane recording launch

This deterministic split covers the 48-state recording cohort exactly once.
The helper targets local port 8001, which the tunnel forwards to the general
Cosmos server.

```bash
# A10 lane 0:
sudo docker exec -d if-robolab-a10-0 bash \
  /research/docs/overnight_2026-09-03/run_recording_queue.sh \
  127.0.0.1 3554 5017 6693 8632

# A10 lane 1:
sudo docker exec -d if-robolab-a10-1 bash \
  /research/docs/overnight_2026-09-03/run_recording_queue.sh \
  127.0.0.1 4828 5428 8281 8901
```

Expected recording wall time is about 46--60 minutes per lane. Cosmos
inference is roughly 1.2 seconds per call; Isaac setup and physical execution
dominate wall time. Port 8002 remains reserved for exactly one K/V client.

## Excluded K/V engineering smoke

Before this smoke, copy only its existing development recording to lane 1:

```bash
mkdir -p ~/imagined-future/results/cosmos3_multitask_screen/cosmos3_screen_v1/BananaInBowlTask
rsync -a ubuntu@68.209.73.251:/lambda/nfs/imagined-future/results/cosmos3_multitask_screen/cosmos3_screen_v1/BananaInBowlTask/\{run_0.hdf5,env_cfg.json\} \
  ~/imagined-future/results/cosmos3_multitask_screen/cosmos3_screen_v1/BananaInBowlTask/
```

Then run the excluded Banana integration smoke against the dedicated tunnel:

```bash
sudo docker exec if-robolab-a10-1 \
  /workspace/isaaclab/isaaclab.sh -p \
  /research/scripts/run_cosmos3_robolab_branches.py \
  --headless --device cuda:0 \
  --task BananaInBowlTask \
  --remote-host 127.0.0.1 --remote-port 8002 \
  --recorded-hdf5 /research/results/cosmos3_multitask_screen/cosmos3_screen_v1/BananaInBowlTask/run_0.hdf5 \
  --output-dir /research/results/overnight_2026_09_03/smoke/kv_banana_development \
  --branch-step 64 \
  --branch-seeds 211 223 227 229 \
  --frozen-recipient-seed 211 --frozen-donor-seed 223 \
  --restore-strategy fresh_replay \
  --target-object-name banana \
  --attention-kv-patch-layers $(seq 0 35) \
  --attention-kv-factorial --minimal-kv-factorial \
  --study-id overnight-2026-09-03-kv-dev-banana
```

This is an excluded engineering state, not a confirmatory outcome. Admit no
frozen outcome until the renderer, camera, replay, and K/V identity gates pass.

## Return artifacts without overwriting

Each A10 writes to its local `~/imagined-future/results`. After a seed/state is
closed and validated, rsync it to a unique Georgia staging directory; never
copy a live HDF5 or overwrite a canonical output:

```text
/lambda/nfs/imagined-future/results/overnight_2026_09_03/.incoming/<a10-hostname>/<attempt-id>/
```

Use `rsync --partial --delay-updates`, verify hashes/schema in staging, then
promote only if the canonical target does not exist. Preserve failed attempts
and logs with explicit invalid labels.

# Isaac Sim H100 diagnosis and A10 recovery

## Conclusion

The new H100 nodes cannot be used as RoboLab / Isaac Sim rendering clients.
This is a hardware-support limitation, not a missing Docker flag. Isaac Sim 5.0
requires an RTX-capable GPU and explicitly says that GPUs without RT cores,
including A100 and H100, are unsupported:

- https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html
- https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/rendering_modes.html

The H100s remain useful for the Cosmos policy servers and other compute-only
work. Do not spend more time changing Vulkan packages, CUDA versions,
`--privileged`, or renderer modes on these H100 clients. RoboLab requires RGB
camera observations, and even Isaac Sim's minimal renderer is an RTX renderer.

## Evidence from the failed nodes

Both newly provisioned H100 hosts report driver `580.105.08`. Inside the pinned
RoboLab containers:

- the NVIDIA container runtime is active;
- the requested GPU is visible as `NVIDIA H100 80GB HBM3`, compute capability
  9.0;
- `NVIDIA_DRIVER_CAPABILITIES=all` is set;
- the correct per-GPU `/dev/nvidia*` devices and `/dev/dri/renderD*` node are
  present;
- `/etc/vulkan/icd.d/nvidia_icd.json` points to the injected
  `libGLX_nvidia.so.0`;
- the host's `libcuda.so.580.105.08` is injected, exports `cuDeviceGetUuid`, and
  `cuInit(0)` succeeds.

Isaac Sim then fails at renderer initialization with
`VkResult: ERROR_INCOMPATIBLE_DRIVER`, followed by `Failed to create any GPU
devices`. This is the expected failure mode when the Omniverse RTX renderer is
given an unsupported non-RT H100.

## Available recovery capacity

The authenticated Lambda capacity check found:

- account allocation: 32 / 32 GPUs;
- graphics-capable capacity: `gpu_1x_a10` in `us-east-1` and `us-west-1`;
- no A6000 or RTX 6000 capacity;
- no currently running graphics-capable instance.

An A10 has RT cores and 24 GB VRAM, enough for this one-environment RoboLab
client while Cosmos inference remains on the separate H100 server. Prefer two
single-A10 instances in `us-east-1` because it is closer to the Georgia Cosmos
server. The Georgia filesystem cannot be attached cross-region, so use SSH
transfer and tunnels.

## Fastest safe recovery sequence

Do not execute steps 2 and 3 until the user authorizes replacing the newly
launched two-H100 client instance. Do not terminate any pre-existing instance.

### 1. Preserve the exact RoboLab runtime before replacement

The four-H100 Georgia server also has the pinned RoboLab image, so the image
remains available even if the two-H100 client is terminated. On
`68.209.73.251`, create one compressed, checksummed archive without disturbing
running Cosmos containers:

```bash
sudo mkdir -p /lambda/nfs/imagined-future/results/overnight_2026_09_03/runtime_artifacts
sudo sh -c 'docker save robolab:overnight-2026-09-03 | zstd -T0 -3 -c > /lambda/nfs/imagined-future/results/overnight_2026_09_03/runtime_artifacts/robolab_f78dad4bc92c.tar.zst'
sudo chmod 0644 /lambda/nfs/imagined-future/results/overnight_2026_09_03/runtime_artifacts/robolab_f78dad4bc92c.tar.zst
sha256sum /lambda/nfs/imagined-future/results/overnight_2026_09_03/runtime_artifacts/robolab_f78dad4bc92c.tar.zst
```

The expected loaded image ID is
`sha256:f78dad4bc92ca4e571f3a4f6cfe1c771f2bbaf18c892277fa5a0f8f759887971`.

### 2. Replace only the unusable rendering capacity

After explicit approval, terminate only the newly launched instance
`d51abf0e889541b2ad5ceacd7e405836` (`if-overnight-robolab-clients`, two H100s).
Leave the four-H100 Cosmos server, two-H100 external-WAM server, and all older
instances untouched. Launch two `gpu_1x_a10` instances in `us-east-1`, with the
existing `harshvardhan-imagined-future` SSH key and no filesystem attachment.

Record both instance IDs, IPs, GPU/driver output, and image ID in the
infrastructure ledger before frozen outputs are generated.

### 3. Give each A10 a direct, scoped SSH path to Georgia

Generate a dedicated Ed25519 key on each A10 and append only its public key to
`ubuntu@68.209.73.251:~/.ssh/authorized_keys`. Never copy a private key between
machines. This key supports both artifact/result transfer and the policy-server
tunnel.

On A10 lane 1, forward the general Cosmos server:

```bash
nohup ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -i ~/.ssh/if_georgia_ed25519 \
  -L 18001:127.0.0.1:8001 \
  ubuntu@68.209.73.251 \
  >~/cosmos_tunnel_8001.log 2>&1 &
```

On A10 lane 2, use `-L 18002:127.0.0.1:8002` for the dedicated K/V server.
Check `nc -vz 127.0.0.1 18001` or `18002` before launching a job. If outbound
SSH from an A10 is blocked, initiate the equivalent reverse tunnel from the
Georgia server; do not expose ports 8001/8002 publicly.

### 4. Load the pinned image and minimal study code

On each A10, pull the archive from Georgia, verify its recorded SHA-256, and
load it:

```bash
mkdir -p ~/runtime_artifacts ~/imagined-future ~/if_results
rsync -a --partial --info=progress2 \
  ubuntu@68.209.73.251:/lambda/nfs/imagined-future/results/overnight_2026_09_03/runtime_artifacts/robolab_f78dad4bc92c.tar.zst \
  ~/runtime_artifacts/
zstd -dc ~/runtime_artifacts/robolab_f78dad4bc92c.tar.zst | sudo docker load
sudo docker image inspect robolab:overnight-2026-09-03 --format '{{.Id}}'
```

Copy only the study code and frozen metadata; the image already contains the
pinned RoboLab source:

```bash
rsync -a ubuntu@68.209.73.251:/lambda/nfs/imagined-future/scripts ~/imagined-future/
rsync -a ubuntu@68.209.73.251:/lambda/nfs/imagined-future/src ~/imagined-future/
rsync -a ubuntu@68.209.73.251:/lambda/nfs/imagined-future/configs ~/imagined-future/
mkdir -p ~/imagined-future/results/overnight_2026_09_03
rsync -a ubuntu@68.209.73.251:/lambda/nfs/imagined-future/results/overnight_2026_09_03/frozen_manifest.json \
  ~/imagined-future/results/overnight_2026_09_03/
```

Start one container per A10. Do not mount over `/workspace/robolab`, because the
image already contains the pinned commit:

```bash
sudo mkdir -p /var/lib/if-overnight-a10/cache/kit /var/lib/if-overnight-a10/cache/ov
sudo docker run -d --name if-robolab-a10 \
  --runtime=nvidia --gpus all --network host --ipc host \
  -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e PYTHONPATH=/research/src \
  -v /var/lib/if-overnight-a10/cache/kit:/isaac-sim/kit/cache \
  -v /var/lib/if-overnight-a10/cache/ov:/root/.cache/ov \
  -v /home/ubuntu/imagined-future:/research \
  robolab:overnight-2026-09-03 -lc 'sleep infinity'
```

### 5. Fail-fast mechanical gate before frozen states

Confirm inside the container that `nvidia-smi` reports A10, `/dev/dri` is
present, and Isaac Sim initializes the renderer. Then run one excluded
development-state Banana recording through the appropriate local tunnel. The
gate passes only if:

- Isaac launches without Vulkan/GPU-foundation errors;
- both camera observations are nonempty;
- the recording HDF5 and `env_cfg.json` are readable;
- the Cosmos client completes a request;
- the native-repeat and clean-self-clamp replay checks pass under their frozen
  definitions.

Do not inspect a frozen intervention outcome during this gate.

### 6. Throughput-optimal two-lane schedule

1. Record seeds `6693, 8281, 8632, 8901` on A10 lane 1.
2. Record seeds `3554, 4828, 5017, 5428` on A10 lane 2.
3. Lane 1 runs the selection-free states for its four seeds against local port
   18001.
4. Lane 2 runs the 24-state K/V factorial for its four seeds against local port
   18002.
5. When K/V finishes, lane 2 runs its 24 selection-free states against a tunnel
   to port 8001.

This avoids moving recordings between A10s and preserves the dedicated,
non-concurrent K/V cache server. With measured task runtimes, recordings should
take roughly 45--60 minutes per lane after cold startup; the full two-lane
experiment remains feasible inside the overnight window.

### 7. Synchronize outputs without overwriting

After each seed or state completes, validate it locally, then rsync it into a
unique incoming/attempt directory on Georgia using `--partial` and
`--delay-updates`. Promote it to the canonical frozen path only after schema and
checksum validation, and only if the canonical target does not already exist.
Never sync a live HDF5 file.

Suggested staging root:

```text
/lambda/nfs/imagined-future/results/overnight_2026_09_03/.incoming/<a10-hostname>/<attempt-id>/
```

Keep failed attempts and logs clearly labeled. Do not delete or overwrite them.


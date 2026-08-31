# Compute environment notes

## 2026-08-31 initial H100 host incident

The assigned host exposes two NVIDIA H100 80 GB GPUs connected by NVLink. The pinned official Cosmos Policy container builds successfully, GPU device nodes are exposed inside Docker, and the intervention suite passes in the official PyTorch 2.7/CUDA 12.8 environment.

CUDA computation was blocked before model loading by `cudaErrorSystemNotReady` (error 802). `nvidia-smi -q` reported `Fabric: State: In Progress` for both GPUs rather than `Completed / Success`. NVIDIA's [Fabric Manager documentation](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html) states that H100 CUDA initialization waits for GPU registration with the NVLink fabric. The following safe recovery attempts did not change the state:

- Verified matching driver and Fabric Manager versions.
- Started Fabric Manager; it reported no NVSwitch device to manage on this direct-NVLink pair.
- Reset both idle NVLinked GPUs together.
- Rebooted the guest host.

This was treated as an infrastructure/provisioning issue, not an experimental result. No checkpoint forward pass or outcome-bearing intervention completed on that host.

## Replacement host validation

The provider replacement exposes the same two NVIDIA H100 80 GB GPUs. Both report `Fabric: Completed / Success`; PyTorch 2.7.0+cu128 detects both devices and completed an explicit CUDA tensor computation. The intervention suite passes 14 tests in the official environment.

The official Cosmos Policy config eagerly resolves a gated base-model path before its public loader replaces that path with the full fine-tuned policy checkpoint. The project defers only that unused config-time lookup. The 508 MB video tokenizer is a real gated dependency and was downloaded after authenticating an account with repository access. The authentication token was removed from the server after the required artifact was cached.

Smoke-test environment:

- Cosmos Policy commit: `18a2accadf4e7a3531e56754102af5a24d2316da`
- Container digest: `sha256:2f5ff2badf82657c82e046bff84cc754acf4a2b3973828f12ab802c751f439e0`
- Policy checkpoint snapshot: `cb689ec0e3347c13667d70a78a3447388f5c3bb8`
- Tokenizer snapshot: `f50c09f5d8ab133a90cac3f4886a6471e9ba3f18`
- PyTorch/CUDA: `2.7.0+cu128`
- GPU/driver: `NVIDIA H100 80GB HBM3`, `580.105.08`

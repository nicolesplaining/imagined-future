# Compute environment notes

## 2026-08-31 H100 host validation

The assigned host exposes two NVIDIA H100 80 GB GPUs connected by NVLink. The pinned official Cosmos Policy container builds successfully, GPU device nodes are exposed inside Docker, and the intervention suite passes in the official PyTorch 2.7/CUDA 12.8 environment.

CUDA computation is currently blocked before model loading by `cudaErrorSystemNotReady` (error 802). `nvidia-smi -q` reports `Fabric: State: In Progress` for both GPUs rather than `Completed / Success`. NVIDIA's [Fabric Manager documentation](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html) states that H100 CUDA initialization waits for GPU registration with the NVLink fabric. The following safe recovery attempts did not change the state:

- Verified matching driver and Fabric Manager versions.
- Started Fabric Manager; it reported no NVSwitch device to manage on this direct-NVLink pair.
- Reset both idle NVLinked GPUs together.
- Rebooted the guest host.

This is treated as an infrastructure/provisioning issue, not an experimental result. No checkpoint forward pass or outcome-bearing intervention has completed. A provider-level power cycle or repair of the passthrough fabric configuration is required before GPU experiments resume.

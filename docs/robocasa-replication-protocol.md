# RoboCasa cross-domain replication

Status: frozen before RoboCasa intervention outcomes.

This secondary study asks whether the LIBERO result transfers to a distinct released Cosmos Policy checkpoint, observation layout, robot embodiment, and simulator. It uses NVIDIA's public RoboCasa evaluation adapter and checkpoint rather than recreating the environment or model interface.

The public RoboCasa fork's 2024 pins (`numpy==1.23.3`, `numba==0.56.4`) cannot import the released Cosmos Policy Megatron runtime, which requires the `numpy.dtypes` API. The replication therefore records a compatibility-only environment override: NumPy 1.26.4, Numba 0.61.2, llvmlite 0.44.0, protobuf 6.33.5, and OpenCV 4.11.0.86. Source commits, simulator assets, evaluator logic, and model weights remain unchanged. Both a rendered simulator smoke test and a checkpoint inference smoke test must pass before natural branch collection.

The six fixed units cross three manipulation types with two deterministic scenes/timings:

- `OpenDrawer`, episode 0, at the first policy query;
- `OpenDrawer`, episode 10, after three deterministic prefix chunks;
- `TurnOffMicrowave`, episode 0, at the first policy query;
- `TurnOffMicrowave`, episode 10, after three deterministic prefix chunks;
- `PnPCounterToCab`, episode 0, at the first policy query;
- `PnPCounterToCab`, episode 10, after three deterministic prefix chunks.

The public evaluator's scene-selection rule maps episode ranges to fixed layout/style pairs. Each branch point is reconstructed from the same environment seed, scene index, model XML, simulator state, and ten-step stabilization sequence. Units failing exact state and rendered-observation replay are excluded without replacement.

The evaluator's environment seed is `195 * episode_index * 256`. Natural branches use the same eight registered model-sampling seeds as LIBERO. Each query is evaluated on the first 16 actions, matching the released evaluator's open-loop execution rule. The physical endpoint vector concatenates every finite non-image observation exposed by the simulator in sorted-key order and the simulator's generalized positions, which ensures articulated drawer, door, and appliance coordinates are represented; the full key-and-offset schema is frozen in each artifact. Pairs are selected from natural normalized-action and physical-endpoint divergence only. Both directions are patched.

The cross-domain semantic test uses all future modalities—proprioception, wrist image, left primary image, and right secondary image—with future-noise seed 401. Recipient/self and exact norm-and-distance-matched Gaussian targets are mandatory controls. Block 27 future-key removal is tested with all-key and equal-count current-key controls.

RoboCasa is a replication stratum, not an enlargement of the LIBERO sample. Its six units are reported separately and are not pooled into the LIBERO confidence interval. A concordant sign supports cross-domain transfer; a null or reversed result limits the claim to the LIBERO checkpoint and task distribution.

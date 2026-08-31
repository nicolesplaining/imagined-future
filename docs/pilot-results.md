# Exact-state semantic-clamp pilot

Status: exploratory calibration, not a confirmatory mediation result.

## Purpose

This pilot tests the concrete failure mode motivating the project: can an intervention substantially change Cosmos Policy's decoded future while leaving task-relevant behavior unchanged? It also validates deterministic LIBERO branching, simulator-grounded semantic labels, and call-aligned interventions through all five denoiser evaluations.

The experiment uses LIBERO-10 task 3, “put the black bowl in the bottom drawer of the cabinet and close it,” initial state 0. The fixed recipient state occurs after 12 deterministic action chunks (192 policy steps), plus the official 10-step warmup. Fresh-environment replay is bitwise exact in the flattened MuJoCo state. At this point:

- `In(akita_black_bowl_1, white_cabinet_1_bottom_region) = true`;
- `Close(white_cabinet_1_bottom_region) = false`;
- the bottom-drawer joint is `-0.1293439351`; and
- the model's predicted value is approximately `0.92285`.

The naturally continued rollout is just before success. After the next chunk, all eight sampled branches remain below the benchmark's closure threshold, with endpoint joint positions from `-0.002487` to `-0.001840`. One chunk later, all eight branches cross the threshold on action step 2 and succeed. This gives a sharply localized open-to-closed transition, although not a naturally mixed success/failure donor set.

## Semantic intervention

The implementation uses the public Cosmos Policy tokenizer and latent-injection code. It places a target observation into the future proprioception, wrist-camera, and primary-camera slots (latent temporal indices 5–7). At every denoiser evaluation, the input is clamped to the noise-matched target and the clean future output is clamped back to the target. Current observation, instruction, action noise, value noise, denoising schedule, and future-noise realization are identical between conditions.

Two targets are compared:

1. **Open target:** the fixed recipient observation itself.
2. **Closed target:** the realized successful endpoint from the continuation at prefix 13, branch seed 195.

The targets are temporally matched observations from one trajectory, not a natural same-state success/failure pair. Robot pose and drawer state both differ, so this run is an implementation and falsification pilot rather than the preregistered confirmatory comparison.

## Manipulation check

The clamp successfully controls the decoded primary-camera future.

| Quantity | Pixel L1 (0–255 scale) |
|---|---:|
| Open decoded future to open target | 2.8483 |
| Closed decoded future to closed target | 2.7172 |
| Open target to closed target | 20.5137 |
| Open decoded future to closed decoded future | 21.2358 |

Thus the intervention produces a large, target-aligned change in the explicit imagined future. The expected-value output is almost invariant: baseline `0.922855`, open clamp `0.922853`, and closed clamp `0.923000`.

## Action and behavior

The open- and closed-target clamps differ by `0.018869` in normalized action-chunk L2 (`0.010048` after conversion to LIBERO environment units). The change is distributed across all 16 steps and all seven action dimensions rather than isolated to the gripper command.

Each action is executed from a separately constructed environment and the exact replayed branch state. The bowl remains inside the drawer in every condition, but none crosses LIBERO's strict `qpos > 0` closure threshold:

| Condition | Endpoint drawer joint | Joint displacement | `In` | `Close` | Success |
|---|---:|---:|:---:|:---:|:---:|
| Baseline | -0.0024332 | +0.1269108 | true | false | false |
| Open-future clamp | -0.0024179 | +0.1269260 | true | false | false |
| Closed-future clamp | -0.0026085 | +0.1267354 | true | false | false |

The closed-future intervention therefore does **not** move behavior toward the transplanted closed state at this unit. Its endpoint is slightly less closed than both baseline and the open-target control.

## Residual-stream coupling scan

An independent exploratory scan transplants only future-token residuals after DiT blocks 0, 7, 14, 21, and 27. A same-run self-transplant at block 0 reproduces the recipient action bitwise exactly across all five denoiser calls. Independent future-noise donors cause large residual differences; full transplants are 22–136% of the recipient future-slice norm at the first denoiser evaluation and frequently overshoot the donor action. These full patches are treated as off-manifold diagnostics, not mediation evidence.

Scaled late-layer patches show dose-dependent donor-direction action movement, confirming a computational path from future-token residuals to the action output. Because the donor was defined by exogenous future noise rather than task semantics, this establishes coupling only.

## Interpretation

This one-state result has the qualitative form of behavioral epiphenomenality: the explicit future is changed substantially and correctly, while the task-relevant action endpoint is not moved toward it. It is not evidence that Cosmos Policy never uses imagined futures. Important remaining limitations are:

- one task, initial state, transition, and model seed;
- no naturally divergent same-state success/failure pair;
- a nearly saturated close action, which may leave little behavioral headroom;
- no modality ablations, shuffled donors, or norm-matched controls yet; and
- no held-out confirmatory states.

The next outcome-bearing stage is broader rollout collection across initial states and seeds to find naturally divergent semantic branches before any confirmatory layer or modality is selected.

# Natural-rollout and matched-branch screen

Status: corrected deterministic exploration complete at one selected state; held-out state screening in progress.

## Reproducibility correction

The first screen and clamps set `PolicyEvalConfig.deterministic=True` but called `get_action` directly. NVIDIA's public evaluator performs an additional side effect: it sets the `DETERMINISTIC=True` environment variable before inference. The public policy tokenizer checks this variable and otherwise samples during VAE encoding. The initial experiments therefore controlled diffusion seeds but not tokenizer sampling. This explains the two cross-process action variants described below and means that semantic clamps could differ in both endpoint content and tokenizer noise.

The shared configuration now reproduces the [official evaluator switch](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py#L803-L808), and every new artifact records whether deterministic tokenization was enabled. A fresh task-4/state-2 branch collection was run independently on both physical H100s; the complete compressed `branches.npz` artifacts have the same SHA-256 digest. Earlier results remain useful exploratory evidence, but none is promoted to a confirmatory claim until it is reproduced from the corrected branch artifact.

## Corrected deterministic result

The corrected branch screen again found a robust failure/success pair, but the donor changed. Branch seed 195 fails under all three shared continuation seeds (0/3), while branch seed 198 succeeds under all three (3/3, at policy steps 213--215). The current simulator state and observation are bitwise identical across branches.

This state was selected using the earlier rollout screen, and donor identities were assigned after observing continuation outcomes. The corrected analysis is therefore a clean exploratory within-unit result, not a held-out confirmatory estimate.

Two all-modality semantic clamps used independent future-noise draws. Both unclamped baselines exactly reproduce the saved recipient action. The clean target-latent norms are nearly matched (`137.6223` recipient and `137.6004` donor), with donor-minus-recipient L2 `9.7361`.

| Future-noise seed | Action donor steering | Executed-state donor steering | Endpoint-image donor preference |
|---|---:|---:|---:|
| 20195 | +0.2306 | +0.1603 | +0.2551 |
| 20196 | +0.2082 | +0.0441 | +0.1569 |

All three directional contrasts are positive in both draws. A rerun that additionally records the official nine-dimensional endpoint proprioception gives a donor-minus-recipient proprio steering contrast of `+0.2664` for noise seed 20195. This is clean evidence that explicitly changing future endpoint content can causally change the jointly generated action and its 16-step physical endpoint at this exact state. The two noise draws are replications of one experimental unit, not independent states, and do not establish a population effect. A robust-failure-to-robust-failure control produces a smaller action contrast (`+0.1151`), a negative executed-state contrast (`-0.0255`), and a positive endpoint-image contrast (`+0.1472`). The robust-success donor exceeds this control on action and simulator-state outcomes in both noise draws, but one control donor at one state is insufficient for a success-specific claim.

The other two robust-success donors also give positive action and endpoint-image contrasts: seed 201 gives `+0.0735` and `+0.1664`, while seed 202 gives `+0.2103` and `+0.4021`. Their full-state contrasts are slightly negative (`-0.0553` and `-0.0515`). Thus action and visual-endpoint steering replicate across all three success donors, whereas full-state steering is donor-dependent. The flattened MuJoCo state includes robot and object coordinates at incompatible scales, so the full-state projection is retained as a broad diagnostic rather than treated as the sole behavioral endpoint.

Corrected single-modality clamps show that the positive action effect is visual:

| Future slots clamped | Action steering | Executed-state steering | Endpoint-image preference |
|---|---:|---:|---:|
| Wrist camera | +0.1439 | +0.1602 | +0.0831 |
| Primary camera | +0.1462 | -0.0686 | +0.2538 |
| Proprioception | -0.0494 | -0.0714 | +0.0093 |

Wrist and primary-camera targets have comparable action effects but different endpoint signatures. The wrist clamp accounts for the positive full-state contrast, while the primary-camera clamp accounts for the primary-image contrast. Future proprioception is directionally negative on both action and state outcomes. Because modalities interact in the joint denoiser, these single-slot effects are not additive decompositions of the all-modality effect.

The versioned machine-readable table is [results/task4_state2_deterministic_clamps.csv](../results/task4_state2_deterministic_clamps.csv).

## Independent-state negative

A corrected deterministic screen at task-8/state-3 provides an independent exact-state endpoint pair. Its outcome labels are not robust: recipient seed 195 fails under continuation seeds 195 and 196 but succeeds under 197, so this unit is excluded from success/failure mediation. It remains a valid test of local future-conditioned behavior.

The clamp passes exactness and manipulation checks. Recipient and donor decoded futures differ by `0.6091` primary-camera pixels and `1.0663` wrist-camera pixels. Nevertheless, donor-minus-recipient action steering is only `+0.0015`. Executed contrasts are likewise near zero: `+0.0094` for full simulator state, `+0.0094` for primary endpoint-image preference, and `-0.0263` for endpoint proprioception. This unit has the epiphenomenal pattern motivating the project: the explicit future changes while behavior does not move coherently toward it. Its machine-readable row is [results/task8_state3_deterministic_clamp.csv](../results/task8_state3_deterministic_clamp.csv).

Task-4/state-9 also failed to yield a robust outcome pair: candidate branch labels changed across continuation seeds. No success-mediation intervention is reported for that state.

## Natural rollout screen

Cosmos Policy was evaluated on all ten LIBERO-10 tasks and initial states 0--9 using model seed 195, the official ten-step no-op warmup, 16-action open-loop chunks, and a 520-step limit. The collector preserves the checkpoint's float32 action output through the official action-unnormalization operation. It records complete simulator states, normalized and environment-space actions, expected values, and simulator-grounded goal predicates at every query.

The policy succeeded in 96/100 episodes. The four failures were:

- task 4, states 2 and 9: the yellow-and-white mug reached the right plate, but the white mug never reached the left plate;
- task 8, states 3 and 8: the second moka pot reached the stove, but the first did not.

Seeds 196--202 were then evaluated only on these four states. Every state showed mixed outcomes. Including seed 195, the success counts were 4/8 and 3/8 for task-4 states 2 and 9, and 5/8 and 6/8 for task-8 states 3 and 8. These are screening frequencies over deliberately selected hard states, not estimates of benchmark success probability.

Full image-preserving reruns were collected for task-4 state 2 (seed 195 failure, seed 196 success) and task-8 state 3 (seed 196 failure, seed 197 success). Every numeric rollout array reproduced bitwise, including actions and simulator states.

## Exact-state branch construction

Episode-level matching is insufficient for mediation because successful and failed trajectories have different current observations after their first action. We therefore restored one simulator state bitwise, sampled eight first action chunks, and replayed each chunk from a fresh environment. A second experiment held all subsequent policy queries to one shared continuation seed. This makes eventual outcome differences attributable to the initial action chunk and the state it induces rather than to different downstream policy noise.

For task-4 state 2, continuation seed 195 produced three successes and five failures. The prespecified continuation seeds 195, 196, and 197 give a robust label to the primary pair: the saved seed-195 initial chunk fails in all three continuations, while the seed-196 initial chunk succeeds in all three at policy steps 216--217. Across all eight initial chunks, three are robust successes, one is a robust failure, and four have continuation-dependent outcomes. For task-8 state 3, the first continuation screen produced five successes and three failures; its two continuation-seed replications remain exploratory follow-up.

An exactness check found that independently generated task-8 actions can differ at approximately `2e-3` in environment units across two otherwise matched script paths. A task-4 clamp moved to the other physical H100 differed from its saved GPU-0 recipient by `0.004727` in normalized action units. The internally replayed branch endpoints remain bitwise stable, but no semantic clamp is interpreted unless its baseline action exactly reproduces the saved recipient action on the same GPU. The clamp runner now enforces zero maximum absolute error and aborts otherwise; the non-matching task-8 clamp and cross-GPU task-4 run were excluded before producing intervention outcomes.

## First matched-state semantic clamp

The first matched intervention uses task-4 state 2 at the initial policy query. The recipient is branch seed 195 and the donor is branch seed 196. They share the exact current simulator state, camera observations, proprioception, and instruction. Only their sampled first action chunks and realized 16-step endpoints differ. Under continuation seed 195, the recipient branch eventually fails and the donor branch succeeds.

The intervention uses the public Cosmos tokenizer to encode each realized endpoint into future proprioception, wrist-camera, and primary-camera latent slots. Recipient and donor clamps share action noise, value noise, future noise, denoising schedule, and all current conditioning. The unclamped baseline reproduces the saved recipient action exactly.

| Outcome | Recipient-target clamp | Donor-target clamp | Donor minus recipient |
|---|---:|---:|---:|
| Action donor steering | 0.3374 | 0.4893 | +0.1519 |
| Executed-state donor steering | 0.5159 | 0.5799 | +0.0640 |
| Endpoint primary-image donor preference | -0.0934 | +0.3356 | +0.4291 |

The endpoint-image preference is distance to the recipient reference minus distance to the donor reference, so positive values favor the donor. The donor clamp endpoint is 0.3243 pixel levels from the donor endpoint and 0.6599 from the recipient endpoint; the recipient clamp shows the opposite ordering (0.4861 to recipient and 0.5796 to donor). Neither first chunk completes a Boolean task predicate, so this is a continuous local-state result rather than a one-chunk success intervention.

Two additional targets were evaluated with the same recipient, future-noise realization, and all-modality clamp. A second robust-success donor (seed 198) gives action, executed-state, and endpoint-image contrasts of `+0.2099`, `+0.1195`, and `+0.2798`. A same-outcome donor that fails under the primary continuation (seed 197) gives corresponding contrasts of `+0.2033`, `+0.0369`, and `+0.1486`. The local endpoint-conditioned effect is therefore not exclusive to eventual-success donors. With only two success donors and one same-outcome control from one simulator state, these contrasts are descriptive and are not used for inference.

The manipulation check is directionally successful but modest. Recipient and donor endpoint targets differ by only 0.7697 pixel levels after preprocessing, while their decoded clamped futures differ by 1.1860. Each decoded future is about 2.9 pixel levels from its own target. Both clamps also cause substantial generic movement toward the donor action relative to the unclamped baseline; the causal contrast is therefore the donor-minus-recipient difference, not either raw clamp displacement.

Exploratory follow-ups before the reproducibility correction were directionally consistent. A second future-noise draw produced action, executed-state, and endpoint-image contrasts of `+0.2012`, `+0.0999`, and `+0.3602`. An independently collected branch reference on the other H100 produced `+0.2390`, `+0.1493`, and `+0.4237`. Single-modality clamps localized the effect mainly to visual future slots:

| Future slots clamped | Action steering | Executed-state steering | Endpoint-image preference |
|---|---:|---:|---:|
| Wrist camera | +0.1324 | +0.1403 | +0.2725 |
| Primary camera | +0.0932 | +0.0723 | +0.0544 |
| Proprioception | -0.0370 | -0.0338 | +0.0162 |

These follow-ups share the uncontrolled-tokenizer limitation and are hypothesis-generating only. They motivate a prespecified visual-versus-proprioception contrast in the corrected pipeline rather than serving as inferential replications.

## Interpretation

Within the original pipeline, these units are evidence against strict behavioral epiphenomenality at the tested state: changing the local endpoint target changes the action and executed endpoint in the donor direction relative to an equally strong recipient-target clamp. The tokenizer omission prevents the stronger statement that endpoint semantics alone caused the contrast. It is also not evidence that Cosmos Policy generally relies on imagined futures or that future latents mediate information about eventual success. Confirmatory interpretation requires reproduction under deterministic tokenization, shuffled and norm-matched controls, additional independent simulator states, and held-out analysis choices.

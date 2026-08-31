# Natural-rollout and matched-branch screen

Status: exploratory donor discovery and one matched-state intervention; not a confirmatory mediation result.

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

## Interpretation

This unit is evidence against strict behavioral epiphenomenality at the tested state: changing only the local endpoint target changes the action and executed endpoint in the donor direction relative to an equally strong recipient-target clamp. It is not yet evidence that Cosmos Policy generally relies on imagined futures, nor that future latents mediate information about eventual success. The result is one state, three donor targets, one all-modality clamp, and one future-noise realization. Confirmatory interpretation requires modality ablations, shuffled and norm-matched controls, additional independent simulator states, future-noise replications, and held-out analysis choices.

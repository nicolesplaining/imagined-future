# Cosmos 3 intervention validation

## Status

These are engineering and synthetic-observation pilot results from the released
Cosmos3-Nano-Policy-DROID checkpoint. They validate the port and provide an
early directional signal. They are not included in confirmatory RoboLab causal
inference because no simulator state was restored or physically executed.

## Exact no-op and token census

The pinned 16B checkpoint was run through the official DROID request transform
with guidance 3, four UniPC steps, shift 5, a 33-frame 540x640 video, and 33
action rows (one current-state row plus 32 generated rows). The prepared flat
state contained:

- vision `[1,48,9,33,40]`, flat coordinates `[0,570240)`;
- action `[33,64]`, flat coordinates `[570240,572352)`.

The 33 RGB frames therefore become nine VAE latent frames. Post-guidance clean
estimates were exposed at sampler sigmas 0.999, 0.937, 0.833, and 0.624. An
identity layout-capture builder plus an identity post-guidance sampler wrapper
reproduced both final action and vision latents bit-for-bit: maximum absolute
error was exactly zero for both modalities.

This test also resolved an architecture-specific trap. The public velocity
postprocess hook runs on the conditional branch before text classifier-free
guidance; treating it as the final clean-video interface would not implement an
exact clamp at guidance 3. The port instead intervenes on the combined velocity
at the sampler boundary.

## Synthetic natural-future pilot

Two native generations from the same bundled banana observation and instruction
used diffusion seeds 0 (recipient) and 1 (donor). Their 32x8 external action
chunks differed by L2 8.080. Future latent frames 1 through 8 were clamped while
the conditioned current frame and every action coordinate were preserved.
Programmatic sentinels measured exactly zero direct action-input and
action-output mutation on every denoising call.

| target | action projection toward donor | action L2 from recipient | decoded L1 to donor |
| --- | ---: | ---: | ---: |
| self future | -0.195 | 2.050 | 0.0998 |
| donor future | **0.325** | 6.358 | **0.0000** |
| matched Gaussian | -0.059 | 1.884 | 0.2000 |

The donor-minus-self contrast is +0.520 and donor-minus-Gaussian is +0.384.
The Gaussian target matched donor norm with relative error `5.42e-7` and
recipient distance with relative error `2.33e-6`. The decoded transplanted
future was identical to the decoded donor under the measured L1 metric.

This is the first positive Cosmos 3 content-specific action signal: a coherent
natural future steered action toward its associated donor more than self or a
geometry-matched nonsemantic target. Its limitations are decisive: the current
observation is checkpoint-bundled, the two futures were not validated as
physically reachable from a saved simulator state, there is one pair, and no
action was executed. It supports proceeding to RoboLab; it does not yet support
a cross-task or physical causal claim.

Machine-readable reports are in `results/cosmos3_noop_v1/`.

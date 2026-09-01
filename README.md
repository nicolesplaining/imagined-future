# Does a world-action model use its imagined future?

This project tests whether Cosmos Policy's generated future-state latents causally influence its actions during direct-policy inference. The central alternative is an observation-to-action shortcut in which future prediction improves training but is behaviorally epiphenomenal at inference.

Cosmos Policy jointly denoises action, future proprioception, future camera images, and value in one latent sequence. Predictive probes can show that a hidden state contains future information, but not that the policy uses that information. The project therefore uses paired simulator branches, common random numbers, semantic future clamping, and future-to-action path interventions.

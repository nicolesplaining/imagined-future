# Does a world-action model use its imagined future?

This project tests whether Cosmos Policy's generated future-state latents causally influence its actions during direct-policy inference. The central alternative is an observation-to-action shortcut in which future prediction improves training but is behaviorally epiphenomenal at inference.

Cosmos Policy jointly denoises action, future proprioception, future camera images, and value in one latent sequence. Predictive probes can show that a hidden state contains future information, but not that the policy uses that information. The project therefore uses paired simulator branches, common random numbers, semantic future clamping, and future-to-action path interventions.

The confirmatory Cosmos 3 replication analyzes 22 saved-state clusters across
six public RoboLab tasks. Coherent predicted and physically executed donor
futures directionally steer both action and execution, and future-token K/V
replacement suppresses that steering. A separate 10-state, six-task factor
study finds that neither isolated robot-motion pixels nor isolated object-state
pixels reproduce a meaningful fraction of the whole-future effect.

Key reproducibility artifacts:

- [Cosmos 3 protocol](docs/cosmos3-replication-protocol.md)
- [Cosmos 3 population results](docs/cosmos3-population-results.md)
- [Frozen manifest](results/cosmos3_population/manifest.json)
- [Machine-readable aggregate](results/cosmos3_population_confirmatory_v1/aggregate_summary.json)
- [Failure report](results/cosmos3_population_confirmatory_v1/failure_report.json)

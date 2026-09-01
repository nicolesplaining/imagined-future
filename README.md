# Does a world-action model use its imagined future?

This project tests whether Cosmos Policy's generated future-state latents causally influence its actions during direct-policy inference. The central alternative is an observation-to-action shortcut in which future prediction improves training but is behaviorally epiphenomenal at inference.

Cosmos Policy jointly denoises action, future proprioception, future camera images, and value in one latent sequence. Predictive probes can show that a hidden state contains future information, but not that the policy uses that information. The project therefore uses paired simulator branches, common random numbers, semantic future clamping, and future-to-action path interventions.

## Current answer

Cosmos Policy causally uses prospective visible robot-motion content, but the
current evidence does not show meaningful use of task-object consequences.

- Reachable natural-future transplantation steers early actions and executed
  endpoints across all ten LIBERO-10 tasks.
- Robot-only natural donors strongly steer action and robot execution in the
  four rare states where robot and object outcomes can be separated; wrist
  imagery is the largest single-modality effect.
- The complete natural screen found a common object/robot-matched recipient in
  4 of 20 trajectories. Robot endpoint steering was 0.315 (95% CI 0.059 to
  0.571), while object goal steering was 0.021 (-0.091 to 0.132) and did not
  beat the matched natural-future control.
- In a prospectively frozen 20-state 2x2 study, decoded futures identify the
  imposed object/robot target with 92.5% primary-camera and 98.75% wrist-camera
  accuracy, while executed endpoints remain at 25% four-cell chance. The object
  main effect is 0.00018 (95% CI -0.00040 to 0.00102), far below the registered
  0.10 meaningful-effect threshold.

The precise claim is endpoint conditioning through prospective robot motion,
not planning, search, goal reasoning, or mediation of task success. See the
[robot-versus-object results](docs/content-factorization-results.md),
[publication assessment](docs/publication-assessment.md), and
[public foundations](docs/references.md).

The Cosmos 3 cross-generation replication is now underway. Its first excluded
RoboLab pilot is positive: a tokenizer-encoded reachable donor moved action
0.999 and physical robot execution 0.998 toward its associated endpoint, while
the matched Gaussian moved them only 0.032 and 0.040. This currently supports
robot-motion/inverse-dynamics use, not object-consequence reasoning. The pinned,
outcome-independent protocol, exact audits, and pilot are in the
[Cosmos 3 protocol](docs/cosmos3-replication-protocol.md) and
[engineering results](docs/cosmos3-engineering-results.md).

## Reproducibility

Prospective protocols and pinned upstream revisions are in `configs/` and
`docs/`. Machine-readable aggregate summaries and complete unit-level repeated
measurements are in `results/`. Large simulator states, decoded images, and
execution artifacts remain outside Git and are addressed by SHA-256 hashes in
the committed manifests.

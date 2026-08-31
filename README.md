# Does a world-action model use its imagined future?

This project tests whether Cosmos Policy's generated future-state latents causally influence its actions during direct-policy inference. The central alternative is an observation-to-action shortcut in which future prediction improves training but is behaviorally epiphenomenal at inference.

## Why this requires interventions

Cosmos Policy jointly denoises action, future proprioception, future camera images, and value in one latent sequence. Predictive probes can show that a hidden state contains future information, but not that the policy uses that information. The project therefore uses paired simulator branches, common random numbers, semantic future clamping, and eventually future-to-action path interventions.

The primary pilot is preregistered in [docs/preregistration.md](docs/preregistration.md). Public papers and code used by the project are catalogued in [docs/references.md](docs/references.md).

## Experimental ladder

1. **Future-noise sensitivity:** keep action noise fixed and resample only future-frame noise.
2. **Semantic sufficiency:** clamp a realized counterfactual future throughout denoising and measure directional action steering.
3. **Natural necessity:** block future-to-action communication while preserving the rest of generation.
4. **Behavior:** execute intervened actions from exactly restored LIBERO states.

The first test detects computational coupling. Only the combination of semantic steering, appropriate controls, and behavioral effects supports the stronger claim that the policy uses an imagined future.

## Upstream provenance

The pilot uses NVIDIA's public [`NVlabs/cosmos-policy`](https://github.com/NVlabs/cosmos-policy) implementation at commit `18a2accadf4e7a3531e56754102af5a24d2316da`, its released `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` checkpoint, and the official [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark. Upstream code is installed separately and is not vendored here.

## Repository layout

- `src/imagined_future/`: intervention primitives, upstream adapters, and estimands.
- `tests/`: CPU unit tests for intervention locality and metrics.
- `configs/pilot.toml`: frozen pilot settings and upstream revisions.
- `docs/preregistration.md`: hypotheses, controls, exclusions, and analysis plan.
- `artifacts/`: ignored machine-generated artifacts; raw data should live in immutable run directories.

## Development check

Run inside an environment containing PyTorch:

```bash
python -m pip install -e '.[test]'
pytest
```

The H100 experiment environment follows the official Cosmos Policy Docker setup. Run manifests must capture the image digest rather than relying on a mutable image tag.

## Current status

The intervention layer and preregistration are in place. A three-seed public-observation smoke test found weak but exactly reproducible future-to-action computational coupling; see [docs/smoke-results.md](docs/smoke-results.md). Exact LIBERO replay and semantic predicate annotation are validated. The first drawer-state clamp strongly changed the decoded future without moving behavior toward closure, an exploratory negative described in [docs/pilot-results.md](docs/pilot-results.md). A subsequent 100-episode screen found exact-state action branches with robustly different outcomes under shared continuation policies. On the first matched state, endpoint clamps steer actions and executed endpoints toward their donors, but a same-outcome donor also produces an effect; see [docs/rollout-screen-results.md](docs/rollout-screen-results.md). This currently supports local future-conditioned behavior, not a general or success-specific mediation claim.

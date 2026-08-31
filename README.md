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

The intervention layer and preregistration are in place. A three-seed public-observation smoke test found weak future-to-action computational coupling; see [docs/smoke-results.md](docs/smoke-results.md). Exact LIBERO replay and semantic predicate annotation are validated. The first drawer-state clamp strongly changed the decoded future without moving behavior toward closure, an exploratory negative described in [docs/pilot-results.md](docs/pilot-results.md). A subsequent screen initially omitted an environment switch required for deterministic VAE encoding in NVIDIA's evaluator; those runs are retained as exploratory evidence. Corrected branch artifacts are bitwise identical across the two physical H100s. On one robust failure/success pair, deterministic endpoint clamps steer actions and executed endpoints toward the donor across noise draws and success donors, primarily through visual future slots. An independent task-8 state instead gives near-zero action and endpoint effects despite a changed decoded future; see [docs/rollout-screen-results.md](docs/rollout-screen-results.md). Selectively removing future keys from action queries in a single DiT block moves action and endpoint proprioception away from the successful donor with bitwise-exact all-key controls; see [docs/necessity-results.md](docs/necessity-results.md). The current result is therefore state-dependent local future use, not a general or uniformly success-specific mediation claim.

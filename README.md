# Does a world-action model use its imagined future?

This project tests whether Cosmos Policy's generated future-state latents causally influence its actions during direct-policy inference. The central alternative is an observation-to-action shortcut in which future prediction improves training but is behaviorally epiphenomenal at inference.

## Why this requires interventions

Cosmos Policy jointly denoises action, future proprioception, future camera images, and value in one latent sequence. Predictive probes can show that a hidden state contains future information, but not that the policy uses that information. The project therefore uses paired simulator branches, common random numbers, semantic future clamping, and eventually future-to-action path interventions.

The primary pilot is preregistered in [docs/preregistration.md](docs/preregistration.md). The held-out multi-task study is frozen in [docs/confirmatory-protocol.md](docs/confirmatory-protocol.md) and [configs/confirmatory.toml](configs/confirmatory.toml). Its results are reported in [docs/confirmatory-results.md](docs/confirmatory-results.md), with claim boundaries in [docs/publication-assessment.md](docs/publication-assessment.md). The separate cross-domain study is frozen in [docs/robocasa-replication-protocol.md](docs/robocasa-replication-protocol.md). Public papers and code used by the project are catalogued in [docs/references.md](docs/references.md).

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

The frozen study is complete for 30 independently reconstructed states spanning all ten LIBERO-10 tasks and three timing strata. At the first policy query, transplanting a model-consistent donor future steers actions toward the donor in all ten tasks and all 60 reciprocal/noise repetitions (state mean +0.4986, 95% bootstrap CI [+0.3360, +0.6785]) and steers executed physical endpoints in all ten tasks (+0.5520, 95% CI [+0.3875, +0.7279]). Exact norm/distance-matched Gaussian, natural-future, shuffled-future, modality, and attention-path controls are complete. Future-key removal has a monotonic action dose response, survives executed-endpoint and no-op controls, but is not more disruptive than current-key removal. The effect is sharply state dependent: mid and late semantic strata are below the registered practical-effect threshold. No screened pair retains a robust binary success/failure contrast across continuation seeds. A separately frozen six-unit RoboCasa replication is positive at physical endpoints in 6/6 states and all three task families; it remains exploratory and is not pooled with LIBERO.

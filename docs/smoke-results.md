# Public-observation coupling smoke test

Status: exploratory implementation validation, not a confirmatory semantic-use result.

## Question and setup

The test asks whether future-frame exogenous noise can affect Cosmos Policy's action when the current observation, instruction, action-frame noise, value-frame noise, denoising schedule, and model seed are fixed.

The input is the official public LIBERO observation for the instruction “put both the alphabet soup and the tomato sauce in the basket.” For each run, the baseline uses the official deterministic inference path. The paired intervention resamples only latent frames 5–7, which the runtime batch identifies as future proprioception, future wrist image, and future primary image. Actions remain normalized so the reported distances are in model output space.

## Results

| Model seed | Future-noise seed | Action L2 | Max absolute action change | Action L2 / baseline norm | Primary-image pixel L1 | Wrist-image pixel L1 |
|---:|---:|---:|---:|---:|---:|---:|
| 195 | 10195 | 0.05289 | 0.02099 | 1.20% | 0.8767 | 1.3608 |
| 196 | 10196 | 0.06734 | 0.02024 | 1.52% | 0.8826 | 1.5252 |
| 197 | 10197 | 0.08958 | 0.02862 | 2.02% | 0.9628 | 1.4172 |

Mean action L2 is 0.06994 with a sample standard deviation of 0.01848. Action displacement occurs throughout the 16-step chunk rather than in a single timestep. Repeating seed 195 in a fresh container produces bitwise-identical baseline and intervened action arrays (maximum absolute difference zero for both), excluding ordinary run-to-run nondeterminism as the source of the paired effect.

Decoded-future differences are small: roughly one intensity level per pixel on a 0–255 scale. The action effect is therefore evidence that future-frame noise is computationally coupled to action generation, but it does not establish that a task-level imagined outcome mediates the chosen action.

## Interpretation limits

- This is one public observation, not a sample of independent simulator states.
- Exogenous noise has no assigned task semantics and can act through arbitrary interactions in the joint denoiser.
- The test does not compare against semantic donors, shuffled donors, random-frame patches, or norm-matched replacements.
- Pixel L1 is a plumbing diagnostic, not a perceptual or task-state measure.
- The three seeds are prespecified replications of one unit, not three independent experimental units.

The supported conclusion is narrow: the direct policy is not perfectly invariant to its jointly generated future frames. The result neither proves nor refutes behaviorally meaningful future mediation. The next stage must construct divergent branches from identical LIBERO states, clamp realized donor futures, and measure donor-directed action steering and executed endpoint changes against all preregistered controls.

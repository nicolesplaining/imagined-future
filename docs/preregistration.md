# Pilot preregistration

Status: design frozen before outcome-bearing model runs.

## Research question

During Cosmos Policy's direct-policy inference, do computations over generated future-state frames causally influence the generated action, or does future prediction help only through training-time representation learning?

This pilot does **not** test whether the explicit best-of-N planner uses its future. That planner mechanically evaluates predicted futures. It tests the stronger and less established claim that an unplanned, jointly denoised action uses the future generated in the same diffusion trajectory.

## Computational causal model

The direct policy does not implement a simple feed-forward chain `observation -> future -> action`. At denoising evaluation `k`, a bidirectional diffusion transformer jointly maps the current action and future latent frames to clean estimates. The numerical solver then constructs the next noisy trajectory. Consequently, future-to-action influence may occur both within a denoiser evaluation and across solver evaluations.

We therefore separate three claims:

1. **Coupling:** future-frame exogenous noise affects the action while action-frame noise and conditioning are fixed.
2. **Semantic sufficiency:** presenting a noise-matched counterfactual future steers the action toward the action associated with that future.
3. **Natural necessity:** removing future-to-action paths from an otherwise unmodified generation degrades action quality or task success.

Claims 1 and 2 are in scope for the first pilot. Claim 3 requires attention-edge interventions and is confirmatory follow-up work.

## Units and branch construction

The experimental unit is a saved LIBERO simulator state, not an episode. For each state:

1. Save the complete simulator state, observation, instruction, task identifier, and timestep.
2. Restore the identical state for each candidate action chunk.
3. Execute at least eight policy-sampled action chunks.
4. Save the resulting observation after the 16-step chunk and the eventual task outcome.
5. Construct donor/recipient branches only within the same saved state and instruction.

Primary pairs have different endpoint task-relevant state and action separation above the prespecified floor. Success/failure labels are secondary: a branch can be scientifically useful when it changes the relevant subgoal without changing terminal success.

We exclude states where restoration is not bitwise stable in simulator state or where two no-op validation rollouts differ beyond numerical tolerance before any policy action is applied.

## Interventions

All paired generations use common random numbers for the current observation, action frame, and value frame.

### I1: future-noise resampling

Resample only the initial Gaussian noise at the future proprioception, wrist-image, and primary-image latent frames. The observation, instruction, action-frame noise, denoising schedule, and all other frames remain fixed.

This is a coupling test, not evidence of semantic future use by itself.

### I2: semantic future clamp

Encode a donor branch's realized endpoint into the corresponding future frames. At every denoiser evaluation with noise level `sigma`, replace the future-frame denoiser input by `z_donor + sigma * epsilon_donor`. Also replace the clean future-frame prediction by `z_donor` before returning it to the solver.

The primary intervention clamps all future-state modalities. Prespecified ablations clamp future proprioception, wrist image, and primary image separately.

### Controls

- Same-outcome donor from the same saved state.
- Donor labels shuffled across saved states within a task.
- Equal-count random temporal frames, excluding current-state and action frames.
- Norm-matched Gaussian replacement at the future frames.
- Action-frame transplantation as a positive implementation control.

Activation distances and decoded endpoint distances are reported for every patch class. Large out-of-distribution patches will not be interpreted as localized mediation evidence.

## Outcomes

### Primary outcome

For recipient action `a_r`, donor action `a_d`, and patched action `a_p`, the donor-steering score is

`S = dot(a_p - a_r, a_d - a_r) / ||a_d - a_r||^2`.

`S = 0` indicates no directional movement and `S = 1` indicates complete recovery of the donor displacement under Euclidean projection. The action chunk is normalized using the checkpoint's training statistics before this calculation. We also report per-timestep and per-action-dimension scores.

### Secondary outcomes

- Euclidean and cosine action displacement.
- Change in endpoint task predicate after executing the patched action.
- Change in full-episode success under patched actions.
- Decoded-future similarity to donor and recipient endpoints.
- Across-noise action variance with action-frame noise fixed.

## Estimation and uncertainty

The primary estimand is the mean within-state difference in donor steering between divergent-future and same-outcome patches. Saved state is the clustering unit. We report a cluster bootstrap 95% confidence interval with 10,000 resamples and the full effect distribution.

Model seeds 195, 196, and 197 are analyzed as prespecified replications, not pooled pseudo-independent samples. Task-level effects are shown separately before any aggregate.

The primary test uses a two-sided familywise alpha of 0.05 across the all-future clamp and three modality ablations using Holm correction. Effect sizes and uncertainty take priority over thresholded significance.

## Interpretation thresholds

- **Evidence for semantic use:** decoded future moves to the donor, donor steering is positive relative to all controls, and executing the patched action changes the task-relevant endpoint in the donor direction.
- **Coupling only:** future noise changes actions, but semantic patches do not outperform norm-matched or shuffled controls.
- **Evidence against runtime mediation at tested sites:** future decoding changes substantially while action effects are practically indistinguishable from zero with a confidence interval excluding the smallest effect of interest.

The pilot does not define a smallest effect of interest until baseline action variability is measured on a held-out calibration set. It will be frozen before divergent-future interventions are evaluated.

## Multiplicity and researcher degrees of freedom

Layer and denoising-step localization is exploratory in the pilot. Any promising site must be evaluated on held-out states in a new confirmatory run. We will not select a site and report its effect on the same states as confirmatory evidence.

Activation patching can mix mediated effects with interactions among multiple mediators. We therefore report group patches, individual-modality patches, pairwise patches, patch distance, and intervention-by-context heterogeneity. We will not call an individual component a unique mediator from a single-node patch effect.

## Reproducibility record

Every run records the project commit, Cosmos Policy commit, checkpoint revision, container digest, CUDA/PyTorch versions, GPU model, complete config, simulator task/state identifiers, and seeds. Raw arrays are immutable; exclusions and derived tables are generated by versioned scripts.

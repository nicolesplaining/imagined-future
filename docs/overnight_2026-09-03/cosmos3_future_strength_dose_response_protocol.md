# Cosmos 3 future-strength dose response: prospective protocol

Status: **design frozen before opening any Cosmos 3 archival v7 evaluation
outcome**. This optional extension may run only after its implementation passes
unit tests, an excluded-state smoke, and an independent outcome-blind GO audit.

## Question and scope

Does donor-directed action steering increase as the imposed future target moves
continuously from the recipient future to the donor future?

This is an action-only intervention-strength experiment. It tests a graded
causal response under the same continuously imposed latent path as the archival
selection-free study. It does not estimate natural mediation, physical task
success, or the effect of a policy-native noisy trajectory. Latent
interpolations may be off manifold, so decoded examples are manipulation checks
rather than evidence criteria.

## Frozen population

- Use exactly the 30 middle-phase states in the immutable archival v7 manifest:
  six tasks x five archived episodes, one state per episode.
- Retain all states, all four branch seeds `211, 223, 227, 229`, and all 12
  ordered off-diagonal recipient-to-donor pairs. No outcome-, separation-,
  success-, or appearance-based filtering is permitted.
- Treat task, archived episode, and state as the inferential hierarchy. Arms
  within a state are repeated measurements, not independent samples.
- Use the same archived H.264 observation reconstruction, recorded
  proprioception, checkpoint content manifest, container image, and four-step
  UniPC schedule as archival v7.

## Intervention

For recipient future target `F_A`, donor target `F_B`, and
`alpha in {0, 0.25, 0.50, 0.75, 1.00}`, construct

`F_alpha = F_A + alpha * (F_B - F_A)`

at future latent frames 1--8 only. Current-frame coordinates remain from the
recipient. At all four denoising calls, use the recipient path noise and impose

`x_sigma = (1 - sigma) * F_alpha + sigma * epsilon_A`

on future coordinates. Retain the model-computed action velocity and overwrite
only selected future-coordinate velocity so that it targets `F_alpha`.

The current observation, instruction, checkpoint, recipient action seed,
recipient initial sampler state, recipient path noise, sampler, schedule, and
all nonfuture coordinates remain fixed within an ordered pair.

## Exact request matrix

Per state, run exactly 92 calls in frozen order:

1. Four native branches and four exact native replays.
2. Four zero-active-site `none` controls.
3. Four recipient self-clamps and four exact self-clamp replays.
4. All 12 ordered off-diagonal pairs at all five alpha values: 60 calls.
5. One exact replay of the `alpha=0.50` arm for each ordered pair: 12 calls.

Total powered cohort: 30 states x 92 = 2,760 model calls. An excluded Bagels
state must first run the identical 92-call matrix.

## Frozen outcomes

For every off-diagonal pair and alpha, compute against unconstrained native
recipient and donor actions:

- distance reduction toward the donor;
- normalized donor-axis projection;
- cosine alignment;
- normalized orthogonal residual;
- correct-donor nearest-neighbor identification among all four native actions.

Primary estimand: the state-averaged ordinary-least-squares slope of distance
reduction versus alpha, aggregated with a task -> episode/state hierarchical
bootstrap. The primary criterion is a 95% interval whose lower bound exceeds
zero.

Confirmatory secondary estimands:

- the `alpha=1` minus `alpha=0` distance-reduction contrast;
- four prespecified adjacent-alpha distance-reduction contrasts;
- slope and endpoint contrast for normalized projection;
- donor-identification rate at each alpha;
- the fraction of ordered pairs with nondecreasing donor proximity across all
  five alpha values.

Report every alpha, every adjacent contrast, and every task. A positive global
slope permits the phrase **graded response**. The stronger phrase **monotonic
dose response** is permitted only if all four adjacent point estimates are
nonnegative; lack of significance for an individual adjacent contrast must be
reported. No layer, task, pair, or alpha may be selected after outcomes.

## Manipulation and identity gates

The excluded smoke and every powered state must pass all of the following:

- full checkpoint-content verification and immutable code/manifest hashes;
- exact 92-call request order and cardinality;
- one transformed-input fingerprint and one expected parameter-probe hash;
- four denoising calls with the frozen sigmas and exact requested/observed
  active-site indices;
- zero model-input target error and zero returned-future-velocity overwrite
  error at every active site;
- zero action-coordinate writes and zero inactive-wrapper writes;
- exact native, self-clamp, midpoint, and zero-site replays;
- exact recipient initial-state and path-noise hashes across alpha within pair;
- exact interpolation formula on future coordinates, exact recipient target on
  nonfuture coordinates, and exact endpoint identities at alpha 0 and 1;
- finite actions, targets, residuals, and all defined metrics. Structurally
  undefined directional values are allowed only where explicitly enumerated by
  the frozen schema; no broad nonfinite exception is allowed.

Final sampler-state residual is finite and descriptive only. It is never an
admission, exclusion, stopping, or evidence criterion.

## Analysis and stopping

- Run the complete 30-state cohort after a GO audit; do not inspect partial
  scientific payloads.
- The analyzer must refuse any missing or extra state file and must revalidate
  every manipulation gate before aggregation.
- Use 10,000 seeded hierarchical bootstrap draws. Provide equal-task means,
  state-weighted sensitivity estimates, per-task values, and leave-one-task-out
  estimates.
- No early stopping for favorable or unfavorable effects. Runtime or mechanical
  failure may stop the run, but any supersession requires a new immutable root,
  manifest, smoke, and audit trail.

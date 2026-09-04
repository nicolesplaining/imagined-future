# Prospective Cosmos 3 single-call timing protocol v2

Status: **prospective and inactive.** This document was written before any
timing-study model call or outcome. It supersedes v1 only if a later,
content-addressed manifest incorporates it verbatim. The v1 document remains
unchanged. No launch is authorized by this protocol alone.

## Scope and question

This audit asks whether changing the registered clean future changes the action
when the future clamp is active at one specified denoising call, and whether
the corresponding effect is larger when the clamp is active at all four calls.
It is an action-space strength/timing test of an imposed intervention. It is not
a test of natural mediation, physical task success, or whether any call is
necessary. A clamp at one call changes the sampler state carried into later
calls, so a `call_i_only` result is attributed to an intervention initiated at
that call, not to an isolated local computation at that call.

## Frozen population

Use exactly the middle-phase state from each of the 30 archived episodes in the
selection-free Cosmos cohort:

- tasks, in frozen order: `BananaInBowlTask`, `RubiksCubeTask`,
  `MustardInLeftBinTask`, `SpoonInMugTask`, `MarkerInMugTask`, and
  `SmartphoneInBinTask`;
- environment seeds, in frozen order: 101, 103, 107, 109, and 113;
- branch seeds, in frozen order: 211, 223, 227, and 229;
- one middle state for every task-environment pair, irrespective of archived
  success, for six tasks times five episodes = 30 states.

The activation manifest must copy the exact state identifiers, branch steps,
input paths, and input hashes from the pre-outcome archival cohort manifest.
There is no replacement, filtering, or adaptation based on separation,
appearance, intervention behavior, or any archival/timing outcome. An
incomplete or mechanically failed run is not analyzable; recovery requires a
new version and all 30 states from scratch unless an outcome-blind resume rule
was already frozen and verifies every completed unit byte-for-byte.

## Frozen call schedule and request matrix

Call indices are chronological invocations of the post-guidance velocity
function. The float32 schedule observed in the excluded development smoke is:

| condition | active call indices | sigma at active call(s) |
|---|---:|---|
| `none` | `[]` | none |
| `call_0_only` | `[0]` | `0.9990000128746033` |
| `call_1_only` | `[1]` | `0.9369999766349792` |
| `call_2_only` | `[2]` | `0.8330000042915344` |
| `call_3_only` | `[3]` | `0.6240000128746033` |
| `all_calls` | `[0,1,2,3]` | all four values above |

Every request must report exactly this complete four-element `research_sigmas`
vector and the same `research_x0_sigmas` vector, in this order, plus the exact
requested and observed active-index lists. Any extra, missing, reordered, or
numerically different call is a gate failure, not a new condition.

For every state, execute exactly:

| request class | calls per state |
|---|---:|
| four native branches | 4 |
| one exact native replay per branch | 4 |
| six complete 4 recipient x 4 source timing grids | 96 |
| one extra exact replay of each `all_calls` diagonal cell | 4 |
| **total** | **108** |

Thus the evaluation contains exactly 30 x 108 = **3,240 model calls**. The
excluded smoke is not part of that count. “At least” repeats are not allowed;
adding or removing a request requires a new prospective protocol and manifest.
The manifest freezes the deterministic request order and output names.

For recipient branch `r` and source branch `q`, the recipient state, prompt,
image, joint/gripper input, recipient sampler seed, initial sampler state, and
recipient path noise are held fixed. Only the registered clean future target
changes from the self target (`q = r`) to the donor target (`q != r`). Source
identifiers and other `research_*` metadata may route the intervention but must
not enter the policy's ordinary sample builder or conditioning inputs.

## Runtime manipulation and no-leakage gates

The activation manifest must hash the runner, analyzer, launcher, server,
intervention implementation, attention/runtime dependencies, upstream sample
builder, image digest, checkpoint, and source commit. It must use a fresh,
isolated server or prove an empty registry before the first request. An excluded
state smoke must exercise every timing condition and both diagonal and
off-diagonal targets after the final code snapshot.

Every evaluation response must make the following checks possible, and the
launcher must fail closed before accepting the unit if any check fails:

1. The transformed policy input fingerprint and model-parameter probe are
   exact within state. The upstream sample builder is pinned and audited to use
   only prompt, image, joint state, and gripper state—not source IDs, target
   hashes, paths, seeds, or other `research_*` fields.
2. For a recipient, the initial-state and path-noise hashes are exact across
   all sources and timing conditions and match its native branch. Donor seeds
   never seed recipient sampling. Target hashes equal the named native source
   future; the self target equals the recipient native future.
3. At every active call and every selected future-vision coordinate, the tensor
   actually presented to the wrapped velocity function is bit-exact to
   `(1-sigma) * target + sigma * recipient_path_noise`. The returned future
   velocity actually presented to the sampler is bit-exact to
   `(sampler_future - target) / sigma`. Captures must use the active data path,
   not merely reconstruct intended values from request metadata.
4. The selected mask is exactly future vision frames 1 through the final
   latent frame. Its shape, count, and coordinate indices are frozen and exact.
   Inactive calls perform zero wrapper writes. All action-input and
   action-output coordinate write errors are exactly zero at every call.
5. `none` is a full no-op. For all four source labels it exactly reproduces the
   recipient native action, final future, call schedule, x0 vision/action
   traces, initial state, and path noise. Its output cannot depend on source
   metadata.
6. Each native replay and each extra `all_calls` diagonal replay is exact for
   action, target, final future, schedule, indices, masks, intervention-site
   captures, initial/path hashes, and x0 signatures.
7. All arrays and audit scalars are finite. Final-sampler distance from target
   is downstream behavior: it is always retained and summarized, but is not an
   admission threshold. No arm or state may be excluded because that residual
   is large.

The excluded smoke must pass these gates under the final immutable snapshot.
Changing code, thresholds, schedule, population, or request order after the
smoke requires a new version and a fresh excluded smoke before any evaluation
call.

## Frozen estimands

Flatten each 32 x 7 action chunk in float64. Let `N_r` be recipient `r`'s
native action and `A[t,r,q]` the action under timing condition `t`, recipient
`r`, and registered source `q`. Nearest-native retrieval uses Euclidean distance
and the frozen branch order 211, 223, 227, 229 as the deterministic tie break.
All six within-state native pair separations must be finite and greater than
`1e-12`; otherwise the complete run fails its evidence gate, the state remains
reported, and no denominator or row is silently dropped.

For each of the 12 ordered off-diagonal pairs `(r,q)`, define timing-matched
source effects:

```
retrieval_gain[t,r,q]
  = 1{nearest(A[t,r,q]) = q} - 1{nearest(A[t,r,r]) = q}

distance_gain[t,r,q]
  = (||A[t,r,r] - N_q|| - ||A[t,r,q] - N_q||) / ||N_r - N_q||
```

The comparator is the self-future clamp at the same call timing, so these
estimands distinguish donor identity from a generic clamp perturbation. Average
the 12 ordered pairs within state. Define the single-call state estimand as the
equal average of `call_0_only` through `call_3_only`. Define the sustained-minus-
single state contrast as `all_calls` minus that four-call average. Do this
separately for retrieval gain and distance gain.

Also report, without replacing the matched primary estimands:

- each timing condition's complete 4 x 4 source-retrieval accuracy; under a
  source-label permutation within each recipient row its expectation is
  exactly 0.25 (and, with distinct natives, `none` is exactly 4/16 = 0.25);
- raw off-diagonal donor retrieval, native-referenced distance reduction,
  donor-axis projection, cosine alignment, normalized orthogonal residual, and
  final target-future distance;
- exact tie counts/margins, minimum native separation, task summaries,
  leave-one-task-out summaries, and the full ordered call profile.

Native-separation quartiles are descriptive. Freeze their three boundaries
globally from all 30 x 12 = 360 directed native separations before within-state
aggregation; assign every pair deterministically, retain boundary ties, and
report pair/state/task denominators. They never control admission or condition
selection.

## Analysis, bootstrap, and decisions

For every estimand, average repeated directions within state, average the five
states within each task, and then average the six task means. Tasks are the
top-level independent units; branches, sources, directions, and timing cells
are repeated measures.

Use NumPy `Generator(PCG64(20260903))` to create one shared table of 10,000
hierarchical draws. In each draw, sample six tasks with replacement; for each
sampled task occurrence, sample five of that task's states with replacement;
then recompute the equal-task estimate. Use the same draw table for every
metric and contrast. Report the point estimate and the 2.5th and 97.5th
percentiles as the two-sided 95% percentile interval, retaining unrounded
values in JSON.

The prespecified primary statement, “a donor-specific single-call action effect
is detectable on average,” is allowed only if the complete runtime/control gate
passes and the 2.5th-percentile bounds for both average-single retrieval gain
and average-single distance gain are strictly greater than zero. Because the
statement requires both component nulls to be rejected, this is an
intersection-union gate; no multiplicity adjustment between its two required
components is needed.

The prespecified sustained-strength statement is allowed only if the
2.5th-percentile bounds for both `all_calls - mean(single_calls)` matched
contrasts are strictly greater than zero. Report the contrast regardless. Do
not substitute a favorable component or metric if the conjunction fails.

For timing-local follow-up, test the four call-specific conjunctions at
familywise alpha 0.05. For each call and component metric, form the one-sided
null-centered bootstrap p-value

```
p = (1 + count((theta_star - theta_hat) >= theta_hat)) / 10001,
```

for `H0: theta <= 0` versus `H1: theta > 0`, using the same 10,000 hierarchical
draws. The raw p-value for a call's conjunctive donor-specific effect is the
maximum of its retrieval-gain and distance-gain component p-values. Apply Holm's
step-down procedure across exactly these four call-level p-values: sort them,
compare in order with `0.05/4`, `0.05/3`, `0.05/2`, and `0.05`, stop at the
first non-rejection, and report both raw and monotone Holm-adjusted p-values.
Individual component p-values and intervals are descriptive unless an
additional multiplicity family was prospectively frozen.

The primary, sustained-strength, and timing-local families are labeled
separately; no omnibus familywise-error claim across those three tiers is made.
Always report six task means and all six leave-one-task-out estimates because
there are only six top-level clusters.

No equivalence test is specified: no scientifically meaningful action-scale
equivalence margin was available independently of these outcomes. Failure of a
lower-bound or Holm gate means only that the corresponding positive effect was
not detected under this design. It cannot support “no effect,” “necessity,”
“only sustained intervention works,” or localization of a natural mechanism.

## Required outputs

The analyzer must refuse incomplete, duplicate, unexpected, nonfinite, or
hash-mismatched inputs and must never convert missing/degenerate values into a
smaller analysis set. Emit machine-readable JSON and CSV, a compact LaTeX
table, and a plot containing all six timing conditions with task-level points.
Record exact counts, exclusions (required to be zero), gates, component and
composite p-values, confidence intervals, task/LOTO results, quartile
boundaries/denominators, controls, and final-target residual distributions.

Before launch, an independent outcome-blind audit must sign the immutable
manifest, code closure, excluded-smoke artifact, and the checklist accompanying
this protocol.

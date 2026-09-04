# Prospective Cosmos 3 single-call timing protocol v2: action-shape amendment

Status: **prospective and inactive.** This clarification was written after the
excluded timing-v4 smoke, but before any admitted timing-evaluation call or
outcome. Only control and schema fields from that excluded smoke were inspected.
Timing v4 is permanently smoke-only and non-admitted.

The v2 protocol says “32 x 7 action chunk,” but the pinned released Cosmos 3
DROID policy returns a 32 x 8 chunk: seven joint coordinates followed by one
gripper coordinate at every timestep. The frozen timing-v4 runner and analyzer
would have flattened all 256 returned coordinates. That mismatch is a
prespecification defect, so timing v4 received an independent **NO-GO** despite
passing its runtime manipulation controls.

## Exact action schema and estimand clarification

Every native, replay, self, donor, and no-op response in the superseding study
must contain an action array of exact shape `[32, 8]`, with exactly 256 finite
numeric coordinates. The last coordinate is the released server's gripper
output after its pinned output conversion; it is part of the policy action and
must not be dropped, rescaled, reweighted, or analyzed separately in the frozen
primary estimands.

Replace the phrase “Flatten each 32 x 7 action chunk in float64” in the v2
protocol with the following operative rule:

> Flatten each full 32 x 8 policy action chunk (seven joint coordinates plus
> one gripper coordinate per timestep) to 256 coordinates in float64.

All Euclidean distances, nearest-native retrievals, normalized donor-distance
gains, directional metrics, task/state averages, bootstrap intervals, and
tests otherwise remain exactly as specified. This amendment changes no
population, request order or count, intervention, comparator, timing condition,
sigma, metric formula, hierarchy, random seed, multiplicity procedure,
evidence criterion, or claim boundary.

## Required fail-closed gates

The superseding manifest must freeze `action_shape: [32, 8]` and
`action_coordinate_count: 256`. Before accepting any response, the runner must
require exact shape `[32, 8]` and all 256 coordinates finite. It must report and
gate an exact census of 108 shape-valid response actions per state, including
the eight replay responses that are not otherwise retained as raw action rows.

The analyzer must independently require shape `[32, 8]` and finite values for
all four stored native actions and all 96 stored timing-grid actions before
recomputing metrics. It must also require the runner's exact 108-response shape
census and may not reshape, truncate, pad, drop, or infer a missing coordinate.
Its completeness output must report 30 x 108 = 3,240 shape-valid response
actions for the admitted cohort and zero shape failures or exclusions.

Tests must fail on `[32, 7]`, `[32, 9]`, transposed `[8, 32]`, ragged, empty,
missing, null, NaN, or infinite action payloads. A valid `[32, 8]` fixture must
independently verify that changing only the gripper coordinate changes the
flattened distance, proving that all eight coordinates enter the estimand.

## Required supersession

The new content-addressed manifest must record and permanently exclude:

- timing-v4 evaluation manifest `cosmos3-timing-89823a363d824c4a`, SHA-256
  `5c4e4d50c8d0132b0ac8d57fe509784b6badc28441c004f0a10311f61deef073`;
- timing-v4 smoke manifest `cosmos3-timing-excluded-smoke-ac67e566390c2204`,
  SHA-256
  `5c924e46fd403fcb32f71367531dfc3f63436099ada6d927e7462ca117c5e7f6`;
- timing-v4 excluded smoke artifact SHA-256
  `17ce67034159bb2d98892e64ece8f3e88e5a6785e9652cf4dea4c07df75e4e66`;
- timing-v4 snapshot checksum-list SHA-256
  `d764bf5b949cfc56f33d16f55db256c8fd63271400b20111eabf6e0274514513`;
  and
- the action-shape prespecification mismatch as the sole launch blocker, with
  zero admitted timing-v4 evaluation calls.

Activation requires a fresh versioned root, immutable snapshot, dedicated
empty-registry server, content-addressed evaluation and excluded-smoke
manifests, and a fresh excluded-state 108-call smoke under the exact final code.
That smoke must pass the original v2 checklist, the structural-null amendment,
and every action-shape gate above before a new independent outcome-blind
launch audit may issue GO.

# Cosmos 3 future-strength dose response: prospective v2 amendment

Status: **prospective and inactive.** This amendment was written without
inspecting any archival-v7 or dose-response evaluation outcome and before any
dose-response implementation or model call. It preserves the original protocol
at SHA-256 `7b559902b640610d88f54b1953151249118ad6d162aa6389ac3cbcaca01cb0cc`
and becomes operative only if a later content-addressed manifest incorporates
both documents verbatim.

The original 92-call arithmetic is unchanged and exact:
`4 native + 4 native replay + 4 none + 4 self + 4 self replay +
(12 pairs x 5 alpha values) + 12 midpoint replay = 92` calls per state and
`30 x 92 = 2,760` admitted calls. This amendment removes analysis and control
ambiguities; it changes no population, intervention sites, alpha grid, request
count, or claim scope.

## Exact action metrics

Use the full released policy action with exact shape `[32, 8]`: seven joint
coordinates plus one gripper coordinate per timestep. Every one of the 92
response actions per state must contain exactly 256 finite coordinates. Flatten
all 256 coordinates in float64 without truncation, padding, rescaling, or
coordinate-specific weighting.

For state `s`, ordered recipient/donor pair `(r,q)`, and
`a in (0, 0.25, 0.50, 0.75, 1)`, let `N_r` and `N_q` be the unconstrained
native actions, `A[s,r,q,a]` the interpolated-clamp action, and
`u = N_q - N_r`. Require `d = ||u||_2` finite and strictly greater than
`1e-12` for all 12 ordered pairs. A degenerate axis fails the complete evidence
gate; it never drops a pair or state.

The frozen pair metrics are:

```text
distance_reduction = (d - ||A - N_q||_2) / d
donor_projection   = dot(A - N_r, u) / d^2
cosine_alignment   = dot(A - N_r, u) / (||A - N_r||_2 d)
orthogonal_residual_normalized
                   = ||(A - N_r) - donor_projection u||_2 / d
```

Set cosine alignment to exactly zero only when `||A-N_r||_2 <= 1e-12`.
Nearest-native retrieval uses Euclidean distance to all four native actions and
the frozen seed order 211, 223, 227, 229 as the deterministic exact-tie break.
Report every tie and top-two margin.

## Primary and alpha-profile estimands

Fit an ordinary least-squares intercept and slope separately to each ordered
pair's five distance-reduction values. With `abar=0.5`, the exact pair slope is

```text
beta[s,r,q] = sum_a ((a - 0.5) * distance_reduction[s,r,q,a]) / 0.625
```

because `sum_a (a-0.5)^2 = 0.625`. Average the 12 pair slopes equally to obtain
one state slope. Equivalently, average the 12 pairs at each alpha first and fit
the same intercept/slope; the analyzer must test this equality.

For each state, also average each alpha equally over the 12 ordered pairs.
Define the endpoint contrast as the state mean at alpha 1 minus alpha 0 and the
four adjacent contrasts as `mean(a_next)-mean(a_previous)` in ascending alpha
order. Define projection slope and endpoint contrast by the same formulas.
Donor-identification rate is the mean of all 12 ordered-pair indicators at each
alpha. The per-state nondecreasing fraction is the fraction of the 12 pair
profiles whose distance reduction is numerically nondecreasing across all five
alphas; it is descriptive and pairs are not independent inferential units.

The sole primary evidence criterion is the distance-reduction slope's
hierarchical 95% lower bound strictly exceeding zero. If it passes with every
runtime/completeness gate, the allowed wording is **positive linear dose trend
in donor-directed action distance under the imposed all-call future
interpolation**. A slope alone does not license “monotonic,” “linear mechanism,”
natural mediation, physical success, or a claim about every task or pair.

A stronger adjacent-step statement is a separate intersection-union gate. It
passes only if the hierarchical 95% lower bound is strictly greater than zero
for each of the four adjacent distance-reduction contrasts. Only then may the
report say that the **task-weighted mean profile increased strictly at every
adjacent alpha step**. No multiplicity adjustment is required among components
of this conjunction because all four must pass. If it fails, the report may
describe a numerically nondecreasing sample mean profile, if observed, but may
not call the response monotonic or infer no effect at a failed step.

Endpoint, projection, cosine, orthogonal-residual, retrieval, pairwise-profile,
task, and leave-one-task-out summaries remain prespecified secondary estimates.
Their intervals are descriptive unless another prospective multiplicity
procedure is frozen before outcomes. No omnibus familywise-error claim is made
across the primary and adjacent-step tiers.

## Exact hierarchical bootstrap

First form one value per state for every estimand. Compute each task mean over
its five states and the point estimate as the equal average of the six task
means. Episodes and states are one-to-one here, so the hierarchy has exactly
two stochastic resampling stages: task, then that task's episode/state.

Create one shared 10,000-draw table with NumPy
`Generator(PCG64(20260903))`. In every draw, sample six tasks with replacement;
for each sampled task occurrence independently sample five of that task's five
states with replacement; then average exactly as for the point estimate. Reuse
this identical task/state draw table for every alpha, metric, slope, and
contrast. Report the 2.5th and 97.5th percentiles using
`numpy.quantile(..., method="linear")`, with unrounded JSON values.

A pooled 30-state point estimate equals the equal-task point estimate because
the design is balanced. Any state-weighted result must be labeled a descriptive
sensitivity; treating 30 states or 360 pairs as independent cannot replace the
frozen hierarchical inference. Always report all six task means and six
leave-one-task-out point estimates.

## Endpoint construction and no-leakage controls

Use the exact ordered alpha grid `[0, 0.25, 0.50, 0.75, 1.00]`. For interior
alphas, compute `F_A + alpha * (F_B - F_A)` in the registered target tensor's
native dtype and this fixed operation order. Do not obtain endpoint targets by
floating-point cancellation: special-case alpha 0 as the registered recipient
tensor `F_A` and alpha 1 as the registered donor tensor `F_B`, with exact target
hash identity.

For every recipient and each of its three donor labels, the alpha-0 response
must be bit-exact to the corresponding explicit self-clamp response in action,
final future, four-call x0 vision/action trace, schedule, mask, intervention-
site capture, recipient initial state, and path noise. The three alpha-0
responses must be mutually identical under a behavior signature that excludes
only intentional source-routing metadata. This is the frozen negative control
that donor ID/hash/registry routing cannot affect the model when its target
reduces to `F_A`.

At alpha 1, the target hash must be exactly the registered donor future hash.
At every interior alpha, store and revalidate recipient, donor, interpolation,
current-frame, and future-mask hashes plus the exact elementwise interpolation
audit. Current-frame and all other nonfuture coordinates must equal the
recipient target exactly. Across alpha within pair, recipient seed,
initial-state hash, path-noise hash, input fingerprint, prompt/image/proprio,
model probe, solver, schedule, and all nonfuture/action coordinates remain
exact.

## Activation gates

The content-addressed manifest, runner, analyzer, launcher, and excluded-smoke
validator must fail closed on any action-shape/count mismatch; missing, extra,
duplicate, reordered, null, NaN, or infinite response; target/interpolation or
identity mismatch; native-axis degeneracy; or failed control. They must report
exactly 92 shape-valid response actions per state, 2,760 for the complete
cohort, zero exclusions, and the exact request-class census.

The manifest must hash this amendment, the original protocol, all executable
dependencies, upstream sample builder, full checkpoint content, image, source
commit, 30 input states/assets, and the final excluded-smoke artifact. A fresh
isolated server with an empty registry and an exact 92-call excluded Bagels
smoke are mandatory. After smoke, restart the exact server and independently
verify its registry is empty again before the first admitted request. An
independent outcome-blind GO audit is mandatory before any admitted call. Any
post-smoke code, schema, threshold, or protocol change requires a new version
and fresh smoke.

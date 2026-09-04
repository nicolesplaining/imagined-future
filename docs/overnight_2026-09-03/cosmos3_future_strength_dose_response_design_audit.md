# Cosmos 3 future-strength dose response: outcome-blind design audit

Verdict: **NO-GO for the original protocol alone; design-GO for implementation
only when the v2 amendment is incorporated verbatim.** This is not a launch
authorization. No archival-v7 or dose-response evaluation outcome was inspected
and no dose-response model call or implementation was made for this audit.

Frozen inputs:

- Original protocol SHA-256
  `7b559902b640610d88f54b1953151249118ad6d162aa6389ac3cbcaca01cb0cc`.
- Prospective v2 amendment SHA-256
  `a02e3a74d0d5b7f4a9d72401c8f869519acf7a4f808067ee2a2180e68775158f`.

The request arithmetic is correct: 4 native + 4 native replay + 4 none + 4
self + 4 self replay + 60 off-diagonal alpha arms + 12 midpoint replays = 92
calls/state, and 30 states x 92 = 2,760 admitted calls. The population is the
selection-free 6-task x 5-episode middle-state cohort, with all 12 directed
pairs retained.

The original text was not implementation-safe because it left the primary OLS
construction and bootstrap seed unspecified, treated episode and its sole state
as if they were separate resampling levels, allowed a monotonicity claim from
nonnegative point estimates alone, did not require alpha-zero donor-metadata
invariance against the explicit self arm, and did not freeze endpoint rounding,
metric formulas/ties/native-axis handling, or the full 32 x 8 action schema.

The amendment resolves these issues by freezing pairwise OLS-with-intercept
slopes averaged 12 pairs to one state; a single shared 10,000-draw
`PCG64(20260903)` task-to-five-state bootstrap with linear percentile intervals;
exact action/metric formulas; `[32,8]`/256-coordinate gates; bit-exact alpha-zero
self and alpha-one donor endpoints; and a four-lower-bound intersection-union
gate for any strict adjacent-step statement. A primary pass supports only a
positive linear dose-trend statement for imposed action-space intervention.

Before launch, a future content-addressed manifest must hash both documents and
the complete implementation closure, enumerate exactly 30 states and 92 unique
request labels/state, and bind all inputs and checkpoint content. A fresh
excluded Bagels smoke must exercise every alpha/control/replay and pass exact
live-site, no-write, RNG, target/interpolation, endpoint, metadata-invariance,
action-shape, finiteness, and completeness gates. The exact server must then be
restarted with an independently verified empty registry and receive a separate
outcome-blind GO audit.

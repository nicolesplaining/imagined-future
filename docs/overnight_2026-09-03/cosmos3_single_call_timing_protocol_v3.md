# Prospective Cosmos 3 single-call timing protocol v3

Status: **prospective and inactive.** This amendment was frozen after the
excluded v2 runtime smoke failed closed and before any timing-evaluation call
or outcome. It incorporates the complete v2 protocol with SHA-256
`526d126a1ffe6cc8216f8be1a0aaa93732faf637932f8155eea3d024fcb38c57`
and changes only the structural representation rule below. The v2 snapshot,
manifest, and failed excluded smoke remain preserved and are not admitted to
the v3 evaluation.

## Exact structural-null rule

The server diagnostic `research_action_donor_projection` divides by the native
recipient-to-donor action distance. Its denominator is exactly zero when and
only when the recipient and source are the same native branch. The server emits
a floating-point NaN for this structurally undefined diagnostic. The v3 runner
must validate the zero-axis condition before canonicalization and then replace
only this exact top-level response field with JSON null. It must also emit the
Boolean sibling `research_action_donor_projection_applicable=false`.

The frozen request matrix therefore requires exactly:

- 24 structural-null timing-grid diagnostics: four diagonal cells under each
  of six timing conditions;
- four structural-null diagnostics in the extra all-calls diagonal replays;
- 72 finite, non-null diagnostics in the off-diagonal timing-grid cells; and
- no such field in the eight native/native-replay responses.

Thus exactly 28 of 100 intervention responses carry the structural null. A
null, NaN, or infinity at any other response path is a gate failure. A finite
projection on a diagonal, a null/nonfinite projection off diagonal, a missing
applicability flag, or a census different from 28/72/8 is also a gate failure.
No NaN or infinity may be serialized to a state artifact.

All actions, off-diagonal scientific metrics, final-target max-absolute and L2
residuals, timing/site captures, masks, initial/path hashes, nonwrite audits,
and confirmatory inputs remain finite-required exactly as in v2. The exception
does not change the population, 108-call request matrix, estimands, statistics,
decision rules, or claim boundary.


# Cosmos archival v7 final: independent raw-output audit

Verdict: **PASS**, with zero numerical, cohort, control, or frozen-summary
discrepancies.

This audit was unblinded only after the exact 90-file package was frozen mode `0444`.
The manifest SHA-256 was
`8ab294c7a581424cfa02faae1db85941e3f43c1e28100e68e21fa5f533703b9e`,
the output-inventory SHA-256 was
`948f443fcd44a34a94a2a93f938a81f89c8affc37d6a7b90def985104c28914e`,
the frozen summary SHA-256 was
`cb914fead7e5fdd1303eb0f8086c8db53b7c0a8b0db1cf36439112e771c89889`,
and the frozen state-row SHA-256 was
`eeef03a495ab0558a4b012b39e295a4f0ee3ea3223d2beeb048c323fc4689247`.

The independent audit reconstructed nearest-native classifications and all
directional quantities from the raw `[32,8]` action arrays without importing the
runner, protocol helpers, or frozen analyzer. It then independently repeated the
task-to-episode-to-state PCG64(20260903) 10,000-draw bootstrap, phase, per-task,
leave-one-task-out, permutation, global pair-quartile, secondary state-quartile,
denominator, and control analyses. Its JSON is
`/lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_archival_selection_v7_final/audit/independent_raw_audit.json`,
SHA-256
`8c6a162a06e134e5bbf4becc7948c6c23e590d41df66a906a75be769cae2747e`.
The separate 90-row exact CSV comparison is
`audit/independent_state_rows_audit.json`, SHA-256
`408da6d9360d88d3506b9d39937dfb081547a5ef2d2d33df19a564fa73fe8eaf`.

## Recomputed results

Equal-task point estimates and hierarchical 95% percentile intervals:

| Estimand | Mean | 95% CI |
|---|---:|---:|
| 4-way future-source retrieval | 1.000000 | [1.000000, 1.000000] |
| donor top-1 | 1.000000 | [1.000000, 1.000000] |
| shuffled-source top-1 | 0.000000 | [0.000000, 0.000000] |
| wrong-donor top-1 | 0.000000 | [0.000000, 0.000000] |
| Gaussian top-1 | 0.000000 | [0.000000, 0.000000] |
| donor distance reduction | 0.763417 | [0.733332, 0.792040] |
| donor cosine alignment | 0.966942 | [0.956897, 0.975315] |
| donor normalized projection | 0.966340 | [0.956303, 0.974469] |
| donor normalized orthogonal residual | 0.226765 | [0.199573, 0.255081] |
| Gaussian normalized projection | 0.017362 | [0.010187, 0.024925] |
| Gaussian distance reduction | -0.068477 | [-0.091347, -0.046384] |

The four-label chance rate is exactly `0.25`. The prespecified source-label Monte
Carlo permutation audit gave `p = 1/10001 = 0.00009999`; its null mean was
`0.250055` and null 95% interval `[0.224306, 0.276389]`.

All 1,080 off-diagonal donor axes were valid and nondegenerate. The global
pair-separation quartiles contained exactly 270 arms each; donor distance-reduction
means increased from `0.64110` to `0.75152`, `0.80769`, and `0.85538`, with every
quartile interval above zero. This is descriptive effect modification, not evidence
that separation was randomized. Phase means were `0.76537` early, `0.76011` middle,
and `0.76477` late, with all phase intervals above zero. Leave-one-task-out distance
means ranged from `0.75402` to `0.77084`; retrieval remained exactly one throughout.

Controls were exact: 4,320/4,320 action-coordinate audits, 15,840/15,840 model-input
sites, 15,840/15,840 returned-velocity sites, and native/self/donor/none replay counts
of 360/360/1,080/360. All frozen evidence criteria passed.

## Claim boundary

The result supports a highly reproducible imposed action-space steering effect in
this archival lossy-input cohort. It does not establish natural mediation,
necessity, semantic planning, fresh physical success, or endpoint behavior. Perfect
nearest-native retrieval should be reported together with continuous displacement,
Gaussian, wrong-label, phase, LOTO, and separation-stratified controls rather than as
a standalone mechanistic equivalence claim.

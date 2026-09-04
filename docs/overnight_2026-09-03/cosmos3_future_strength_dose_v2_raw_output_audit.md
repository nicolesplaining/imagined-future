# Cosmos 3 future-strength dose v2: independent raw-output audit

Verdict: **PASS**. The independently recomputed raw results agree with the frozen
analysis with zero discrepancies. This audit opened scientific payloads only after
the exact 30-file package was mode-frozen and the frozen summary was hash-sealed.

## Frozen evidence

- Manifest: `cosmos3-dose-c00d81db8d910603`, SHA-256
  `1a37bdba796b580c905970c441aededccf4193c51fe370f0f3557eccd5a9ee7d`.
- Raw inventory: SHA-256
  `8fa279868725119522bbcf1ecf5c3a3375833aadf3110b63d89a099720a600d8`;
  exact 30 manifest-derived files, all mode `0444` and post-chmod rehashed.
- Frozen summary: SHA-256
  `741307dff8ffac6458a65678db1f713aee3d74eb2b4ed11436d8345ad4c44a69`.
- Independent audit JSON: SHA-256
  `c12cb8080df4af4f179a231cdd040936f9d693b8dd6c4b3bdb08be52f57c11ed`,
  mode `0444` at
  `/lambda/nfs/imagined-future/results/overnight_2026_09_03/cosmos3_future_strength_dose_v2/evaluation/audit/independent_raw_audit.json`.
- Independent auditor: SHA-256
  `a9ac9cb1eca2e44c7194b0ac1bfad8efe414a2a0c88693a94d5ff691eea2e927`;
  inventory-gate helper: SHA-256
  `59351b76ff45c02ca2ef26f006b3e10edf51dd9077173df96cdd4da59991707e`.
  Both were frozen mode `0444` before cohort completion. Neither imports the
  frozen runner, analyzer, or scientific helper.

## Independent recomputation

The audit revalidated all 2,760 finite `[32,8]` action chunks, recomputed all
1,800 dose-cell action metrics directly from raw actions, constructed 360
recipient-donor profiles, averaged pairwise OLS-with-intercept slopes into 30
state rows, and reproduced the shared 10,000-draw `PCG64(20260903)`
task-to-five-state hierarchical bootstrap. It also reproduced every alpha
profile, endpoint and adjacent contrast, per-task result, leave-one-task-out
result, and nested summary field.

| Estimand | Estimate | Hierarchical 95% CI |
|---|---:|---:|
| Primary distance-reduction slope | 0.899434 | [0.851514, 0.936352] |
| Distance endpoint, alpha 1 minus alpha 0 | 0.784550 | [0.746037, 0.812606] |
| Projection slope | 1.104676 | [1.072601, 1.135047] |
| Projection endpoint, alpha 1 minus alpha 0 | 0.970414 | [0.945496, 0.997806] |

The task-weighted distance-reduction means by alpha were `-0.021644`,
`0.032795`, `0.231858`, `0.712281`, and `0.762906`. All four adjacent mean
increases had positive hierarchical lower bounds:

- alpha 0 to 0.25: 0.054439 [0.041823, 0.067447]
- alpha 0.25 to 0.50: 0.199063 [0.168768, 0.225568]
- alpha 0.50 to 0.75: 0.480422 [0.426739, 0.534382]
- alpha 0.75 to 1.00: 0.050626 [0.039326, 0.062492]

All six task slopes were positive (range 0.822359 to 0.943460), all six
leave-one-task-out slopes were positive (range 0.890629 to 0.914849), and all
30 state slopes and 360 pairwise slopes were positive. The equal-task and pooled
30-state primary points were identical because the design has five states per
task.

## Controls and claim boundary

All independently checked controls pass: 2,760/2,760 action shapes; 2,520
intervention responses; 2,400 active responses and 9,600 active sites; 360
structural projection nulls, 2,160 finite projections, and 240 native-absent
projections; exact action nonwrite, replay, source/target, alpha-zero, and
recipient-RNG/schedule identities. Model-input clamp and returned-velocity site
errors were exactly zero, within the frozen `1e-7` tolerance.

The supported statement is a positive, graded donor-directed **action-space**
response under experimentally imposed all-denoising-call latent-future
interpolation. The task-weighted mean profile increased at every adjacent alpha
step. This is not evidence that every pair is monotone: only 257/360 pair
profiles (71.39%) were nondecreasing at every step. It also does not establish
natural mediation, policy-native future strength, physical task success, or
cross-model equivalence. Final-sampler target residuals remain descriptive only;
their maximum absolute value was 0.028369, while live intervention-site errors
were exactly zero.
